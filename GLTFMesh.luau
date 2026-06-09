--!strict
-- GLTFMesh
-- Turns a parsed glTF primitive into GPU-ready buffers:
--   * one interleaved vertex buffer, stride 72 bytes:
--       pos.xyz (0) | normal.xyz (12) | uv0.xy (24) | tangent.xyzw (32)
--       | uv1.xy (48) | color.rgba (56)
--   * one index buffer (uint16 unless an index exceeds 65535, then uint32)
-- Every primitive shares ONE vertex layout (one GPU pipeline). Missing
-- attributes get neutral fallbacks so any glTF works:
--   * NORMAL missing      -> smooth normals are synthesised
--   * TEXCOORD_0 missing  -> (0,0)
--   * TANGENT missing     -> computed from positions+uv0 (or a safe default)
--   * TEXCOORD_1 missing  -> copies uv0
--   * COLOR_0 missing     -> white (1,1,1,1)

local GLBParser = require('GLBParser')

type GLBData = GLBParser.GLBData
type Primitive = GLBParser.Primitive

export type DrawItem = {
  vertexData: buffer, -- interleaved, stride = vertexStride
  vertexCount: number,
  vertexStride: number, -- bytes per vertex (32)
  indexData: buffer,
  indexCount: number,
  indexFormat: string, -- "uint16" | "uint32"
  materialIndex: number?, -- glTF 0-based material index, or nil
}

-- CPU-side skinning inputs for a primitive: flat bind-pose arrays. The static
-- attributes (tangent/uv1/color) are carried verbatim so the skinned vertex
-- buffer keeps the full layout; only positions/normals are re-skinned per frame.
export type SkinData = {
  positions: { number }, -- bind-pose, flat xyz
  normals: { number }, -- bind-pose, flat xyz
  tangents: { number }, -- flat xyzw (per-vertex), supplied or computed
  uvs: { number }, -- flat uv0 (may be empty)
  uvs1: { number }, -- flat uv1 (may be empty -> falls back to uv0)
  colors: { number }, -- flat rgba 0..1 (may be empty -> white)
  joints: { number }, -- 4 joint indices per vertex (0-based)
  weights: { number }, -- 4 weights per vertex
}

local FLOAT = 4
-- pos(3) + normal(3) + uv0(2) + tangent(4) + uv1(2) + color(4) = 18 floats
local STRIDE = 18 * FLOAT -- 72 bytes
-- Byte offsets of each attribute within a vertex.
local OFF_POS = 0
local OFF_NORMAL = 12
local OFF_UV0 = 24
local OFF_TANGENT = 32
local OFF_UV1 = 48
local OFF_COLOR = 56

local GLTFMesh = {}

-- Shared vertex layout, exported so other modules (e.g. the skinning writer in
-- GLBModel) can build vertex buffers that match buildPrimitive's output exactly.
GLTFMesh.STRIDE = STRIDE
GLTFMesh.OFF_POS = OFF_POS
GLTFMesh.OFF_NORMAL = OFF_NORMAL
GLTFMesh.OFF_UV0 = OFF_UV0
GLTFMesh.OFF_TANGENT = OFF_TANGENT
GLTFMesh.OFF_UV1 = OFF_UV1
GLTFMesh.OFF_COLOR = OFF_COLOR

-- Little-endian integer writes composed from writeu8 (mirrors the loader's read
-- hedge: only u8 byte ops are assumed present on every Rive runtime build).
local function writeU16LE(b: buffer, off: number, v: number)
  buffer.writeu8(b, off, v % 256)
  buffer.writeu8(b, off + 1, math.floor(v / 256) % 256)
end
local function writeU32LE(b: buffer, off: number, v: number)
  buffer.writeu8(b, off, v % 256)
  buffer.writeu8(b, off + 1, math.floor(v / 256) % 256)
  buffer.writeu8(b, off + 2, math.floor(v / 65536) % 256)
  buffer.writeu8(b, off + 3, math.floor(v / 16777216) % 256)
end

-- Smooth normals: accumulate triangle face normals per vertex, then normalise.
-- `positions` and `indices` are flat 0-based arrays; returns flat {nx,ny,nz,...}.
function GLTFMesh.computeNormals(
  positions: { number },
  indices: { number }
): { number }
  local vcount = math.floor(#positions / 3)
  local nx, ny, nz = {}, {}, {}
  for i = 1, vcount do
    nx[i], ny[i], nz[i] = 0, 0, 0
  end
  local triCount = math.floor(#indices / 3)
  for t = 0, triCount - 1 do
    local i0 = indices[t * 3 + 1]
    local i1 = indices[t * 3 + 2]
    local i2 = indices[t * 3 + 3]
    local a, b, c = i0 * 3, i1 * 3, i2 * 3
    local e1x = positions[b + 1] - positions[a + 1]
    local e1y = positions[b + 2] - positions[a + 2]
    local e1z = positions[b + 3] - positions[a + 3]
    local e2x = positions[c + 1] - positions[a + 1]
    local e2y = positions[c + 2] - positions[a + 2]
    local e2z = positions[c + 3] - positions[a + 3]
    local fx = e1y * e2z - e1z * e2y
    local fy = e1z * e2x - e1x * e2z
    local fz = e1x * e2y - e1y * e2x
    nx[i0 + 1] += fx
    ny[i0 + 1] += fy
    nz[i0 + 1] += fz
    nx[i1 + 1] += fx
    ny[i1 + 1] += fy
    nz[i1 + 1] += fz
    nx[i2 + 1] += fx
    ny[i2 + 1] += fy
    nz[i2 + 1] += fz
  end
  local out: { number } = {}
  for i = 1, vcount do
    local x, y, z = nx[i], ny[i], nz[i]
    local len = math.sqrt(x * x + y * y + z * z)
    if len > 1e-8 then
      out[(i - 1) * 3 + 1] = x / len
      out[(i - 1) * 3 + 2] = y / len
      out[(i - 1) * 3 + 3] = z / len
    else
      out[(i - 1) * 3 + 1] = 0
      out[(i - 1) * 3 + 2] = 0
      out[(i - 1) * 3 + 3] = 1
    end
  end
  return out
end

-- Compute per-vertex tangents (xyzw, w = bitangent handedness) from positions,
-- normals, uvs and indices using the standard Lengyel method. Returns a flat
-- {tx,ty,tz,tw,...}. Used only when the glTF has no TANGENT attribute but does
-- have UVs (needed for normal mapping). If uvs are empty, returns a safe default.
function GLTFMesh.computeTangents(
  positions: { number },
  normals: { number },
  uvs: { number },
  indices: { number }
): { number }
  local vcount = math.floor(#positions / 3)
  local out: { number } = {}
  if #uvs < 2 then
    -- No UVs: tangents are meaningless. Emit a stable default per vertex.
    for i = 0, vcount - 1 do
      out[i * 4 + 1] = 1
      out[i * 4 + 2] = 0
      out[i * 4 + 3] = 0
      out[i * 4 + 4] = 1
    end
    return out
  end
  -- Accumulators for tangent (T) and bitangent (B) per vertex.
  local tx, ty, tz = {}, {}, {}
  local bx, by, bz = {}, {}, {}
  for i = 1, vcount do
    tx[i], ty[i], tz[i] = 0, 0, 0
    bx[i], by[i], bz[i] = 0, 0, 0
  end
  local triCount = math.floor(#indices / 3)
  for t = 0, triCount - 1 do
    local i0 = indices[t * 3 + 1]
    local i1 = indices[t * 3 + 2]
    local i2 = indices[t * 3 + 3]
    local p0 = i0 * 3
    local p1 = i1 * 3
    local p2 = i2 * 3
    local u0 = i0 * 2
    local u1 = i1 * 2
    local u2 = i2 * 2
    local e1x = positions[p1 + 1] - positions[p0 + 1]
    local e1y = positions[p1 + 2] - positions[p0 + 2]
    local e1z = positions[p1 + 3] - positions[p0 + 3]
    local e2x = positions[p2 + 1] - positions[p0 + 1]
    local e2y = positions[p2 + 2] - positions[p0 + 2]
    local e2z = positions[p2 + 3] - positions[p0 + 3]
    local du1 = uvs[u1 + 1] - uvs[u0 + 1]
    local dv1 = uvs[u1 + 2] - uvs[u0 + 2]
    local du2 = uvs[u2 + 1] - uvs[u0 + 1]
    local dv2 = uvs[u2 + 2] - uvs[u0 + 2]
    local denom = du1 * dv2 - du2 * dv1
    local r = if math.abs(denom) > 1e-12 then 1 / denom else 0
    local sx = (dv2 * e1x - dv1 * e2x) * r
    local sy = (dv2 * e1y - dv1 * e2y) * r
    local sz = (dv2 * e1z - dv1 * e2z) * r
    local tcx = (du1 * e2x - du2 * e1x) * r
    local tcy = (du1 * e2y - du2 * e1y) * r
    local tcz = (du1 * e2z - du2 * e1z) * r
    for _, vi in { i0, i1, i2 } do
      tx[vi + 1] += sx
      ty[vi + 1] += sy
      tz[vi + 1] += sz
      bx[vi + 1] += tcx
      by[vi + 1] += tcy
      bz[vi + 1] += tcz
    end
  end
  for i = 0, vcount - 1 do
    local nX = normals[i * 3 + 1] or 0
    local nY = normals[i * 3 + 2] or 0
    local nZ = normals[i * 3 + 3] or 1
    local tX, tY, tZ = tx[i + 1], ty[i + 1], tz[i + 1]
    -- Gram-Schmidt orthogonalise T against N.
    local ndt = nX * tX + nY * tY + nZ * tZ
    tX = tX - nX * ndt
    tY = tY - nY * ndt
    tZ = tZ - nZ * ndt
    local len = math.sqrt(tX * tX + tY * tY + tZ * tZ)
    if len > 1e-8 then
      tX, tY, tZ = tX / len, tY / len, tZ / len
    else
      tX, tY, tZ = 1, 0, 0
    end
    -- Handedness: sign of dot(cross(N, T), B).
    local cx = nY * tZ - nZ * tY
    local cy = nZ * tX - nX * tZ
    local cz = nX * tY - nY * tX
    local handed = (cx * bx[i + 1] + cy * by[i + 1] + cz * bz[i + 1]) < 0
    out[i * 4 + 1] = tX
    out[i * 4 + 2] = tY
    out[i * 4 + 3] = tZ
    out[i * 4 + 4] = if handed then -1 else 1
  end
  return out
end

-- Build one DrawItem from a primitive. Returns nil if POSITION is missing.
function GLTFMesh.buildPrimitive(glb: GLBData, prim: Primitive): DrawItem?
  local posIdx = prim.attributes.POSITION
  if posIdx == nil then
    return nil
  end
  -- readers are 1-based over glb.accessors, which mirrors glTF order, so +1.
  local positions = GLBParser.readAccessorFloats(glb, posIdx + 1)
  if not positions then
    return nil
  end
  local vcount = math.floor(#positions / 3)

  local indices: { number }
  if prim.indices ~= nil then
    indices = GLBParser.readIndices(glb, prim.indices + 1) or {}
  else
    indices = {}
    for i = 0, vcount - 1 do
      indices[i + 1] = i
    end
  end

  local normals: { number }? = nil
  if prim.attributes.NORMAL ~= nil then
    normals = GLBParser.readAccessorFloats(glb, prim.attributes.NORMAL + 1)
  end
  if not normals then
    normals = GLTFMesh.computeNormals(positions, indices)
  end
  local nrm = normals :: { number }

  local uvs: { number }? = nil
  if prim.attributes.TEXCOORD_0 ~= nil then
    uvs = GLBParser.readAccessorFloats(glb, prim.attributes.TEXCOORD_0 + 1)
  end

  -- Second UV set (lightmap / AO). Falls back to uv0 when absent.
  local uvs1: { number }? = nil
  if prim.attributes.TEXCOORD_1 ~= nil then
    uvs1 = GLBParser.readAccessorFloats(glb, prim.attributes.TEXCOORD_1 + 1)
  end

  -- Vertex colours (COLOR_0). May be VEC3 or VEC4; default white. glTF allows
  -- normalised ubyte/ushort here, which readAccessorFloats returns as raw
  -- integers, so we normalise those component types to 0..1.
  local colors: { number }? = nil
  local colorComps = 4
  local colorScale = 1
  if prim.attributes.COLOR_0 ~= nil then
    local accIdx = prim.attributes.COLOR_0 + 1
    colors = GLBParser.readAccessorFloats(glb, accIdx)
    local acc = glb.accessors[accIdx]
    if acc then
      colorComps = if acc.accessorType == 'VEC3' then 3 else 4
      if acc.componentType == 5121 then
        colorScale = 1 / 255 -- normalised UBYTE
      elseif acc.componentType == 5123 then
        colorScale = 1 / 65535 -- normalised USHORT
      end
    end
  end

  -- Tangents: use the supplied TANGENT (VEC4) when present, otherwise compute
  -- them from positions + uv0 so normal mapping still works.
  local tangents: { number }? = nil
  if prim.attributes.TANGENT ~= nil then
    tangents = GLBParser.readAccessorFloats(glb, prim.attributes.TANGENT + 1)
  end
  if not tangents then
    tangents =
      GLTFMesh.computeTangents(positions, nrm, uvs or {}, indices)
  end
  local tan = tangents :: { number }

  -- interleave
  local vbo = buffer.create(vcount * STRIDE)
  for v = 0, vcount - 1 do
    local base = v * STRIDE
    -- position
    buffer.writef32(vbo, base + OFF_POS + 0, positions[v * 3 + 1])
    buffer.writef32(vbo, base + OFF_POS + 4, positions[v * 3 + 2])
    buffer.writef32(vbo, base + OFF_POS + 8, positions[v * 3 + 3])
    -- normal
    buffer.writef32(vbo, base + OFF_NORMAL + 0, nrm[v * 3 + 1] or 0)
    buffer.writef32(vbo, base + OFF_NORMAL + 4, nrm[v * 3 + 2] or 0)
    buffer.writef32(vbo, base + OFF_NORMAL + 8, nrm[v * 3 + 3] or 1)
    -- uv0
    local u0 = if uvs then (uvs[v * 2 + 1] or 0) else 0
    local v0 = if uvs then (uvs[v * 2 + 2] or 0) else 0
    buffer.writef32(vbo, base + OFF_UV0 + 0, u0)
    buffer.writef32(vbo, base + OFF_UV0 + 4, v0)
    -- tangent (xyzw)
    buffer.writef32(vbo, base + OFF_TANGENT + 0, tan[v * 4 + 1] or 1)
    buffer.writef32(vbo, base + OFF_TANGENT + 4, tan[v * 4 + 2] or 0)
    buffer.writef32(vbo, base + OFF_TANGENT + 8, tan[v * 4 + 3] or 0)
    buffer.writef32(vbo, base + OFF_TANGENT + 12, tan[v * 4 + 4] or 1)
    -- uv1 (falls back to uv0)
    local u1 = if uvs1 then (uvs1[v * 2 + 1] or u0) else u0
    local v1 = if uvs1 then (uvs1[v * 2 + 2] or v0) else v0
    buffer.writef32(vbo, base + OFF_UV1 + 0, u1)
    buffer.writef32(vbo, base + OFF_UV1 + 4, v1)
    -- color (rgba), default white
    if colors then
      local cb = v * colorComps
      buffer.writef32(vbo, base + OFF_COLOR + 0, (colors[cb + 1] or 1) * colorScale)
      buffer.writef32(vbo, base + OFF_COLOR + 4, (colors[cb + 2] or 1) * colorScale)
      buffer.writef32(vbo, base + OFF_COLOR + 8, (colors[cb + 3] or 1) * colorScale)
      buffer.writef32(
        vbo,
        base + OFF_COLOR + 12,
        if colorComps == 4 then (colors[cb + 4] or 1) * colorScale else 1
      )
    else
      buffer.writef32(vbo, base + OFF_COLOR + 0, 1)
      buffer.writef32(vbo, base + OFF_COLOR + 4, 1)
      buffer.writef32(vbo, base + OFF_COLOR + 8, 1)
      buffer.writef32(vbo, base + OFF_COLOR + 12, 1)
    end
  end

  -- pick index width
  local maxIdx = 0
  for _, ix in ipairs(indices) do
    if ix > maxIdx then
      maxIdx = ix
    end
  end
  local use32 = maxIdx > 65535
  local icount = #indices
  local ibo: buffer
  if use32 then
    ibo = buffer.create(icount * 4)
    for i = 0, icount - 1 do
      writeU32LE(ibo, i * 4, indices[i + 1])
    end
  else
    ibo = buffer.create(icount * 2)
    for i = 0, icount - 1 do
      writeU16LE(ibo, i * 2, indices[i + 1])
    end
  end

  return {
    vertexData = vbo,
    vertexCount = vcount,
    vertexStride = STRIDE,
    indexData = ibo,
    indexCount = icount,
    indexFormat = if use32 then 'uint32' else 'uint16',
    materialIndex = prim.material,
  }
end

-- Build every primitive of every mesh into a flat draw list.
function GLTFMesh.buildAll(glb: GLBData): { DrawItem }
  local items: { DrawItem } = {}
  for _, mesh in ipairs(glb.meshes) do
    for _, prim in ipairs(mesh.primitives) do
      local item = GLTFMesh.buildPrimitive(glb, prim)
      if item then
        table.insert(items, item)
      end
    end
  end
  return items
end

-- Extract CPU skinning data for a primitive, or nil if it isn't skinned
-- (no JOINTS_0 / WEIGHTS_0). Bind-pose positions/normals are local (mesh space).
function GLTFMesh.buildSkinData(glb: GLBData, prim: Primitive): SkinData?
  local jIdx = prim.attributes.JOINTS_0
  local wIdx = prim.attributes.WEIGHTS_0
  local pIdx = prim.attributes.POSITION
  if jIdx == nil or wIdx == nil or pIdx == nil then
    return nil
  end
  local positions = GLBParser.readAccessorFloats(glb, pIdx + 1)
  if not positions then
    return nil
  end
  local vcount = math.floor(#positions / 3)
  local indices: { number }
  if prim.indices ~= nil then
    indices = GLBParser.readIndices(glb, prim.indices + 1) or {}
  else
    indices = {}
    for i = 0, vcount - 1 do
      indices[i + 1] = i
    end
  end
  local normals: { number }? = nil
  if prim.attributes.NORMAL ~= nil then
    normals = GLBParser.readAccessorFloats(glb, prim.attributes.NORMAL + 1)
  end
  if not normals then
    normals = GLTFMesh.computeNormals(positions, indices)
  end
  local nrm = normals :: { number }
  local uvs: { number } = {}
  if prim.attributes.TEXCOORD_0 ~= nil then
    uvs = GLBParser.readAccessorFloats(glb, prim.attributes.TEXCOORD_0 + 1)
      or {}
  end
  local uvs1: { number } = {}
  if prim.attributes.TEXCOORD_1 ~= nil then
    uvs1 = GLBParser.readAccessorFloats(glb, prim.attributes.TEXCOORD_1 + 1)
      or {}
  end
  -- Colours, normalised to 0..1 rgba (same logic as buildPrimitive).
  local colors: { number } = {}
  if prim.attributes.COLOR_0 ~= nil then
    local accIdx = prim.attributes.COLOR_0 + 1
    local raw = GLBParser.readAccessorFloats(glb, accIdx) or {}
    local acc = glb.accessors[accIdx]
    local comps = if acc and acc.accessorType == 'VEC3' then 3 else 4
    local scale = 1
    if acc then
      if acc.componentType == 5121 then
        scale = 1 / 255
      elseif acc.componentType == 5123 then
        scale = 1 / 65535
      end
    end
    local vc = if comps > 0 then math.floor(#raw / comps) else 0
    local function rc(i: number): number
      local x = raw[i]
      return if x ~= nil then (x :: number) else 1
    end
    for v = 0, vc - 1 do
      colors[v * 4 + 1] = rc(v * comps + 1) * scale
      colors[v * 4 + 2] = rc(v * comps + 2) * scale
      colors[v * 4 + 3] = rc(v * comps + 3) * scale
      colors[v * 4 + 4] = if comps == 4 then rc(v * comps + 4) * scale else 1
    end
  end
  -- Tangents: supplied or computed from bind-pose positions + uv0.
  local tangents: { number }? = nil
  if prim.attributes.TANGENT ~= nil then
    tangents = GLBParser.readAccessorFloats(glb, prim.attributes.TANGENT + 1)
  end
  if not tangents then
    tangents = GLTFMesh.computeTangents(positions, nrm, uvs, indices)
  end
  return {
    positions = positions,
    normals = nrm,
    tangents = tangents :: { number },
    uvs = uvs,
    uvs1 = uvs1,
    colors = colors,
    joints = GLBParser.readJoints(glb, jIdx + 1) or {},
    weights = GLBParser.readAccessorFloats(glb, wIdx + 1) or {},
  }
end

return GLTFMesh
