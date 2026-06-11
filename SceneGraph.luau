--!strict
-- SceneGraph
-- Node hierarchy -> world matrices, world-space AABB (for camera fit), and
-- glTF animation sampling (TRS + morph "weights" keyframes: LINEAR / STEP /
-- CUBICSPLINE). All maths via Math3D so it stays headless-testable.

local GLBParser = require('GLBParser')
local M3 = require('Math3D')

type GLBData = GLBParser.GLBData
type Node = GLBParser.Node
type Mat = M3.Mat

export type AABB = { min: { number }, max: { number } }
export type TRSOverride = {
  translation: { number }?,
  rotation: { number }?,
  scale: { number }?,
}

local SceneGraph = {}

-- Local matrix of a node: node.matrix verbatim, else TRS compose. An animation
-- override (per channel) replaces the relevant TRS component.
function SceneGraph.localMatrix(node: Node, override: TRSOverride?): Mat
  if node.matrix and not override then
    return node.matrix
  end
  local t = (override and override.translation) or node.translation
  local r = (override and override.rotation) or node.rotation
  local s = (override and override.scale) or node.scale
  return M3.composeTRS(t, r, s)
end

-- Root node indices (0-based) of the default scene, with sensible fallbacks.
function SceneGraph.rootNodes(glb: GLBData): { number }
  if glb.scenes then
    local idx = glb.scene or 0
    local sc = glb.scenes[idx + 1] or glb.scenes[1]
    if sc and sc.nodes then
      return sc.nodes
    end
  end
  local all: { number } = {}
  if glb.nodes then
    for i = 0, #glb.nodes - 1 do
      all[#all + 1] = i
    end
  end
  return all
end

-- World matrix per node index (0-based key). `overrides` is optional, keyed by
-- 0-based node index (from sampleClip).
function SceneGraph.computeWorldMatrices(
  glb: GLBData,
  overrides: { [number]: TRSOverride }?
): { [number]: Mat }
  local world: { [number]: Mat } = {}
  local nodes: { Node } = glb.nodes or {}
  local function visit(idx: number, parent: Mat)
    local node = nodes[idx + 1]
    if not node then
      return
    end
    local ov = overrides and overrides[idx]
    local localM = SceneGraph.localMatrix(node, ov)
    local w = M3.multiply(parent, localM)
    world[idx] = w
    if node.children then
      for _, c in ipairs(node.children) do
        visit(c, w)
      end
    end
  end
  for _, r in ipairs(SceneGraph.rootNodes(glb)) do
    visit(r, M3.identity())
  end
  return world
end

-- World-space AABB over every mesh primitive. Uses accessor min/max (mandatory
-- on glTF POSITION accessors) and transforms the 8 corners; falls back to a full
-- vertex scan if min/max are missing.
function SceneGraph.computeAABB(glb: GLBData, world: { [number]: Mat }): AABB
  local minx, miny, minz = math.huge, math.huge, math.huge
  local maxx, maxy, maxz = -math.huge, -math.huge, -math.huge
  local function acc(x: number, y: number, z: number)
    if x < minx then
      minx = x
    end
    if y < miny then
      miny = y
    end
    if z < minz then
      minz = z
    end
    if x > maxx then
      maxx = x
    end
    if y > maxy then
      maxy = y
    end
    if z > maxz then
      maxz = z
    end
  end
  local nodes: { Node } = glb.nodes or {}
  for idx = 0, #nodes - 1 do
    local node = nodes[idx + 1]
    local w = world[idx]
    if node.mesh ~= nil and w then
      local mesh = glb.meshes[node.mesh + 1]
      if mesh then
        for _, prim in ipairs(mesh.primitives) do
          local posIdx = prim.attributes.POSITION
          if posIdx ~= nil then
            local accessor = glb.accessors[posIdx + 1]
            if accessor and accessor.min and accessor.max then
              local mn, mx = accessor.min, accessor.max
              for _, cx in { mn[1], mx[1] } do
                for _, cy in { mn[2], mx[2] } do
                  for _, cz in { mn[3], mx[3] } do
                    acc(M3.transformPoint(w, cx, cy, cz))
                  end
                end
              end
            else
              local p = GLBParser.readAccessorFloats(glb, posIdx + 1)
              if p then
                for v = 0, math.floor(#p / 3) - 1 do
                  acc(
                    M3.transformPoint(
                      w,
                      p[v * 3 + 1],
                      p[v * 3 + 2],
                      p[v * 3 + 3]
                    )
                  )
                end
              end
            end
          end
        end
      end
    end
  end
  return { min = { minx, miny, minz }, max = { maxx, maxy, maxz } }
end

-- World-space AABB for a single node's mesh geometry (its primitives only,
-- not its children). Returns nil if the node has no drawable geometry.
function SceneGraph.computeNodeAABB(
  glb: GLBData,
  nodeIndex: number,
  world: { [number]: Mat }
): AABB?
  local nodes: { Node } = glb.nodes or {}
  local node = nodes[nodeIndex + 1]
  local w = world[nodeIndex]
  if not node or node.mesh == nil or not w then
    return nil
  end
  local mesh = glb.meshes[node.mesh + 1]
  if not mesh then
    return nil
  end
  local minx, miny, minz = math.huge, math.huge, math.huge
  local maxx, maxy, maxz = -math.huge, -math.huge, -math.huge
  local got = false
  local function acc(x: number, y: number, z: number)
    got = true
    if x < minx then minx = x end
    if y < miny then miny = y end
    if z < minz then minz = z end
    if x > maxx then maxx = x end
    if y > maxy then maxy = y end
    if z > maxz then maxz = z end
  end
  for _, prim in ipairs(mesh.primitives) do
    local posIdx = prim.attributes.POSITION
    if posIdx ~= nil then
      local accessor = glb.accessors[posIdx + 1]
      if accessor and accessor.min and accessor.max then
        local mn, mx = accessor.min, accessor.max
        for _, cx in { mn[1], mx[1] } do
          for _, cy in { mn[2], mx[2] } do
            for _, cz in { mn[3], mx[3] } do
              acc(M3.transformPoint(w, cx, cy, cz))
            end
          end
        end
      else
        local p = GLBParser.readAccessorFloats(glb, posIdx + 1)
        if p then
          for v = 0, math.floor(#p / 3) - 1 do
            acc(M3.transformPoint(w, p[v * 3 + 1], p[v * 3 + 2], p[v * 3 + 3]))
          end
        end
      end
    end
  end
  if not got then
    return nil
  end
  return { min = { minx, miny, minz }, max = { maxx, maxy, maxz } }
end

-- Slab-method ray/AABB intersection. ox,oy,oz = ray origin; dx,dy,dz =
-- normalised ray direction. Returns the entry distance t along the ray (>= 0)
-- or nil if there is no hit in front of the origin.
function SceneGraph.rayAABB(
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  aabb: AABB
): number?
  local tmin = -math.huge
  local tmax = math.huge
  local o = { ox, oy, oz }
  local d = { dx, dy, dz }
  for i = 1, 3 do
    local di = d[i]
    local lo = aabb.min[i]
    local hi = aabb.max[i]
    if math.abs(di) < 1e-9 then
      -- Ray parallel to this slab: miss if origin is outside the slab.
      if o[i] < lo or o[i] > hi then
        return nil
      end
    else
      local inv = 1 / di
      local t1 = (lo - o[i]) * inv
      local t2 = (hi - o[i]) * inv
      if t1 > t2 then
        t1, t2 = t2, t1
      end
      if t1 > tmin then tmin = t1 end
      if t2 < tmax then tmax = t2 end
      if tmin > tmax then
        return nil
      end
    end
  end
  -- If tmin < 0 the origin is inside the box; use tmax (still a valid hit).
  if tmax < 0 then
    return nil
  end
  if tmin < 0 then
    return tmax
  end
  return tmin
end

-- ---- animation sampling ----------------------------------------------------

local function readTuple(
  values: { number },
  base: number,
  comps: number
): { number }
  local out = table.create(comps, 0)
  for c = 1, comps do
    out[c] = values[base + c]
  end
  return out
end

local function normalizeQuat(q: { number }): { number }
  local len = math.sqrt(q[1] * q[1] + q[2] * q[2] + q[3] * q[3] + q[4] * q[4])
  if len > 1e-8 then
    return { q[1] / len, q[2] / len, q[3] / len, q[4] / len }
  end
  return { 0, 0, 0, 1 }
end

local function nlerpQuat(a: { number }, b: { number }, s: number): { number }
  local dot = a[1] * b[1] + a[2] * b[2] + a[3] * b[3] + a[4] * b[4]
  local sign = if dot < 0 then -1 else 1
  return normalizeQuat({
    a[1] + (b[1] * sign - a[1]) * s,
    a[2] + (b[2] * sign - a[2]) * s,
    a[3] + (b[3] * sign - a[3]) * s,
    a[4] + (b[4] * sign - a[4]) * s,
  })
end

-- Sample one channel's value at time t. comps: 3 (T/S) or 4 (rotation).
local function sampleChannel(
  times: { number },
  values: { number },
  comps: number,
  t: number,
  interp: string,
  isRot: boolean
): { number }
  local n = #times
  local cubic = interp == 'CUBICSPLINE'
  local strideK = if cubic then 3 * comps else comps
  local valueOffset = if cubic then comps else 0 -- middle triplet for cubic
  if n == 1 or t <= times[1] then
    return readTuple(values, valueOffset, comps)
  end
  if t >= times[n] then
    return readTuple(values, (n - 1) * strideK + valueOffset, comps)
  end
  local i = 1
  while i < n and not (times[i] <= t and t < times[i + 1]) do
    i += 1
  end
  local ki = i - 1 -- 0-based lower keyframe
  local dt = times[i + 1] - times[i]
  local s = if dt > 0 then (t - times[i]) / dt else 0
  if interp == 'STEP' then
    return readTuple(values, ki * strideK + valueOffset, comps)
  elseif cubic then
    local vi = readTuple(values, ki * strideK + comps, comps)
    local bi = readTuple(values, ki * strideK + 2 * comps, comps) -- out-tangent of i
    local ai1 = readTuple(values, (ki + 1) * strideK, comps) -- in-tangent of i+1
    local vi1 = readTuple(values, (ki + 1) * strideK + comps, comps)
    local s2, s3 = s * s, s * s * s
    local h00 = 2 * s3 - 3 * s2 + 1
    local h10 = s3 - 2 * s2 + s
    local h01 = -2 * s3 + 3 * s2
    local h11 = s3 - s2
    local out = table.create(comps, 0)
    for c = 1, comps do
      out[c] = h00 * vi[c] + h10 * dt * bi[c] + h01 * vi1[c] + h11 * dt * ai1[c]
    end
    return if isRot then normalizeQuat(out) else out
  else -- LINEAR
    local v0 = readTuple(values, ki * comps, comps)
    local v1 = readTuple(values, (ki + 1) * comps, comps)
    if isRot then
      return nlerpQuat(v0, v1, s)
    end
    local out = table.create(comps, 0)
    for c = 1, comps do
      out[c] = v0[c] + (v1[c] - v0[c]) * s
    end
    return out
  end
end

-- Number of morph targets on the mesh of a node (0-based node index). Per the
-- glTF spec every primitive of a mesh declares the same target count, so the
-- first primitive is authoritative.
local function morphTargetCount(glb: GLBData, nodeIndex: number): number
  local nodes = glb.nodes
  if not nodes then
    return 0
  end
  local node = nodes[nodeIndex + 1]
  if not node or node.mesh == nil then
    return 0
  end
  local mesh = glb.meshes[node.mesh + 1]
  if not mesh then
    return 0
  end
  local prim = mesh.primitives[1]
  if not prim or not prim.targets then
    return 0
  end
  return #(prim.targets :: { { [string]: number } })
end

-- Sample an animation clip at time t. Returns
--   1) per-node TRS overrides (node index 0-based), and
--   2) per-node interpolated morph weights for "weights" channels, or nil
--      when the clip animates no morph weights.
-- A "weights" sampler stores numTargets scalars per keyframe (3x that for
-- CUBICSPLINE: in-tangent, value, out-tangent), so it is sampled as one
-- numTargets-wide tuple through the same interpolation code as TRS.
function SceneGraph.sampleClip(
  glb: GLBData,
  animIndex: number,
  t: number
): ({ [number]: TRSOverride }, { [number]: { number } }?)
  local overrides: { [number]: TRSOverride } = {}
  local weights: { [number]: { number } }? = nil
  if not glb.animations then
    return overrides, nil
  end
  local anim = glb.animations[animIndex]
  if not anim then
    return overrides, nil
  end
  for _, channel in ipairs(anim.channels) do
    local target = channel.target
    local path = target.path
    if target.node ~= nil then
      local sampler = anim.samplers[channel.sampler + 1]
      if sampler then
        local times = GLBParser.readAccessorFloats(glb, sampler.input + 1)
        local values = GLBParser.readAccessorFloats(glb, sampler.output + 1)
        if times and values then
          if path == 'weights' then
            local numTargets = morphTargetCount(glb, target.node)
            if numTargets > 0 then
              local result = sampleChannel(
                times,
                values,
                numTargets,
                t,
                sampler.interpolation or 'LINEAR',
                false
              )
              local w = weights or {}
              w[target.node] = result
              weights = w
            end
          elseif
            path == 'translation'
            or path == 'rotation'
            or path == 'scale'
          then
            local isRot = path == 'rotation'
            local comps = if isRot then 4 else 3
            local result = sampleChannel(
              times,
              values,
              comps,
              t,
              sampler.interpolation or 'LINEAR',
              isRot
            )
            local ov: TRSOverride = overrides[target.node] or {}
            if path == 'translation' then
              ov.translation = result
            elseif path == 'rotation' then
              ov.rotation = result
            else
              ov.scale = result
            end
            overrides[target.node] = ov
          end
        end
      end
    end
  end
  return overrides, weights
end

-- ---- morph targets (CPU) -----------------------------------------------------

-- Deform a primitive's LOCAL positions/normals with its morph targets:
-- result = base + sum(weights[i] * delta_i). Deltas are read through
-- readAccessorFloats, so sparse / Draco-overridden delta accessors work.
-- TANGENT deltas are ignored. Returns (positions?, normals?); both nil when
-- the primitive has no morph targets, and normals nil when the primitive has
-- no NORMAL attribute.
function SceneGraph.applyMorphs(
  glb: GLBData,
  prim: GLBParser.Primitive,
  weights: { number }
): ({ number }?, { number }?)
  local targets = prim.targets
  if not targets or #targets == 0 then
    return nil, nil
  end
  local tgs: { { [string]: number } } = targets
  local function deform(attrIdx: number?, semantic: string): { number }?
    if attrIdx == nil then
      return nil
    end
    local base = GLBParser.readAccessorFloats(glb, attrIdx + 1)
    if not base then
      return nil
    end
    local out = table.clone(base :: { number })
    for ti, tmap in ipairs(tgs) do
      local w = weights[ti] or 0
      if w ~= 0 then
        local deltaIdx = tmap[semantic]
        if deltaIdx ~= nil then
          local delta = GLBParser.readAccessorFloats(glb, deltaIdx + 1)
          if delta then
            local d = delta :: { number }
            local n = math.min(#out, #d)
            for i = 1, n do
              out[i] += w * d[i]
            end
          end
        end
      end
    end
    return out
  end
  return deform(prim.attributes.POSITION, 'POSITION'),
    deform(prim.attributes.NORMAL, 'NORMAL')
end

-- ---- skinning (CPU) --------------------------------------------------------

-- Skinning matrices: skin[j] = inverse(meshWorld) * jointWorld[j] * inverseBind[j].
-- All inputs/outputs are flat col-major Mats; meshWorld may be nil (treated as identity).
function SceneGraph.jointMatrices(
  jointWorlds: { Mat },
  inverseBinds: { Mat },
  meshWorld: Mat?
): { Mat }
  local invMesh = if meshWorld
    then (M3.invert(meshWorld) or M3.identity())
    else M3.identity()
  local out: { Mat } = {}
  for j = 1, #jointWorlds do
    local ib = inverseBinds[j] or M3.identity()
    out[j] = M3.multiply(invMesh, M3.multiply(jointWorlds[j], ib))
  end
  return out
end

-- Linear-blend skinning of positions and normals. positions/normals are flat
-- {x,y,z,...}; joints/weights are flat 4-per-vertex; jointMats is keyed by
-- joint index + 1. Returns skinned (positions, normals).
function SceneGraph.skinVertices(
  positions: { number },
  normals: { number },
  joints: { number },
  weights: { number },
  jointMats: { Mat }
): ({ number }, { number })
  local vcount = math.floor(#positions / 3)
  local outP: { number } = {}
  local outN: { number } = {}
  for v = 0, vcount - 1 do
    local px, py, pz =
      positions[v * 3 + 1], positions[v * 3 + 2], positions[v * 3 + 3]
    local nx, ny, nz =
      normals[v * 3 + 1] or 0, normals[v * 3 + 2] or 0, normals[v * 3 + 3] or 1
    local ax, ay, az = 0, 0, 0
    local bx, by, bz = 0, 0, 0
    for k = 1, 4 do
      local w = weights[v * 4 + k]
      if w and w > 0 then
        local m = jointMats[joints[v * 4 + k] + 1]
        if m then
          local tx, ty, tz = M3.transformPoint(m, px, py, pz)
          ax += tx * w
          ay += ty * w
          az += tz * w
          local rx, ry, rz = M3.transformDir(m, nx, ny, nz)
          bx += rx * w
          by += ry * w
          bz += rz * w
        end
      end
    end
    outP[v * 3 + 1], outP[v * 3 + 2], outP[v * 3 + 3] = ax, ay, az
    local nl = math.sqrt(bx * bx + by * by + bz * bz)
    if nl > 1e-8 then
      outN[v * 3 + 1], outN[v * 3 + 2], outN[v * 3 + 3] =
        bx / nl, by / nl, bz / nl
    else
      outN[v * 3 + 1], outN[v * 3 + 2], outN[v * 3 + 3] = 0, 0, 1
    end
  end
  return outP, outN
end

-- A CPU triangle soup used for UV-accurate ray picking: world-space positions
-- (9 numbers per triangle) and matching uv0 coordinates (6 per triangle).
export type TriMesh = {
  positions: { number },
  uvs: { number },
  triCount: number,
}

-- Build the world-space TriMesh of every primitive that uses the given
-- material (rest pose): positions transformed by the node world matrices,
-- uv0 carried per corner. Returns nil when no UV-mapped triangle matches.
function SceneGraph.buildMaterialTriMesh(
  glb: GLBData,
  matIndex: number
): TriMesh?
  local world = SceneGraph.computeWorldMatrices(glb)
  local nodes: { GLBParser.Node } = glb.nodes or {}
  local positions: { number } = {}
  local uvs: { number } = {}
  local triCount = 0
  for idx = 0, #nodes - 1 do
    local node = nodes[idx + 1]
    local w = world[idx]
    local meshIdx = node.mesh
    if meshIdx ~= nil and w then
      local mesh = glb.meshes[meshIdx + 1]
      if mesh then
        for _, prim in ipairs(mesh.primitives) do
          local posIdx = prim.attributes.POSITION
          local uvIdx = prim.attributes.TEXCOORD_0
          if prim.material == matIndex and posIdx ~= nil then
            local pos = GLBParser.readAccessorFloats(glb, posIdx + 1)
            local uv: { number }? = nil
            if uvIdx ~= nil then
              uv = GLBParser.readAccessorFloats(glb, uvIdx + 1)
            end
            if pos and uv then
              local p = pos :: { number }
              local u = uv :: { number }
              local indices: { number }
              local idxAcc = prim.indices
              if idxAcc ~= nil then
                indices = GLBParser.readIndices(glb, idxAcc + 1) or {}
              else
                indices = {}
                for i = 0, math.floor(#p / 3) - 1 do
                  indices[i + 1] = i
                end
              end
              for t = 0, math.floor(#indices / 3) - 1 do
                for k = 1, 3 do
                  local vi = indices[t * 3 + k]
                  local wx, wy, wz = M3.transformPoint(
                    w,
                    p[vi * 3 + 1],
                    p[vi * 3 + 2],
                    p[vi * 3 + 3]
                  )
                  table.insert(positions, wx)
                  table.insert(positions, wy)
                  table.insert(positions, wz)
                  table.insert(uvs, u[vi * 2 + 1] or 0)
                  table.insert(uvs, u[vi * 2 + 2] or 0)
                end
                triCount += 1
              end
            end
          end
        end
      end
    end
  end
  if triCount == 0 then
    return nil
  end
  return { positions = positions, uvs = uvs, triCount = triCount }
end

-- Ray / triangle-mesh intersection (Moller-Trumbore over a TriMesh).
-- Returns (t, u, v) for the NEAREST hit, where (u, v) is the interpolated
-- uv0 coordinate at the hit point -- or nil when nothing is hit.
function SceneGraph.rayMeshUV(
  mesh: TriMesh,
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number
): (number?, number, number)
  local P = mesh.positions
  local U = mesh.uvs
  local bestT: number? = nil
  local bestU, bestV = 0, 0
  for tri = 0, mesh.triCount - 1 do
    local p = tri * 9
    local ax, ay, az = P[p + 1], P[p + 2], P[p + 3]
    local e1x = P[p + 4] - ax
    local e1y = P[p + 5] - ay
    local e1z = P[p + 6] - az
    local e2x = P[p + 7] - ax
    local e2y = P[p + 8] - ay
    local e2z = P[p + 9] - az
    -- h = d x e2
    local hx = dy * e2z - dz * e2y
    local hy = dz * e2x - dx * e2z
    local hz = dx * e2y - dy * e2x
    local det = e1x * hx + e1y * hy + e1z * hz
    if math.abs(det) > 1e-12 then
      local inv = 1 / det
      local sx, sy, sz = ox - ax, oy - ay, oz - az
      local bu = (sx * hx + sy * hy + sz * hz) * inv
      if bu >= 0 and bu <= 1 then
        -- q = s x e1
        local qx = sy * e1z - sz * e1y
        local qy = sz * e1x - sx * e1z
        local qz = sx * e1y - sy * e1x
        local bv = (dx * qx + dy * qy + dz * qz) * inv
        if bv >= 0 and bu + bv <= 1 then
          local t = (e2x * qx + e2y * qy + e2z * qz) * inv
          if t > 1e-6 and (bestT == nil or t < bestT :: number) then
            bestT = t
            local q = tri * 6
            local w = 1 - bu - bv
            bestU = w * U[q + 1] + bu * U[q + 3] + bv * U[q + 5]
            bestV = w * U[q + 2] + bu * U[q + 4] + bv * U[q + 6]
          end
        end
      end
    end
  end
  return bestT, bestU, bestV
end

-- A CPU triangle soup over the WHOLE scene for triangle-precise picking:
-- world-space positions (9 numbers per triangle) and the owning glTF node
-- index (0-based) per triangle.
export type PickMesh = {
  positions: { number },
  triNode: { number },
  triCount: number,
}

-- Build the world-space PickMesh of every mesh primitive (all nodes, all
-- materials). Local POSITION data is read via readAccessorFloats and
-- transformed by world[nodeIndex]. overridePositions (optional, keyed by
-- 0-based node index) supplies a flat ALREADY-WORLD-SPACE position array
-- indexed exactly like the primitive's POSITION accessor (count*3 numbers,
-- e.g. skinned or morphed vertices); when present for a node it is used
-- verbatim, with no further transform.
function SceneGraph.buildPickMesh(
  glb: GLBData,
  world: { [number]: Mat },
  overridePositions: { [number]: { number } }?
): PickMesh
  local nodes: { GLBParser.Node } = glb.nodes or {}
  local positions: { number } = {}
  local triNode: { number } = {}
  local triCount = 0
  for idx = 0, #nodes - 1 do
    local node = nodes[idx + 1]
    local w = world[idx]
    local meshIdx = node.mesh
    if meshIdx ~= nil and w then
      local mesh = glb.meshes[meshIdx + 1]
      if mesh then
        local override = if overridePositions
          then overridePositions[idx]
          else nil
        for _, prim in ipairs(mesh.primitives) do
          local posIdx = prim.attributes.POSITION
          if posIdx ~= nil then
            local pos = override
              or GLBParser.readAccessorFloats(glb, posIdx + 1)
            if pos then
              local p = pos :: { number }
              local indices: { number }
              local idxAcc = prim.indices
              if idxAcc ~= nil then
                indices = GLBParser.readIndices(glb, idxAcc + 1) or {}
              else
                indices = {}
                for i = 0, math.floor(#p / 3) - 1 do
                  indices[i + 1] = i
                end
              end
              for t = 0, math.floor(#indices / 3) - 1 do
                for k = 1, 3 do
                  local vi = indices[t * 3 + k]
                  local wx, wy, wz
                  if override then
                    -- already world space
                    wx, wy, wz =
                      p[vi * 3 + 1], p[vi * 3 + 2], p[vi * 3 + 3]
                  else
                    wx, wy, wz = M3.transformPoint(
                      w,
                      p[vi * 3 + 1],
                      p[vi * 3 + 2],
                      p[vi * 3 + 3]
                    )
                  end
                  table.insert(positions, wx)
                  table.insert(positions, wy)
                  table.insert(positions, wz)
                end
                triCount += 1
                triNode[triCount] = idx
              end
            end
          end
        end
      end
    end
  end
  return { positions = positions, triNode = triNode, triCount = triCount }
end

-- Ray / PickMesh intersection (Moller-Trumbore over every triangle).
-- Returns (t, nodeIndex) for the NEAREST hit, where nodeIndex is the 0-based
-- glTF node owning the hit triangle -- or (nil, -1) when nothing is hit.
function SceneGraph.rayPickMesh(
  mesh: PickMesh,
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number
): (number?, number)
  local P = mesh.positions
  local bestT: number? = nil
  local bestNode = -1
  for tri = 0, mesh.triCount - 1 do
    local p = tri * 9
    local ax, ay, az = P[p + 1], P[p + 2], P[p + 3]
    local e1x = P[p + 4] - ax
    local e1y = P[p + 5] - ay
    local e1z = P[p + 6] - az
    local e2x = P[p + 7] - ax
    local e2y = P[p + 8] - ay
    local e2z = P[p + 9] - az
    -- h = d x e2
    local hx = dy * e2z - dz * e2y
    local hy = dz * e2x - dx * e2z
    local hz = dx * e2y - dy * e2x
    local det = e1x * hx + e1y * hy + e1z * hz
    if math.abs(det) > 1e-12 then
      local inv = 1 / det
      local sx, sy, sz = ox - ax, oy - ay, oz - az
      local bu = (sx * hx + sy * hy + sz * hz) * inv
      if bu >= 0 and bu <= 1 then
        -- q = s x e1
        local qx = sy * e1z - sz * e1y
        local qy = sz * e1x - sx * e1z
        local qz = sx * e1y - sy * e1x
        local bv = (dx * qx + dy * qy + dz * qz) * inv
        if bv >= 0 and bu + bv <= 1 then
          local t = (e2x * qx + e2y * qy + e2z * qz) * inv
          if t > 1e-6 and (bestT == nil or t < bestT :: number) then
            bestT = t
            bestNode = mesh.triNode[tri + 1]
          end
        end
      end
    end
  end
  return bestT, bestNode
end

return SceneGraph
