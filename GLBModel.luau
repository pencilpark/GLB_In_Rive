--!strict
-- GLBModel (Node Script) -- renders a .glb in 3D and exposes its properties.
--   P3: PBR-lite + base-colour via embedded GLB textures or a databound ViewModel Image.
--   P5: TRS animation (sampled each frame, per-node matrices rewritten).
--   P7: model stats written to a bound ViewModel (+ logged).
-- One pipeline; per-material bind group (group 1) holding a 256-byte dynamic
-- per-object UBO + base-colour texture + sampler. Camera auto-frames the AABB.
-- cullMode = "none" for now (flip to "back" once orientation is confirmed).
--
-- Assets (exact names): Blank GLBParser/GLTFMesh/Math3D/SceneGraph, WGSL "model",
-- Node "GLBModel". Add the .glb as a blob, set exactName, size the artboard ~ canvasW x canvasH.

local GLBParser = require('GLBParser')
local GLTFMesh = require('GLTFMesh')
local Math3D = require('Math3D')
local SceneGraph = require('SceneGraph')

-- ---------------------------------------------------------------------------
-- GPU API shims (type-checker only).
-- The GPU/WGSL scripting surface (GPUCanvas, GPUBuffer, GPUTexture, GPUPipeline,
-- GPUBindGroup*, GPUSampler, the descriptor entry records, and the
-- context:gpuCanvas / context:shader / context:decodeImage methods) exists at
-- RUNTIME in this build but is not present in the editor's type definitions.
-- These aliases/declarations exist purely so the analyzer stops reporting
-- "Unknown type / Unknown global / Key not found". They emit no code and do not
-- change runtime behavior.
-- ---------------------------------------------------------------------------
type GPUBuffer = any
type GPUCanvas = any
type GPUTexture = any
type GPUTextureView = any
type GPUPipeline = any
type GPUSampler = any
type GPUBindGroup = any
type GPUBindGroupLayout = any
type UBOEntry = any
type TextureEntry = any
type SamplerEntry = any
type VertexAttribute = any
type VertexBufferLayout = any
type ColorTarget = any

-- Runtime-provided globals (constructors live on these). Declared as `any` so
-- the analyzer recognizes them; the real implementations are injected by the
-- runtime.
local GPUBuffer = (_G :: any).GPUBuffer
local GPUTexture = (_G :: any).GPUTexture
local GPUSampler = (_G :: any).GPUSampler
local GPUPipeline = (_G :: any).GPUPipeline
local GPUBindGroup = (_G :: any).GPUBindGroup
local GPUBindGroupLayout = (_G :: any).GPUBindGroupLayout

local STRIDE = 256
-- Frame uniform layout (16-byte aligned, must match the WGSL `Frame` struct):
--   view @0, proj @64, camPos @128, params @144, surf @160, tint @176,
--   misc @192, texControl @208, replaceColor @224,
--   lights[3] @240 (each Light = dir vec4 + color vec4 = 32 bytes):
--     light0 @240/@256, light1 @272/@288, light2 @304/@320,
--   hilite @336 (.rgb highlight colour, .w strength)  -> ends at 352.
local FRAME_SIZE = 352

type SkinRT = {
  data: GLTFMesh.SkinData,
  skinIndex: number,
  meshNode: number,
  invBinds: { { number } },
}

type DrawGPU = {
  vbo: GPUBuffer,
  ibo: GPUBuffer,
  indexCount: number,
  indexFormat: string,
  recordIndex: number, -- ordinal in the per-object UBO
  nodeIndex: number, -- for animation
  materialIndex: number?,
  skin: SkinRT?, -- present only for skinned primitives (CPU skinning)
}

-- A pickable "part" of the model: one drawable node. Groups every per-object
-- record that belongs to that node so we can highlight/hit-test it as a unit.
type Part = {
  name: string, -- node name (or a synthesised "Part N")
  nodeIndex: number,
  recordIndices: { number }, -- per-object UBO ordinals belonging to this part
}

type Model = {
  -- ======================================================================
  -- SETUP -- assets & pipeline config (read at init; changing these re-runs init)
  -- ======================================================================
  exactName: Input<string>, -- GLB blob asset name
  shaderName: Input<string>, -- WGSL Shader asset name (no .wgsl)
  canvasW: Input<number>,
  canvasH: Input<number>,
  msaa: Input<boolean>, -- 4x anti-aliasing (false if it misbehaves)
  cullMode: Input<number>, -- 0 none, 1 back, 2 front (read at init)

  -- ======================================================================
  -- CAMERA / AXES (live / keyframeable)
  -- ======================================================================
  yaw: Input<number>, -- orbit around vertical axis (deg)
  pitch: Input<number>, -- orbit up/down (deg)
  roll: Input<number>, -- camera roll (deg)
  autoSpin: Input<number>, -- deg/s added to yaw
  fovY: Input<number>, -- vertical field of view (degrees)
  distanceFactor: Input<number>, -- zoom: multiplies the auto-fit distance
  panX: Input<number>,
  panY: Input<number>,
  panZ: Input<number>,

  -- ======================================================================
  -- LIGHTING -- global look + 3 directional lights (key / fill / back).
  -- Each light has a yaw/pitch direction, an intensity (0 = OFF), and a colour.
  -- The classic 3-point setup: light1 = key, light2 = fill, light3 = back/rim.
  -- ======================================================================
  ambient: Input<number>, -- global ambient fill
  exposure: Input<number>, -- overall brightness multiplier
  reflectivity: Input<number>, -- sky reflection weight (key for metals)
  specularStrength: Input<number>,
  rimStrength: Input<number>,

  -- Light 1 (KEY -- main light)
  light1Intensity: Input<number>, -- 0 disables this light
  light1Yaw: Input<number>,
  light1Pitch: Input<number>,
  light1Color: Input<Color>,

  -- Light 2 (FILL -- softens shadows from the opposite side)
  light2Intensity: Input<number>, -- 0 disables this light
  light2Yaw: Input<number>,
  light2Pitch: Input<number>,
  light2Color: Input<Color>,

  -- Light 3 (BACK / RIM -- separates the model from the background)
  light3Intensity: Input<number>, -- 0 disables this light
  light3Yaw: Input<number>,
  light3Pitch: Input<number>,
  light3Color: Input<Color>,

  -- ======================================================================
  -- BACKGROUND
  -- ======================================================================
  bgColor: Input<Color>,
  bgAlpha: Input<number>, -- 0 = transparent background

  -- ======================================================================
  -- MATERIAL overrides (live / keyframeable)
  -- ======================================================================
  tint: Input<Color>, -- multiplies base colour
  metallicMul: Input<number>,
  roughnessMul: Input<number>,
  emissiveBoost: Input<number>,

  -- ======================================================================
  -- TEXTURE controls (live / keyframeable) -- global, apply to every material
  -- ======================================================================
  showBaseTexture: Input<boolean>, -- false -> replace the base texture with replaceColor
  textureMix: Input<number>, -- 0..1: 0 = pure replaceColor, 1 = pure texture (blend/tint between)
  replaceColor: Input<Color>, -- solid colour used when the texture is hidden or mixed out
  showEmissive: Input<boolean>, -- toggle the emissive contribution
  showMetallicRoughness: Input<boolean>, -- toggle the metallic-roughness map (off -> scalar factors only)
  flipV: Input<boolean>, -- flip the V texture axis (glTF top-left vs GPU bottom-left)
  alphaBlend: Input<boolean>, -- enable straight-alpha blending for transparent models

  -- per-material base-colour replacement (B1: databound Image -> one material)
  -- materialIndex selects WHICH material the databound image is applied to.
  -- The image comes from a ViewModel Image property named by texturePropName.
  -- Each material keeps its own override, so changing materialIndex does not
  -- lose what you assigned to the others.
  materialIndex: Input<number>, -- 0-based; which material receives the databound image
  texturePropName: Input<string>, -- name of the ViewModel Image property to read

  -- ======================================================================
  -- ANIMATION (live / keyframeable)
  -- ======================================================================
  animationIndex: Input<number>, -- 0-based
  animationSpeed: Input<number>,
  playAnimation: Input<boolean>, -- false -> use animationTime to scrub
  animationTime: Input<number>, -- seconds (used when playAnimation = false)

  -- ======================================================================
  -- INTERACTIVITY -- mouse orbit + part picking (hitboxes)
  -- ======================================================================
  interactive: Input<boolean>, -- true -> drag to orbit (adds to yaw/pitch, keyframeable offset)
  dragSpeed: Input<number>, -- degrees of rotation per pixel dragged
  pickEnabled: Input<boolean>, -- true -> clicking a part selects/highlights it
  pickPropName: Input<string>, -- ViewModel String property to receive the picked part name
  pickIndexPropName: Input<string>, -- ViewModel Number property to receive the picked part index (-1 = none)
  hoverHighlight: Input<boolean>, -- true -> highlight the part under the pointer on move
  highlightColor: Input<Color>, -- colour used to highlight the hovered/selected part
  highlightStrength: Input<number>, -- 0..1 blend strength of the highlight

  gpu: GPUCanvas?,
  depth: GPUTexture?,
  msaaColor: GPUTexture?,
  pipeline: GPUPipeline?,
  sampler: ImageSampler?,
  gpuSampler: GPUSampler?,
  layout1: GPUBindGroupLayout?,
  frameBuf: GPUBuffer?,
  frameBind: GPUBindGroup?,
  objBuf: GPUBuffer?,
  objBytes: buffer?,
  matBinds: { GPUBindGroup },
  fallbackBind: GPUBindGroup?,

  -- texture system (kept so async decodes can rebuild the binds)
  context: Context?,
  whiteView: GPUTextureView?,
  -- per-GLB-image cache: image index -> decoded texture view (+ kept-alive textures)
  imageViews: { [number]: GPUTextureView },
  imageTextures: { GPUTexture },

  -- per-material base-colour overrides from databound Images (B1).
  -- matIndex -> decoded texture view; takes priority over the embedded GLB texture.
  overrideBasePerMat: { [number]: GPUTextureView },
  matImageTextures: { GPUTexture }, -- keep-alive for the uploaded textures
  imageProp: any, -- Property<Image> from the ViewModel (kept alive)
  lastBoundMatIndex: number, -- detect materialIndex changes in update()

  glb: GLBParser.GLBData?,
  draws: { DrawGPU },
  center: { number },
  radius: number,
  angle: number,
  hasAnim: boolean,
  animDuration: number,
  animTime: number,

  -- interactivity / picking state
  parts: { Part }, -- one entry per drawable node
  partAABBs: { SceneGraph.AABB? }, -- world-space AABB per part (index matches parts)
  camWorld: { number }?, -- camera world matrix (inverse view) for ray picking
  fovYRad: number, -- last vertical fov used (radians) for ray picking
  dragYaw: number, -- accumulated drag offset added to yaw (degrees)
  dragPitch: number, -- accumulated drag offset added to pitch (degrees)
  dragging: boolean,
  lastPointerX: number,
  lastPointerY: number,
  selectedPart: number, -- 1-based index into parts, or 0 = none
  hoveredPart: number, -- 1-based index into parts, or 0 = none
  pickProp: any, -- Property<string> for the picked part name (kept alive)
  pickIndexProp: any, -- Property<number> for the picked part index (kept alive)
  debug: Input<number>, -- 0 normal, 1 raw base-colour, 2 UVs, 3 metal/rough, 4 emissive
}

local function writeVec4(
  buf: buffer,
  off: number,
  x: number,
  y: number,
  z: number,
  w: number
)
  buffer.writef32(buf, off, x)
  buffer.writef32(buf, off + 4, y)
  buffer.writef32(buf, off + 8, z)
  buffer.writef32(buf, off + 12, w)
end

local function packRecord(
  bytes: buffer,
  off: number,
  model: { number },
  baseColor: { number },
  emissive: { number },
  metal: number,
  rough: number,
  hasTex: number
)
  Math3D.writeToBuffer(bytes, off, model)
  Math3D.writeToBuffer(bytes, off + 64, Math3D.normalMatrix(model))
  writeVec4(
    bytes,
    off + 128,
    baseColor[1],
    baseColor[2],
    baseColor[3],
    baseColor[4] or 1
  )
  writeVec4(
    bytes,
    off + 144,
    emissive[1] or 0,
    emissive[2] or 0,
    emissive[3] or 0,
    0
  )
  writeVec4(bytes, off + 160, metal, rough, hasTex, 0.5)
end

local function makeMatBind(
  objBuf: GPUBuffer,
  layout1: GPUBindGroupLayout,
  baseView: GPUTextureView,
  mrView: GPUTextureView,
  emisView: GPUTextureView,
  normalView: GPUTextureView,
  occlusionView: GPUTextureView,
  sampler: GPUSampler
): GPUBindGroup
  local ubos: { UBOEntry } =
    { { slot = 0, buffer = objBuf, offset = 0, size = STRIDE } :: UBOEntry }
  local texs: { TextureEntry } = {
    { slot = 1, view = baseView } :: TextureEntry,
    { slot = 3, view = mrView } :: TextureEntry,
    { slot = 4, view = emisView } :: TextureEntry,
    { slot = 5, view = normalView } :: TextureEntry,
    { slot = 6, view = occlusionView } :: TextureEntry,
  }
  local samps: { SamplerEntry } =
    { { slot = 2, sampler = sampler } :: SamplerEntry }
  return GPUBindGroup.new({
    layout = layout1,
    ubos = ubos,
    textures = texs,
    samplers = samps,
  })
end

-- Resolved per-material texture views + the flags the shader needs.
type MatViews = {
  base: GPUTextureView,
  mr: GPUTextureView,
  emis: GPUTextureView,
  normal: GPUTextureView,
  occlusion: GPUTextureView,
  hasMR: number,
  hasEmis: number,
  hasNormal: number,
  hasOcclusion: number,
  occlusionUV: number, -- 0 = uv0, 1 = uv1
}

-- Resolve every texture view for a material index, using:
--   1. per-material databound image override (overrideBasePerMat) for base
--   2. the material's own embedded GLB textures (decoded into self.imageViews)
--   3. the 1x1 white fallback
-- A texture index resolves to a decoded image via texture -> source image.
local function textureView(
  self: Model,
  texInfo: GLBParser.TextureInfo?
): GPUTextureView?
  local glb = self.glb
  if not glb or not texInfo then
    return nil
  end
  local tex = glb.textures[texInfo.index + 1]
  if tex and tex.source ~= nil then
    return self.imageViews[tex.source]
  end
  return nil
end

local function materialViews(self: Model, matIndex: number?): MatViews
  local white = self.whiteView :: GPUTextureView
  local glb = self.glb
  local baseV: GPUTextureView? = nil
  local mrV: GPUTextureView? = nil
  local emisV: GPUTextureView? = nil
  local normV: GPUTextureView? = nil
  local occV: GPUTextureView? = nil
  local occUV = 0

  -- Per-material databound base-colour override (B1).
  if matIndex ~= nil then
    baseV = self.overrideBasePerMat[matIndex]
  end

  if glb and matIndex ~= nil then
    local mat = glb.materials[matIndex + 1]
    if mat then
      if baseV == nil then
        baseV = textureView(self, mat.baseColorTexture)
      end
      mrV = textureView(self, mat.metallicRoughnessTexture)
      emisV = textureView(self, mat.emissiveTexture)
      normV = textureView(self, mat.normalTexture)
      occV = textureView(self, mat.occlusionTexture)
      if mat.occlusionTexture and mat.occlusionTexture.texCoord == 1 then
        occUV = 1
      end
    end
  end

  return {
    base = baseV or white,
    mr = mrV or white,
    emis = emisV or white,
    normal = normV or white,
    occlusion = occV or white,
    hasMR = if mrV ~= nil then 1 else 0,
    hasEmis = if emisV ~= nil then 1 else 0,
    hasNormal = if normV ~= nil then 1 else 0,
    hasOcclusion = if occV ~= nil then 1 else 0,
    occlusionUV = occUV,
  }
end

-- Rebuild every per-material bind group from the views currently available
-- (decoded GLB images + white fallback). Also refreshes the per-record flags
-- (flags2/flags3/flags4) so the shader knows which maps exist and their scales.
-- Called once in init and again whenever an async texture decode completes.
local function rebuildMatBinds(self: Model)
  local objBuf, layout1, sampler = self.objBuf, self.layout1, self.gpuSampler
  local glb, objBytes = self.glb, self.objBytes
  if not objBuf or not layout1 or not sampler or not glb or not objBytes then
    return
  end

  local function bindFor(matIndex: number?): GPUBindGroup
    local v = materialViews(self, matIndex)
    return makeMatBind(
      objBuf,
      layout1,
      v.base,
      v.mr,
      v.emis,
      v.normal,
      v.occlusion,
      sampler
    )
  end

  -- One bind group per material.
  self.matBinds = {}
  for m = 0, #glb.materials - 1 do
    self.matBinds[m + 1] = bindFor(m)
  end
  -- Fallback for primitives without a material.
  self.fallbackBind = bindFor(nil)

  -- Refresh per-record flags per material so the shader picks up which maps are
  -- present plus their per-material scales/strengths.
  for _, d in ipairs(self.draws) do
    local v = materialViews(self, d.materialIndex)
    local mat = if d.materialIndex ~= nil
      then glb.materials[d.materialIndex + 1]
      else nil
    local off = d.recordIndex * STRIDE
    -- flags2 @176: hasMR, hasEmissive, emissiveBoost, (highlight set elsewhere)
    buffer.writef32(objBytes, off + 176, v.hasMR)
    buffer.writef32(objBytes, off + 180, v.hasEmis)
    buffer.writef32(objBytes, off + 184, self.emissiveBoost)
    -- flags3 @192: hasNormal, hasOcclusion, normalScale, occlusionStrength
    buffer.writef32(objBytes, off + 192, v.hasNormal)
    buffer.writef32(objBytes, off + 196, v.hasOcclusion)
    buffer.writef32(objBytes, off + 200, if mat then mat.normalScale else 1)
    buffer.writef32(
      objBytes,
      off + 204,
      if mat then mat.occlusionStrength else 1
    )
    -- flags4 @208: emissiveStrength, occlusionUV, unused, unused
    buffer.writef32(
      objBytes,
      off + 208,
      if mat then mat.emissiveStrength else 1
    )
    buffer.writef32(objBytes, off + 212, v.occlusionUV)
  end
  objBuf:write(objBytes, 0)
end

-- Decode one embedded GLB image (by glTF image index) to RGBA and upload it to a
-- GPUTexture, caching the view in self.imageViews. Async via context:decodeImage
-- (reliable + alpha-preserving). Rebuilds the binds when each image arrives.
local function decodeGLBImage(self: Model, context: Context, imageIndex: number)
  local glb = self.glb
  if not glb then
    return
  end
  if self.imageViews[imageIndex] ~= nil then
    return -- already decoded / in flight
  end
  local bytes = GLBParser.imageBytes(glb, imageIndex)
  if not bytes then
    print(
      string.format(
        '[GLBModel.tex] GLB image %d has no embedded bytes (external uri?)',
        imageIndex
      )
    )
    return
  end
  (context :: any):decodeImage(bytes):andThen(function(decoded)
    local tex = GPUTexture.new({
      width = decoded.width,
      height = decoded.height,
      format = 'rgba8unorm',
    })
    tex:upload({ data = decoded.data })
    table.insert(self.imageTextures, tex) -- keep alive
    self.imageViews[imageIndex] = tex:view()
    rebuildMatBinds(self)
    if self.context then
      self.context:markNeedsUpdate()
    end
    print(
      string.format(
        '[GLBModel.tex] decoded GLB image %d -> %dx%d',
        imageIndex,
        decoded.width,
        decoded.height
      )
    )
    return nil
  end, function(err)
    print(
      '[GLBModel.tex] decodeImage failed for GLB image '
        .. tostring(imageIndex)
        .. ': '
        .. tostring(err)
    )
    return nil
  end)
end

-- Kick off decoding of every embedded image referenced by a material's
-- base/MR/emissive texture slots. Each decode is async and self-caching.
local function loadEmbeddedTextures(self: Model, context: Context)
  local glb = self.glb
  if not glb then
    return
  end
  local function queue(t: GLBParser.TextureInfo?)
    if t then
      local tex = glb.textures[t.index + 1]
      if tex and tex.source ~= nil then
        decodeGLBImage(self, context, tex.source)
      end
    end
  end
  for _, mat in ipairs(glb.materials) do
    queue(mat.baseColorTexture)
    queue(mat.metallicRoughnessTexture)
    queue(mat.emissiveTexture)
    queue(mat.normalTexture)
    queue(mat.occlusionTexture)
  end
end

-- Inverse bind matrices for a skin, as a list of flat Mats (identity fallback).
local function readInvBinds(
  glb: GLBParser.GLBData,
  skin: GLBParser.Skin
): { { number } }
  local invBinds: { { number } } = {}
  local jointCount = #skin.joints
  if skin.inverseBindMatrices ~= nil then
    local flat = GLBParser.readMat4Array(glb, skin.inverseBindMatrices + 1)
    if flat then
      for j = 0, jointCount - 1 do
        local m: { number } = {}
        for k = 1, 16 do
          m[k] = flat[j * 16 + k]
        end
        invBinds[j + 1] = m
      end
    end
  end
  for j = 1, jointCount do
    if not invBinds[j] then
      invBinds[j] = Math3D.identity()
    end
  end
  return invBinds
end

-- Build one DrawGPU per primitive (placed by its node world matrix) + the
-- packed per-object UBO bytes + bounds + counts + the pickable parts list.
local function buildDraws(glb: GLBParser.GLBData): (
  { DrawGPU },
  buffer,
  { number },
  number,
  SceneGraph.AABB,
  number,
  number,
  { Part },
  { SceneGraph.AABB? }
)
  local world = SceneGraph.computeWorldMatrices(glb)
  local aabb = SceneGraph.computeAABB(glb, world)
  local nodes: { GLBParser.Node } = glb.nodes or {}
  type Entry = {
    item: GLTFMesh.DrawItem,
    node: number,
    w: { number },
    skin: SkinRT?,
  }
  local list: { Entry } = {}
  for idx = 0, #nodes - 1 do
    local node = nodes[idx + 1]
    local w = world[idx]
    if node.mesh ~= nil and w then
      local mesh = glb.meshes[node.mesh + 1]
      if mesh then
        for _, prim in ipairs(mesh.primitives) do
          local item = GLTFMesh.buildPrimitive(glb, prim)
          if item then
            local skinRT: SkinRT? = nil
            if node.skin ~= nil and glb.skins then
              local sd = GLTFMesh.buildSkinData(glb, prim)
              local sk = glb.skins[node.skin + 1]
              if sd and sk then
                skinRT = {
                  data = sd,
                  skinIndex = node.skin,
                  meshNode = idx,
                  invBinds = readInvBinds(glb, sk),
                }
              end
            end
            table.insert(
              list,
              { item = item, node = idx, w = w, skin = skinRT }
            )
          end
        end
      end
    end
  end

  local draws: { DrawGPU } = {}
  local objBytes = buffer.create(math.max(1, #list) * STRIDE)
  local verts, tris = 0, 0
  for i, e in ipairs(list) do
    local item = e.item
    local mat = if item.materialIndex ~= nil
      then glb.materials[item.materialIndex + 1]
      else nil
    local baseColor = if mat then mat.baseColorFactor else { 1, 1, 1, 1 }
    local emissive = if mat then mat.emissiveFactor else { 0, 0, 0 }
    local metal = if mat then mat.metallicFactor else 1
    local rough = if mat then mat.roughnessFactor else 1
    local hasTex = if mat and mat.baseColorTexture then 1 else 0
    local recModel = if e.skin then Math3D.identity() else e.w
    packRecord(
      objBytes,
      (i - 1) * STRIDE,
      recModel,
      baseColor,
      emissive,
      metal,
      rough,
      hasTex
    )
    table.insert(draws, {
      vbo = GPUBuffer.new({
        size = buffer.len(item.vertexData),
        usage = 'vertex',
        data = item.vertexData,
        immutable = e.skin == nil,
      }),
      ibo = GPUBuffer.new({
        size = buffer.len(item.indexData),
        usage = 'index',
        data = item.indexData,
        immutable = true,
      }),
      indexCount = item.indexCount,
      indexFormat = item.indexFormat,
      recordIndex = i - 1,
      nodeIndex = e.node,
      materialIndex = item.materialIndex,
      skin = e.skin,
    })
    verts += item.vertexCount
    tris += math.floor(item.indexCount / 3)
  end

  local center = {
    (aabb.min[1] + aabb.max[1]) * 0.5,
    (aabb.min[2] + aabb.max[2]) * 0.5,
    (aabb.min[3] + aabb.max[3]) * 0.5,
  }
  local dx, dy, dz =
    aabb.max[1] - aabb.min[1],
    aabb.max[2] - aabb.min[2],
    aabb.max[3] - aabb.min[3]
  local radius = 0.5 * math.sqrt(dx * dx + dy * dy + dz * dz)
  if radius < 1e-4 then
    radius = 1
  end

  -- Group draws into pickable "parts", one per drawable node. A node's name
  -- becomes the part name; unnamed nodes get a synthesised label. Each part
  -- records which per-object UBO ordinals it owns (for highlight) and gets a
  -- world-space AABB (for ray picking).
  local parts: { Part } = {}
  local partAABBs: { SceneGraph.AABB? } = {}
  local nodeToPart: { [number]: number } = {}
  for _, d in ipairs(draws) do
    local pi = nodeToPart[d.nodeIndex]
    if pi == nil then
      local node = nodes[d.nodeIndex + 1]
      local name = (node and node.name) or ('Part ' .. tostring(#parts + 1))
      local recordIndices: { number } = {}
      local newPart: Part = {
        name = name,
        nodeIndex = d.nodeIndex,
        recordIndices = recordIndices,
      }
      table.insert(parts, newPart)
      pi = #parts
      nodeToPart[d.nodeIndex] = pi
      partAABBs[pi] = SceneGraph.computeNodeAABB(glb, d.nodeIndex, world)
    end
    table.insert(parts[pi].recordIndices, d.recordIndex)
  end

  return draws, objBytes, center, radius, aabb, verts, tris, parts, partAABBs
end

-- CPU linear-blend skinning: re-skin each skinned primitive's vertices for the
-- given world matrices and rewrite its (mutable) vertex buffer. Skinned vertices
-- come out in world space, so those draws use an identity model matrix.
local function applySkinning(self: Model, world: { [number]: { number } })
  local glb = self.glb
  if not glb or not glb.skins then
    return
  end
  for _, d in ipairs(self.draws) do
    local s = d.skin
    if s then
      local skin = glb.skins[s.skinIndex + 1]
      if skin then
        local jointWorlds: { { number } } = {}
        for ji, jn in ipairs(skin.joints) do
          jointWorlds[ji] = world[jn] or Math3D.identity()
        end
        local jointMats = SceneGraph.jointMatrices(jointWorlds, s.invBinds, nil)
        local sp, sn = SceneGraph.skinVertices(
          s.data.positions,
          s.data.normals,
          s.data.joints,
          s.data.weights,
          jointMats
        )
        -- Skin the tangent xyz the same way as normals (direction transform);
        -- the handedness sign (tangent.w) is preserved from the bind pose.
        local tanXYZ: { number } = {}
        local tcount = math.floor(#s.data.tangents / 4)
        for v = 0, tcount - 1 do
          tanXYZ[v * 3 + 1] = s.data.tangents[v * 4 + 1]
          tanXYZ[v * 3 + 2] = s.data.tangents[v * 4 + 2]
          tanXYZ[v * 3 + 3] = s.data.tangents[v * 4 + 3]
        end
        local _, st = SceneGraph.skinVertices(
          s.data.positions,
          tanXYZ,
          s.data.joints,
          s.data.weights,
          jointMats
        )
        local uvs = s.data.uvs
        local uvs1 = s.data.uvs1
        local colors = s.data.colors
        local tangents = s.data.tangents
        local stride = GLTFMesh.STRIDE
        local vcount = math.floor(#sp / 3)
        local bytes = buffer.create(vcount * stride)
        for v = 0, vcount - 1 do
          local b = v * stride
          -- position
          buffer.writef32(bytes, b + GLTFMesh.OFF_POS + 0, sp[v * 3 + 1])
          buffer.writef32(bytes, b + GLTFMesh.OFF_POS + 4, sp[v * 3 + 2])
          buffer.writef32(bytes, b + GLTFMesh.OFF_POS + 8, sp[v * 3 + 3])
          -- normal
          buffer.writef32(bytes, b + GLTFMesh.OFF_NORMAL + 0, sn[v * 3 + 1])
          buffer.writef32(bytes, b + GLTFMesh.OFF_NORMAL + 4, sn[v * 3 + 2])
          buffer.writef32(bytes, b + GLTFMesh.OFF_NORMAL + 8, sn[v * 3 + 3])
          -- uv0
          local u0 = uvs[v * 2 + 1] or 0
          local v0 = uvs[v * 2 + 2] or 0
          buffer.writef32(bytes, b + GLTFMesh.OFF_UV0 + 0, u0)
          buffer.writef32(bytes, b + GLTFMesh.OFF_UV0 + 4, v0)
          -- tangent (skinned xyz + preserved handedness w)
          buffer.writef32(
            bytes,
            b + GLTFMesh.OFF_TANGENT + 0,
            st[v * 3 + 1] or 1
          )
          buffer.writef32(
            bytes,
            b + GLTFMesh.OFF_TANGENT + 4,
            st[v * 3 + 2] or 0
          )
          buffer.writef32(
            bytes,
            b + GLTFMesh.OFF_TANGENT + 8,
            st[v * 3 + 3] or 0
          )
          buffer.writef32(
            bytes,
            b + GLTFMesh.OFF_TANGENT + 12,
            tangents[v * 4 + 4] or 1
          )
          -- uv1 (fallback to uv0)
          buffer.writef32(
            bytes,
            b + GLTFMesh.OFF_UV1 + 0,
            uvs1[v * 2 + 1] or u0
          )
          buffer.writef32(
            bytes,
            b + GLTFMesh.OFF_UV1 + 4,
            uvs1[v * 2 + 2] or v0
          )
          -- color (fallback white)
          buffer.writef32(
            bytes,
            b + GLTFMesh.OFF_COLOR + 0,
            colors[v * 4 + 1] or 1
          )
          buffer.writef32(
            bytes,
            b + GLTFMesh.OFF_COLOR + 4,
            colors[v * 4 + 2] or 1
          )
          buffer.writef32(
            bytes,
            b + GLTFMesh.OFF_COLOR + 8,
            colors[v * 4 + 3] or 1
          )
          buffer.writef32(
            bytes,
            b + GLTFMesh.OFF_COLOR + 12,
            colors[v * 4 + 4] or 1
          )
        end
        d.vbo:write(bytes, 0)
      end
    end
  end
end

-- (loadTexture removed: the async decodeImage path crashes the editor on large
-- textures. Textures now go through texBlob -- a pre-baked RGBA blob -- in init.)

local function updateCamera(self: Model)
  local gpu, frameBuf = self.gpu, self.frameBuf
  if not gpu or not frameBuf or gpu.width <= 0 or gpu.height <= 0 then
    return
  end
  local aspect = gpu.width / gpu.height
  local fovY = (math.rad(self.fovY) / 100) + 0.01
  local cx, cy, cz =
    self.center[1] + self.panX,
    self.center[2] + self.panY,
    self.center[3] + self.panZ
  local dist = Math3D.fitDistance(self.radius, fovY * 10)
    * (self.distanceFactor * 0.01)
  -- Drag offsets (dragYaw/dragPitch) are added on top of the keyframeable
  -- yaw/pitch inputs only when interactive mode is on, so they survive as a
  -- keyframeable internal offset and never fight the input values.
  local extraYaw = if self.interactive then math.rad(self.dragYaw) else 0
  local extraPitch = if self.interactive then math.rad(self.dragPitch) else 0
  local yawTotal = math.rad(self.yaw) + self.angle + extraYaw
  local pitchTotal =
    math.clamp(math.rad(self.pitch) + extraPitch, -math.rad(89), math.rad(89))
  local view, ex, ey, ez =
    Math3D.orbitCamera(cx, cy, cz, yawTotal, pitchTotal, dist)
  -- Store the camera world matrix (inverse view, pre-roll) + fov for ray picking.
  self.camWorld = Math3D.invert(view)
  self.fovYRad = fovY
  if self.roll ~= 0 then
    view = Math3D.multiply(Math3D.fromRotationZ(math.rad(self.roll)), view)
  end
  -- Keep a sane far/near ratio to preserve depth precision (avoid z-fighting
  -- flicker). Floor the near plane relative to the model size rather than at a
  -- tiny fixed 0.01, which would blow up the ratio when the camera is close.
  local far = dist + self.radius * 2
  local near = math.max(self.radius * 0.02, dist - self.radius * 2)
  if near < far * 0.0005 then
    near = far * 0.0005
  end
  local proj = Math3D.perspectiveReverseZ(fovY, aspect, near, far)
  local b = buffer.create(FRAME_SIZE)
  Math3D.writeToBuffer(b, 0, view)
  Math3D.writeToBuffer(b, 64, proj)
  writeVec4(b, 128, ex, ey, ez, 1)
  -- params @144: (.x ambient, .y exposure, .z debug, .w reflectivity)
  writeVec4(
    b,
    144,
    self.ambient * 0.01,
    self.exposure * 0.01,
    self.debug,
    self.reflectivity * 0.01
  )
  -- surf @160: (.x specularStrength, .y rimStrength, .z metallicMul, .w roughnessMul)
  writeVec4(
    b,
    160,
    self.specularStrength,
    self.rimStrength * 0.01,
    self.metallicMul * 0.01,
    self.roughnessMul * 0.01
  )
  -- tint @176
  writeVec4(
    b,
    176,
    Color.red(self.tint) / 255,
    Color.green(self.tint) / 255,
    Color.blue(self.tint) / 255,
    Color.alpha(self.tint) / 255
  )
  -- misc @192: (.x flipV, .y alphaBlend, .z width, .w height)
  writeVec4(
    b,
    192,
    if self.flipV then 1 else 0,
    if self.alphaBlend then 1 else 0,
    gpu.width,
    gpu.height
  )
  -- texControl @208: (showBase, textureMix, showEmissive, showMR)
  writeVec4(
    b,
    208,
    if self.showBaseTexture then 1 else 0,
    math.clamp(self.textureMix, 0, 1),
    if self.showEmissive then 1 else 0,
    if self.showMetallicRoughness then 1 else 0
  )
  -- replaceColor @224: solid colour used when the texture is hidden / mixed out
  writeVec4(
    b,
    224,
    Color.red(self.replaceColor) / 255,
    Color.green(self.replaceColor) / 255,
    Color.blue(self.replaceColor) / 255,
    Color.alpha(self.replaceColor) / 255
  )
  -- lights[3] starting @240. Each Light is dir(vec4) + color(vec4) = 32 bytes.
  -- A yaw/pitch pair becomes a unit direction pointing TOWARDS the light;
  -- intensity goes in dir.w (0 = light off).
  local function writeLight(
    off: number,
    yaw: number,
    pitch: number,
    intensity: number,
    color: Color
  )
    local ry, rp = math.rad(yaw), math.rad(pitch)
    local cp = math.cos(rp)
    writeVec4(
      b,
      off,
      cp * math.sin(ry),
      math.sin(rp),
      cp * math.cos(ry),
      math.max(intensity, 0)
    )
    writeVec4(
      b,
      off + 16,
      Color.red(color) / 255,
      Color.green(color) / 255,
      Color.blue(color) / 255,
      0
    )
  end
  writeLight(
    240,
    self.light1Yaw,
    self.light1Pitch,
    self.light1Intensity * 0.01,
    self.light1Color
  )
  writeLight(
    272,
    self.light2Yaw,
    self.light2Pitch,
    self.light2Intensity * 0.01,
    self.light2Color
  )
  writeLight(
    304,
    self.light3Yaw,
    self.light3Pitch,
    self.light3Intensity * 0.01,
    self.light3Color
  )
  -- hilite @336: (.rgb highlight colour, .w strength)
  writeVec4(
    b,
    336,
    Color.red(self.highlightColor) / 255,
    Color.green(self.highlightColor) / 255,
    Color.blue(self.highlightColor) / 255,
    math.clamp(self.highlightStrength, 0, 1)
  )
  frameBuf:write(b, 0)
end

-- Set the per-object highlight flag (flags2.w @ byte 188 within each record) so
-- that the selected and/or hovered part is tinted by the shader. All other
-- records are cleared. Cheap: only rewrites the objBytes + re-uploads.
local function refreshHighlight(self: Model)
  local objBuf, objBytes = self.objBuf, self.objBytes
  if not objBuf or not objBytes then
    return
  end
  -- Clear every record's highlight flag.
  for i = 0, #self.draws - 1 do
    buffer.writef32(objBytes, i * STRIDE + 188, 0)
  end
  local function flagPart(partIdx: number)
    local part = self.parts[partIdx]
    if part then
      for _, rec in ipairs(part.recordIndices) do
        buffer.writef32(objBytes, rec * STRIDE + 188, 1)
      end
    end
  end
  if self.selectedPart > 0 then
    flagPart(self.selectedPart)
  end
  if self.hoveredPart > 0 and self.hoveredPart ~= self.selectedPart then
    flagPart(self.hoveredPart)
  end
  objBuf:write(objBytes, 0)
  if self.context then
    self.context:markNeedsUpdate()
  end
end

-- Ray-pick the closest part under a pointer position (in canvas pixels).
-- Returns the 1-based part index, or 0 if nothing was hit. Uses the per-part
-- world-space AABBs and the stored camera matrix.
local function pickPart(self: Model, px: number, py: number): number
  local gpu = self.gpu
  local camWorld = self.camWorld
  if not gpu or not camWorld or gpu.width <= 0 or gpu.height <= 0 then
    return 0
  end
  -- Pointer pixels -> normalised device coords (-1..1), Y up.
  local ndcX = (px / gpu.width) * 2 - 1
  local ndcY = 1 - (py / gpu.height) * 2
  local aspect = gpu.width / gpu.height
  local ox, oy, oz, dx, dy, dz =
    Math3D.screenRay(camWorld, self.fovYRad, aspect, ndcX, ndcY)
  local best = 0
  local bestT = math.huge
  for i, box in ipairs(self.partAABBs) do
    if box then
      local t = SceneGraph.rayAABB(ox, oy, oz, dx, dy, dz, box)
      if t and t < bestT then
        bestT = t
        best = i
      end
    end
  end
  return best
end

-- Report the currently selected part to the ViewModel (name + index).
local function reportPick(self: Model)
  local part = if self.selectedPart > 0
    then self.parts[self.selectedPart]
    else nil
  if self.pickProp ~= nil then
    self.pickProp.value = if part then part.name else ''
  end
  if self.pickIndexProp ~= nil then
    self.pickIndexProp.value = if part then (self.selectedPart - 1) else -1
  end
end

local function byte01(v: number): number
  return math.clamp(math.floor(v * 255 + 0.5), 0, 255)
end

local function exposeStats(
  context: Context,
  glb: GLBParser.GLBData,
  draws: { DrawGPU },
  aabb: SceneGraph.AABB,
  verts: number,
  tris: number,
  center: { number },
  parts: { Part }
)
  local animCount = if glb.animations then #glb.animations else 0
  local animDur = if (glb.animations and #glb.animations > 0)
    then GLBParser.getAnimationDuration(glb, 1)
    else 0
  print(
    '[GLBModel] expected ViewModel properties (create them to receive data):'
  )
  print(
    '  Number: bboxSizeX/Y/Z bboxCenterX/Y/Z vertexCount triangleCount meshCount primitiveCount materialCount textureCount animationCount animationDuration partCount'
  )
  print(
    '  Color:  dominantColor1 dominantColor2 dominantColor3   String: modelName meshNames materialNames animationNames partNames'
  )
  print(
    '  Picking (optional): String pickedPart, Number pickedPartIndex (configurable via inputs)'
  )

  local vm = context:viewModel()
  if not vm then
    print('[GLBModel] no bound ViewModel -- stats logged only')
    return
  end
  local function setNum(name: string, v: number)
    local p = vm:getNumber(name)
    if p then
      p.value = v
    end
  end
  local function setStr(name: string, v: string)
    local p = vm:getString(name)
    if p then
      p.value = v
    end
  end
  setNum('bboxSizeX', aabb.max[1] - aabb.min[1])
  setNum('bboxSizeY', aabb.max[2] - aabb.min[2])
  setNum('bboxSizeZ', aabb.max[3] - aabb.min[3])
  setNum('bboxCenterX', center[1])
  setNum('bboxCenterY', center[2])
  setNum('bboxCenterZ', center[3])
  setNum('vertexCount', verts)
  setNum('triangleCount', tris)
  setNum('meshCount', #glb.meshes)
  setNum('primitiveCount', #draws)
  setNum('materialCount', #glb.materials)
  setNum('textureCount', #glb.textures)
  setNum('animationCount', animCount)
  setNum('animationDuration', animDur)
  setNum('partCount', #parts)
  for i = 1, 3 do
    local m = glb.materials[i]
    if m then
      local p = vm:getColor('dominantColor' .. i)
      if p then
        local b = m.baseColorFactor
        p.value =
          Color.rgba(byte01(b[1]), byte01(b[2]), byte01(b[3]), byte01(b[3]))
      end
    end
  end
  local meshNames, matNames, animNames = {}, {}, {}
  for _, mh in ipairs(glb.meshes) do
    table.insert(meshNames, mh.name or 'mesh')
  end
  for _, m in ipairs(glb.materials) do
    table.insert(matNames, m.name or 'material')
  end
  if glb.animations then
    for _, a in ipairs(glb.animations) do
      table.insert(animNames, a.name or 'anim')
    end
  end
  local partNames = {}
  for _, p in ipairs(parts) do
    table.insert(partNames, p.name)
  end
  setStr('modelName', (glb.meshes[1] and glb.meshes[1].name) or 'model')
  setStr('meshNames', table.concat(meshNames, '\n'))
  setStr('materialNames', table.concat(matNames, '\n'))
  setStr('animationNames', table.concat(animNames, '\n'))
  setStr('partNames', table.concat(partNames, '\n'))
end

-- Convert a ViewModel Image (from data binding) into a GPUTextureView.
-- The GPU scripting API is untyped here and the exact Image->texture call can
-- vary between runtime builds, so we try the common conventions in order and
-- keep the created GPUTexture alive in self.matImageTextures. Returns nil if
-- none of the conventions work (a clear log is printed so it can be adjusted).
local function imageToView(self: Model, img: any): GPUTextureView?
  if img == nil then
    return nil
  end
  -- 1) GPUTexture.new({ image = img }) then :view()
  do
    local ok, v = pcall(function(): any
      local tex = (GPUTexture :: any).new({ image = img })
      table.insert(self.matImageTextures, tex)
      return tex:view()
    end)
    if ok and v ~= nil then
      return v
    end
  end
  -- 2) GPUTexture.new({ w, h, format }) then tex:upload({ image = img })
  do
    local ok, v = pcall(function(): any
      local tex = (GPUTexture :: any).new({
        width = img.width,
        height = img.height,
        format = 'rgba8unorm',
      })
      tex:upload({ image = img })
      table.insert(self.matImageTextures, tex)
      return tex:view()
    end)
    if ok and v ~= nil then
      return v
    end
  end
  -- 3) GPUTexture.fromImage(img)
  do
    local ok, v = pcall(function(): any
      local tex = (GPUTexture :: any).fromImage(img)
      table.insert(self.matImageTextures, tex)
      return tex:view()
    end)
    if ok and v ~= nil then
      return v
    end
  end
  print(
    '[GLBModel.tex] could not convert the databound Image to a GPU texture; '
      .. 'adjust imageToView() to match this runtime\'s Image->GPUTexture API.'
  )
  return nil
end

-- Apply the currently databound Image to the material selected by materialIndex.
-- Stores the resulting view in overrideBasePerMat[matIndex] (or clears it when
-- the image is nil) and rebuilds the binds so the change is visible immediately.
local function applyMaterialImage(self: Model)
  local glb = self.glb
  if not glb then
    return
  end
  local matIndex = math.floor(self.materialIndex)
  if matIndex < 0 or matIndex >= #glb.materials then
    print(
      string.format(
        '[GLBModel.tex] materialIndex %d out of range (0..%d) -- ignored',
        matIndex,
        math.max(0, #glb.materials - 1)
      )
    )
    return
  end
  self.lastBoundMatIndex = matIndex

  local prop = self.imageProp
  local img: any = if prop ~= nil then prop.value else nil
  if img == nil then
    -- No image bound: clear any override for this material (revert to embedded).
    if self.overrideBasePerMat[matIndex] ~= nil then
      self.overrideBasePerMat[matIndex] = nil
      rebuildMatBinds(self)
      if self.context then
        self.context:markNeedsUpdate()
      end
    end
    return
  end

  local view = imageToView(self, img)
  if view == nil then
    return
  end
  self.overrideBasePerMat[matIndex] = view
  rebuildMatBinds(self)
  if self.context then
    self.context:markNeedsUpdate()
  end
  print(
    string.format(
      '[GLBModel.tex] applied databound image to material %d (%dx%d)',
      matIndex,
      (img.width or 0) :: number,
      (img.height or 0) :: number
    )
  )
end

function init(self: Model, context: Context): boolean
  local gpuContext = context :: any -- GPU methods live at runtime; see shim block above
  local shader = gpuContext:shader(self.shaderName)
  if not shader then
    shader = gpuContext:shader(self.shaderName .. '.wgsl') -- some builds may keep the extension
    if shader then
      print(
        '[GLBModel] note: shader resolved with the \'.wgsl\' extension in its name'
      )
    end
  end
  if not shader then
    print(
      '[GLBModel] shader \''
        .. tostring(self.shaderName)
        .. '\' not found via context:shader.'
    )
    print(
      '[GLBModel] Probe: set the shaderName input to a KNOWN-WORKING WGSL Shader Script name. If that also returns nil, context:shader cannot resolve WGSL Shader Scripts in this build (they may need to be a separate .wgsl asset).'
    )
    return false
  end
  print('[GLBModel] shader \'' .. tostring(self.shaderName) .. '\' resolved OK')
  local gpu =
    gpuContext:gpuCanvas({ width = self.canvasW, height = self.canvasH })
  self.gpu = gpu
  self.sampler = ImageSampler('clamp', 'clamp', 'bilinear')
  local samples = if self.msaa then 4 else 1
  -- This build only supports 'depth24plus-stencil8'. We don't use the stencil,
  -- but the format carries a stencil aspect that MUST be cleared every frame --
  -- the render pass below now sets stencilLoadOp/stencilStoreOp. Leaving it
  -- undefined was the cause of the random per-triangle flicker.
  self.depth = GPUTexture.new({
    width = self.canvasW,
    height = self.canvasH,
    format = 'depth24plus-stencil8',
    renderTarget = true,
    sampleCount = samples,
  })
  if samples > 1 then
    self.msaaColor = GPUTexture.new({
      width = self.canvasW,
      height = self.canvasH,
      format = gpu.format,
      renderTarget = true,
      sampleCount = samples,
    })
  end

  local blob = context:blob(self.exactName)
  if not blob then
    print('[GLBModel] blob not found: \'' .. tostring(self.exactName) .. '\'')
    return true
  end
  local res = GLBParser.parse(blob.data)
  if not res.success then
    print('[GLBModel] parse failed: ' .. tostring((res :: any).error))
    return true
  end
  local glb: GLBParser.GLBData = (res :: any).data
  self.glb = glb

  local draws, objBytes, center, radius, aabb, verts, tris, parts, partAABBs =
    buildDraws(glb)
  self.draws = draws
  self.objBytes = objBytes
  self.center = center
  self.radius = radius
  self.parts = parts
  self.partAABBs = partAABBs
  print(string.format('[GLBModel] parts (pickable nodes)=%d', #parts))
  self.hasAnim = glb.animations ~= nil and #glb.animations > 0
  self.animDuration = if self.hasAnim
    then GLBParser.getAnimationDuration(glb, self.animationIndex + 1)
    else 0
  self.animTime = 0
  applySkinning(self, SceneGraph.computeWorldMatrices(glb))
  print(
    string.format(
      '[GLBModel] draws=%d verts=%d tris=%d center=(%.2f,%.2f,%.2f) radius=%.2f format=%s anim=%s',
      #draws,
      verts,
      tris,
      center[1],
      center[2],
      center[3],
      radius,
      tostring(gpu.format),
      tostring(self.hasAnim)
    )
  )

  -- 1x1 white fallback texture (always-valid base-colour binding)
  local whiteTex =
    GPUTexture.new({ width = 1, height = 1, format = 'rgba8unorm' })
  local wb = buffer.create(4)
  buffer.writeu8(wb, 0, 255)
  buffer.writeu8(wb, 1, 255)
  buffer.writeu8(wb, 2, 255)
  buffer.writeu8(wb, 3, 255)
  whiteTex:upload({ data = wb })
  local whiteView = whiteTex:view()
  local gpuSampler = GPUSampler.new({
    min = 'linear',
    mag = 'linear',
    wrapU = 'repeat',
    wrapV = 'repeat',
  })
  self.gpuSampler = gpuSampler

  -- Vertex layout matches GLTFMesh's interleaved output (stride 72):
  --   0 pos.xyz | 1 normal.xyz | 2 uv0.xy | 3 tangent.xyzw | 4 uv1.xy | 5 color.rgba
  local attrs: { VertexAttribute } = {
    { slot = 0, format = 'float32x3', offset = GLTFMesh.OFF_POS } :: VertexAttribute,
    { slot = 1, format = 'float32x3', offset = GLTFMesh.OFF_NORMAL } :: VertexAttribute,
    { slot = 2, format = 'float32x2', offset = GLTFMesh.OFF_UV0 } :: VertexAttribute,
    { slot = 3, format = 'float32x4', offset = GLTFMesh.OFF_TANGENT } :: VertexAttribute,
    { slot = 4, format = 'float32x2', offset = GLTFMesh.OFF_UV1 } :: VertexAttribute,
    { slot = 5, format = 'float32x4', offset = GLTFMesh.OFF_COLOR } :: VertexAttribute,
  }
  local vlayout: { VertexBufferLayout } = {
    { stride = GLTFMesh.STRIDE, attributes = attrs } :: VertexBufferLayout,
  }
  -- PREMULTIPLIED alpha blending for transparent models. The shader now outputs
  -- premultiplied RGB (rgb already scaled by alpha), so the blend is
  -- srcColor * 1 + dstColor * (1 - srcAlpha); alpha = src + dst*(1-src).
  local ctargets: { ColorTarget }
  if self.alphaBlend then
    ctargets = {
      {
        format = gpu.format,
        blend = {
          srcColor = 'one',
          dstColor = 'one-minus-src-alpha',
          colorOp = 'add',
          srcAlpha = 'one',
          dstAlpha = 'one-minus-src-alpha',
          alphaOp = 'add',
        },
      } :: ColorTarget,
    }
  else
    ctargets = {
      { format = gpu.format } :: ColorTarget,
    }
  end
  local layout0 = GPUBindGroupLayout.new({ shader = shader, groupIndex = 0 })
  local layout1 = GPUBindGroupLayout.new({
    shader = shader,
    groupIndex = 1,
    dynamicUBOs = { 0 },
  })
  self.layout1 = layout1
  local bgls: { GPUBindGroupLayout } = { layout0, layout1 }
  local cullStr = if self.cullMode == 1
    then 'back'
    elseif self.cullMode == 2 then 'front'
    else 'none'
  self.pipeline = GPUPipeline.new({
    vertex = { module = shader, entryPoint = 'vs_main' },
    fragment = { module = shader, entryPoint = 'fs_main' },
    vertexLayout = vlayout,
    colorTargets = ctargets,
    depthStencil = {
      format = 'depth24plus-stencil8',
      -- Reverse-Z: near maps to 1, far to 0, so closer fragments have a LARGER
      -- depth value -> the test must be 'greater' and the buffer cleared to 0.
      compare = 'greater',
      write = true,
    },
    cullMode = cullStr,
    topology = 'triangle-list',
    sampleCount = samples,
    bindGroupLayouts = bgls,
  })

  local frameBuf = GPUBuffer.new({ size = FRAME_SIZE, usage = 'uniform' })
  self.frameBuf = frameBuf
  local objBuf = GPUBuffer.new({
    size = buffer.len(objBytes),
    usage = 'uniform',
    data = objBytes,
  })
  self.objBuf = objBuf

  local frameUbos: { UBOEntry } = {
    { slot = 0, buffer = frameBuf, offset = 0, size = FRAME_SIZE } :: UBOEntry,
  }
  self.frameBind = GPUBindGroup.new({ layout = layout0, ubos = frameUbos })

  -- Texture system: textures come from the GLB itself (embedded PNG/JPEG, decoded
  -- per material slot -- base / metallic-roughness / emissive). Works for ANY
  -- textured GLB with no extra assets. Decodes are async; binds + flags are
  -- rebuilt as each image arrives. Unbound slots fall back to the 1x1 white texture.
  self.context = context
  self.whiteView = whiteView
  self.imageViews = {}
  self.imageTextures = {}
  self.overrideBasePerMat = {}
  self.matImageTextures = {}
  self.lastBoundMatIndex = math.floor(self.materialIndex)

  -- B1: databound base-colour image. Read the named ViewModel Image property and
  -- listen for changes so reassigning the asset (or animating materialIndex)
  -- swaps the texture live. Applied to the material selected by materialIndex.
  local vm = context:viewModel()
  if vm and self.texturePropName ~= '' then
    local prop = vm:getImage(self.texturePropName)
    if prop then
      self.imageProp = prop
      prop:addListener(prop, function()
        applyMaterialImage(self)
      end)
    else
      print(
        '[GLBModel.tex] ViewModel has no Image property named \''
          .. tostring(self.texturePropName)
          .. '\' (per-material image override disabled)'
      )
    end
  end

  -- Picking: resolve the optional ViewModel String/Number properties that
  -- receive the picked part's name and index. Missing properties are fine --
  -- the picked part is still highlighted, just not reported to the VM.
  if vm and self.pickPropName ~= '' then
    self.pickProp = vm:getString(self.pickPropName)
    if self.pickProp == nil then
      print(
        '[GLBModel.pick] ViewModel has no String property named \''
          .. tostring(self.pickPropName)
          .. '\' (picked name not reported)'
      )
    end
  end
  if vm and self.pickIndexPropName ~= '' then
    self.pickIndexProp = vm:getNumber(self.pickIndexPropName)
  end

  -- Bind immediately (overrides + white fallback), then decode embedded GLB
  -- textures in the background and rebind as they finish.
  rebuildMatBinds(self)
  applyMaterialImage(self) -- apply any already-bound databound image
  loadEmbeddedTextures(self, context)

  print(
    string.format(
      '[GLBModel.tex] materials=%d images=%d (embedded textures decode async)',
      #glb.materials,
      #glb.images
    )
  )

  updateCamera(self)
  refreshHighlight(self) -- initialise per-object highlight flags to 0
  exposeStats(context, glb, draws, aabb, verts, tris, center, parts)
  return true
end

function update(self: Model)
  if self.glb and self.hasAnim then
    self.animDuration =
      GLBParser.getAnimationDuration(self.glb, self.animationIndex + 1)
  end
  -- live emissive boost (per-object flags2.z at byte 176+8) so it's keyframeable
  local objBuf, objBytes = self.objBuf, self.objBytes
  if objBuf and objBytes then
    for i = 0, #self.draws - 1 do
      buffer.writef32(objBytes, i * STRIDE + 184, self.emissiveBoost)
    end
    objBuf:write(objBytes, 0)
  end
  -- If materialIndex changed, re-apply the databound image to the new target.
  if math.floor(self.materialIndex) ~= self.lastBoundMatIndex then
    applyMaterialImage(self)
  end
  updateCamera(self) -- re-push camera/light/material params on any input change
end

function advance(self: Model, seconds: number): boolean
  self.angle += math.rad(self.autoSpin) * seconds
  local glb, objBuf, objBytes = self.glb, self.objBuf, self.objBytes
  if self.hasAnim and self.animDuration > 0 and glb and objBuf and objBytes then
    if self.playAnimation then
      self.animTime = (self.animTime + seconds * self.animationSpeed)
        % self.animDuration
    else
      self.animTime = self.animationTime % self.animDuration -- scrub / keyframe
    end
    local overrides =
      SceneGraph.sampleClip(glb, self.animationIndex + 1, self.animTime)
    local world = SceneGraph.computeWorldMatrices(glb, overrides)
    for _, d in ipairs(self.draws) do
      if not d.skin then
        local w = world[d.nodeIndex]
        if w then
          local off = d.recordIndex * STRIDE
          Math3D.writeToBuffer(objBytes, off, w)
          Math3D.writeToBuffer(objBytes, off + 64, Math3D.normalMatrix(w))
        end
      end
    end
    objBuf:write(objBytes, 0)
    applySkinning(self, world)
  end
  updateCamera(self)
  return true
end

function drawCanvas(self: Model)
  local gpu, pipeline, depth, frameBind =
    self.gpu, self.pipeline, self.depth, self.frameBind
  if not gpu or not pipeline or not depth or not frameBind then
    return
  end
  if gpu.width <= 0 or gpu.height <= 0 or #self.draws == 0 then
    return
  end
  local msaaC = self.msaaColor
  -- Premultiplied clear color: the canvas is composited with 'srcOver' (which
  -- expects premultiplied alpha) and the shader now outputs premultiplied RGB,
  -- so the background must match -- multiply RGB by bgAlpha. With bgAlpha = 0
  -- this yields (0,0,0,0): a truly transparent background.
  local ba = math.clamp(self.bgAlpha, 0, 1)
  local clearCol = {
    (Color.red(self.bgColor) / 255) * ba,
    (Color.green(self.bgColor) / 255) * ba,
    (Color.blue(self.bgColor) / 255) * ba,
    ba,
  }
  local colorArr: any
  if msaaC then
    colorArr = {
      {
        view = msaaC:view(),
        resolveTarget = gpu:colorView(),
        loadOp = 'clear',
        storeOp = 'discard',
        clearColor = clearCol,
      },
    }
  else
    colorArr =
      { { loadOp = 'clear', storeOp = 'store', clearColor = clearCol } }
  end
  local pass = gpu:beginRenderPass({
    color = colorArr,
    depthStencil = {
      view = depth:view(),
      depthLoadOp = 'clear',
      depthStoreOp = 'store',
      -- Reverse-Z: clear to 0 (the far value) instead of 1.0.
      depthClearValue = 0.0,
      -- The depth24plus-stencil8 format has a stencil aspect that must be
      -- cleared/stored every frame or its contents are undefined (random
      -- per-triangle flicker). We don't use stencil, so just clear it to 0.
      stencilLoadOp = 'clear',
      stencilStoreOp = 'store',
      stencilClearValue = 0,
    },
  })
  pass:setViewport(0, 0, gpu.width, gpu.height)
  pass:setPipeline(pipeline)
  pass:setBindGroup(0, frameBind)
  for _, d in ipairs(self.draws) do
    local bg = if d.materialIndex ~= nil
      then self.matBinds[d.materialIndex + 1]
      else self.fallbackBind
    if bg then
      pass:setVertexBuffer(0, d.vbo)
      pass:setIndexBuffer(d.ibo, d.indexFormat :: 'uint16' | 'uint32')
      pass:setBindGroup(1, bg, { d.recordIndex * STRIDE })
      pass:drawIndexed(d.indexCount)
    end
  end
  pass:finish()
end

function draw(self: Model, renderer: Renderer)
  local gpu, sampler = self.gpu, self.sampler
  if not gpu or not sampler then
    return
  end
  renderer:drawImage(gpu.image, sampler, 'srcOver', 1)
end

-- Map a pointer event (local node space) to canvas pixel coordinates. The image
-- is drawn 1:1 from the GPU canvas, so the node's local pixels already match the
-- canvas resolution when the node is sized to canvasW x canvasH. If the node is
-- scaled, this still works proportionally because pickPart normalises by the
-- canvas dimensions.
local function eventToCanvas(self: Model, event: PointerEvent): (number, number)
  return event.position.x, event.position.y
end

function pointerDown(self: Model, event: PointerEvent)
  local handled = false
  -- Start an orbit drag (interactive mode only).
  if self.interactive then
    self.dragging = true
    self.lastPointerX = event.position.x
    self.lastPointerY = event.position.y
    handled = true
  end
  -- Pick the part under the pointer (select + highlight + report).
  if self.pickEnabled then
    local px, py = eventToCanvas(self, event)
    local hit = pickPart(self, px, py)
    self.selectedPart = hit
    refreshHighlight(self)
    reportPick(self)
    if hit > 0 then
      handled = true
    end
  end
  if handled then
    event:hit()
  end
end

function pointerMove(self: Model, event: PointerEvent)
  -- Orbit drag: accumulate the delta into the keyframeable drag offsets.
  if self.interactive and self.dragging then
    local dx = event.position.x - self.lastPointerX
    local dy = event.position.y - self.lastPointerY
    self.lastPointerX = event.position.x
    self.lastPointerY = event.position.y
    self.dragYaw += dx * self.dragSpeed
    self.dragPitch += dy * self.dragSpeed
    updateCamera(self)
    if self.context then
      self.context:markNeedsUpdate()
    end
    event:hit()
  elseif self.pickEnabled and self.hoverHighlight then
    -- Hover highlight (only when not dragging).
    local px, py = eventToCanvas(self, event)
    local hit = pickPart(self, px, py)
    if hit ~= self.hoveredPart then
      self.hoveredPart = hit
      refreshHighlight(self)
    end
    if hit > 0 then
      event:hit()
    end
  end
end

function pointerUp(self: Model, event: PointerEvent)
  if self.dragging then
    self.dragging = false
    event:hit()
  end
end

function pointerExit(self: Model, event: PointerEvent)
  self.dragging = false
  -- Clear hover highlight when the pointer leaves.
  if self.hoveredPart ~= 0 then
    self.hoveredPart = 0
    refreshHighlight(self)
  end
end

return function(): Node<Model>
  return {
    -- SETUP
    exactName = 'model',
    shaderName = 'model',
    canvasW = 1024,
    canvasH = 1024,

    -- CAMERA / AXES
    yaw = 0,
    pitch = 0,
    roll = 0,
    autoSpin = 15,
    fovY = 100,
    distanceFactor = 1000,
    panX = 0,
    panY = 0,
    panZ = 0,

    -- LIGHTING (global look)
    msaa = true,
    cullMode = 0,
    ambient = 30,
    exposure = 100,
    reflectivity = 55,
    specularStrength = 1,
    rimStrength = 0.1,

    -- Light 1 (KEY): strong, warm-ish white, high from the front-right.
    light1Color = Color.rgba(255, 250, 240, 255),
    light1Intensity = 90,
    light1Yaw = 40,
    light1Pitch = 55,

    -- Light 2 (FILL): dim, cool, from the opposite (front-left) side, low.
    light2Color = Color.rgba(220, 230, 255, 255),
    light2Intensity = 35,
    light2Yaw = -60,
    light2Pitch = 15,

    -- Light 3 (BACK / RIM): from behind to pop the silhouette. Off by default.
    light3Color = Color.rgba(255, 255, 255, 255),
    light3Intensity = 0,
    light3Yaw = 180,
    light3Pitch = 35,

    -- BACKGROUND
    bgColor = Color.rgba(13, 15, 20, 1),
    bgAlpha = 1,

    -- MATERIAL
    tint = Color.rgba(255, 255, 255, 1),
    metallicMul = 0.1,
    roughnessMul = 0.1,
    emissiveBoost = 0.1,

    -- TEXTURE (defaults preserve the original look: texture fully shown)
    showBaseTexture = true,
    textureMix = 1,
    replaceColor = Color.rgba(200, 200, 200, 255),
    showEmissive = true,
    showMetallicRoughness = true,
    flipV = false,
    alphaBlend = false,

    -- per-material databound base-colour image (B1)
    materialIndex = 0,
    texturePropName = '',

    -- ANIMATION
    animationIndex = 0,
    animationSpeed = 0,
    playAnimation = true,
    animationTime = 0,

    -- INTERACTIVITY
    interactive = false,
    dragSpeed = 0.4,
    pickEnabled = false,
    pickPropName = '',
    pickIndexPropName = '',
    hoverHighlight = false,
    highlightColor = Color.rgba(255, 200, 40, 255),
    highlightStrength = 0.6,

    gpu = late(),
    depth = late(),
    msaaColor = late(),
    pipeline = late(),
    sampler = late(),
    gpuSampler = late(),
    layout1 = late(),
    frameBuf = late(),
    frameBind = late(),
    objBuf = late(),
    objBytes = late(),
    matBinds = {},
    fallbackBind = late(),
    context = late(),
    whiteView = late(),
    imageViews = {},
    imageTextures = {},
    overrideBasePerMat = {},
    matImageTextures = {},
    imageProp = late(),
    lastBoundMatIndex = 0,
    glb = late(),
    draws = {},
    center = { 0, 0, 0 },
    radius = 1,
    angle = 0,
    hasAnim = false,
    animDuration = 0,
    animTime = 0,

    -- interactivity / picking state
    parts = {},
    partAABBs = {},
    camWorld = late(),
    fovYRad = 0,
    dragYaw = 0,
    dragPitch = 0,
    dragging = false,
    lastPointerX = 0,
    lastPointerY = 0,
    selectedPart = 0,
    hoveredPart = 0,
    pickProp = late(),
    pickIndexProp = late(),

    init = init,
    update = update,
    advance = advance,
    drawCanvas = drawCanvas,
    draw = draw,
    pointerDown = pointerDown,
    pointerMove = pointerMove,
    pointerUp = pointerUp,
    pointerExit = pointerExit,
    --debug mode
    debug = 0,
  }
end
