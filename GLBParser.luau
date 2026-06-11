--!strict
--[[
  GLB Loader Module
  Parses GLB (Binary glTF) files and extracts mesh data for rendering.

  GLB Structure:
  - Header (12 bytes): magic, version, total length
  - JSON chunk: scene description, mesh metadata, accessors, buffer views
  - BIN chunk: raw binary data (vertices, indices, etc.)

  Compressed GLBs are decoded in pure Luau at parse time:
  - EXT_meshopt_compression  -> bufferViews decoded via MeshoptDecoder
  - KHR_draco_mesh_compression -> primitives decoded via DracoDecoder
  KHR_texture_basisu / EXT_texture_webp images are decoded later, on the
  texture path (see GLBModel), since they only affect images.
]]

local MeshoptDecoder = require('MeshoptDecoder')
local DracoDecoder = require('DracoDecoder')

export type BufferView = {
	byteOffset: number,
	byteLength: number,
	byteStride: number?,
	target: number?,
}

-- Sparse accessor override: a subset of elements stored separately as
-- (indices, values) pairs that replace the base bufferView data (or zeros when
-- the accessor has no bufferView). See glTF 2.0 spec "Sparse Accessors".
export type AccessorSparse = {
	count: number, -- number of overridden elements
	indicesBufferView: number,
	indicesByteOffset: number,
	indicesComponentType: number, -- 5121/5123/5125 (ubyte/ushort/uint)
	valuesBufferView: number,
	valuesByteOffset: number,
}

export type Accessor = {
	bufferView: number?, -- absent for sparse-only / Draco-compressed accessors
	byteOffset: number,
	componentType: number,
	count: number,
	accessorType: string,
	min: { number }?,
	max: { number }?,
	sparse: AccessorSparse?,
	normalized: boolean?,
}

-- KHR_draco_mesh_compression payload on a primitive: the compressed
-- bufferView plus the semantic -> draco-attribute-unique-id map.
export type DracoExt = {
	bufferView: number,
	attributes: { [string]: number },
}

export type Primitive = {
	attributes: { [string]: number },
	indices: number?,
	material: number?,
	mode: number?,
	draco: DracoExt?,
	-- Morph targets: one map per target, semantic -> accessor index
	-- (POSITION / NORMAL / TANGENT deltas).
	targets: { { [string]: number } }?,
}

export type Mesh = {
	name: string?,
	primitives: { Primitive },
	weights: { number }?, -- default morph target weights
}

-- Node in the scene graph (used for skeleton hierarchy)
export type Node = {
	name: string?,
	children: { number }?, -- indices of child nodes
	mesh: number?, -- index of mesh (if any)
	skin: number?, -- index of skin (if any)
	camera: number?, -- index of camera (if any)
	translation: { number }?, -- [x, y, z]
	rotation: { number }?, -- [x, y, z, w] quaternion
	scale: { number }?, -- [x, y, z]
	matrix: { number }?, -- 4x4 column-major matrix (alternative to TRS)
	weights: { number }?, -- morph weights overriding the mesh defaults
}

-- glTF camera, with the perspective/orthographic sub-object flattened.
-- type is "perspective" (yfov/znear/zfar/aspectRatio set) or "orthographic"
-- (xmag/ymag/znear/zfar set).
export type Camera = {
	name: string?,
	type: string,
	yfov: number?, -- vertical field of view, radians
	znear: number?,
	zfar: number?, -- nil = infinite far plane (perspective only)
	aspectRatio: number?,
	xmag: number?, -- half horizontal orthographic magnification
	ymag: number?, -- half vertical orthographic magnification
}

-- Skin defines the skeleton for mesh skinning
export type Skin = {
	name: string?,
	joints: { number }, -- indices of nodes used as joints
	inverseBindMatrices: number?, -- accessor index for inverse bind matrices
	skeleton: number?, -- index of root node (optional)
}

-- Animation channel target
export type AnimationTarget = {
	node: number?, -- index of target node
	path: string, -- "translation", "rotation", "scale", or "weights"
}

-- Animation sampler (defines keyframes)
export type AnimationSampler = {
	input: number, -- accessor index for keyframe times
	output: number, -- accessor index for keyframe values
	interpolation: string?, -- "LINEAR", "STEP", or "CUBICSPLINE"
}

-- Animation channel (connects sampler to target)
export type AnimationChannel = {
	sampler: number, -- index into animation's samplers array
	target: AnimationTarget,
}

-- Animation clip
export type Animation = {
	name: string?,
	channels: { AnimationChannel },
	samplers: { AnimationSampler },
}

-- KHR_texture_transform: affine UV transform attached to a texture slot.
export type UVTransform = {
	offsetX: number,
	offsetY: number,
	scaleX: number,
	scaleY: number,
	rotation: number, -- radians
}

export type TextureInfo = {
	index: number, -- index into textures
	texCoord: number, -- UV set, default 0
	transform: UVTransform?, -- KHR_texture_transform, if present
}

export type Material = {
	name: string?,
	baseColorFactor: { number }, -- {r,g,b,a} 0..1
	baseColorTexture: TextureInfo?,
	metallicFactor: number,
	roughnessFactor: number,
	metallicRoughnessTexture: TextureInfo?,
	normalTexture: TextureInfo?,
	normalScale: number, -- KHR normalTexture.scale (default 1)
	occlusionTexture: TextureInfo?,
	occlusionStrength: number, -- occlusionTexture.strength (default 1)
	emissiveFactor: { number }, -- {r,g,b} 0..1
	emissiveTexture: TextureInfo?,
	emissiveStrength: number, -- KHR_materials_emissive_strength (default 1)
	alphaMode: string, -- "OPAQUE" | "MASK" | "BLEND"
	alphaCutoff: number,
	doubleSided: boolean,
	-- Material extension factors (textures of these extensions are not parsed).
	clearcoatFactor: number, -- KHR_materials_clearcoat (default 0)
	clearcoatRoughnessFactor: number, -- KHR_materials_clearcoat (default 0)
	transmissionFactor: number, -- KHR_materials_transmission (default 0)
	sheenColorFactor: { number }, -- KHR_materials_sheen {r,g,b} (default {0,0,0})
	sheenRoughnessFactor: number, -- KHR_materials_sheen (default 0)
	specularFactor: number, -- KHR_materials_specular (default 1)
	specularColorFactor: { number }, -- KHR_materials_specular {r,g,b} (default {1,1,1})
	ior: number, -- KHR_materials_ior (default 1.5)
}

export type TextureDef = {
	source: number?, -- index into images
	sampler: number?, -- index into samplers
}

export type ImageDef = {
	name: string?,
	uri: string?, -- external (rare in GLB)
	mimeType: string?, -- e.g. "image/png" for embedded
	bufferView: number?, -- GLB-embedded image data
}

export type SamplerDef = {
	magFilter: number?,
	minFilter: number?,
	wrapS: number?, -- 10497 repeat, 33071 clamp, 33648 mirror
	wrapT: number?,
}

export type GLBData = {
	version: number,
	meshes: { Mesh },
	accessors: { Accessor },
	bufferViews: { BufferView },
	binChunk: buffer,
	-- Decoded bytes for compressed bufferViews (1-based view index). When a
	-- view has an entry here, readers use it (offset 0) instead of binChunk.
	viewData: { [number]: buffer },
	-- Decoded values for compressed accessors (1-based accessor index), flat
	-- component arrays produced by the Draco decoder. Takes priority over the
	-- accessor's bufferView. Values are final (dequantized) for float types.
	accessorOverrides: { [number]: { number } },
	nodes: { Node }?,
	skins: { Skin }?,
	animations: { Animation }?,
	cameras: { Camera }?,
	scene: number?, -- index of default scene
	scenes: { { nodes: { number }? } }?, -- scene definitions
	materials: { Material },
	textures: { TextureDef },
	images: { ImageDef },
	samplers: { SamplerDef },
}

export type ParseResult = { success: true, data: GLBData } | { success: false, error: string }

-- Component type sizes in bytes
local COMPONENT_SIZES: { [number]: number } = {
	[5120] = 1, -- BYTE
	[5121] = 1, -- UNSIGNED_BYTE
	[5122] = 2, -- SHORT
	[5123] = 2, -- UNSIGNED_SHORT
	[5125] = 4, -- UNSIGNED_INT
	[5126] = 4, -- FLOAT
}

-- Number of components per accessor type
local TYPE_COMPONENTS: { [string]: number } = {
	["SCALAR"] = 1,
	["VEC2"] = 2,
	["VEC3"] = 3,
	["VEC4"] = 4,
	["MAT2"] = 4,
	["MAT3"] = 9,
	["MAT4"] = 16,
}

-- GLB constants
local GLB_MAGIC = 0x46546C67 -- "glTF" in little-endian
local CHUNK_TYPE_JSON = 0x4E4F534A -- "JSON" in little-endian
local CHUNK_TYPE_BIN = 0x004E4942 -- "BIN\0" in little-endian

-- Read a 32-bit unsigned integer (little-endian)
local function readU32(data: buffer, offset: number): number
	local b0 = buffer.readu8(data, offset)
	local b1 = buffer.readu8(data, offset + 1)
	local b2 = buffer.readu8(data, offset + 2)
	local b3 = buffer.readu8(data, offset + 3)
	return b0 + b1 * 256 + b2 * 65536 + b3 * 16777216
end

-- Read a float32 (little-endian)
local function readF32(data: buffer, offset: number): number
	return buffer.readf32(data, offset)
end

-- Read a 16-bit unsigned integer (little-endian)
local function readU16(data: buffer, offset: number): number
	local b0 = buffer.readu8(data, offset)
	local b1 = buffer.readu8(data, offset + 1)
	return b0 + b1 * 256
end

-- Extract a string from buffer
local function readString(data: buffer, offset: number, length: number): string
	return buffer.readstring(data, offset, length)
end

-- Simple JSON parser for glTF JSON chunk
-- This is a minimal parser that handles the subset needed for mesh extraction
local function parseJSON(jsonStr: string): any
	local pos = 1
	local len = #jsonStr

	local function skipWhitespace()
		while pos <= len do
			local c = string.sub(jsonStr, pos, pos)
			if c == " " or c == "\t" or c == "\n" or c == "\r" then
				pos = pos + 1
			else
				break
			end
		end
	end

	local function parseValue(): any
		skipWhitespace()
		if pos > len then
			return nil
		end

		local c = string.sub(jsonStr, pos, pos)

		if c == "{" then
			-- Parse object
			pos = pos + 1
			local obj: { [string]: any } = {}
			skipWhitespace()
			if string.sub(jsonStr, pos, pos) == "}" then
				pos = pos + 1
				return obj
			end
			while true do
				skipWhitespace()
				-- Parse key (must be string)
				if string.sub(jsonStr, pos, pos) ~= '"' then
					return nil
				end
				pos = pos + 1
				local keyStart = pos
				while pos <= len and string.sub(jsonStr, pos, pos) ~= '"' do
					if string.sub(jsonStr, pos, pos) == "\\" then
						pos = pos + 2
					else
						pos = pos + 1
					end
				end
				local key = string.sub(jsonStr, keyStart, pos - 1)
				pos = pos + 1 -- skip closing quote

				skipWhitespace()
				if string.sub(jsonStr, pos, pos) ~= ":" then
					return nil
				end
				pos = pos + 1

				local value = parseValue()
				obj[key] = value

				skipWhitespace()
				local sep = string.sub(jsonStr, pos, pos)
				if sep == "}" then
					pos = pos + 1
					return obj
				elseif sep == "," then
					pos = pos + 1
				else
					return nil
				end
			end
		elseif c == "[" then
			-- Parse array
			pos = pos + 1
			local arr: { any } = {}
			skipWhitespace()
			if string.sub(jsonStr, pos, pos) == "]" then
				pos = pos + 1
				return arr
			end
			while true do
				local value = parseValue()
				table.insert(arr, value)

				skipWhitespace()
				local sep = string.sub(jsonStr, pos, pos)
				if sep == "]" then
					pos = pos + 1
					return arr
				elseif sep == "," then
					pos = pos + 1
				else
					return nil
				end
			end
		elseif c == '"' then
			-- Parse string
			pos = pos + 1
			local start = pos
			while pos <= len and string.sub(jsonStr, pos, pos) ~= '"' do
				if string.sub(jsonStr, pos, pos) == "\\" then
					pos = pos + 2
				else
					pos = pos + 1
				end
			end
			local str = string.sub(jsonStr, start, pos - 1)
			pos = pos + 1
			return str
		elseif c == "t" then
			-- true
			pos = pos + 4
			return true
		elseif c == "f" then
			-- false
			pos = pos + 5
			return false
		elseif c == "n" then
			-- null
			pos = pos + 4
			return nil
		else
			-- Parse number
			local start = pos
			if string.sub(jsonStr, pos, pos) == "-" then
				pos = pos + 1
			end
			while pos <= len do
				local ch = string.sub(jsonStr, pos, pos)
				if (ch >= "0" and ch <= "9") or ch == "." or ch == "e" or ch == "E" or ch == "+" or ch == "-" then
					pos = pos + 1
				else
					break
				end
			end
			local numStr = string.sub(jsonStr, start, pos - 1)
			return tonumber(numStr)
		end
	end

	return parseValue()
end

-- Parse the GLB header
local function parseHeader(data: buffer): { magic: number, version: number, length: number }?
	if buffer.len(data) < 12 then
		return nil
	end

	return {
		magic = readU32(data, 0),
		version = readU32(data, 4),
		length = readU32(data, 8),
	}
end

-- Main parse function
local function parse(data: buffer): ParseResult
	local dataLen = buffer.len(data)

	-- Parse header
	local header = parseHeader(data)
	if not header then
		return { success = false, error = "Failed to parse GLB header" }
	end

	if header.magic ~= GLB_MAGIC then
		return {
			success = false,
			error = string.format("Invalid GLB magic number: 0x%08X (expected 0x%08X)", header.magic, GLB_MAGIC),
		}
	end

	if header.version ~= 2 then
		return {
			success = false,
			error = string.format("Unsupported GLB version: %d (only version 2 supported)", header.version),
		}
	end

	-- Parse chunks
	local offset = 12
	local jsonData: any = nil
	local binChunk: buffer? = nil

	while offset < dataLen do
		if offset + 8 > dataLen then
			break
		end

		local chunkLength = readU32(data, offset)
		local chunkType = readU32(data, offset + 4)
		offset = offset + 8

		if offset + chunkLength > dataLen then
			return { success = false, error = "Chunk extends beyond file" }
		end

		if chunkType == CHUNK_TYPE_JSON then
			local jsonStr = readString(data, offset, chunkLength)
			jsonData = parseJSON(jsonStr)
			if not jsonData then
				return { success = false, error = "Failed to parse JSON chunk" }
			end
			-- Extension compatibility report. Compressed-data extensions are
			-- DECODED in pure Luau (meshopt/Draco at parse time here; Basis-KTX2
			-- and WebP later on the texture path). Additive material extensions
			-- are safely IGNORED; anything unknown is logged.
			do
				-- Extensions decoded by this loader.
				local DECODED: { [string]: string } = {
					["KHR_draco_mesh_compression"] = "Draco mesh compression",
					["EXT_meshopt_compression"] = "meshopt compression",
					["KHR_texture_basisu"] = "Basis/KTX2 compressed textures",
					["EXT_texture_webp"] = "WebP textures",
					["KHR_texture_transform"] = "UV transforms (baked into UVs)",
					["KHR_mesh_quantization"] = "quantized attributes",
				}
				-- Material extensions whose scalar/colour FACTORS are parsed onto
				-- Material (their texture slots are not).
				local FACTORS_PARSED: { [string]: boolean } = {
					["KHR_materials_ior"] = true,
					["KHR_materials_clearcoat"] = true,
					["KHR_materials_sheen"] = true,
					["KHR_materials_transmission"] = true,
					["KHR_materials_specular"] = true,
				}
				-- Extensions that are safe to ignore (rendering still works, just
				-- without that specific embellishment).
				local KNOWN_IGNORED: { [string]: boolean } = {
					["KHR_materials_unlit"] = true,
					["KHR_materials_emissive_strength"] = true,
					["KHR_materials_volume"] = true,
					["KHR_lights_punctual"] = true,
					["KHR_materials_variants"] = true,
				}

				local jd = jsonData :: any
				local used = jd.extensionsUsed :: { string }?
				if used then
					for _, ext in ipairs(used) do
						if DECODED[ext] then
							print(
								"[GLBParser] extension '"
									.. tostring(ext)
									.. "' ("
									.. DECODED[ext]
									.. ") -- decoded."
							)
						elseif FACTORS_PARSED[ext] then
							print(
								"[GLBParser] extension '"
									.. tostring(ext)
									.. "' -- factors parsed (textures ignored)."
							)
						elseif KNOWN_IGNORED[ext] then
							print(
								"[GLBParser] note: extension '"
									.. tostring(ext)
									.. "' ignored (model renders without it)."
							)
						else
							print(
								"[GLBParser] note: extension '"
									.. tostring(ext)
									.. "' is used but not implemented -- ignored."
							)
						end
					end
				end
			end
		elseif chunkType == CHUNK_TYPE_BIN then
			binChunk = buffer.create(chunkLength)
			buffer.copy(binChunk, 0, data, offset, chunkLength)
		end

		offset = offset + chunkLength
	end

	if not jsonData then
		return { success = false, error = "No JSON chunk found" }
	end

	-- Type aliases for JSON structures
	type JsonPrimitive = {
		attributes: { [string]: number }?,
		indices: number?,
		material: number?,
		mode: number?,
		extensions: any?,
		targets: { { [string]: number } }?,
	}
	type JsonMesh = {
		name: string?,
		primitives: { JsonPrimitive }?,
		weights: { number }?,
	}
	type JsonAccessor = {
		bufferView: number?,
		byteOffset: number?,
		componentType: number,
		count: number,
		type: string,
		min: { number }?,
		max: { number }?,
		normalized: boolean?,
		sparse: {
			count: number,
			indices: { bufferView: number, byteOffset: number?, componentType: number },
			values: { bufferView: number, byteOffset: number? },
		}?,
	}
	type JsonBufferView = {
		byteOffset: number?,
		byteLength: number,
		byteStride: number?,
		target: number?,
		extensions: any?,
	}
	type JsonNode = {
		name: string?,
		children: { number }?,
		mesh: number?,
		skin: number?,
		camera: number?,
		translation: { number }?,
		rotation: { number }?,
		scale: { number }?,
		matrix: { number }?,
		weights: { number }?,
	}
	type JsonSkin = {
		name: string?,
		joints: { number },
		inverseBindMatrices: number?,
		skeleton: number?,
	}
	type JsonAnimationTarget = {
		node: number?,
		path: string,
	}
	type JsonAnimationSampler = {
		input: number,
		output: number,
		interpolation: string?,
	}
	type JsonAnimationChannel = {
		sampler: number,
		target: JsonAnimationTarget,
	}
	type JsonAnimation = {
		name: string?,
		channels: { JsonAnimationChannel }?,
		samplers: { JsonAnimationSampler }?,
	}
	type JsonScene = {
		nodes: { number }?,
	}

	-- Extract mesh data from JSON
	local meshes: { Mesh } = {}
	if jsonData.meshes then
		for _, meshJsonRaw in ipairs(jsonData.meshes) do
			local meshJson = meshJsonRaw :: JsonMesh
			local mesh: Mesh = {
				name = meshJson.name,
				primitives = {},
				weights = meshJson.weights,
			}

			if meshJson.primitives then
				for _, primJsonRaw in ipairs(meshJson.primitives) do
					local primJson = primJsonRaw :: JsonPrimitive
					local draco: DracoExt? = nil
					local pexts = primJson.extensions
					if pexts and pexts.KHR_draco_mesh_compression then
						local de = pexts.KHR_draco_mesh_compression :: any
						if de.bufferView ~= nil and de.attributes ~= nil then
							draco = {
								bufferView = de.bufferView :: number,
								attributes = de.attributes :: { [string]: number },
							}
						end
					end
					local prim: Primitive = {
						attributes = primJson.attributes or {},
						indices = primJson.indices,
						material = primJson.material,
						mode = primJson.mode or 4, -- Default to TRIANGLES
						draco = draco,
						targets = primJson.targets,
					}
					table.insert(mesh.primitives, prim)
				end
			end

			table.insert(meshes, mesh)
		end
	end

	-- Extract accessors
	local accessors: { Accessor } = {}
	if jsonData.accessors then
		for _, accJsonRaw in ipairs(jsonData.accessors) do
			local accJson = accJsonRaw :: JsonAccessor
			local sparse: AccessorSparse? = nil
			if accJson.sparse then
				local sp = accJson.sparse
				sparse = {
					count = sp.count,
					indicesBufferView = sp.indices.bufferView,
					indicesByteOffset = sp.indices.byteOffset or 0,
					indicesComponentType = sp.indices.componentType,
					valuesBufferView = sp.values.bufferView,
					valuesByteOffset = sp.values.byteOffset or 0,
				}
			end
			local accessor: Accessor = {
				-- Keep absence as nil: a sparse accessor without a bufferView has
				-- an all-zero base, which must not alias bufferView 0.
				bufferView = accJson.bufferView,
				byteOffset = accJson.byteOffset or 0,
				componentType = accJson.componentType,
				count = accJson.count,
				accessorType = accJson.type,
				min = accJson.min,
				max = accJson.max,
				normalized = accJson.normalized,
				sparse = sparse,
			}
			table.insert(accessors, accessor)
		end
	end

	-- Extract buffer views. Views compressed with EXT_meshopt_compression are
	-- decoded right here into viewData (the readers then use the decoded bytes
	-- at offset 0 instead of binChunk). The view's declared buffer is the
	-- optional fallback buffer, which GLBs produced by gltfpack don't carry.
	local bufferViews: { BufferView } = {}
	local viewData: { [number]: buffer } = {}
	if jsonData.bufferViews then
		for bvIndex, bvJsonRaw in ipairs(jsonData.bufferViews) do
			local bvJson = bvJsonRaw :: JsonBufferView
			local bv: BufferView = {
				byteOffset = bvJson.byteOffset or 0,
				byteLength = bvJson.byteLength,
				byteStride = bvJson.byteStride,
				target = bvJson.target,
			}
			local exts = bvJson.extensions
			local mo = if exts then exts.EXT_meshopt_compression :: any else nil
			if mo and binChunk then
				local count = mo.count :: number
				local stride = mo.byteStride :: number
				local mode = (mo.mode or "ATTRIBUTES") :: string
				local filter = mo.filter :: string?
				local srcOffset = (mo.byteOffset or 0) :: number
				local srcLength = mo.byteLength :: number
				if (mo.buffer or 0) ~= 0 then
					print("[GLBParser] meshopt: compressed data in buffer "
						.. tostring(mo.buffer)
						.. " unsupported (only the GLB BIN chunk); view "
						.. tostring(bvIndex - 1)
						.. " left undecoded.")
				else
					local src = buffer.create(srcLength)
					buffer.copy(src, 0, binChunk :: buffer, srcOffset, srcLength)
					local decoded, derr =
						MeshoptDecoder.decodeBufferView(src, count, stride, mode, filter)
					if decoded then
						viewData[bvIndex] = decoded
						-- The decoded bytes are tightly packed at `stride`; make the
						-- view describe them so the readers stay consistent.
						bv.byteOffset = 0
						bv.byteLength = count * stride
						if mode == "ATTRIBUTES" then
							bv.byteStride = stride
						end
					else
						return {
							success = false,
							error = "EXT_meshopt_compression decode failed on bufferView "
								.. tostring(bvIndex - 1)
								.. ": "
								.. tostring(derr),
						}
					end
				end
			end
			table.insert(bufferViews, bv)
		end
	end

	-- Extract nodes (scene graph / skeleton hierarchy)
	local nodes: { Node }? = nil
	if jsonData.nodes then
		nodes = {}
		for _, nodeJsonRaw in ipairs(jsonData.nodes) do
			local nodeJson = nodeJsonRaw :: JsonNode
			local node: Node = {
				name = nodeJson.name,
				children = nodeJson.children,
				mesh = nodeJson.mesh,
				skin = nodeJson.skin,
				camera = nodeJson.camera,
				translation = nodeJson.translation,
				rotation = nodeJson.rotation,
				scale = nodeJson.scale,
				matrix = nodeJson.matrix,
				weights = nodeJson.weights,
			}
			table.insert(nodes :: { Node }, node)
		end
	end

	-- Extract skins (skeleton definitions)
	local skins: { Skin }? = nil
	if jsonData.skins then
		skins = {}
		for _, skinJsonRaw in ipairs(jsonData.skins) do
			local skinJson = skinJsonRaw :: JsonSkin
			local skin: Skin = {
				name = skinJson.name,
				joints = skinJson.joints,
				inverseBindMatrices = skinJson.inverseBindMatrices,
				skeleton = skinJson.skeleton,
			}
			table.insert(skins :: { Skin }, skin)
		end
	end

	-- Extract animations
	local animations: { Animation }? = nil
	if jsonData.animations then
		animations = {}
		for _, animJsonRaw in ipairs(jsonData.animations) do
			local animJson = animJsonRaw :: JsonAnimation
			local anim: Animation = {
				name = animJson.name,
				channels = {},
				samplers = {},
			}

			if animJson.samplers then
				for _, samplerRaw in ipairs(animJson.samplers) do
					local sampler = samplerRaw :: JsonAnimationSampler
					table.insert(anim.samplers, {
						input = sampler.input,
						output = sampler.output,
						interpolation = sampler.interpolation or "LINEAR",
					})
				end
			end

			if animJson.channels then
				for _, channelRaw in ipairs(animJson.channels) do
					local channel = channelRaw :: JsonAnimationChannel
					table.insert(anim.channels, {
						sampler = channel.sampler,
						target = {
							node = channel.target.node,
							path = channel.target.path,
						},
					})
				end
			end

			table.insert(animations :: { Animation }, anim)
		end
	end

	-- Extract cameras, flattening the perspective/orthographic sub-object
	-- into one record (camera.type says which fields are meaningful).
	local cameras: { Camera }? = nil
	if jsonData.cameras then
		cameras = {}
		for _, cRaw in ipairs(jsonData.cameras) do
			local c = cRaw :: any
			local cam: Camera = {
				name = c.name,
				type = (c.type or "") :: string,
			}
			local persp = c.perspective :: any
			local ortho = c.orthographic :: any
			if persp then
				cam.yfov = persp.yfov :: number?
				cam.znear = persp.znear :: number?
				cam.zfar = persp.zfar :: number?
				cam.aspectRatio = persp.aspectRatio :: number?
			elseif ortho then
				cam.xmag = ortho.xmag :: number?
				cam.ymag = ortho.ymag :: number?
				cam.znear = ortho.znear :: number?
				cam.zfar = ortho.zfar :: number?
			end
			table.insert(cameras :: { Camera }, cam)
		end
	end

	-- Extract scenes
	local scenes: { { nodes: { number }? } }? = nil
	local sceneIndex: number? = jsonData.scene
	if jsonData.scenes then
		scenes = {}
		for _, sceneRaw in ipairs(jsonData.scenes) do
			local sceneJson = sceneRaw :: JsonScene
			table.insert(scenes :: { { nodes: { number }? } }, {
				nodes = sceneJson.nodes,
			})
		end
	end

	-- Extract materials (PBR metallic-roughness core + common texture slots)
	local function texInfo(t: any): TextureInfo?
		if t and t.index ~= nil then
			local transform: UVTransform? = nil
			local texCoord = (t.texCoord or 0) :: number
			local textInfoExts = t.extensions :: any
			if textInfoExts and textInfoExts.KHR_texture_transform then
				local tt = textInfoExts.KHR_texture_transform :: any
				local off = tt.offset or {}
				local sc = tt.scale or {}
				transform = {
					offsetX = (off[1] or 0) :: number,
					offsetY = (off[2] or 0) :: number,
					scaleX = (sc[1] or 1) :: number,
					scaleY = (sc[2] or 1) :: number,
					rotation = (tt.rotation or 0) :: number,
				}
				if tt.texCoord ~= nil then
					texCoord = tt.texCoord :: number
				end
			end
			return {
				index = t.index :: number,
				texCoord = texCoord,
				transform = transform,
			}
		end
		return nil
	end
	local materials: { Material } = {}
	if jsonData.materials then
		for _, mRaw in ipairs(jsonData.materials) do
			local m = mRaw :: any
			local pbr = m.pbrMetallicRoughness or {}
			-- KHR_materials_emissive_strength: HDR emissive multiplier.
			local emisStrength: number = 1
			local exts = m.extensions
			if exts and exts.KHR_materials_emissive_strength then
				local esExt = exts.KHR_materials_emissive_strength :: any
				local es = esExt.emissiveStrength
				if es ~= nil then
					emisStrength = es :: number
				end
			end
			-- normalTexture.scale and occlusionTexture.strength live on the
			-- texture-info object itself (not the global texture).
			local normScale: number = 1
			if m.normalTexture and m.normalTexture.scale ~= nil then
				normScale = m.normalTexture.scale :: number
			end
			local occStrength: number = 1
			if m.occlusionTexture and m.occlusionTexture.strength ~= nil then
				occStrength = m.occlusionTexture.strength :: number
			end
			-- Material extension FACTORS (defaults per each extension's spec).
			-- Their texture slots are intentionally not parsed.
			local clearcoatFactor: number = 0
			local clearcoatRoughnessFactor: number = 0
			local transmissionFactor: number = 0
			local sheenColorFactor: { number } = { 0, 0, 0 }
			local sheenRoughnessFactor: number = 0
			local specularFactor: number = 1
			local specularColorFactor: { number } = { 1, 1, 1 }
			local ior: number = 1.5
			if exts then
				local cc = exts.KHR_materials_clearcoat :: any
				if cc then
					clearcoatFactor = (cc.clearcoatFactor or 0) :: number
					clearcoatRoughnessFactor = (cc.clearcoatRoughnessFactor or 0) :: number
				end
				local tr = exts.KHR_materials_transmission :: any
				if tr then
					transmissionFactor = (tr.transmissionFactor or 0) :: number
				end
				local sh = exts.KHR_materials_sheen :: any
				if sh then
					sheenColorFactor = (sh.sheenColorFactor or { 0, 0, 0 }) :: { number }
					sheenRoughnessFactor = (sh.sheenRoughnessFactor or 0) :: number
				end
				local sp = exts.KHR_materials_specular :: any
				if sp then
					specularFactor = (sp.specularFactor or 1) :: number
					specularColorFactor = (sp.specularColorFactor or { 1, 1, 1 }) :: { number }
				end
				local io = exts.KHR_materials_ior :: any
				if io and io.ior ~= nil then
					ior = io.ior :: number
				end
			end
			-- Note: 0 is truthy in Lua, so `or` correctly preserves 0 factors.
			local mat: Material = {
				name = m.name,
				baseColorFactor = pbr.baseColorFactor or { 1, 1, 1, 1 },
				baseColorTexture = texInfo(pbr.baseColorTexture),
				metallicFactor = pbr.metallicFactor or 1,
				roughnessFactor = pbr.roughnessFactor or 1,
				metallicRoughnessTexture = texInfo(pbr.metallicRoughnessTexture),
				normalTexture = texInfo(m.normalTexture),
				normalScale = normScale,
				occlusionTexture = texInfo(m.occlusionTexture),
				occlusionStrength = occStrength,
				emissiveFactor = m.emissiveFactor or { 0, 0, 0 },
				emissiveTexture = texInfo(m.emissiveTexture),
				emissiveStrength = emisStrength,
				alphaMode = m.alphaMode or "OPAQUE",
				alphaCutoff = m.alphaCutoff or 0.5,
				doubleSided = m.doubleSided or false,
				clearcoatFactor = clearcoatFactor,
				clearcoatRoughnessFactor = clearcoatRoughnessFactor,
				transmissionFactor = transmissionFactor,
				sheenColorFactor = sheenColorFactor,
				sheenRoughnessFactor = sheenRoughnessFactor,
				specularFactor = specularFactor,
				specularColorFactor = specularColorFactor,
				ior = ior,
			}
			table.insert(materials, mat)
		end
	end

	-- Extract textures / images / samplers (indices stay glTF 0-based).
	-- KHR_texture_basisu / EXT_texture_webp move the image reference into a
	-- texture extension; resolve it here so texture.source always points at
	-- the actual image (KTX2/WebP bytes are decoded on the texture path).
	local textures: { TextureDef } = {}
	if jsonData.textures then
		for _, tRaw in ipairs(jsonData.textures) do
			local t = tRaw :: any
			local source = t.source
			local texts = t.extensions :: any
			if texts then
				local basisu = texts.KHR_texture_basisu :: any
				local webp = texts.EXT_texture_webp :: any
				if basisu and basisu.source ~= nil then
					source = basisu.source
				elseif webp and webp.source ~= nil then
					source = webp.source
				end
			end
			table.insert(textures, { source = source, sampler = t.sampler })
		end
	end
	local images: { ImageDef } = {}
	if jsonData.images then
		for _, iRaw in ipairs(jsonData.images) do
			local im = iRaw :: any
			table.insert(images, { name = im.name, uri = im.uri, mimeType = im.mimeType, bufferView = im.bufferView })
		end
	end
	local samplers: { SamplerDef } = {}
	if jsonData.samplers then
		for _, sRaw in ipairs(jsonData.samplers) do
			local s = sRaw :: any
			table.insert(samplers, { magFilter = s.magFilter, minFilter = s.minFilter, wrapS = s.wrapS, wrapT = s.wrapT })
		end
	end

	-- Decode KHR_draco_mesh_compression primitives. Each compressed primitive
	-- owns one bufferView with a Draco stream; decoding yields the indices plus
	-- one value array per attribute (keyed by the draco attribute unique id,
	-- which the extension maps from the glTF semantic). Decoded arrays override
	-- the primitive's accessors (whose bufferView is absent for Draco data).
	local accessorOverrides: { [number]: { number } } = {}
	for meshIndex, mesh in ipairs(meshes) do
		for primIndex, prim in ipairs(mesh.primitives) do
			local de = prim.draco
			if de and binChunk then
				local bv = bufferViews[de.bufferView + 1]
				if not bv then
					return {
						success = false,
						error = "Draco bufferView " .. de.bufferView .. " missing",
					}
				end
				local src = buffer.create(bv.byteLength)
				buffer.copy(src, 0, binChunk :: buffer, bv.byteOffset, bv.byteLength)
				local decoded, derr = DracoDecoder.decode(src)
				if not decoded then
					return {
						success = false,
						error = string.format(
							"Draco decode failed (mesh %d, primitive %d): %s",
							meshIndex - 1,
							primIndex - 1,
							tostring(derr)
						),
					}
				end
				local dm = decoded :: DracoDecoder.DracoMesh
				if prim.indices ~= nil then
					accessorOverrides[prim.indices + 1] = dm.indices
				end
				for sem, uid in pairs(de.attributes) do
					local accIndex = prim.attributes[sem]
					local attr = dm.attributes[uid]
					if accIndex ~= nil and attr then
						accessorOverrides[accIndex + 1] = attr.values
					else
						print(
							"[GLBParser] draco: attribute '"
								.. tostring(sem)
								.. "' (unique id "
								.. tostring(uid)
								.. ") not found in the decoded stream -- skipped."
						)
					end
				end
			end
		end
	end

	local result: GLBData = {
		version = header.version,
		meshes = meshes,
		accessors = accessors,
		bufferViews = bufferViews,
		binChunk = binChunk :: buffer,
		viewData = viewData,
		accessorOverrides = accessorOverrides,
		nodes = nodes,
		skins = skins,
		animations = animations,
		cameras = cameras,
		scene = sceneIndex,
		scenes = scenes,
		materials = materials,
		textures = textures,
		images = images,
		samplers = samplers,
	}

	return { success = true, data = result }
end

-- Helper to get component size
local function getComponentSize(componentType: number): number
	return COMPONENT_SIZES[componentType] or 0
end

-- Helper to get type component count
local function getTypeComponents(accessorType: string): number
	return TYPE_COMPONENTS[accessorType] or 0
end

-- Read a single typed scalar from a byte buffer by glTF componentType.
local function readComponent(buf: buffer, componentType: number, offset: number): number
	if componentType == 5126 then
		return readF32(buf, offset) -- FLOAT
	elseif componentType == 5123 then
		return readU16(buf, offset) -- UNSIGNED_SHORT
	elseif componentType == 5125 then
		return readU32(buf, offset) -- UNSIGNED_INT
	elseif componentType == 5121 then
		return buffer.readu8(buf, offset) -- UNSIGNED_BYTE
	elseif componentType == 5122 then
		-- SHORT (signed)
		local v = readU16(buf, offset)
		if v >= 32768 then
			v = v - 65536
		end
		return v
	elseif componentType == 5120 then
		-- BYTE (signed)
		local v = buffer.readu8(buf, offset)
		if v >= 128 then
			v = v - 256
		end
		return v
	end
	return 0
end

-- Resolve a bufferView (1-based index) to the byte buffer holding its data
-- plus the view's base offset inside that buffer. Decoded compressed views
-- live in glb.viewData (base offset 0); everything else reads the BIN chunk.
local function viewSource(glb: GLBData, viewIndex: number): (buffer, number)
	local decoded = glb.viewData[viewIndex]
	if decoded then
		return decoded, 0
	end
	local bv = glb.bufferViews[viewIndex]
	return glb.binChunk, if bv then bv.byteOffset else 0
end

-- glTF `normalized` integer -> float conversion factor/clamping per spec.
local function normalizeComponent(componentType: number, v: number): number
	if componentType == 5121 then
		return v / 255
	elseif componentType == 5123 then
		return v / 65535
	elseif componentType == 5120 then
		return math.max(v / 127, -1)
	elseif componentType == 5122 then
		return math.max(v / 32767, -1)
	end
	return v
end

-- Apply a sparse accessor's (indices, values) overrides onto a flat result
-- array, in place. componentCount = scalars per element.
local function applySparse(glb: GLBData, accessor: Accessor, componentCount: number, result: { number })
	local sparse = accessor.sparse
	if not sparse then
		return
	end
	local idxBV = glb.bufferViews[sparse.indicesBufferView + 1]
	local valBV = glb.bufferViews[sparse.valuesBufferView + 1]
	if not idxBV or not valBV then
		return
	end
	local idxBuf, idxViewBase = viewSource(glb, sparse.indicesBufferView + 1)
	local valBuf, valViewBase = viewSource(glb, sparse.valuesBufferView + 1)
	local idxCompSize = getComponentSize(sparse.indicesComponentType)
	local valCompSize = getComponentSize(accessor.componentType)
	local idxBase = idxViewBase + sparse.indicesByteOffset
	local valBase = valViewBase + sparse.valuesByteOffset
	for k = 0, sparse.count - 1 do
		local elemIndex = readComponent(idxBuf, sparse.indicesComponentType, idxBase + k * idxCompSize)
		for j = 0, componentCount - 1 do
			local v = readComponent(valBuf, accessor.componentType, valBase + (k * componentCount + j) * valCompSize)
			result[elemIndex * componentCount + j + 1] = v
		end
	end
end

-- Read accessor data as array of numbers (for positions, normals, etc.).
-- Supports sparse accessors and accessors with no bufferView (all-zero base
-- that is then overridden by the sparse data). Draco-decoded accessors come
-- from glb.accessorOverrides. Integer accessors marked `normalized` are
-- converted to floats per the glTF spec (KHR_mesh_quantization).
local function readAccessorFloats(glb: GLBData, accessorIndex: number): { number }?
	if accessorIndex < 1 or accessorIndex > #glb.accessors then
		return nil
	end

	local accessor = glb.accessors[accessorIndex]
	local componentSize = getComponentSize(accessor.componentType)
	local componentCount = getTypeComponents(accessor.accessorType)

	if componentSize == 0 or componentCount == 0 then
		return nil
	end

	local result: { number } = {}
	local override = glb.accessorOverrides[accessorIndex]

	if override then
		-- Draco-decoded values (already dequantized for float accessor types).
		if accessor.normalized then
			for i = 1, #override do
				result[i] = normalizeComponent(accessor.componentType, override[i])
			end
		else
			table.move(override, 1, #override, 1, result)
		end
		return result
	end

	local bvIndex = accessor.bufferView
	local bufferView = if bvIndex ~= nil
		then glb.bufferViews[bvIndex + 1] -- Convert to 1-indexed
		else nil

	if bufferView and bvIndex ~= nil then
		-- Normal path: read the base bufferView data.
		local buf, viewBase = viewSource(glb, bvIndex + 1)
		local stride = bufferView.byteStride or (componentSize * componentCount)
		local baseOffset = viewBase + accessor.byteOffset
		for i = 0, accessor.count - 1 do
			local elementOffset = baseOffset + i * stride
			for j = 0, componentCount - 1 do
				result[i * componentCount + j + 1] =
					readComponent(buf, accessor.componentType, elementOffset + j * componentSize)
			end
		end
	elseif accessor.sparse then
		-- Sparse accessor with no bufferView: the base is all zeros.
		for i = 1, accessor.count * componentCount do
			result[i] = 0
		end
	else
		-- No bufferView and no sparse data: nothing to read.
		return nil
	end

	-- Overlay any sparse overrides on top of the base values.
	if accessor.sparse then
		applySparse(glb, accessor, componentCount, result)
	end

	if accessor.normalized then
		for i = 1, #result do
			result[i] = normalizeComponent(accessor.componentType, result[i])
		end
	end

	return result
end

-- Read indices from accessor
local function readIndices(glb: GLBData, accessorIndex: number): { number }?
	if accessorIndex < 1 or accessorIndex > #glb.accessors then
		return nil
	end

	local accessor = glb.accessors[accessorIndex]

	local override = glb.accessorOverrides[accessorIndex]
	if override then
		local result: { number } = {}
		table.move(override, 1, #override, 1, result)
		return result
	end

	local bvIndex = accessor.bufferView
	if bvIndex == nil then
		return nil
	end
	local bufferView = glb.bufferViews[bvIndex + 1]

	if not bufferView then
		return nil
	end

	local buf, viewBase = viewSource(glb, bvIndex + 1)
	local componentSize = getComponentSize(accessor.componentType)
	local baseOffset = viewBase + accessor.byteOffset
	local result: { number } = {}

	for i = 0, accessor.count - 1 do
		local valueOffset = baseOffset + i * componentSize
		local value: number

		if accessor.componentType == 5123 then
			-- UNSIGNED_SHORT
			value = readU16(buf, valueOffset)
		elseif accessor.componentType == 5125 then
			-- UNSIGNED_INT
			value = readU32(buf, valueOffset)
		elseif accessor.componentType == 5121 then
			-- UNSIGNED_BYTE
			value = buffer.readu8(buf, valueOffset)
		else
			value = 0
		end

		table.insert(result, value)
	end

	return result
end

-- Read joint indices (VEC4 of unsigned bytes or shorts)
local function readJoints(glb: GLBData, accessorIndex: number): { number }?
	if accessorIndex < 1 or accessorIndex > #glb.accessors then
		return nil
	end

	local accessor = glb.accessors[accessorIndex]

	local override = glb.accessorOverrides[accessorIndex]
	if override then
		local result: { number } = {}
		table.move(override, 1, #override, 1, result)
		return result
	end

	local bvIndex = accessor.bufferView
	if bvIndex == nil then
		return nil
	end
	local bufferView = glb.bufferViews[bvIndex + 1]

	if not bufferView then
		return nil
	end

	local buf, viewBase = viewSource(glb, bvIndex + 1)
	local componentSize = getComponentSize(accessor.componentType)
	local componentCount = getTypeComponents(accessor.accessorType) -- Should be 4 for VEC4

	if componentSize == 0 or componentCount == 0 then
		return nil
	end

	local stride = bufferView.byteStride or (componentSize * componentCount)
	local baseOffset = viewBase + accessor.byteOffset
	local result: { number } = {}

	for i = 0, accessor.count - 1 do
		local elementOffset = baseOffset + i * stride
		for j = 0, componentCount - 1 do
			local valueOffset = elementOffset + j * componentSize
			local value: number

			if accessor.componentType == 5121 then
				-- UNSIGNED_BYTE
				value = buffer.readu8(buf, valueOffset)
			elseif accessor.componentType == 5123 then
				-- UNSIGNED_SHORT
				value = readU16(buf, valueOffset)
			else
				value = 0
			end

			table.insert(result, value)
		end
	end

	return result
end

-- Read inverse bind matrices (array of MAT4)
-- Returns flat array: 16 floats per matrix, column-major order
local function readMat4Array(glb: GLBData, accessorIndex: number): { number }?
	return readAccessorFloats(glb, accessorIndex)
end

-- Get animation duration by finding max keyframe time
local function getAnimationDuration(glb: GLBData, animIndex: number): number
	if not glb.animations then
		return 0
	end

	local anim = glb.animations[animIndex]
	if not anim then
		return 0
	end

	local maxTime = 0
	for _, sampler in ipairs(anim.samplers) do
		local times = readAccessorFloats(glb, sampler.input + 1)
		if times and #times > 0 then
			local lastTime = times[#times]
			if lastTime and lastTime > maxTime then
				maxTime = lastTime
			end
		end
	end

	return maxTime
end

-- Extract embedded image bytes (PNG/JPEG/WebP/KTX2) for a glTF image index
-- (0-based).
local function imageBytes(glb: GLBData, imageIndex: number): buffer?
	local img = glb.images[imageIndex + 1]
	if not img or img.bufferView == nil then
		return nil
	end
	local bv = glb.bufferViews[img.bufferView + 1]
	if not bv then
		return nil
	end
	local buf, viewBase = viewSource(glb, img.bufferView + 1)
	local out = buffer.create(bv.byteLength)
	buffer.copy(out, 0, buf, viewBase, bv.byteLength)
	return out
end

-- Resolve a glTF texture index (0-based, e.g. material.baseColorTexture.index)
-- to its embedded PNG/JPEG bytes: texture -> source image -> imageBytes.
-- Returns nil if the texture/image is external or missing.
local function textureImageBytes(glb: GLBData, textureIndex: number): buffer?
	local tex = glb.textures[textureIndex + 1]
	if not tex or tex.source == nil then
		return nil
	end
	return imageBytes(glb, tex.source)
end

return {
	parse = parse,
	readAccessorFloats = readAccessorFloats,
	readIndices = readIndices,
	readJoints = readJoints,
	readMat4Array = readMat4Array,
	getAnimationDuration = getAnimationDuration,
	imageBytes = imageBytes,
	textureImageBytes = textureImageBytes,
	getComponentSize = getComponentSize,
	getTypeComponents = getTypeComponents,
	COMPONENT_SIZES = COMPONENT_SIZES,
	TYPE_COMPONENTS = TYPE_COMPONENTS,
}
