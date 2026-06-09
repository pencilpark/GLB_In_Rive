--!strict
--[[
  GLB Loader Module
  Parses GLB (Binary glTF) files and extracts mesh data for rendering.

  GLB Structure:
  - Header (12 bytes): magic, version, total length
  - JSON chunk: scene description, mesh metadata, accessors, buffer views
  - BIN chunk: raw binary data (vertices, indices, etc.)
]]

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
	bufferView: number,
	byteOffset: number,
	componentType: number,
	count: number,
	accessorType: string,
	min: { number }?,
	max: { number }?,
	sparse: AccessorSparse?,
	normalized: boolean?,
}

export type Primitive = {
	attributes: { [string]: number },
	indices: number?,
	material: number?,
	mode: number?,
}

export type Mesh = {
	name: string?,
	primitives: { Primitive },
}

-- Node in the scene graph (used for skeleton hierarchy)
export type Node = {
	name: string?,
	children: { number }?, -- indices of child nodes
	mesh: number?, -- index of mesh (if any)
	skin: number?, -- index of skin (if any)
	translation: { number }?, -- [x, y, z]
	rotation: { number }?, -- [x, y, z, w] quaternion
	scale: { number }?, -- [x, y, z]
	matrix: { number }?, -- 4x4 column-major matrix (alternative to TRS)
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

export type TextureInfo = {
	index: number, -- index into textures
	texCoord: number, -- UV set, default 0
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
	nodes: { Node }?,
	skins: { Skin }?,
	animations: { Animation }?,
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
			-- Extension compatibility report. Some extensions store data in a way
			-- our raw byte reads cannot decode (compressed geometry/textures) --
			-- those are BLOCKING and we fail loudly with actionable advice. Others
			-- are purely additive (extra material features) -- those are safely
			-- IGNORED, and we log them so the user knows what was skipped.
			do
				-- Extensions we cannot decode in pure Luau: fail with guidance.
				local BLOCKING: { [string]: string } = {
					["KHR_draco_mesh_compression"] = "Draco mesh compression",
					["EXT_meshopt_compression"] = "meshopt compression",
					["KHR_texture_basisu"] = "Basis/KTX2 compressed textures",
					["EXT_texture_webp"] = "WebP textures",
				}
				-- Extensions that are safe to ignore (rendering still works, just
				-- without that specific embellishment).
				local KNOWN_IGNORED: { [string]: boolean } = {
					["KHR_materials_unlit"] = true,
					["KHR_materials_emissive_strength"] = true,
					["KHR_materials_ior"] = true,
					["KHR_materials_clearcoat"] = true,
					["KHR_materials_sheen"] = true,
					["KHR_materials_transmission"] = true,
					["KHR_materials_volume"] = true,
					["KHR_materials_specular"] = true,
					["KHR_lights_punctual"] = true,
					["KHR_texture_transform"] = true,
					["KHR_materials_variants"] = true,
					["KHR_mesh_quantization"] = true,
				}

				local jd = jsonData :: any
				-- A required extension that is blocking aborts the load.
				local req = jd.extensionsRequired :: { string }?
				if req then
					for _, ext in ipairs(req) do
						local label = BLOCKING[ext]
						if label then
							return {
								success = false,
								error = "GLB requires '"
									.. ext
									.. "' ("
									.. label
									.. ") which cannot be decoded in this runtime. "
									.. "Re-export the model WITHOUT that feature "
									.. "(e.g. `gltf-transform decompress`/`uncompress`, or disable "
									.. "compression/Basis textures on export).",
							}
						end
					end
				end
				-- Report every extension that is merely USED (not required) but that
				-- we don't implement, so the user understands any visual difference.
				local used = jd.extensionsUsed :: { string }?
				if used then
					for _, ext in ipairs(used) do
						if not BLOCKING[ext] and not KNOWN_IGNORED[ext] then
							print(
								"[GLBParser] note: extension '"
									.. tostring(ext)
									.. "' is used but not implemented -- ignored."
							)
						elseif KNOWN_IGNORED[ext] then
							print(
								"[GLBParser] note: extension '"
									.. tostring(ext)
									.. "' ignored (model renders without it)."
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
	}
	type JsonMesh = { name: string?, primitives: { JsonPrimitive }? }
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
	}
	type JsonNode = {
		name: string?,
		children: { number }?,
		mesh: number?,
		skin: number?,
		translation: { number }?,
		rotation: { number }?,
		scale: { number }?,
		matrix: { number }?,
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
			}

			if meshJson.primitives then
				for _, primJsonRaw in ipairs(meshJson.primitives) do
					local primJson = primJsonRaw :: JsonPrimitive
					local prim: Primitive = {
						attributes = primJson.attributes or {},
						indices = primJson.indices,
						material = primJson.material,
						mode = primJson.mode or 4, -- Default to TRIANGLES
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
				bufferView = accJson.bufferView or 0,
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

	-- Extract buffer views
	local bufferViews: { BufferView } = {}
	if jsonData.bufferViews then
		for _, bvJsonRaw in ipairs(jsonData.bufferViews) do
			local bvJson = bvJsonRaw :: JsonBufferView
			local bv: BufferView = {
				byteOffset = bvJson.byteOffset or 0,
				byteLength = bvJson.byteLength,
				byteStride = bvJson.byteStride,
				target = bvJson.target,
			}
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
				translation = nodeJson.translation,
				rotation = nodeJson.rotation,
				scale = nodeJson.scale,
				matrix = nodeJson.matrix,
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
			return { index = t.index :: number, texCoord = (t.texCoord or 0) :: number }
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
			}
			table.insert(materials, mat)
		end
	end

	-- Extract textures / images / samplers (indices stay glTF 0-based)
	local textures: { TextureDef } = {}
	if jsonData.textures then
		for _, tRaw in ipairs(jsonData.textures) do
			local t = tRaw :: any
			table.insert(textures, { source = t.source, sampler = t.sampler })
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

	local result: GLBData = {
		version = header.version,
		meshes = meshes,
		accessors = accessors,
		bufferViews = bufferViews,
		binChunk = binChunk :: buffer,
		nodes = nodes,
		skins = skins,
		animations = animations,
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

-- Read a single typed scalar from the bin chunk by glTF componentType.
local function readComponent(glb: GLBData, componentType: number, offset: number): number
	if componentType == 5126 then
		return readF32(glb.binChunk, offset) -- FLOAT
	elseif componentType == 5123 then
		return readU16(glb.binChunk, offset) -- UNSIGNED_SHORT
	elseif componentType == 5125 then
		return readU32(glb.binChunk, offset) -- UNSIGNED_INT
	elseif componentType == 5121 then
		return buffer.readu8(glb.binChunk, offset) -- UNSIGNED_BYTE
	elseif componentType == 5122 then
		-- SHORT (signed)
		local v = readU16(glb.binChunk, offset)
		if v >= 32768 then
			v = v - 65536
		end
		return v
	elseif componentType == 5120 then
		-- BYTE (signed)
		local v = buffer.readu8(glb.binChunk, offset)
		if v >= 128 then
			v = v - 256
		end
		return v
	end
	return 0
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
	local idxCompSize = getComponentSize(sparse.indicesComponentType)
	local valCompSize = getComponentSize(accessor.componentType)
	local idxBase = idxBV.byteOffset + sparse.indicesByteOffset
	local valBase = valBV.byteOffset + sparse.valuesByteOffset
	for k = 0, sparse.count - 1 do
		local elemIndex = readComponent(glb, sparse.indicesComponentType, idxBase + k * idxCompSize)
		for j = 0, componentCount - 1 do
			local v = readComponent(glb, accessor.componentType, valBase + (k * componentCount + j) * valCompSize)
			result[elemIndex * componentCount + j + 1] = v
		end
	end
end

-- Read accessor data as array of numbers (for positions, normals, etc.).
-- Supports sparse accessors and accessors with no bufferView (all-zero base
-- that is then overridden by the sparse data).
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
	local bufferView = glb.bufferViews[accessor.bufferView + 1] -- Convert to 1-indexed

	if bufferView then
		-- Normal path: read the base bufferView data.
		local stride = bufferView.byteStride or (componentSize * componentCount)
		local baseOffset = bufferView.byteOffset + accessor.byteOffset
		for i = 0, accessor.count - 1 do
			local elementOffset = baseOffset + i * stride
			for j = 0, componentCount - 1 do
				result[i * componentCount + j + 1] =
					readComponent(glb, accessor.componentType, elementOffset + j * componentSize)
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

	return result
end

-- Read indices from accessor
local function readIndices(glb: GLBData, accessorIndex: number): { number }?
	if accessorIndex < 1 or accessorIndex > #glb.accessors then
		return nil
	end

	local accessor = glb.accessors[accessorIndex]
	local bufferView = glb.bufferViews[accessor.bufferView + 1]

	if not bufferView then
		return nil
	end

	local componentSize = getComponentSize(accessor.componentType)
	local baseOffset = bufferView.byteOffset + accessor.byteOffset
	local result: { number } = {}

	for i = 0, accessor.count - 1 do
		local valueOffset = baseOffset + i * componentSize
		local value: number

		if accessor.componentType == 5123 then
			-- UNSIGNED_SHORT
			value = readU16(glb.binChunk, valueOffset)
		elseif accessor.componentType == 5125 then
			-- UNSIGNED_INT
			value = readU32(glb.binChunk, valueOffset)
		elseif accessor.componentType == 5121 then
			-- UNSIGNED_BYTE
			value = buffer.readu8(glb.binChunk, valueOffset)
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
	local bufferView = glb.bufferViews[accessor.bufferView + 1]

	if not bufferView then
		return nil
	end

	local componentSize = getComponentSize(accessor.componentType)
	local componentCount = getTypeComponents(accessor.accessorType) -- Should be 4 for VEC4

	if componentSize == 0 or componentCount == 0 then
		return nil
	end

	local stride = bufferView.byteStride or (componentSize * componentCount)
	local baseOffset = bufferView.byteOffset + accessor.byteOffset
	local result: { number } = {}

	for i = 0, accessor.count - 1 do
		local elementOffset = baseOffset + i * stride
		for j = 0, componentCount - 1 do
			local valueOffset = elementOffset + j * componentSize
			local value: number

			if accessor.componentType == 5121 then
				-- UNSIGNED_BYTE
				value = buffer.readu8(glb.binChunk, valueOffset)
			elseif accessor.componentType == 5123 then
				-- UNSIGNED_SHORT
				value = readU16(glb.binChunk, valueOffset)
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

-- Extract embedded image bytes (PNG/JPEG) for a glTF image index (0-based).
local function imageBytes(glb: GLBData, imageIndex: number): buffer?
	local img = glb.images[imageIndex + 1]
	if not img or img.bufferView == nil then
		return nil
	end
	local bv = glb.bufferViews[img.bufferView + 1]
	if not bv then
		return nil
	end
	local out = buffer.create(bv.byteLength)
	buffer.copy(out, 0, glb.binChunk, bv.byteOffset, bv.byteLength)
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
