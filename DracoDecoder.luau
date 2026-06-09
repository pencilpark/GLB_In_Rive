--!strict
--[[
  DracoDecoder -- pure-Luau decoder for KHR_draco_mesh_compression.
  (stub -- implementation follows)
]]

export type DracoAttribute = {
	values: { number }, -- flat component array, count * componentCount entries
	componentCount: number,
}

export type DracoMesh = {
	numPoints: number,
	numFaces: number,
	indices: { number }, -- flat triangle indices (0-based values)
	attributes: { [number]: DracoAttribute }, -- keyed by draco unique id
}

local function decode(src: buffer): (DracoMesh?, string?)
	return nil, "Draco decoding not implemented yet"
end

return {
	decode = decode,
}
