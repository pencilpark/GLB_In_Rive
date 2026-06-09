--!strict
--[[
  DracoDecoder -- pure-Luau decoder for KHR_draco_mesh_compression.
  Port of the Draco decoder (bitstream version 2.2) following the reference
  C++ implementation (github.com/google/draco, Apache-2.0) and the Draco
  Bitstream Specification.

  Supports triangular meshes encoded with:
   - sequential connectivity (compressed + raw indices)
   - edgebreaker connectivity (standard + valence traversal decoders)
   - attribute decoders: generic / integer / quantization / normals
   - prediction: difference, parallelogram, constrained multi-parallelogram,
     portable texcoords, geometric normals
   - transforms: wrap, normal-octahedron-canonicalized
   - traversals: depth-first and max-prediction-degree

  decode(buffer) -> (DracoMesh?, string?)
]]

local band = bit32.band
local rshift = bit32.rshift

export type DracoAttribute = {
	values: { number }, -- flat per-point component array (count*comps entries)
	componentCount: number,
}

export type DracoMesh = {
	numPoints: number,
	numFaces: number,
	indices: { number }, -- flat triangle point indices (0-based values)
	attributes: { [number]: DracoAttribute }, -- keyed by draco unique id
}

local INVALID = -1

-- ---------------------------------------------------------------------------
-- float32 helpers (the reference decoder computes float paths in float32)
-- ---------------------------------------------------------------------------
local f32buf = buffer.create(4)
local function f32(x: number): number
	buffer.writef32(f32buf, 0, x)
	return buffer.readf32(f32buf, 0)
end

-- truncating integer division (C semantics), exact for |a| < 2^53
local function truncDiv(a: number, b: number): number
	local q = math.floor(math.abs(a) / math.abs(b))
	-- guard against double rounding at the boundary
	local ab, bb = math.abs(a), math.abs(b)
	while q * bb > ab do
		q -= 1
	end
	while (q + 1) * bb <= ab do
		q += 1
	end
	if (a < 0) ~= (b < 0) then
		q = -q
	end
	return q
end

-- Exact floor(sqrt(a * b)) for non-negative integers a, b < 2^53 (the product
-- may exceed double precision, so comparisons use 26-bit limb arithmetic).
local LIMB = 67108864 -- 2^26
local function mulLimbs(a: number, b: number): (number, number, number, number)
	-- a, b < 2^53 -> 26-bit limbs a = a1*2^26 + a0 (a1 < 2^27)
	local a1 = math.floor(a / LIMB)
	local a0 = a - a1 * LIMB
	local b1 = math.floor(b / LIMB)
	local b0 = b - b1 * LIMB
	local p0 = a0 * b0 -- < 2^52
	local p1 = a1 * b0 + a0 * b1 -- < 2^54 (ok, exact integers)
	local p2 = a1 * b1 -- < 2^54
	-- normalize to limbs l0..l3 (26-bit each)
	local l0 = p0 % LIMB
	local c = math.floor(p0 / LIMB) + p1 % LIMB
	local l1 = c % LIMB
	c = math.floor(c / LIMB) + math.floor(p1 / LIMB) + p2 % LIMB
	local l2 = c % LIMB
	local l3 = math.floor(c / LIMB) + math.floor(p2 / LIMB)
	return l3, l2, l1, l0
end

local function mulLE(a1: number, a2: number, b1: number, b2: number): boolean
	-- is a1*a2 <= b1*b2 ?
	local x3, x2, x1, x0 = mulLimbs(a1, a2)
	local y3, y2, y1, y0 = mulLimbs(b1, b2)
	if x3 ~= y3 then
		return x3 < y3
	end
	if x2 ~= y2 then
		return x2 < y2
	end
	if x1 ~= y1 then
		return x1 < y1
	end
	return x0 <= y0
end

local function intSqrtProduct(a: number, b: number): number
	if a == 0 or b == 0 then
		return 0
	end
	local s = math.floor(math.sqrt(a) * math.sqrt(b))
	-- adjust for floating point error: find max s with s*s <= a*b
	while not mulLE(s, s, a, b) do
		s -= 1
	end
	while mulLE(s + 1, s + 1, a, b) do
		s += 1
	end
	return s
end

-- ---------------------------------------------------------------------------
-- byte reader
-- ---------------------------------------------------------------------------
type Reader = {
	buf: buffer,
	pos: number,
	len: number,
}

local function mkReader(buf: buffer, pos: number, len: number?): Reader
	return { buf = buf, pos = pos, len = len or buffer.len(buf) }
end

local function rU8(r: Reader): number
	local v = buffer.readu8(r.buf, r.pos)
	r.pos += 1
	return v
end

local function rI8(r: Reader): number
	local v = buffer.readu8(r.buf, r.pos)
	r.pos += 1
	return if v >= 128 then v - 256 else v
end

local function rU16(r: Reader): number
	local v = buffer.readu16(r.buf, r.pos)
	r.pos += 2
	return v
end

local function rI16(r: Reader): number
	local v = buffer.readu16(r.buf, r.pos)
	r.pos += 2
	return if v >= 32768 then v - 65536 else v
end

local function rU32(r: Reader): number
	local v = buffer.readu32(r.buf, r.pos)
	r.pos += 4
	return v
end

local function rI32(r: Reader): number
	local v = buffer.readu32(r.buf, r.pos)
	r.pos += 4
	return if v >= 2147483648 then v - 4294967296 else v
end

local function rF32(r: Reader): number
	local v = buffer.readf32(r.buf, r.pos)
	r.pos += 4
	return v
end

-- LEB128 (7-bit groups, least significant first)
local function rVarint(r: Reader): number
	local result = 0
	local shift = 1
	while true do
		local b = buffer.readu8(r.buf, r.pos)
		r.pos += 1
		result += band(b, 127) * shift
		if b < 128 then
			break
		end
		shift *= 128
	end
	return result
end

-- LSB-first bit reader over the byte stream (DecoderBuffer::BitDecoder).
type BitReader = {
	r: Reader,
	base: number, -- byte position where bit decoding started
	bitOffset: number,
}

local function startBits(r: Reader): BitReader
	return { r = r, base = r.pos, bitOffset = 0 }
end

local function readBit(br: BitReader): number
	local off = br.bitOffset
	local byteOff = br.base + rshift(off, 3)
	local bit = 0
	if byteOff < br.r.len then
		bit = band(rshift(buffer.readu8(br.r.buf, byteOff), band(off, 7)), 1)
	end
	br.bitOffset = off + 1
	return bit
end

-- value |= bit << i (LSB-first assembly, like DecodeLeastSignificantBits32)
local function readBitsLSB(br: BitReader, nbits: number): number
	local v = 0
	local m = 1
	for _ = 1, nbits do
		v += readBit(br) * m
		m *= 2
	end
	return v
end

local function endBits(br: BitReader)
	br.r.pos = br.base + math.floor((br.bitOffset + 7) / 8)
end

-- ---------------------------------------------------------------------------
-- rANS decoders
-- ---------------------------------------------------------------------------
type AnsState = {
	buf: buffer,
	base: number, -- byte offset of the ans data start
	bufOffset: number, -- bytes remaining (offset from base)
	state: number,
}

-- ANS decoder init: reads the final state (1-4 bytes, tagged by the top two
-- bits of the last byte) and re-adds the encoder's normalization base.
-- The rabs bit decoder uses lBase = 4096 (3-byte states max); the rANS symbol
-- decoder uses lBase = rans_precision * 4 (4-byte states allowed).
local function ansReadInit(buf: buffer, base: number, offset: number, lBase: number): (AnsState?, string?)
	if offset < 1 then
		return { buf = buf, base = base, bufOffset = 0, state = lBase }, nil
	end
	local last = buffer.readu8(buf, base + offset - 1)
	local x = rshift(last, 6)
	local st: AnsState = { buf = buf, base = base, bufOffset = 0, state = 0 }
	if x == 0 then
		st.bufOffset = offset - 1
		st.state = band(last, 0x3F)
	elseif x == 1 then
		if offset < 2 then
			return nil, "ans init underflow"
		end
		st.bufOffset = offset - 2
		st.state = band(buffer.readu16(buf, base + offset - 2), 0x3FFF)
	elseif x == 2 then
		if offset < 3 then
			return nil, "ans init underflow"
		end
		st.bufOffset = offset - 3
		st.state = band(
			buffer.readu8(buf, base + offset - 3)
				+ buffer.readu8(buf, base + offset - 2) * 256
				+ buffer.readu8(buf, base + offset - 1) * 65536,
			0x3FFFFF
		)
	else
		if offset < 4 then
			return nil, "ans init underflow"
		end
		st.bufOffset = offset - 4
		st.state = buffer.readu32(buf, base + offset - 4) % 1073741824 -- & 0x3FFFFFFF
	end
	st.state += lBase
	if st.state >= lBase * 256 then
		return nil, "ans init state invalid"
	end
	return st, nil
end

-- rabs (binary) read with probability of zero p0 (8-bit precision).
local function rabsRead(st: AnsState, p0: number): number
	local p = 256 - p0
	if st.state < 4096 and st.bufOffset > 0 then
		st.bufOffset -= 1
		st.state = st.state * 256 + buffer.readu8(st.buf, st.base + st.bufOffset)
	end
	local x = st.state
	local quot = math.floor(x / 256)
	local rem = x - quot * 256
	local xn = quot * p
	if rem < p then
		st.state = xn + rem
		return 1
	else
		st.state = x - xn - p
		return 0
	end
end

-- RAnsBitDecoder: prob_zero u8 + varint size + ans data.
type RansBitDecoder = { st: AnsState, probZero: number }

local function startRansBitDecoder(r: Reader): (RansBitDecoder?, string?)
	local probZero = rU8(r)
	local size = rVarint(r)
	if r.pos + size > r.len then
		return nil, "rans bit decoder size overflow"
	end
	local st, err = ansReadInit(r.buf, r.pos, size, 4096)
	if not st then
		return nil, err
	end
	r.pos += size
	return { st = st :: AnsState, probZero = probZero }, nil
end

local function decodeNextBit(d: RansBitDecoder): number
	return rabsRead(d.st, d.probZero)
end

-- rANS symbol decoder (RAnsSymbolDecoder<L>).
type RansSymbolDecoder = {
	st: AnsState,
	precision: number,
	lBase: number,
	lut: { number }, -- rem -> symbol (1-based array over 0..precision-1)
	probs: { number }, -- per symbol
	cumProbs: { number },
	numSymbols: number,
}

local function ransPrecisionBits(uniqueSymbolsBitLength: number): number
	local p = math.floor((3 * uniqueSymbolsBitLength) / 2)
	if p < 12 then
		p = 12
	end
	if p > 20 then
		p = 20
	end
	return p
end

local function createRansSymbolDecoder(r: Reader, uniqueBitLength: number): (RansSymbolDecoder?, string?)
	local precision = 2 ^ ransPrecisionBits(uniqueBitLength)
	local numSymbols = rVarint(r)
	local probs: { number } = table.create(numSymbols, 0)
	local i = 0
	while i < numSymbols do
		local probData = rU8(r)
		local token = band(probData, 3)
		if token == 3 then
			local offset = rshift(probData, 2)
			if i + offset >= numSymbols then
				return nil, "rans prob table overflow"
			end
			for j = 0, offset do
				probs[i + j + 1] = 0
			end
			i += offset + 1
		else
			local prob = rshift(probData, 2)
			local m = 64
			for _ = 1, token do
				local eb = rU8(r)
				prob += eb * m
				m *= 256
			end
			probs[i + 1] = prob
			i += 1
		end
	end
	-- build lookup table
	local lut = table.create(precision, 0)
	local cumProbs: { number } = table.create(numSymbols, 0)
	local cum = 0
	for s = 1, numSymbols do
		cumProbs[s] = cum
		local p = probs[s]
		if p > 0 then
			for j = cum, cum + p - 1 do
				if j >= precision then
					return nil, "rans prob table sum overflow"
				end
				lut[j + 1] = s - 1
			end
		end
		cum += p
	end
	if numSymbols > 0 and cum ~= precision then
		return nil, "rans prob table does not sum to precision"
	end
	-- start decoding: varUI64 byte count + ans init (with l_base added)
	local bytesEncoded = rVarint(r)
	if r.pos + bytesEncoded > r.len then
		return nil, "rans data overflow"
	end
	local lBase = precision * 4
	local st, err = ansReadInit(r.buf, r.pos, bytesEncoded, lBase)
	if not st then
		return nil, err
	end
	r.pos += bytesEncoded
	local stt = st :: AnsState
	return {
		st = stt,
		precision = precision,
		lBase = lBase,
		lut = lut,
		probs = probs,
		cumProbs = cumProbs,
		numSymbols = numSymbols,
	}, nil
end

local function ransReadSymbol(d: RansSymbolDecoder): number
	local st = d.st
	local state = st.state
	while state < d.lBase and st.bufOffset > 0 do
		st.bufOffset -= 1
		state = state * 256 + buffer.readu8(st.buf, st.base + st.bufOffset)
	end
	local quo = math.floor(state / d.precision)
	local rem = state - quo * d.precision
	local sym = d.lut[rem + 1]
	st.state = quo * d.probs[sym + 1] + rem - d.cumProbs[sym + 1]
	return sym
end

-- DecodeSymbols: scheme u8 (0 tagged / 1 raw) + payload -> out array (0-based
-- contents in a 1-based Lua array).
local function decodeSymbols(r: Reader, numValues: number, numComponents: number): ({ number }?, string?)
	local out: { number } = table.create(numValues, 0)
	if numValues == 0 then
		return out, nil
	end
	local scheme = rU8(r)
	if scheme == 0 then
		-- tagged
		local tagDecoder, err = createRansSymbolDecoder(r, 5)
		if not tagDecoder then
			return nil, err
		end
		local br = startBits(r)
		local idx = 1
		local i = 0
		while i < numValues do
			local bitLength = ransReadSymbol(tagDecoder :: RansSymbolDecoder)
			for _ = 1, numComponents do
				out[idx] = readBitsLSB(br, bitLength)
				idx += 1
			end
			i += numComponents
		end
		endBits(br)
		return out, nil
	elseif scheme == 1 then
		-- raw
		local maxBitLength = rU8(r)
		if maxBitLength < 1 or maxBitLength > 18 then
			return nil, "invalid raw symbol bit length " .. maxBitLength
		end
		local dec, err = createRansSymbolDecoder(r, maxBitLength)
		if not dec then
			return nil, err
		end
		local d = dec :: RansSymbolDecoder
		if numValues > 0 and d.numSymbols == 0 then
			return nil, "no symbols"
		end
		for i = 1, numValues do
			out[i] = ransReadSymbol(d)
		end
		return out, nil
	end
	return nil, "unknown symbol coding scheme " .. scheme
end

-- ---------------------------------------------------------------------------
-- corner table helpers
-- ---------------------------------------------------------------------------
local function cNext(c: number): number
	if c < 0 then
		return c
	end
	return if c % 3 == 2 then c - 2 else c + 1
end

local function cPrev(c: number): number
	if c < 0 then
		return c
	end
	return if c % 3 == 0 then c + 2 else c - 1
end

-- Decoder context.
type Ctx = {
	r: Reader,
	-- connectivity
	encoderMethod: number,
	numFaces: number, -- total faces (corner table)
	numEncVerts: number,
	numAttributeData: number,
	numSymbols: number,
	numSplitSymbols: number,
	-- base corner table
	cornerToVert: { number }, -- per corner (1-based array, 0-based values)
	opposite: { number },
	vertexCorners: { number }, -- left-most corner per vertex
	numVertices: number,
	isVertHole: { boolean },
	facePoints: { number }, -- per corner: point id (after AssignPointsToCorners)
	numPoints: number,
	-- attribute connectivity (per attribute data index, 1-based)
	attSeamCorners: { { number } }, -- seam corner lists
	attEdgeOnSeam: { { boolean } },
	attVertOnSeam: { { boolean } },
	attCornerToVert: { { number } },
	attVertexCorners: { { number } }, -- left-most corner per attribute vertex
	attNumVertices: { number },
	-- traversal decoder
	traversalType: number, -- 0 standard, 2 valence
	symbolBits: BitReader?, -- standard symbol bit reader
	startFaceDecoder: RansBitDecoder?,
	seamDecoders: { RansBitDecoder },
	-- valence
	contextSymbols: { { number } },
	contextCounters: { number },
	vertexValences: { number },
	lastSymbol: number,
	activeContext: number,
	-- splits
	splitSourceIds: { number },
	splitSplitIds: { number },
	splitEdges: { number },
}

-- ---------------------------------------------------------------------------
-- edgebreaker connectivity
-- ---------------------------------------------------------------------------
local TOPOLOGY_C = 0
local TOPOLOGY_S = 1
local TOPOLOGY_L = 3
local TOPOLOGY_R = 5
local TOPOLOGY_E = 7
local SYMBOL_TO_TOPOLOGY = { [0] = TOPOLOGY_C, [1] = TOPOLOGY_S, [2] = TOPOLOGY_L, [3] = TOPOLOGY_R, [4] = TOPOLOGY_E }

local function ebDecodeSymbol(ctx: Ctx): number
	if ctx.traversalType == 2 then
		-- valence
		if ctx.activeContext >= 0 then
			local cidx = ctx.activeContext + 1
			local counter = ctx.contextCounters[cidx] - 1
			ctx.contextCounters[cidx] = counter
			if counter < 0 then
				return -1
			end
			local symbolId = ctx.contextSymbols[cidx][counter + 1]
			if symbolId == nil or symbolId > 4 then
				return -1
			end
			ctx.lastSymbol = SYMBOL_TO_TOPOLOGY[symbolId]
		else
			ctx.lastSymbol = TOPOLOGY_E
		end
		return ctx.lastSymbol
	else
		local br = ctx.symbolBits :: BitReader
		local symbol = readBit(br)
		if symbol ~= TOPOLOGY_C then
			local suffix = readBitsLSB(br, 2)
			symbol += suffix * 2
		end
		ctx.lastSymbol = symbol
		return symbol
	end
end

local function valenceNewActiveCorner(ctx: Ctx, corner: number)
	if ctx.traversalType ~= 2 then
		return
	end
	local nextC = cNext(corner)
	local prevC = cPrev(corner)
	local vv = ctx.vertexValences
	local vNext = ctx.cornerToVert[nextC + 1]
	local vPrev = ctx.cornerToVert[prevC + 1]
	local vTip = ctx.cornerToVert[corner + 1]
	local s = ctx.lastSymbol
	if s == TOPOLOGY_C or s == TOPOLOGY_S then
		vv[vNext + 1] += 1
		vv[vPrev + 1] += 1
	elseif s == TOPOLOGY_R then
		vv[vTip + 1] += 1
		vv[vNext + 1] += 1
		vv[vPrev + 1] += 2
	elseif s == TOPOLOGY_L then
		vv[vTip + 1] += 1
		vv[vNext + 1] += 2
		vv[vPrev + 1] += 1
	elseif s == TOPOLOGY_E then
		vv[vTip + 1] += 2
		vv[vNext + 1] += 2
		vv[vPrev + 1] += 2
	end
	local activeValence = vv[vNext + 1]
	local clamped = math.clamp(activeValence, 2, 7)
	ctx.activeContext = clamped - 2
end

local function setOpposite(ctx: Ctx, a: number, b: number)
	ctx.opposite[a + 1] = b
	ctx.opposite[b + 1] = a
end

local function addNewVertex(ctx: Ctx): number
	ctx.numVertices += 1
	ctx.vertexCorners[ctx.numVertices] = INVALID
	return ctx.numVertices - 1
end

local function isTopologySplit(ctx: Ctx, encoderSymbolId: number): (boolean, number, number)
	local n = #ctx.splitSourceIds
	if n == 0 then
		return false, 0, 0
	end
	if ctx.splitSourceIds[n] > encoderSymbolId then
		return true, 0, -1 -- corrupted
	end
	if ctx.splitSourceIds[n] ~= encoderSymbolId then
		return false, 0, 0
	end
	local edge = ctx.splitEdges[n]
	local splitId = ctx.splitSplitIds[n]
	table.remove(ctx.splitSourceIds, n)
	table.remove(ctx.splitSplitIds, n)
	table.remove(ctx.splitEdges, n)
	return true, edge, splitId
end

local function decodeEdgebreakerConnectivity(ctx: Ctx): string?
	local r = ctx.r
	-- header
	ctx.traversalType = rU8(r)
	ctx.numEncVerts = rVarint(r)
	local numFaces = rVarint(r)
	ctx.numAttributeData = rU8(r)
	ctx.numSymbols = rVarint(r)
	ctx.numSplitSymbols = rVarint(r)
	ctx.numFaces = numFaces
	if ctx.traversalType ~= 0 and ctx.traversalType ~= 2 then
		return "unsupported edgebreaker traversal type " .. ctx.traversalType
	end
	if ctx.numSymbols > numFaces then
		return "more symbols than faces"
	end

	-- topology split events
	local numSplits = rVarint(r)
	local lastSource = 0
	for _ = 1, numSplits do
		local delta = rVarint(r)
		local source = delta + lastSource
		local delta2 = rVarint(r)
		if delta2 > source then
			return "invalid topology split data"
		end
		table.insert(ctx.splitSourceIds, source)
		table.insert(ctx.splitSplitIds, source - delta2)
		lastSource = source
	end
	if numSplits > 0 then
		local br = startBits(r)
		for i = 1, numSplits do
			ctx.splitEdges[i] = readBit(br)
		end
		endBits(br)
	end

	-- traversal start: symbol buffer (standard only), start faces, seams
	if ctx.traversalType == 0 then
		local traversalSize = rVarint(r)
		if r.pos + traversalSize > r.len then
			return "symbol buffer overflow"
		end
		local symReader = mkReader(r.buf, r.pos, r.pos + traversalSize)
		ctx.symbolBits = startBits(symReader)
		r.pos += traversalSize
	end
	do
		local sfd, err = startRansBitDecoder(r)
		if not sfd then
			return err
		end
		ctx.startFaceDecoder = sfd
	end
	for i = 1, ctx.numAttributeData do
		local sd, err = startRansBitDecoder(r)
		if not sd then
			return err
		end
		ctx.seamDecoders[i] = sd :: RansBitDecoder
	end
	if ctx.traversalType == 2 then
		-- valence context symbols
		ctx.vertexValences = table.create(ctx.numEncVerts + ctx.numSplitSymbols, 0)
		for i = 1, 6 do
			local numCtxSymbols = rVarint(r)
			if numCtxSymbols > numFaces then
				return "too many valence context symbols"
			end
			if numCtxSymbols > 0 then
				local syms, err = decodeSymbols(r, numCtxSymbols, 1)
				if not syms then
					return err
				end
				ctx.contextSymbols[i] = syms :: { number }
				ctx.contextCounters[i] = numCtxSymbols
			else
				ctx.contextSymbols[i] = {}
				ctx.contextCounters[i] = 0
			end
		end
	end

	-- corner table
	local numCorners = numFaces * 3
	ctx.cornerToVert = table.create(numCorners, INVALID)
	ctx.opposite = table.create(numCorners, INVALID)
	ctx.vertexCorners = {}
	ctx.numVertices = 0
	local maxNumVertices = ctx.numEncVerts + ctx.numSplitSymbols
	ctx.isVertHole = table.create(maxNumVertices, true)

	local activeCornerStack: { number } = {}
	local topologySplitActiveCorners: { [number]: number } = {}
	local invalidVertices: { number } = {}
	local removeInvalidVertices = ctx.numAttributeData == 0

	local facesCreated = 0
	for symbolId = 0, ctx.numSymbols - 1 do
		local face = facesCreated
		facesCreated += 1
		local checkTopologySplit = false
		local symbol = ebDecodeSymbol(ctx)
		local corner = 3 * face
		if symbol == TOPOLOGY_C then
			if #activeCornerStack == 0 then
				return "edgebreaker: empty stack on C"
			end
			local cornerA = activeCornerStack[#activeCornerStack]
			local vertexX = ctx.cornerToVert[cNext(cornerA) + 1]
			local cornerB = cNext(ctx.vertexCorners[vertexX + 1])
			if cornerA == cornerB then
				return "edgebreaker: C corners equal"
			end
			if ctx.opposite[cornerA + 1] >= 0 or ctx.opposite[cornerB + 1] >= 0 then
				return "edgebreaker: C corner already has opposite"
			end
			setOpposite(ctx, cornerA, corner + 1)
			setOpposite(ctx, cornerB, corner + 2)
			local vertAPrev = ctx.cornerToVert[cPrev(cornerA) + 1]
			local vertBNext = ctx.cornerToVert[cNext(cornerB) + 1]
			if vertexX == vertAPrev or vertexX == vertBNext then
				return "edgebreaker: degenerate C face"
			end
			ctx.cornerToVert[corner + 1] = vertexX
			ctx.cornerToVert[corner + 2] = vertBNext
			ctx.cornerToVert[corner + 3] = vertAPrev
			ctx.vertexCorners[vertAPrev + 1] = corner + 2
			ctx.isVertHole[vertexX + 1] = false
			activeCornerStack[#activeCornerStack] = corner
		elseif symbol == TOPOLOGY_R or symbol == TOPOLOGY_L then
			if #activeCornerStack == 0 then
				return "edgebreaker: empty stack on R/L"
			end
			local cornerA = activeCornerStack[#activeCornerStack]
			if ctx.opposite[cornerA + 1] >= 0 then
				return "edgebreaker: R/L corner already has opposite"
			end
			local oppCorner: number, cornerL: number, cornerR: number
			if symbol == TOPOLOGY_R then
				oppCorner = corner + 2
				cornerL = corner + 1
				cornerR = corner
			else
				oppCorner = corner + 1
				cornerL = corner
				cornerR = corner + 2
			end
			setOpposite(ctx, oppCorner, cornerA)
			local newVert = addNewVertex(ctx)
			if ctx.numVertices > maxNumVertices then
				return "edgebreaker: too many vertices"
			end
			ctx.cornerToVert[oppCorner + 1] = newVert
			ctx.vertexCorners[newVert + 1] = oppCorner
			local vertexR = ctx.cornerToVert[cPrev(cornerA) + 1]
			ctx.cornerToVert[cornerR + 1] = vertexR
			ctx.vertexCorners[vertexR + 1] = cornerR
			ctx.cornerToVert[cornerL + 1] = ctx.cornerToVert[cNext(cornerA) + 1]
			activeCornerStack[#activeCornerStack] = corner
			checkTopologySplit = true
		elseif symbol == TOPOLOGY_S then
			if #activeCornerStack == 0 then
				return "edgebreaker: empty stack on S"
			end
			local cornerB = table.remove(activeCornerStack) :: number
			local split = topologySplitActiveCorners[symbolId]
			if split ~= nil then
				table.insert(activeCornerStack, split)
			end
			if #activeCornerStack == 0 then
				return "edgebreaker: empty stack on S (post split)"
			end
			local cornerA = activeCornerStack[#activeCornerStack]
			if cornerA == cornerB then
				return "edgebreaker: S corners equal"
			end
			if ctx.opposite[cornerA + 1] >= 0 or ctx.opposite[cornerB + 1] >= 0 then
				return "edgebreaker: S corner already has opposite"
			end
			setOpposite(ctx, cornerA, corner + 2)
			setOpposite(ctx, cornerB, corner + 1)
			local vertexP = ctx.cornerToVert[cPrev(cornerA) + 1]
			ctx.cornerToVert[corner + 1] = vertexP
			ctx.cornerToVert[corner + 2] = ctx.cornerToVert[cNext(cornerA) + 1]
			local vertBPrev = ctx.cornerToVert[cPrev(cornerB) + 1]
			ctx.cornerToVert[corner + 3] = vertBPrev
			ctx.vertexCorners[vertBPrev + 1] = corner + 2
			local cornerN = cNext(cornerB)
			local vertexN = ctx.cornerToVert[cornerN + 1]
			if ctx.traversalType == 2 then
				-- MergeVertices valence update
				ctx.vertexValences[vertexP + 1] += ctx.vertexValences[vertexN + 1]
			end
			ctx.vertexCorners[vertexP + 1] = ctx.vertexCorners[vertexN + 1]
			-- update vertex id on all corners connected CCW to corner_n
			local firstCorner = cornerN
			while cornerN >= 0 do
				ctx.cornerToVert[cornerN + 1] = vertexP
				-- SwingLeft on base table
				local oppN = ctx.opposite[cNext(cornerN) + 1]
				cornerN = if oppN == nil or oppN < 0 then INVALID else cNext(oppN)
				if cornerN == firstCorner then
					return "edgebreaker: S vertex loop"
				end
			end
			ctx.vertexCorners[vertexN + 1] = INVALID
			if removeInvalidVertices then
				table.insert(invalidVertices, vertexN)
			end
			activeCornerStack[#activeCornerStack] = corner
		elseif symbol == TOPOLOGY_E then
			local v0 = addNewVertex(ctx)
			local v1 = addNewVertex(ctx)
			local v2 = addNewVertex(ctx)
			if ctx.numVertices > maxNumVertices then
				return "edgebreaker: too many vertices (E)"
			end
			ctx.cornerToVert[corner + 1] = v0
			ctx.cornerToVert[corner + 2] = v1
			ctx.cornerToVert[corner + 3] = v2
			ctx.vertexCorners[v0 + 1] = corner
			ctx.vertexCorners[v1 + 1] = corner + 1
			ctx.vertexCorners[v2 + 1] = corner + 2
			table.insert(activeCornerStack, corner)
			checkTopologySplit = true
		else
			return "edgebreaker: invalid symbol"
		end

		valenceNewActiveCorner(ctx, activeCornerStack[#activeCornerStack])

		if checkTopologySplit then
			local encoderSymbolId = ctx.numSymbols - symbolId - 1
			while true do
				local isSplit, splitEdge, encSplitId = isTopologySplit(ctx, encoderSymbolId)
				if not isSplit then
					break
				end
				if encSplitId < 0 then
					return "edgebreaker: invalid split symbol id"
				end
				local actTopCorner = activeCornerStack[#activeCornerStack]
				local newActiveCorner: number
				if splitEdge == 1 then -- RIGHT_FACE_EDGE
					newActiveCorner = cNext(actTopCorner)
				else
					newActiveCorner = cPrev(actTopCorner)
				end
				local decSplitId = ctx.numSymbols - encSplitId - 1
				topologySplitActiveCorners[decSplitId] = newActiveCorner
			end
		end
	end

	if ctx.numVertices > maxNumVertices then
		return "edgebreaker: unexpected vertex count"
	end

	-- start faces
	local sfd = ctx.startFaceDecoder :: RansBitDecoder
	while #activeCornerStack > 0 do
		local corner = table.remove(activeCornerStack) :: number
		local interior = decodeNextBit(sfd) > 0
		if interior then
			if facesCreated >= numFaces then
				return "edgebreaker: too many faces"
			end
			local cornerA = corner
			local vertN = ctx.cornerToVert[cNext(cornerA) + 1]
			local cornerB = cNext(ctx.vertexCorners[vertN + 1])
			local vertX = ctx.cornerToVert[cNext(cornerB) + 1]
			local cornerC = cNext(ctx.vertexCorners[vertX + 1])
			if corner == cornerB or corner == cornerC or cornerB == cornerC then
				return "edgebreaker: invalid start face"
			end
			if
				ctx.opposite[corner + 1] >= 0
				or ctx.opposite[cornerB + 1] >= 0
				or ctx.opposite[cornerC + 1] >= 0
			then
				return "edgebreaker: start face corner already opposite"
			end
			local vertP = ctx.cornerToVert[cNext(cornerC) + 1]
			local face = facesCreated
			facesCreated += 1
			local newCorner = 3 * face
			setOpposite(ctx, newCorner, corner)
			setOpposite(ctx, newCorner + 1, cornerB)
			setOpposite(ctx, newCorner + 2, cornerC)
			ctx.cornerToVert[newCorner + 1] = vertX
			ctx.cornerToVert[newCorner + 2] = vertP
			ctx.cornerToVert[newCorner + 3] = vertN
			ctx.isVertHole[vertX + 1] = false
			ctx.isVertHole[vertP + 1] = false
			ctx.isVertHole[vertN + 1] = false
		end
	end
	if facesCreated ~= numFaces then
		return "edgebreaker: face count mismatch (" .. facesCreated .. " vs " .. numFaces .. ")"
	end

	-- remove isolated vertices created by S symbols (only when no attribute data)
	local numVertices = ctx.numVertices
	for _, invalidVert in ipairs(invalidVertices) do
		local srcVert = numVertices - 1
		while ctx.vertexCorners[srcVert + 1] == INVALID do
			numVertices -= 1
			srcVert = numVertices - 1
		end
		if srcVert >= invalidVert then
			-- remap all corners on srcVert to invalidVert
			local first = ctx.vertexCorners[srcVert + 1]
			-- iterate the corner ring around srcVert (VertexCornersIterator):
			-- walk left then right from the left-most corner
			local c = first
			while c >= 0 do
				if ctx.cornerToVert[c + 1] ~= srcVert then
					return "edgebreaker: corrupted vertex remap"
				end
				ctx.cornerToVert[c + 1] = invalidVert
				-- swing right on base table
				local oppP = ctx.opposite[cPrev(c) + 1]
				c = if oppP == nil or oppP < 0 then INVALID else cPrev(oppP)
				if c == first then
					break
				end
			end
			ctx.vertexCorners[invalidVert + 1] = first
			ctx.vertexCorners[srcVert + 1] = INVALID
			ctx.isVertHole[invalidVert + 1] = ctx.isVertHole[srcVert + 1]
			ctx.isVertHole[srcVert + 1] = false
			numVertices -= 1
		end
	end
	ctx.numVertices = numVertices

	-- attribute seams
	if ctx.numAttributeData > 0 then
		for i = 1, ctx.numAttributeData do
			ctx.attSeamCorners[i] = {}
		end
		for c = 0, numFaces * 3 - 1, 3 do
			local srcFace = math.floor(c / 3)
			for k = 0, 2 do
				local corner = c + k
				local oppCorner = ctx.opposite[corner + 1]
				if oppCorner < 0 then
					for i = 1, ctx.numAttributeData do
						table.insert(ctx.attSeamCorners[i], corner)
					end
				else
					local oppFace = math.floor(oppCorner / 3)
					if oppFace >= srcFace then
						for i = 1, ctx.numAttributeData do
							local isSeam = decodeNextBit(ctx.seamDecoders[i]) > 0
							if isSeam then
								table.insert(ctx.attSeamCorners[i], corner)
							end
						end
					end
				end
			end
		end
	end

	return nil
end

-- Build the per-attribute corner tables (seams) and recompute vertices.
local function buildAttributeCornerTables(ctx: Ctx): string?
	local numCorners = ctx.numFaces * 3
	for i = 1, ctx.numAttributeData do
		local edgeOnSeam = table.create(numCorners, false)
		local vertOnSeam = table.create(ctx.numVertices, false)
		for _, c in ipairs(ctx.attSeamCorners[i]) do
			edgeOnSeam[c + 1] = true
			vertOnSeam[ctx.cornerToVert[cNext(c) + 1] + 1] = true
			vertOnSeam[ctx.cornerToVert[cPrev(c) + 1] + 1] = true
			local opp = ctx.opposite[c + 1]
			if opp >= 0 then
				edgeOnSeam[opp + 1] = true
				vertOnSeam[ctx.cornerToVert[cNext(opp) + 1] + 1] = true
				vertOnSeam[ctx.cornerToVert[cPrev(opp) + 1] + 1] = true
			end
		end
		ctx.attEdgeOnSeam[i] = edgeOnSeam
		ctx.attVertOnSeam[i] = vertOnSeam

		-- RecomputeVerticesInternal (identity mapping path)
		local attCornerToVert = table.create(numCorners, INVALID)
		local attVertexCorners: { number } = {}
		local numNewVertices = 0

		-- attribute table SwingLeft/SwingRight respect seam edges
		local opposite = ctx.opposite
		local function attOpposite(c: number): number
			if c < 0 or edgeOnSeam[c + 1] then
				return INVALID
			end
			local o = opposite[c + 1]
			return if o == nil then INVALID else o
		end
		local function attSwingLeft(c: number): number
			local o = attOpposite(cNext(c))
			return if o < 0 then INVALID else cNext(o)
		end

		for v = 0, ctx.numVertices - 1 do
			local c = ctx.vertexCorners[v + 1]
			if c == INVALID then
				continue
			end
			local firstVertId = numNewVertices
			numNewVertices += 1
			local firstC = c
			if vertOnSeam[v + 1] then
				local actC = attSwingLeft(firstC)
				while actC >= 0 do
					firstC = actC
					actC = attSwingLeft(actC)
					if actC == c then
						return "attribute corner table: seam loop"
					end
				end
			end
			attCornerToVert[firstC + 1] = firstVertId
			attVertexCorners[firstVertId + 1] = firstC
			-- swing right on BASE table
			local function baseSwingRight(cc: number): number
				local o = opposite[cPrev(cc) + 1]
				return if o == nil or o < 0 then INVALID else cPrev(o)
			end
			local actC = baseSwingRight(firstC)
			while actC >= 0 and actC ~= firstC do
				if edgeOnSeam[cNext(actC) + 1] then
					-- IsCornerOppositeToSeamEdge(Next(act_c))
					firstVertId = numNewVertices
					numNewVertices += 1
					attVertexCorners[firstVertId + 1] = actC
				end
				attCornerToVert[actC + 1] = firstVertId
				actC = baseSwingRight(actC)
			end
		end
		ctx.attCornerToVert[i] = attCornerToVert
		ctx.attVertexCorners[i] = attVertexCorners
		ctx.attNumVertices[i] = numNewVertices
	end
	return nil
end

-- AssignPointsToCorners: deduplicate corners into point ids -> faces.
local function assignPointsToCorners(ctx: Ctx): string?
	local numCorners = ctx.numFaces * 3
	ctx.facePoints = table.create(numCorners, 0)
	if ctx.numAttributeData == 0 then
		for c = 0, numCorners - 1 do
			ctx.facePoints[c + 1] = ctx.cornerToVert[c + 1]
		end
		ctx.numPoints = ctx.numVertices
		return nil
	end

	local opposite = ctx.opposite
	local function baseSwingRight(cc: number): number
		local o = opposite[cPrev(cc) + 1]
		return if o == nil or o < 0 then INVALID else cPrev(o)
	end

	local cornerToPoint = table.create(numCorners, INVALID)
	local numPoints = 0
	for v = 0, ctx.numVertices - 1 do
		local c = ctx.vertexCorners[v + 1]
		if c == INVALID then
			continue
		end
		local dedupFirstCorner = c
		if not ctx.isVertHole[v + 1] then
			for i = 1, ctx.numAttributeData do
				if not ctx.attVertOnSeam[i][ctx.cornerToVert[c + 1] + 1] then
					continue
				end
				local vertId = ctx.attCornerToVert[i][c + 1]
				local actC = baseSwingRight(c)
				local seamFound = false
				while actC ~= c do
					if actC < 0 then
						return "assign points: invalid swing"
					end
					if ctx.attCornerToVert[i][actC + 1] ~= vertId then
						dedupFirstCorner = actC
						seamFound = true
						break
					end
					actC = baseSwingRight(actC)
				end
				if seamFound then
					break
				end
			end
		end

		c = dedupFirstCorner
		cornerToPoint[c + 1] = numPoints
		numPoints += 1
		local prevC = c
		c = baseSwingRight(c)
		while c >= 0 and c ~= dedupFirstCorner do
			local attributeSeam = false
			for i = 1, ctx.numAttributeData do
				if ctx.attCornerToVert[i][c + 1] ~= ctx.attCornerToVert[i][prevC + 1] then
					attributeSeam = true
					break
				end
			end
			if attributeSeam then
				cornerToPoint[c + 1] = numPoints
				numPoints += 1
			else
				cornerToPoint[c + 1] = cornerToPoint[prevC + 1]
			end
			prevC = c
			c = baseSwingRight(c)
		end
	end
	for i = 1, numCorners do
		ctx.facePoints[i] = cornerToPoint[i]
	end
	ctx.numPoints = numPoints
	return nil
end

-- ---------------------------------------------------------------------------
-- traversal (sequence generation) for attribute decoding
-- ---------------------------------------------------------------------------
type EncData = {
	valueToCorner: { number }, -- attribute value index -> corner
	vertToValue: { number }, -- (attribute) vertex -> value index
	numValues: number,
}

type TraversalTable = {
	numFaces: number,
	numVertices: number,
	vertexAt: (c: number) -> number,
	oppositeOf: (c: number) -> number,
	leftMost: (v: number) -> number,
}

local function traversalIsOnBoundary(t: TraversalTable, v: number): boolean
	local corner = t.leftMost(v)
	if corner < 0 then
		return true
	end
	-- SwingLeft == invalid?
	local o = t.oppositeOf(cNext(corner))
	return o < 0
end

local function traverseDepthFirst(t: TraversalTable, enc: EncData, isFaceVisited: { boolean }, isVertexVisited: { boolean })
	local function getRight(c: number): number
		return t.oppositeOf(cNext(c))
	end
	local function getLeft(c: number): number
		return t.oppositeOf(cPrev(c))
	end
	local function onNewVertexVisited(v: number, corner: number)
		table.insert(enc.valueToCorner, corner)
		enc.vertToValue[v + 1] = enc.numValues
		enc.numValues += 1
	end

	local stack: { number } = {}
	for f = 0, t.numFaces - 1 do
		local cornerId = 3 * f
		if isFaceVisited[f + 1] then
			continue
		end
		table.clear(stack)
		table.insert(stack, cornerId)
		local nextVert = t.vertexAt(cNext(cornerId))
		local prevVert = t.vertexAt(cPrev(cornerId))
		if nextVert < 0 or prevVert < 0 then
			return "traversal: invalid vertex"
		end
		if not isVertexVisited[nextVert + 1] then
			isVertexVisited[nextVert + 1] = true
			onNewVertexVisited(nextVert, cNext(cornerId))
		end
		if not isVertexVisited[prevVert + 1] then
			isVertexVisited[prevVert + 1] = true
			onNewVertexVisited(prevVert, cPrev(cornerId))
		end
		while #stack > 0 do
			cornerId = stack[#stack]
			local faceId = math.floor(cornerId / 3)
			if cornerId < 0 or isFaceVisited[faceId + 1] then
				table.remove(stack)
				continue
			end
			while true do
				faceId = math.floor(cornerId / 3)
				isFaceVisited[faceId + 1] = true
				local vertId = t.vertexAt(cornerId)
				if vertId < 0 then
					return "traversal: invalid vertex id"
				end
				if not isVertexVisited[vertId + 1] then
					local onBoundary = traversalIsOnBoundary(t, vertId)
					isVertexVisited[vertId + 1] = true
					onNewVertexVisited(vertId, cornerId)
					if not onBoundary then
						cornerId = getRight(cornerId)
						continue
					end
				end
				local rightCorner = getRight(cornerId)
				local leftCorner = getLeft(cornerId)
				local rightFace = if rightCorner < 0 then -1 else math.floor(rightCorner / 3)
				local leftFace = if leftCorner < 0 then -1 else math.floor(leftCorner / 3)
				local rightVisited = rightFace < 0 or isFaceVisited[rightFace + 1]
				local leftVisited = leftFace < 0 or isFaceVisited[leftFace + 1]
				if rightVisited then
					if leftVisited then
						table.remove(stack)
						break
					else
						cornerId = leftCorner
					end
				else
					if leftVisited then
						cornerId = rightCorner
					else
						stack[#stack] = leftCorner
						table.insert(stack, rightCorner)
						break
					end
				end
			end
		end
	end
	return nil
end

local function traverseMaxPredictionDegree(
	t: TraversalTable,
	enc: EncData,
	isFaceVisited: { boolean },
	isVertexVisited: { boolean }
)
	local kMaxPriority = 3
	local predictionDegree = table.create(t.numVertices, 0)
	local stacks: { { number } } = { {}, {}, {} }
	local bestPriority = 0

	local function getRight(c: number): number
		return t.oppositeOf(cNext(c))
	end
	local function getLeft(c: number): number
		return t.oppositeOf(cPrev(c))
	end
	local function onNewVertexVisited(v: number, corner: number)
		table.insert(enc.valueToCorner, corner)
		enc.vertToValue[v + 1] = enc.numValues
		enc.numValues += 1
	end
	local function computePriority(cornerId: number): number
		local vTip = t.vertexAt(cornerId)
		local priority = 0
		if not isVertexVisited[vTip + 1] then
			predictionDegree[vTip + 1] += 1
			priority = if predictionDegree[vTip + 1] > 1 then 1 else 2
		end
		if priority >= kMaxPriority then
			priority = kMaxPriority - 1
		end
		return priority
	end
	local function popNext(): number
		for i = bestPriority, kMaxPriority - 1 do
			local s = stacks[i + 1]
			if #s > 0 then
				local ret = table.remove(s) :: number
				bestPriority = i
				return ret
			end
		end
		return INVALID
	end
	local function pushStack(ci: number, priority: number)
		table.insert(stacks[priority + 1], ci)
		if priority < bestPriority then
			bestPriority = priority
		end
	end

	for f = 0, t.numFaces - 1 do
		local cornerId = 3 * f
		if isFaceVisited[f + 1] then
			continue
		end
		table.insert(stacks[1], cornerId)
		bestPriority = 0
		local nextVert = t.vertexAt(cNext(cornerId))
		local prevVert = t.vertexAt(cPrev(cornerId))
		if not isVertexVisited[nextVert + 1] then
			isVertexVisited[nextVert + 1] = true
			onNewVertexVisited(nextVert, cNext(cornerId))
		end
		if not isVertexVisited[prevVert + 1] then
			isVertexVisited[prevVert + 1] = true
			onNewVertexVisited(prevVert, cPrev(cornerId))
		end
		local tipVertex = t.vertexAt(cornerId)
		if not isVertexVisited[tipVertex + 1] then
			isVertexVisited[tipVertex + 1] = true
			onNewVertexVisited(tipVertex, cornerId)
		end
		while true do
			cornerId = popNext()
			if cornerId < 0 then
				break
			end
			local faceId = math.floor(cornerId / 3)
			if isFaceVisited[faceId + 1] then
				continue
			end
			while true do
				faceId = math.floor(cornerId / 3)
				isFaceVisited[faceId + 1] = true
				local vertId = t.vertexAt(cornerId)
				if not isVertexVisited[vertId + 1] then
					isVertexVisited[vertId + 1] = true
					onNewVertexVisited(vertId, cornerId)
				end
				local rightCorner = getRight(cornerId)
				local leftCorner = getLeft(cornerId)
				local rightFace = if rightCorner < 0 then -1 else math.floor(rightCorner / 3)
				local leftFace = if leftCorner < 0 then -1 else math.floor(leftCorner / 3)
				local rightVisited = rightFace < 0 or isFaceVisited[rightFace + 1]
				local leftVisited = leftFace < 0 or isFaceVisited[leftFace + 1]
				local continued = false
				if not leftVisited then
					local priority = computePriority(leftCorner)
					if rightVisited and priority <= bestPriority then
						cornerId = leftCorner
						continued = true
					else
						pushStack(leftCorner, priority)
					end
				end
				if not continued and not rightVisited then
					local priority = computePriority(rightCorner)
					if priority <= bestPriority then
						cornerId = rightCorner
						continued = true
					else
						pushStack(rightCorner, priority)
					end
				end
				if not continued then
					break
				end
			end
		end
	end
	return nil
end

-- ---------------------------------------------------------------------------
-- prediction transforms
-- ---------------------------------------------------------------------------
type AttState = {
	-- parsed attribute header
	attType: number,
	dataType: number,
	numComponents: number,
	normalized: boolean,
	uniqueId: number,
	seqType: number, -- sequential decoder type
	-- prediction
	scheme: number,
	transformType: number,
	-- wrap transform
	wrapMin: number,
	wrapMax: number,
	-- octahedron transform (prediction)
	octMaxQ: number,
	-- quantization (per component mins + range + bits)
	quantBits: number,
	quantMin: { number },
	quantRange: number,
	-- normals octahedral storage bits
	normQuantBits: number,
	-- decoded data
	values: { number }, -- portable int values, then transformed in place / replaced
	-- portable (pre-dequantization) integer values, kept for prediction
	-- schemes of child attributes (they must read quantized parent data)
	portableValues: { number }?,
	numValues: number,
}

local function wrapComputeOriginal(att: AttState, comps: number, pred: { number }, corr: { number }, corrOff: number, out: { number }, outOff: number)
	local minV = att.wrapMin
	local maxV = att.wrapMax
	local maxDif = 1 + maxV - minV
	for i = 1, comps do
		local p = pred[i]
		if p > maxV then
			p = maxV
		elseif p < minV then
			p = minV
		end
		local v = p + corr[corrOff + i]
		if v > maxV then
			v -= maxDif
		elseif v < minV then
			v += maxDif
		end
		out[outOff + i] = v
	end
end

-- normal octahedron canonicalized transform: diamond inversion. The C++
-- version computes in unsigned 32-bit then divides as signed; for valid data
-- the intermediate values stay small, so plain arithmetic with truncating
-- division matches it.
local function octInvertDiamondInt(s: number, t: number, centerValue: number): (number, number)
	local signS: number
	local signT: number
	if s >= 0 and t >= 0 then
		signS = 1
		signT = 1
	elseif s <= 0 and t <= 0 then
		signS = -1
		signT = -1
	else
		signS = if s > 0 then 1 else -1
		signT = if t > 0 then 1 else -1
	end
	local cornerS = signS * centerValue
	local cornerT = signT * centerValue
	local us = 2 * s - cornerS
	local ut = 2 * t - cornerT
	if signS * signT >= 0 then
		local temp = us
		us = -ut
		ut = -temp
	else
		local temp = us
		us = ut
		ut = temp
	end
	us += cornerS
	ut += cornerT
	return truncDiv(us, 2), truncDiv(ut, 2)
end

local function octModMax(x: number, centerValue: number, maxQuantized: number): number
	if x > centerValue then
		return x - maxQuantized
	end
	if x < -centerValue then
		return x + maxQuantized
	end
	return x
end

local function octGetRotationCount(sx: number, sy: number): number
	if sx == 0 then
		if sy == 0 then
			return 0
		elseif sy > 0 then
			return 3
		else
			return 1
		end
	elseif sx > 0 then
		if sy >= 0 then
			return 2
		else
			return 1
		end
	else
		if sy <= 0 then
			return 0
		else
			return 3
		end
	end
end

local function octRotate(px: number, py: number, count: number): (number, number)
	if count == 1 then
		return py, -px
	elseif count == 2 then
		return -px, -py
	elseif count == 3 then
		return -py, px
	end
	return px, py
end

local function canonicalizedComputeOriginal(
	att: AttState,
	predS: number,
	predT: number,
	corrS: number,
	corrT: number
): (number, number)
	local quantBits = 0
	local v = att.octMaxQ
	while v ~= 0 do
		quantBits += 1
		v = math.floor(v / 2)
	end
	local maxQuantized = 2 ^ quantBits - 1
	local maxValue = maxQuantized - 1
	local centerValue = math.floor(maxValue / 2)

	local ps = predS - centerValue
	local pt = predT - centerValue
	local predIsInDiamond = math.abs(ps) + math.abs(pt) <= centerValue
	if not predIsInDiamond then
		ps, pt = octInvertDiamondInt(ps, pt, centerValue)
	end
	local predIsInBottomLeft = (ps == 0 and pt == 0) or (ps < 0 and pt <= 0)
	local rotationCount = octGetRotationCount(ps, pt)
	if not predIsInBottomLeft then
		ps, pt = octRotate(ps, pt, rotationCount)
	end
	local origS = ps + corrS
	local origT = pt + corrT
	origS = octModMax(origS, centerValue, maxQuantized)
	origT = octModMax(origT, centerValue, maxQuantized)
	if not predIsInBottomLeft then
		local reverse = (4 - rotationCount) % 4
		origS, origT = octRotate(origS, origT, reverse)
	end
	if not predIsInDiamond then
		origS, origT = octInvertDiamondInt(origS, origT, centerValue)
	end
	return origS + centerValue, origT + centerValue
end

local function transformComputeOriginal(
	att: AttState,
	comps: number,
	pred: { number },
	corr: { number },
	corrOff: number,
	out: { number },
	outOff: number
)
	if att.transformType == 1 then
		wrapComputeOriginal(att, comps, pred, corr, corrOff, out, outOff)
	elseif att.transformType == 3 then
		local s, t = canonicalizedComputeOriginal(att, pred[1], pred[2], corr[corrOff + 1], corr[corrOff + 2])
		out[outOff + 1] = s
		out[outOff + 2] = t
	end
end

-- ---------------------------------------------------------------------------
-- main decoder
-- ---------------------------------------------------------------------------
local DT_SIZES: { [number]: number } = {
	[1] = 1, -- INT8
	[2] = 1, -- UINT8
	[3] = 2, -- INT16
	[4] = 2, -- UINT16
	[5] = 4, -- INT32
	[6] = 4, -- UINT32
	[7] = 8, -- INT64
	[8] = 8, -- UINT64
	[9] = 4, -- FLOAT32
	[10] = 8, -- FLOAT64
	[11] = 1, -- BOOL
}

local function readDataValue(r: Reader, dataType: number): number
	if dataType == 1 then
		return rI8(r)
	elseif dataType == 2 or dataType == 11 then
		return rU8(r)
	elseif dataType == 3 then
		return rI16(r)
	elseif dataType == 4 then
		return rU16(r)
	elseif dataType == 5 then
		return rI32(r)
	elseif dataType == 6 then
		return rU32(r)
	elseif dataType == 9 then
		return rF32(r)
	elseif dataType == 10 then
		local v = buffer.readf64(r.buf, r.pos)
		r.pos += 8
		return v
	end
	-- 64-bit ints: read as double-precision approximation
	local lo = rU32(r)
	local hi = rU32(r)
	if dataType == 7 and hi >= 2147483648 then
		return (hi - 4294967296) * 4294967296 + lo
	end
	return hi * 4294967296 + lo
end

local function skipMetadataElement(r: Reader)
	local numEntries = rVarint(r)
	for _ = 1, numEntries do
		local keySize = rU8(r)
		r.pos += keySize
		local valueSize = rU8(r)
		r.pos += valueSize
	end
	local numSub = rVarint(r)
	for _ = 1, numSub do
		local keySize = rU8(r)
		r.pos += keySize
		skipMetadataElement(r)
	end
end

local function decode(src: buffer): (DracoMesh?, string?)
	local ok, result, err = pcall(function(): (DracoMesh?, string?)
		local r = mkReader(src, 0)
		if r.len < 11 then
			return nil, "draco stream too short"
		end
		local magic = buffer.readstring(src, 0, 5)
		if magic ~= "DRACO" then
			return nil, "bad draco magic"
		end
		r.pos = 5
		local major = rU8(r)
		local minor = rU8(r)
		local encoderType = rU8(r)
		local encoderMethod = rU8(r)
		local flags = rU16(r)
		if encoderType ~= 1 then
			return nil, "only triangular meshes are supported (encoder_type=" .. encoderType .. ")"
		end
		if major ~= 2 or minor < 2 then
			return nil, string.format("unsupported draco bitstream version %d.%d (need 2.2+)", major, minor)
		end
		if band(flags, 32768) ~= 0 then
			-- metadata: attribute metadata entries + file metadata
			local numAttMetadata = rVarint(r)
			for _ = 1, numAttMetadata do
				rVarint(r) -- att metadata id
				skipMetadataElement(r)
			end
			skipMetadataElement(r)
		end

		local ctx: Ctx = {
			r = r,
			encoderMethod = encoderMethod,
			numFaces = 0,
			numEncVerts = 0,
			numAttributeData = 0,
			numSymbols = 0,
			numSplitSymbols = 0,
			cornerToVert = {},
			opposite = {},
			vertexCorners = {},
			numVertices = 0,
			isVertHole = {},
			facePoints = {},
			numPoints = 0,
			attSeamCorners = {},
			attEdgeOnSeam = {},
			attVertOnSeam = {},
			attCornerToVert = {},
			attVertexCorners = {},
			attNumVertices = {},
			traversalType = 0,
			symbolBits = nil,
			startFaceDecoder = nil,
			seamDecoders = {},
			contextSymbols = {},
			contextCounters = {},
			vertexValences = {},
			lastSymbol = -1,
			activeContext = -1,
			splitSourceIds = {},
			splitSplitIds = {},
			splitEdges = {},
		}

		if encoderMethod == 1 then
			local cerr = decodeEdgebreakerConnectivity(ctx)
			if cerr then
				return nil, cerr
			end
			local aerr = buildAttributeCornerTables(ctx)
			if aerr then
				return nil, aerr
			end
			local perr = assignPointsToCorners(ctx)
			if perr then
				return nil, perr
			end
		elseif encoderMethod == 0 then
			-- sequential connectivity
			local numFaces = rVarint(r)
			local numPoints = rVarint(r)
			local method = rU8(r)
			ctx.numFaces = numFaces
			ctx.numPoints = numPoints
			ctx.facePoints = table.create(numFaces * 3, 0)
			if method == 0 then
				-- compressed indices
				local syms, serr = decodeSymbols(r, numFaces * 3, 1)
				if not syms then
					return nil, serr
				end
				local s = syms :: { number }
				local last = 0
				for i = 1, numFaces * 3 do
					local v = s[i]
					local diff = rshift(v, 1)
					if band(v, 1) == 1 then
						diff = -diff
					end
					last += diff
					ctx.facePoints[i] = last
				end
			elseif method == 1 then
				if numPoints < 256 then
					for i = 1, numFaces * 3 do
						ctx.facePoints[i] = rU8(r)
					end
				elseif numPoints < 65536 then
					for i = 1, numFaces * 3 do
						ctx.facePoints[i] = rU16(r)
					end
				elseif numPoints < 2097152 then
					for i = 1, numFaces * 3 do
						ctx.facePoints[i] = rVarint(r)
					end
				else
					for i = 1, numFaces * 3 do
						ctx.facePoints[i] = rU32(r)
					end
				end
			else
				return nil, "unknown sequential connectivity method"
			end
		else
			return nil, "unknown encoder method " .. encoderMethod
		end

		-- =====================================================================
		-- attribute decoding
		-- =====================================================================
		local numAttDecoders = rU8(r)
		type DecoderInfo = {
			attDataId: number,
			decoderType: number, -- 0 vertex, 1 corner
			traversalMethod: number,
			atts: { AttState },
			encData: EncData,
			indicesMap: { number }, -- point -> value index
		}
		local decoders: { DecoderInfo } = {}
		for i = 1, numAttDecoders do
			local info: DecoderInfo = {
				attDataId = -1,
				decoderType = 0,
				traversalMethod = 0,
				atts = {},
				encData = { valueToCorner = {}, vertToValue = {}, numValues = 0 },
				indicesMap = {},
			}
			if encoderMethod == 1 then
				info.attDataId = rI8(r)
				info.decoderType = rU8(r)
				local tm = rU8(r)
				if tm >= 2 then
					return nil, "invalid traversal method"
				end
				info.traversalMethod = tm
			end
			decoders[i] = info
		end
		-- per decoder: attribute descriptors
		for i = 1, numAttDecoders do
			local info = decoders[i]
			local numAtts = rVarint(r)
			if numAtts == 0 then
				return nil, "attributes decoder with zero attributes"
			end
			for j = 1, numAtts do
				local attType = rU8(r)
				local dataType = rU8(r)
				local numComponents = rU8(r)
				local normalized = rU8(r)
				local uniqueId = rVarint(r)
				if dataType == 0 or dataType > 11 then
					return nil, "invalid attribute data type"
				end
				info.atts[j] = {
					attType = attType,
					dataType = dataType,
					numComponents = numComponents,
					normalized = normalized > 0,
					uniqueId = uniqueId,
					seqType = 0,
					scheme = -2,
					transformType = -1,
					wrapMin = 0,
					wrapMax = 0,
					octMaxQ = 0,
					quantBits = 0,
					quantMin = {},
					quantRange = 1,
					normQuantBits = 0,
					values = {},
					numValues = 0,
				}
			end
			for j = 1, numAtts do
				info.atts[j].seqType = rU8(r)
			end
		end

		-- generate sequences (traversal) + point -> value maps
		for i = 1, numAttDecoders do
			local info = decoders[i]
			local enc = info.encData
			if encoderMethod == 0 then
				-- identity sequence
				enc.numValues = ctx.numPoints
				for p = 0, ctx.numPoints - 1 do
					info.indicesMap[p + 1] = p
				end
			else
				-- build the traversal table for this decoder
				local t: TraversalTable
				if info.decoderType == 0 or info.attDataId < 0 then
					-- vertex attribute: base corner table
					local cornerToVert = ctx.cornerToVert
					local opposite = ctx.opposite
					local vertexCorners = ctx.vertexCorners
					t = {
						numFaces = ctx.numFaces,
						numVertices = ctx.numVertices,
						vertexAt = function(c: number): number
							if c < 0 then
								return INVALID
							end
							return cornerToVert[c + 1]
						end,
						oppositeOf = function(c: number): number
							if c < 0 then
								return INVALID
							end
							local o = opposite[c + 1]
							return if o == nil then INVALID else o
						end,
						leftMost = function(v: number): number
							local c = vertexCorners[v + 1]
							return if c == nil then INVALID else c
						end,
					}
				else
					-- corner attribute: attribute corner table
					local ai = info.attDataId + 1
					local attCornerToVert = ctx.attCornerToVert[ai]
					local edgeOnSeam = ctx.attEdgeOnSeam[ai]
					local opposite = ctx.opposite
					local attVertexCorners = ctx.attVertexCorners[ai]
					t = {
						numFaces = ctx.numFaces,
						numVertices = ctx.attNumVertices[ai],
						vertexAt = function(c: number): number
							if c < 0 then
								return INVALID
							end
							return attCornerToVert[c + 1]
						end,
						oppositeOf = function(c: number): number
							if c < 0 or edgeOnSeam[c + 1] then
								return INVALID
							end
							local o = opposite[c + 1]
							return if o == nil then INVALID else o
						end,
						leftMost = function(v: number): number
							local c = attVertexCorners[v + 1]
							return if c == nil then INVALID else c
						end,
					}
				end
				local maxVerts = math.max(t.numVertices, ctx.numVertices)
				enc.vertToValue = table.create(maxVerts, INVALID)
				local isFaceVisited = table.create(ctx.numFaces, false)
				local isVertexVisited = table.create(maxVerts, false)
				local terr: string?
				if info.traversalMethod == 1 then
					if info.decoderType ~= 0 and info.attDataId >= 0 then
						return nil, "prediction-degree traversal on corner attribute"
					end
					terr = traverseMaxPredictionDegree(t, enc, isFaceVisited, isVertexVisited)
				else
					terr = traverseDepthFirst(t, enc, isFaceVisited, isVertexVisited)
				end
				if terr then
					return nil, terr
				end
				-- UpdatePointToAttributeIndexMapping
				for f = 0, ctx.numFaces - 1 do
					for p = 0, 2 do
						local corner = f * 3 + p
						local pointId = ctx.facePoints[corner + 1]
						local vert = t.vertexAt(corner)
						if vert < 0 then
							return nil, "invalid vertex in mapping"
						end
						local entry = enc.vertToValue[vert + 1]
						info.indicesMap[pointId + 1] = entry
					end
				end
			end
		end

		-- find the position attribute decoder/attribute (parent for predictions)
		local posDecoder: DecoderInfo? = nil
		local posAtt: AttState? = nil
		for i = 1, numAttDecoders do
			for _, att in ipairs(decoders[i].atts) do
				if att.attType == 0 then -- POSITION
					posDecoder = decoders[i]
					posAtt = att
					break
				end
			end
			if posAtt then
				break
			end
		end

		-- decode each decoder's attributes
		for i = 1, numAttDecoders do
			local info = decoders[i]
			local enc = info.encData
			local numValues = enc.numValues

			-- DecodePortableAttributes
			for _, att in ipairs(info.atts) do
				att.numValues = numValues
				if att.seqType == 0 then
					-- GENERIC: raw values in original format
					local comps = att.numComponents
					local total = numValues * comps
					local vals = table.create(total, 0)
					for k = 1, total do
						vals[k] = readDataValue(r, att.dataType)
					end
					att.values = vals
				elseif att.seqType >= 1 and att.seqType <= 3 then
					-- INTEGER / QUANTIZATION / NORMALS
					local comps = if att.seqType == 3 then 2 else att.numComponents
					local scheme = rI8(r)
					if scheme < -2 or scheme >= 7 then
						return nil, "invalid prediction scheme " .. scheme
					end
					att.scheme = scheme
					if scheme ~= -2 then
						local transformType = rI8(r)
						if transformType < -1 or transformType > 3 then
							return nil, "invalid prediction transform " .. transformType
						end
						att.transformType = transformType
						if transformType ~= 1 and transformType ~= 3 then
							return nil, "unsupported prediction transform " .. transformType
						end
					end
					local total = numValues * comps
					local vals: { number }
					local compressed = rU8(r)
					if compressed > 0 then
						local syms, serr = decodeSymbols(r, total, comps)
						if not syms then
							return nil, serr
						end
						vals = syms :: { number }
					else
						local numBytes = rU8(r)
						vals = table.create(total, 0)
						if numBytes == 4 then
							for k = 1, total do
								vals[k] = rU32(r)
							end
						else
							for k = 1, total do
								local v = 0
								local m = 1
								for _ = 1, numBytes do
									v += rU8(r) * m
									m *= 256
								end
								vals[k] = v
							end
						end
					end
					-- ConvertSymbolsToSignedInts unless corrections are positive
					local correctionsPositive = scheme ~= -2 and att.transformType == 3
					if total > 0 and not correctionsPositive then
						for k = 1, total do
							local v = vals[k]
							if band(v, 1) == 0 then
								vals[k] = rshift(v, 1)
							else
								vals[k] = -rshift(v, 1) - 1
							end
						end
					end
					att.values = vals

					if scheme ~= -2 then
						-- DecodePredictionData
						local creaseBits: { { number } } = {}
						local orientations: { number } = {}
						local flipDecoder: RansBitDecoder? = nil
						if scheme == 4 then
							-- constrained multi: 4 contexts of crease bits
							for ctxI = 1, 4 do
								local numFlags = rVarint(r)
								local bits = table.create(numFlags, 0)
								if numFlags > 0 then
									local bd, berr = startRansBitDecoder(r)
									if not bd then
										return nil, berr
									end
									for k = 1, numFlags do
										bits[k] = decodeNextBit(bd :: RansBitDecoder)
									end
								end
								creaseBits[ctxI] = bits
							end
						elseif scheme == 5 then
							-- tex coords portable: orientations
							local numOrientations = rI32(r)
							if numOrientations < 0 then
								return nil, "invalid orientation count"
							end
							local bd, berr = startRansBitDecoder(r)
							if not bd then
								return nil, berr
							end
							local lastOrientation = 1
							for k = 1, numOrientations do
								if decodeNextBit(bd :: RansBitDecoder) == 0 then
									lastOrientation = 1 - lastOrientation
								end
								orientations[k] = lastOrientation
							end
						elseif scheme == 6 then
							-- geometric normal: transform data first, then flip bits
							att.octMaxQ = rI32(r)
							rI32(r) -- unused center value
							local bd, berr = startRansBitDecoder(r)
							if not bd then
								return nil, berr
							end
							flipDecoder = bd
						end
						if scheme ~= 6 then
							-- transform data
							if att.transformType == 1 then
								att.wrapMin = rI32(r)
								att.wrapMax = rI32(r)
							elseif att.transformType == 3 then
								att.octMaxQ = rI32(r)
								rI32(r) -- unused center value
							end
						end

						-- ComputeOriginalValues
						local signed = vals
						local out = table.create(total, 0)
						local pred = table.create(comps, 0)

						local function attVertexAt(c: number): number
							-- vertex lookup for THIS decoder's traversal table
							if c < 0 then
								return INVALID
							end
							if encoderMethod == 0 then
								return INVALID
							end
							if info.decoderType == 0 or info.attDataId < 0 then
								return ctx.cornerToVert[c + 1]
							end
							return ctx.attCornerToVert[info.attDataId + 1][c + 1]
						end
						local function attOppositeOf(c: number): number
							if c < 0 then
								return INVALID
							end
							if info.decoderType ~= 0 and info.attDataId >= 0 then
								if ctx.attEdgeOnSeam[info.attDataId + 1][c + 1] then
									return INVALID
								end
							end
							local o = ctx.opposite[c + 1]
							return if o == nil then INVALID else o
						end
						local function attSwingLeft(c: number): number
							local o = attOppositeOf(cNext(c))
							return if o < 0 then INVALID else cNext(o)
						end
						local function attSwingRight(c: number): number
							local o = attOppositeOf(cPrev(c))
							return if o < 0 then INVALID else cPrev(o)
						end

						local function parallelogramPred(p: number, ci: number, data: { number }): boolean
							local oci = attOppositeOf(ci)
							if oci < 0 then
								return false
							end
							local vOpp = attVertexAt(oci)
							local vNext = attVertexAt(cNext(oci))
							local vPrev = attVertexAt(cPrev(oci))
							local eOpp = enc.vertToValue[vOpp + 1]
							local eNext = enc.vertToValue[vNext + 1]
							local ePrev = enc.vertToValue[vPrev + 1]
							if eOpp < p and eNext < p and ePrev < p then
								local oOff = eOpp * comps
								local nOff = eNext * comps
								local pOff = ePrev * comps
								for c2 = 1, comps do
									pred[c2] = data[nOff + c2] + data[pOff + c2] - data[oOff + c2]
								end
								return true
							end
							return false
						end

						if scheme == 0 then
							-- DIFFERENCE
							local zero = table.create(comps, 0)
							transformComputeOriginal(att, comps, zero, signed, 0, out, 0)
							for p = 1, numValues - 1 do
								local off = p * comps
								for c2 = 1, comps do
									pred[c2] = out[off - comps + c2]
								end
								transformComputeOriginal(att, comps, pred, signed, off, out, off)
							end
						elseif scheme == 1 then
							-- PARALLELOGRAM
							for c2 = 1, comps do
								pred[c2] = 0
							end
							transformComputeOriginal(att, comps, pred, signed, 0, out, 0)
							for p = 1, numValues - 1 do
								local cornerId = enc.valueToCorner[p + 1]
								local off = p * comps
								if parallelogramPred(p, cornerId, out) then
									transformComputeOriginal(att, comps, pred, signed, off, out, off)
								else
									for c2 = 1, comps do
										pred[c2] = out[off - comps + c2]
									end
									transformComputeOriginal(att, comps, pred, signed, off, out, off)
								end
							end
						elseif scheme == 4 then
							-- CONSTRAINED MULTI PARALLELOGRAM
							local predVals: { { number } } = {}
							for k = 1, 4 do
								predVals[k] = table.create(comps, 0)
							end
							local creasePos = { 0, 0, 0, 0 }
							for c2 = 1, comps do
								pred[c2] = 0
							end
							transformComputeOriginal(att, comps, pred, signed, 0, out, 0)
							for p = 1, numValues - 1 do
								local startCorner = enc.valueToCorner[p + 1]
								local cornerId = startCorner
								local numParallelograms = 0
								local firstPass = true
								while cornerId >= 0 do
									-- try the parallelogram for this corner
									local oci = attOppositeOf(cornerId)
									local okPred = false
									if oci >= 0 then
										local vOpp = attVertexAt(oci)
										local vNext = attVertexAt(cNext(oci))
										local vPrev = attVertexAt(cPrev(oci))
										local eOpp = enc.vertToValue[vOpp + 1]
										local eNext = enc.vertToValue[vNext + 1]
										local ePrev = enc.vertToValue[vPrev + 1]
										if eOpp < p and eNext < p and ePrev < p then
											local oOff = eOpp * comps
											local nOff = eNext * comps
											local pOff = ePrev * comps
											local pv = predVals[numParallelograms + 1]
											for c2 = 1, comps do
												pv[c2] = out[nOff + c2] + out[pOff + c2] - out[oOff + c2]
											end
											okPred = true
										end
									end
									if okPred then
										numParallelograms += 1
										if numParallelograms == 4 then
											break
										end
									end
									if firstPass then
										cornerId = attSwingLeft(cornerId)
									else
										cornerId = attSwingRight(cornerId)
									end
									if cornerId == startCorner then
										break
									end
									if cornerId < 0 and firstPass then
										firstPass = false
										cornerId = attSwingRight(startCorner)
									end
								end
								local numUsed = 0
								local multi = table.create(comps, 0)
								if numParallelograms > 0 then
									local context = numParallelograms -- 1-based ctx index = (n-1)+1
									for k = 1, numParallelograms do
										creasePos[context] += 1
										local isCrease = creaseBits[context][creasePos[context]]
										if isCrease == 0 then
											numUsed += 1
											local pv = predVals[k]
											for c2 = 1, comps do
												multi[c2] += pv[c2]
											end
										end
									end
								end
								local off = p * comps
								if numUsed == 0 then
									for c2 = 1, comps do
										pred[c2] = out[off - comps + c2]
									end
									transformComputeOriginal(att, comps, pred, signed, off, out, off)
								else
									for c2 = 1, comps do
										pred[c2] = truncDiv(multi[c2], numUsed)
									end
									transformComputeOriginal(att, comps, pred, signed, off, out, off)
								end
							end
						elseif scheme == 5 then
							-- TEX COORDS PORTABLE
							if not posDecoder or not posAtt then
								return nil, "texcoords prediction without positions"
							end
							local pd = posDecoder :: DecoderInfo
							local pa = posAtt :: AttState
							local paVals = pa.portableValues or pa.values
							local orientationIdx = #orientations
							local function getPosForEntry(entryId: number): (number, number, number)
								local corner = enc.valueToCorner[entryId + 1]
								local pointId = ctx.facePoints[corner + 1]
								local mapped = pd.indicesMap[pointId + 1]
								local pOff = mapped * 3
								return paVals[pOff + 1], paVals[pOff + 2], paVals[pOff + 3]
							end
							for p = 0, numValues - 1 do
								local off = p * comps
								local predOk = false
								local cornerId = enc.valueToCorner[p + 1]
								local vNext = attVertexAt(cNext(cornerId))
								local vPrev = attVertexAt(cPrev(cornerId))
								local nextEntry = if vNext >= 0 then enc.vertToValue[vNext + 1] else INVALID
								local prevEntry = if vPrev >= 0 then enc.vertToValue[vPrev + 1] else INVALID
								if prevEntry < p and nextEntry < p and prevEntry >= 0 and nextEntry >= 0 then
									local nU = out[nextEntry * 2 + 1]
									local nV = out[nextEntry * 2 + 2]
									local pU = out[prevEntry * 2 + 1]
									local pV = out[prevEntry * 2 + 2]
									if pU == nU and pV == nV then
										pred[1] = pU
										pred[2] = pV
										predOk = true
									else
										local tipX, tipY, tipZ = getPosForEntry(p)
										local nX, nY, nZ = getPosForEntry(nextEntry)
										local pX, pY, pZ = getPosForEntry(prevEntry)
										local pnX, pnY, pnZ = pX - nX, pY - nY, pZ - nZ
										local pnNorm2 = pnX * pnX + pnY * pnY + pnZ * pnZ
										if pnNorm2 ~= 0 then
											local cnX, cnY, cnZ = tipX - nX, tipY - nY, tipZ - nZ
											local cnDotPn = pnX * cnX + pnY * cnY + pnZ * cnZ
											local pnuU = pU - nU
											local pnuV = pV - nV
											local xUVu = nU * pnNorm2 + cnDotPn * pnuU
											local xUVv = nV * pnNorm2 + cnDotPn * pnuV
											local xPosX = nX + truncDiv(cnDotPn * pnX, pnNorm2)
											local xPosY = nY + truncDiv(cnDotPn * pnY, pnNorm2)
											local xPosZ = nZ + truncDiv(cnDotPn * pnZ, pnNorm2)
											local cxX, cxY, cxZ = tipX - xPosX, tipY - xPosY, tipZ - xPosZ
											local cxNorm2 = cxX * cxX + cxY * cxY + cxZ * cxZ
											local normSquared = intSqrtProduct(cxNorm2, pnNorm2)
											local cxU = pnuV * normSquared
											local cxV = -pnuU * normSquared
											if orientationIdx < 1 then
												return nil, "texcoords: orientation underflow"
											end
											local orientation = orientations[orientationIdx]
											orientationIdx -= 1
											if orientation == 1 then
												pred[1] = truncDiv(xUVu + cxU, pnNorm2)
												pred[2] = truncDiv(xUVv + cxV, pnNorm2)
											else
												pred[1] = truncDiv(xUVu - cxU, pnNorm2)
												pred[2] = truncDiv(xUVv - cxV, pnNorm2)
											end
											predOk = true
										end
									end
								end
								if not predOk then
									local dataOffset = 0
									if prevEntry >= 0 and prevEntry < p then
										dataOffset = prevEntry * 2
									end
									if nextEntry >= 0 and nextEntry < p then
										dataOffset = nextEntry * 2
									elseif p > 0 then
										dataOffset = (p - 1) * 2
									else
										pred[1] = 0
										pred[2] = 0
										dataOffset = -1
									end
									if dataOffset >= 0 then
										pred[1] = out[dataOffset + 1]
										pred[2] = out[dataOffset + 2]
									end
								end
								wrapComputeOriginal(att, comps, pred, signed, off, out, off)
							end
						elseif scheme == 6 then
							-- GEOMETRIC NORMAL
							if not posDecoder or not posAtt then
								return nil, "normal prediction without positions"
							end
							local pd = posDecoder :: DecoderInfo
							local pa = posAtt :: AttState
							local paVals = pa.portableValues or pa.values
							local fd = flipDecoder :: RansBitDecoder
							-- octahedron parameters from the transform data
							local quantBits = 0
							local mq = att.octMaxQ
							while mq ~= 0 do
								quantBits += 1
								mq = math.floor(mq / 2)
							end
							local maxQuantized = 2 ^ quantBits - 1
							local maxValue = maxQuantized - 1
							local centerValue = math.floor(maxValue / 2)

							local function getPosForCorner(ci: number): (number, number, number)
								local vert = attVertexAt(ci)
								local dataId = enc.vertToValue[vert + 1]
								local corner = enc.valueToCorner[dataId + 1]
								local pointId = ctx.facePoints[corner + 1]
								local mapped = pd.indicesMap[pointId + 1]
								local pOff = mapped * 3
								return paVals[pOff + 1], paVals[pOff + 2], paVals[pOff + 3]
							end

							for p = 0, numValues - 1 do
								local cornerId = enc.valueToCorner[p + 1]
								local cx, cy, cz = getPosForCorner(cornerId)
								local nx, ny, nz = 0, 0, 0
								local corner = cornerId
								local startCorner = corner
								local leftTraversal = true
								while corner >= 0 do
									local cN = cNext(corner)
									local cP = cPrev(corner)
									local nxp, nyp, nzp = getPosForCorner(cN)
									local pxp, pyp, pzp = getPosForCorner(cP)
									local dnx, dny, dnz = nxp - cx, nyp - cy, nzp - cz
									local dpx, dpy, dpz = pxp - cx, pyp - cy, pzp - cz
									nx += dny * dpz - dnz * dpy
									ny += dnz * dpx - dnx * dpz
									nz += dnx * dpy - dny * dpx
									if leftTraversal then
										local leftC = attSwingLeft(corner)
										corner = leftC
										if corner < 0 then
											corner = attSwingRight(startCorner)
											leftTraversal = false
										elseif corner == startCorner then
											corner = INVALID
										end
									else
										corner = attSwingRight(corner)
									end
								end
								local absSum = math.abs(nx) + math.abs(ny) + math.abs(nz)
								local upperBound = 536870912 -- 1 << 29
								if absSum > upperBound then
									local quotient = truncDiv(absSum, upperBound)
									nx = truncDiv(nx, quotient)
									ny = truncDiv(ny, quotient)
									nz = truncDiv(nz, quotient)
								end
								-- CanonicalizeIntegerVector
								absSum = math.abs(nx) + math.abs(ny) + math.abs(nz)
								if absSum == 0 then
									nx = centerValue
									ny = 0
									nz = 0
								else
									nx = truncDiv(nx * centerValue, absSum)
									ny = truncDiv(ny * centerValue, absSum)
									if nz >= 0 then
										nz = centerValue - math.abs(nx) - math.abs(ny)
									else
										nz = -(centerValue - math.abs(nx) - math.abs(ny))
									end
								end
								local flip = decodeNextBit(fd)
								if flip == 1 then
									nx, ny, nz = -nx, -ny, -nz
								end
								-- IntegerVectorToQuantizedOctahedralCoords
								local s: number, t2: number
								if nx >= 0 then
									s = ny + centerValue
									t2 = nz + centerValue
								else
									if ny < 0 then
										s = math.abs(nz)
									else
										s = maxValue - math.abs(nz)
									end
									if nz < 0 then
										t2 = math.abs(ny)
									else
										t2 = maxValue - math.abs(ny)
									end
								end
								-- CanonicalizeOctahedralCoords
								if (s == 0 and t2 == 0) or (s == 0 and t2 == maxValue) or (s == maxValue and t2 == 0) then
									s = maxValue
									t2 = maxValue
								elseif s == 0 and t2 > centerValue then
									t2 = centerValue - (t2 - centerValue)
								elseif s == maxValue and t2 < centerValue then
									t2 = centerValue + (centerValue - t2)
								elseif t2 == maxValue and s < centerValue then
									s = centerValue + (centerValue - s)
								elseif t2 == 0 and s > centerValue then
									s = centerValue - (s - centerValue)
								end
								local off = p * 2
								local oS, oT = canonicalizedComputeOriginal(att, s, t2, signed[off + 1], signed[off + 2])
								out[off + 1] = oS
								out[off + 2] = oT
							end
						else
							return nil, "unsupported prediction scheme " .. scheme
						end
						att.values = out
					end
				else
					return nil, "unknown sequential attribute decoder type " .. att.seqType
				end
			end

			-- DecodeDataNeededByPortableTransforms
			for _, att in ipairs(info.atts) do
				if att.seqType == 3 then
					att.normQuantBits = rU8(r)
				elseif att.seqType == 2 then
					local comps = att.numComponents
					for c2 = 1, comps do
						att.quantMin[c2] = rF32(r)
					end
					att.quantRange = rF32(r)
					att.quantBits = rU8(r)
				end
			end

			-- TransformAttributesToOriginalFormat
			for _, att in ipairs(info.atts) do
				if att.seqType == 2 then
					-- dequantize (float32 math like the reference); the integer
					-- (portable) values are kept for child prediction schemes.
					local comps = att.numComponents
					local maxQuantized = 2 ^ att.quantBits - 1
					local delta = f32(att.quantRange / maxQuantized)
					local vals = att.values
					att.portableValues = vals
					local total = att.numValues * comps
					local outVals = table.create(total, 0)
					local k = 1
					while k <= total do
						for c2 = 1, comps do
							outVals[k] = f32(f32(vals[k] * delta) + att.quantMin[c2])
							k += 1
						end
					end
					att.values = outVals
				elseif att.seqType == 3 then
					-- octahedral normals -> unit vectors (3 components)
					local qb = att.normQuantBits
					local maxQuantized = 2 ^ qb - 1
					local maxValue = maxQuantized - 1
					local scale = f32(2 / maxValue)
					local vals = att.values
					local outVals = table.create(att.numValues * 3, 0)
					for p = 0, att.numValues - 1 do
						local s = vals[p * 2 + 1]
						local t2 = vals[p * 2 + 2]
						local y = f32(f32(s * scale) - 1)
						local z = f32(f32(t2 * scale) - 1)
						local x = f32(1 - f32(math.abs(y)) - f32(math.abs(z)))
						local xOffset = if x < 0 then f32(-x) else 0
						if xOffset < 0 then
							xOffset = 0
						end
						y += if y < 0 then xOffset else -xOffset
						y = f32(y)
						z += if z < 0 then xOffset else -xOffset
						z = f32(z)
						local normSquared = f32(f32(f32(x * x) + f32(y * y)) + f32(z * z))
						if normSquared < 1e-6 then
							outVals[p * 3 + 1] = 0
							outVals[p * 3 + 2] = 0
							outVals[p * 3 + 3] = 0
						else
							local d = f32(1 / f32(math.sqrt(normSquared)))
							outVals[p * 3 + 1] = f32(x * d)
							outVals[p * 3 + 2] = f32(y * d)
							outVals[p * 3 + 3] = f32(z * d)
						end
					end
					att.values = outVals
				end
			end
		end

		-- =====================================================================
		-- output: per-point attribute arrays + indices
		-- =====================================================================
		local outAttributes: { [number]: DracoAttribute } = {}
		for i = 1, numAttDecoders do
			local info = decoders[i]
			for _, att in ipairs(info.atts) do
				local comps = if att.seqType == 3 then 3 else att.numComponents
				local vals = att.values
				local outVals = table.create(ctx.numPoints * comps, 0)
				for p = 0, ctx.numPoints - 1 do
					local srcIdx = info.indicesMap[p + 1]
					if srcIdx == nil or srcIdx < 0 then
						srcIdx = 0
					end
					local sOff = srcIdx * comps
					local dOff = p * comps
					for c2 = 1, comps do
						outVals[dOff + c2] = vals[sOff + c2] or 0
					end
				end
				outAttributes[att.uniqueId] = {
					values = outVals,
					componentCount = comps,
				}
			end
		end

		local indices = table.create(ctx.numFaces * 3, 0)
		for k = 1, ctx.numFaces * 3 do
			indices[k] = ctx.facePoints[k]
		end

		local mesh: DracoMesh = {
			numPoints = ctx.numPoints,
			numFaces = ctx.numFaces,
			indices = indices,
			attributes = outAttributes,
		}
		return mesh, nil
	end)
	if not ok then
		return nil, "draco decode error: " .. tostring(result)
	end
	return result, err
end

-- test hook: decode `count` rabs bits from a stream of [prob_zero u8][varint size][data]
local function _testDecodeRabs(stream: buffer, count: number): (string?, string?)
	local r = mkReader(stream, 0)
	local d, err = startRansBitDecoder(r)
	if not d then
		return nil, err
	end
	local out = table.create(count, 0)
	for i = 1, count do
		out[i] = decodeNextBit(d :: RansBitDecoder)
	end
	return table.concat(out, ""), nil
end

return {
	decode = decode,
	_testDecodeRabs = _testDecodeRabs,
}
