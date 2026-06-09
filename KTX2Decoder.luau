--!strict
--[[
  KTX2Decoder -- pure-Luau decoder for KHR_texture_basisu textures.
  Decodes KTX2 containers holding Basis Universal data to RGBA8:
   - ETC1S / BasisLZ (supercompression scheme 1): codebook palettes,
     huffman-coded slices, optional alpha slice
   - UASTC LDR 4x4 (vkFormat 0, scheme 0 none / 2 zstd): full 19-mode
     block decode (BISE trits/quints, 2/3-subset partitions, dual plane)
  Ported from basis_universal's transcoder (Apache-2.0).

  decode(bytes) -> ({ width, height, rgba: buffer }?, string?)
  Only mip level 0 is decoded (the GPU path here doesn't use mips).
]]

local ZstdDecoder = require('ZstdDecoder')

local band = bit32.band
local bor = bit32.bor
local bxor = bit32.bxor
local rshift = bit32.rshift
local lshift = bit32.lshift

export type DecodedTexture = {
	width: number,
	height: number,
	rgba: buffer,
}

local PATTERNS2 = {
	0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
	0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
	1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
	0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0,
	0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
	1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1,
	1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0,
	1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
	1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1,
	0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0,
	0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0,
	1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1,
	1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0,
	0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
	1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1,
	0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0,
	1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
	1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
	1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
	1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0,
}

local PATTERNS3 = {
	0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2,
	1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2,
	1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2,
	1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0,
	1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0,
	0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2,
	0, 2, 1, 1, 0, 2, 1, 1, 0, 2, 1, 1, 0, 2, 1, 1,
	2, 0, 0, 0, 2, 0, 0, 0, 2, 1, 1, 1, 2, 1, 1, 1,
	2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2,
	1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 1, 1, 1, 1,
	0, 0, 2, 2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 2,
}

local PATTERNS2_M7 = {
	0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
	1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
	0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
	0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
	1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
	0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0,
	1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
	0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0,
	1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0,
	0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0,
	1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
}

local ANCHORS2 = {
	0, 2, 0, 0, 3, 0, 1, 0, 0, 0, 3, 0,
	7, 0, 0, 0, 2, 0, 3, 0, 0, 7, 0, 0,
	0, 11, 0, 2, 0, 0, 0, 7, 0, 11, 0, 0,
	3, 0, 0, 8, 0, 0, 0, 4, 0, 12, 0, 0,
	1, 0, 0, 8, 0, 0, 0, 1, 0, 0, 2, 0,
	0, 4, 0, 8, 0, 0, 1, 0, 0, 0, 2, 0,
	4, 0, 0, 0, 1, 0, 4, 0, 0, 1, 0, 0,
	4, 0, 0, 1, 0, 0,
}

local ANCHORS3 = {
	0, 8, 10, 8, 0, 12, 4, 0, 12, 8, 0, 4,
	3, 0, 2, 0, 1, 3, 0, 2, 1, 1, 9, 0,
	1, 2, 0, 4, 0, 8, 0, 6, 2,
}

local ANCHORS2_M7 = {
	0, 4, 0, 0, 2, 0, 2, 0, 0, 0, 7, 0,
	8, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0,
	2, 0, 0, 0, 1, 0, 0, 8, 0, 2, 0, 0,
	0, 1, 0, 0, 7, 0, 12, 0, 0, 2, 0, 0,
	9, 0, 0, 0, 2, 0, 4, 0, 0,
}

local HUFF_MODES = {
	11, 0, 10, 3, 11, 15, 12, 7, 11, 18, 10, 5, 11, 14, 12, 9,
	11, 0, 10, 4, 11, 16, 12, 8, 11, 18, 10, 6, 11, 2, 12, 13,
	11, 0, 10, 3, 11, 17, 12, 7, 11, 18, 10, 5, 11, 14, 12, 9,
	11, 0, 10, 4, 11, 1, 12, 8, 11, 18, 10, 6, 11, 2, 12, 13,
	11, 0, 10, 3, 11, 19, 12, 7, 11, 18, 10, 5, 11, 14, 12, 9,
	11, 0, 10, 4, 11, 16, 12, 8, 11, 18, 10, 6, 11, 2, 12, 13,
	11, 0, 10, 3, 11, 17, 12, 7, 11, 18, 10, 5, 11, 14, 12, 9,
	11, 0, 10, 4, 11, 1, 12, 8, 11, 18, 10, 6, 11, 2, 12, 13,
}

local MODE_HUFF_CODES = {
	1, 4, 53, 6, 29, 5, 3, 5,
	19, 5, 11, 5, 27, 5, 7, 5,
	23, 5, 15, 5, 2, 3, 0, 2,
	6, 3, 31, 5, 13, 5, 5, 7,
	21, 6, 37, 6, 9, 4, 69, 7,
}

local BISE_RANGES = {
	1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 1,
	1, 1, 0, 3, 0, 0, 1, 0, 1, 2, 1, 0,
	4, 0, 0, 2, 0, 1, 3, 1, 0, 5, 0, 0,
	3, 0, 1, 4, 1, 0, 6, 0, 0, 4, 0, 1,
	5, 1, 0, 7, 0, 0, 5, 0, 1, 6, 1, 0,
	8, 0, 0,
}

local MODE_WEIGHT_BITS = {
	4, 2, 3, 2, 2, 3, 2, 2, 0, 2, 4, 2, 3, 1, 2, 4, 2, 2, 5,
}

local MODE_EP_RANGES = {
	19, 20, 8, 7, 12, 20, 18, 12, 0, 8, 13, 13, 19, 20, 20, 20, 20, 20, 11,
}

local MODE_SUBSETS = {
	1, 1, 2, 3, 2, 1, 1, 2, 0, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1,
}

local MODE_PLANES = {
	1, 1, 1, 1, 1, 1, 2, 1, 0, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1,
}

local MODE_COMPS = {
	3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 3,
}

local MODE_HAS_BIAS = {
	1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1,
}

local MODE_HAS_HINT0 = {
	1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
}

local MODE_HAS_HINT1 = {
	1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1,
}

local MODE_HAS_ALPHA = {
	0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
}

local MODE_HINT_BITS = {
	15, 15, 15, 15, 15, 15, 15, 15, 0, 23, 17, 17, 17, 23, 23, 23, 23, 23, 15,
}

local UNQUANT_B = {
	"",
	"",
	"",
	"",
	"000000000",
	"",
	"000000000",
	"b000b0bb0",
	"",
	"b0000bb00",
	"cb000cbcb",
	"",
	"cb0000cbc",
	"dcb000dcb",
	"",
	"dcb0000dc",
	"edcb000ed",
	"",
	"edcb0000e",
	"fedcb000f",
	"",
}
local UNQUANT_C = { 0, 0, 0, 0, 204, 0, 113, 93, 0, 54, 44, 0, 26, 22, 0, 13, 11, 0, 6, 5, 0 }

local BC7_WEIGHTS1 = {
	0, 64,
}

local BC7_WEIGHTS2 = {
	0, 21, 43, 64,
}

local BC7_WEIGHTS3 = {
	0, 9, 18, 27, 37, 46, 55, 64,
}

local ASTC_WEIGHTS4 = {
	0, 4, 8, 12, 17, 21, 25, 29, 35, 39, 43, 47, 52, 56, 60, 64,
}

local ASTC_WEIGHTS5 = {
	0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
	34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
}


local ETC1_INTEN: { { number } } = {
	{ -8, -2, 2, 8 },
	{ -17, -5, 5, 17 },
	{ -29, -9, 9, 29 },
	{ -42, -13, 13, 42 },
	{ -60, -18, 18, 60 },
	{ -80, -24, 24, 80 },
	{ -106, -33, 33, 106 },
	{ -183, -47, 47, 183 },
}

-- ---------------------------------------------------------------------------
-- forward LSB bit reader (basis bitwise_decoder)
-- ---------------------------------------------------------------------------
type BitsIn = {
	buf: buffer,
	pos: number, -- next byte
	endPos: number,
	bitBuf: number,
	bitCount: number,
}

local function bitsInit(buf: buffer, base: number, len: number): BitsIn
	return { buf = buf, pos = base, endPos = base + len, bitBuf = 0, bitCount = 0 }
end

local function bitsPeek(b: BitsIn, n: number): number
	while b.bitCount < n do
		local c = 0
		if b.pos < b.endPos then
			c = buffer.readu8(b.buf, b.pos)
			b.pos += 1
		end
		b.bitBuf += c * lshift(1, b.bitCount)
		b.bitCount += 8
	end
	return band(b.bitBuf, lshift(1, n) - 1)
end

local function bitsRemove(b: BitsIn, n: number)
	b.bitBuf = math.floor(b.bitBuf / lshift(1, n))
	b.bitCount -= n
end

local function bitsGet(b: BitsIn, n: number): number
	if n == 0 then
		return 0
	end
	if n > 25 then
		local lo = bitsPeek(b, 25)
		bitsRemove(b, 25)
		local hi = bitsPeek(b, n - 25)
		bitsRemove(b, n - 25)
		return lo + hi * 33554432
	end
	local v = bitsPeek(b, n)
	bitsRemove(b, n)
	return v
end

local function bitsGetVlc(b: BitsIn, chunkBits: number): number
	local chunkSize = lshift(1, chunkBits)
	local chunkMask = chunkSize - 1
	local v = 0
	local ofs = 0
	while true do
		local s = bitsGet(b, chunkBits + 1)
		v += band(s, chunkMask) * lshift(1, ofs)
		ofs += chunkBits
		if band(s, chunkSize) == 0 then
			break
		end
		if ofs >= 32 then
			break
		end
	end
	return v
end

-- ---------------------------------------------------------------------------
-- canonical huffman (basis flavour: codes read LSB-first)
-- ---------------------------------------------------------------------------
type Huff = {
	-- maps (len * 2^17 + reversedCodePrefix) -> symbol
	map: { [number]: number },
	maxLen: number,
}

local function huffInit(codeSizes: { number }, totalSyms: number): (Huff?, string?)
	local sizeCount = table.create(17, 0)
	for i = 1, totalSyms do
		local s = codeSizes[i]
		if s > 16 then
			return nil, "huffman code size too large"
		end
		if s > 0 then
			sizeCount[s] += 1
		end
	end
	local nextCode = table.create(17, 0)
	local total = 0
	local maxLen = 0
	for i = 1, 16 do
		if sizeCount[i] > 0 then
			maxLen = i
		end
		total = (total + sizeCount[i]) * 2
		nextCode[i + 1] = total
	end
	local map: { [number]: number } = {}
	for sym = 1, totalSyms do
		local size = codeSizes[sym]
		if size > 0 then
			local code = nextCode[size]
			nextCode[size] = code + 1
			-- reverse the code bits (LSB-first storage)
			local rev = 0
			local c = code
			for _ = 1, size do
				rev = rev * 2 + band(c, 1)
				c = rshift(c, 1)
			end
			map[size * 131072 + rev] = sym - 1
		end
	end
	return { map = map, maxLen = maxLen }, nil
end

local function huffDecode(b: BitsIn, h: Huff): number
	local v = 0
	local m = 1
	for len = 1, h.maxLen do
		v += bitsGet(b, 1) * m
		m *= 2
		local sym = h.map[len * 131072 + v]
		if sym ~= nil then
			return sym
		end
	end
	return -1
end

-- read_huffman_table: code length codes + RLE-coded code sizes
local SORTED_CODELENGTH_CODES = { 17, 18, 19, 20, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15, 16 }

local function readHuffTable(b: BitsIn): (Huff?, string?)
	local totalUsedSyms = bitsGet(b, 14)
	if totalUsedSyms == 0 then
		return { map = {}, maxLen = 0 }, nil
	end
	if totalUsedSyms > 16384 then
		return nil, "too many huffman symbols"
	end
	local clcSizes = table.create(21, 0)
	local numClc = bitsGet(b, 5)
	if numClc < 1 or numClc > 21 then
		return nil, "invalid code length code count"
	end
	for i = 1, numClc do
		clcSizes[SORTED_CODELENGTH_CODES[i] + 1] = bitsGet(b, 3)
	end
	local clcTable, cerr = huffInit(clcSizes, 21)
	if not clcTable then
		return nil, cerr
	end
	local codeSizes = table.create(totalUsedSyms, 0)
	local cur = 0
	while cur < totalUsedSyms do
		local c = huffDecode(b, clcTable :: Huff)
		if c < 0 then
			return nil, "bad huffman code length stream"
		end
		if c <= 16 then
			cur += 1
			codeSizes[cur] = c
		elseif c == 17 then
			cur += bitsGet(b, 3) + 3
		elseif c == 18 then
			cur += bitsGet(b, 7) + 11
		else
			if cur == 0 then
				return nil, "huffman repeat with no previous size"
			end
			local l: number
			if c == 19 then
				l = bitsGet(b, 2) + 3
			else
				l = bitsGet(b, 7) + 7
			end
			local prev = codeSizes[cur]
			if prev == 0 then
				return nil, "huffman repeat of zero size"
			end
			for _ = 1, l do
				if cur >= totalUsedSyms then
					return nil, "huffman repeat overflow"
				end
				cur += 1
				codeSizes[cur] = prev
			end
		end
	end
	if cur ~= totalUsedSyms then
		return nil, "huffman code size count mismatch"
	end
	return huffInit(codeSizes, totalUsedSyms)
end

-- ---------------------------------------------------------------------------
-- ETC1S / BasisLZ
-- ---------------------------------------------------------------------------
type Etc1sCodebooks = {
	-- endpoints: per index: { r5, g5, b5, inten }
	endR: { number },
	endG: { number },
	endB: { number },
	endI: { number },
	-- selectors: per index: 4 row bytes (2 bits per texel)
	selRows: { { number } },
	numEndpoints: number,
	numSelectors: number,
}

local function decodePalettes(
	numEndpoints: number,
	epBuf: buffer,
	epBase: number,
	epLen: number,
	numSelectors: number,
	selBuf: buffer,
	selBase: number,
	selLen: number
): (Etc1sCodebooks?, string?)
	local b = bitsInit(epBuf, epBase, epLen)
	local m0, e0 = readHuffTable(b)
	if not m0 then
		return nil, e0
	end
	local m1, e1 = readHuffTable(b)
	if not m1 then
		return nil, e1
	end
	local m2, e2 = readHuffTable(b)
	if not m2 then
		return nil, e2
	end
	local mInten, e3 = readHuffTable(b)
	if not mInten then
		return nil, e3
	end
	local color5DeltaModel0 = m0 :: Huff
	local color5DeltaModel1 = m1 :: Huff
	local color5DeltaModel2 = m2 :: Huff
	local intenDeltaModel = mInten :: Huff

	local grayscale = bitsGet(b, 1) ~= 0

	local endR = table.create(numEndpoints, 0)
	local endG = table.create(numEndpoints, 0)
	local endB = table.create(numEndpoints, 0)
	local endI = table.create(numEndpoints, 0)

	local prevR, prevG, prevB = 16, 16, 16
	local prevInten = 0

	for i = 1, numEndpoints do
		local intenDelta = huffDecode(b, intenDeltaModel)
		prevInten = band(intenDelta + prevInten, 7)
		endI[i] = prevInten
		local numComps = if grayscale then 1 else 3
		for c = 1, numComps do
			local prev = if c == 1 then prevR elseif c == 2 then prevG else prevB
			local delta: number
			if prev <= 9 then
				delta = huffDecode(b, color5DeltaModel0)
			elseif prev <= 21 then
				delta = huffDecode(b, color5DeltaModel1)
			else
				delta = huffDecode(b, color5DeltaModel2)
			end
			local v = band(prev + delta, 31)
			if c == 1 then
				prevR = v
				endR[i] = v
			elseif c == 2 then
				prevG = v
				endG[i] = v
			else
				prevB = v
				endB[i] = v
			end
		end
		if grayscale then
			endG[i] = endR[i]
			endB[i] = endR[i]
			prevG = prevR
			prevB = prevR
		end
	end

	-- selectors
	local sb = bitsInit(selBuf, selBase, selLen)
	local usedGlobal = bitsGet(sb, 1) == 1
	if usedGlobal then
		return nil, "global selector codebooks are unsupported"
	end
	local usedHybrid = bitsGet(sb, 1) == 1
	if usedHybrid then
		return nil, "hybrid selector codebooks are unsupported"
	end
	local usedRaw = bitsGet(sb, 1) == 1

	local selRows: { { number } } = table.create(numSelectors)
	if usedRaw then
		for i = 1, numSelectors do
			local rows = table.create(4, 0)
			for j = 1, 4 do
				rows[j] = bitsGet(sb, 8)
			end
			selRows[i] = rows
		end
	else
		local dm, derr = readHuffTable(sb)
		if not dm then
			return nil, derr
		end
		local deltaModel = dm :: Huff
		local prevBytes = { 0, 0, 0, 0 }
		for i = 1, numSelectors do
			local rows = table.create(4, 0)
			if i == 1 then
				for j = 1, 4 do
					local cur = bitsGet(sb, 8)
					prevBytes[j] = cur
					rows[j] = cur
				end
			else
				for j = 1, 4 do
					local delta = huffDecode(sb, deltaModel)
					local cur = bxor(delta, prevBytes[j])
					prevBytes[j] = cur
					rows[j] = cur
				end
			end
			selRows[i] = rows
		end
	end

	return {
		endR = endR,
		endG = endG,
		endB = endB,
		endI = endI,
		selRows = selRows,
		numEndpoints = numEndpoints,
		numSelectors = numSelectors,
	}, nil
end

type Etc1sTables = {
	endpointPred: Huff,
	deltaEndpoint: Huff,
	selectorModel: Huff,
	selectorHistoryRle: Huff,
	selectorHistoryBufSize: number,
}

local function decodeTables(buf: buffer, base: number, len: number): (Etc1sTables?, string?)
	local b = bitsInit(buf, base, len)
	local t1, e1 = readHuffTable(b)
	if not t1 then
		return nil, e1
	end
	local t2, e2 = readHuffTable(b)
	if not t2 then
		return nil, e2
	end
	local t3, e3 = readHuffTable(b)
	if not t3 then
		return nil, e3
	end
	local t4, e4 = readHuffTable(b)
	if not t4 then
		return nil, e4
	end
	local bufSize = bitsGet(b, 13)
	if bufSize == 0 then
		return nil, "zero selector history buffer size"
	end
	return {
		endpointPred = t1 :: Huff,
		deltaEndpoint = t2 :: Huff,
		selectorModel = t3 :: Huff,
		selectorHistoryRle = t4 :: Huff,
		selectorHistoryBufSize = bufSize,
	}, nil
end

local function clamp255(v: number): number
	return if v < 0 then 0 elseif v > 255 then 255 else v
end

-- Transcode one ETC1S slice. writeBlock(blockX, blockY, endpointIndex,
-- selectorIndex) emits a fully decoded 4x4 block.
local function transcodeSlice(
	cb: Etc1sCodebooks,
	tb: Etc1sTables,
	buf: buffer,
	base: number,
	len: number,
	numBlocksX: number,
	numBlocksY: number,
	writeBlock: (number, number, number, number) -> ()
): string?
	local b = bitsInit(buf, base, len)

	-- approx move to front history
	local historySize = tb.selectorHistoryBufSize
	local history = table.create(historySize, 0)
	local historyRover = math.floor(historySize / 2)

	local curSelectorRleCount = 0
	local curPredBits = 0
	local prevEndpointPredSym = 0
	local endpointPredRepeatCount = 0
	local prevEndpointIndex = 0

	local numEndpoints = cb.numEndpoints
	local numSelectors = cb.numSelectors
	local totalBlocks = numBlocksX * numBlocksY
	local RLE_SYMBOL_INDEX = historySize + numSelectors

	-- per-column predictors for two rows (pred bits + endpoint index)
	local predBitsRow: { { number } } = { table.create(numBlocksX, 0), table.create(numBlocksX, 0) }
	local endpointRow: { { number } } = { table.create(numBlocksX, 0), table.create(numBlocksX, 0) }

	for blockY = 0, numBlocksY - 1 do
		local curRow = band(blockY, 1)
		for blockX = 0, numBlocksX - 1 do
			if band(blockX, 1) == 0 then
				if band(blockY, 1) == 0 then
					if endpointPredRepeatCount > 0 then
						endpointPredRepeatCount -= 1
						curPredBits = prevEndpointPredSym
					else
						curPredBits = huffDecode(b, tb.endpointPred)
						if curPredBits == 256 then -- ENDPOINT_PRED_REPEAT_LAST_SYMBOL (4^4)
							endpointPredRepeatCount = bitsGetVlc(b, 4) + 3 - 1
							curPredBits = prevEndpointPredSym
						else
							prevEndpointPredSym = curPredBits
						end
					end
					predBitsRow[bxor(curRow, 1) + 1][blockX + 1] = rshift(curPredBits, 4)
				else
					curPredBits = predBitsRow[curRow + 1][blockX + 1]
				end
			end

			local pred = band(curPredBits, 3)
			curPredBits = rshift(curPredBits, 2)

			local endpointIndex: number
			if pred == 0 then
				if blockX == 0 then
					return "etc1s: invalid left prediction"
				end
				endpointIndex = prevEndpointIndex
			elseif pred == 1 then
				if blockY == 0 then
					return "etc1s: invalid upper prediction"
				end
				endpointIndex = endpointRow[bxor(curRow, 1) + 1][blockX + 1]
			elseif pred == 2 then
				if blockX == 0 or blockY == 0 then
					return "etc1s: invalid upper-left prediction"
				end
				endpointIndex = endpointRow[bxor(curRow, 1) + 1][blockX]
			else
				local deltaSym = huffDecode(b, tb.deltaEndpoint)
				endpointIndex = deltaSym + prevEndpointIndex
				if endpointIndex >= numEndpoints then
					endpointIndex -= numEndpoints
				end
			end
			endpointRow[curRow + 1][blockX + 1] = endpointIndex
			prevEndpointIndex = endpointIndex

			local selectorIndex = 0
			do
				local selectorSym: number
				if curSelectorRleCount > 0 then
					curSelectorRleCount -= 1
					selectorSym = numSelectors
				else
					selectorSym = huffDecode(b, tb.selectorModel)
					if selectorSym == RLE_SYMBOL_INDEX then
						local runSym = huffDecode(b, tb.selectorHistoryRle)
						if runSym == 63 then -- SELECTOR_HISTORY_BUF_RLE_COUNT_TOTAL - 1
							curSelectorRleCount = bitsGetVlc(b, 7) + 3
						else
							curSelectorRleCount = runSym + 3
						end
						if curSelectorRleCount > totalBlocks then
							return "etc1s: invalid selector RLE count"
						end
						selectorSym = numSelectors
						curSelectorRleCount -= 1
					end
				end
				if selectorSym >= numSelectors then
					local historyIndex = selectorSym - numSelectors
					if historyIndex >= historySize then
						return "etc1s: invalid history index"
					end
					selectorIndex = history[historyIndex + 1]
					if historyIndex ~= 0 then
						-- approx_move_to_front use(): swap with index/2
						local half = math.floor(historyIndex / 2)
						local x = history[half + 1]
						history[half + 1] = history[historyIndex + 1]
						history[historyIndex + 1] = x
					end
				else
					selectorIndex = selectorSym
					if historySize > 0 then
						history[historyRover + 1] = selectorIndex
						historyRover += 1
						if historyRover == historySize then
							historyRover = math.floor(historySize / 2)
						end
					end
				end
			end

			if endpointIndex >= numEndpoints or selectorIndex >= numSelectors then
				return "etc1s: invalid indices"
			end

			writeBlock(blockX, blockY, endpointIndex, selectorIndex)
		end
	end
	return nil
end

-- ---------------------------------------------------------------------------
-- UASTC
-- ---------------------------------------------------------------------------

-- per-range unquantization lookup, built lazily: UNQ[range+1][packedVal+1]
local UNQ_CACHE: { [number]: { number } } = {}

local function astcLevels(range: number): number
	local bits = BISE_RANGES[range * 3 + 1]
	local trits = BISE_RANGES[range * 3 + 2]
	local quints = BISE_RANGES[range * 3 + 3]
	return (1 + 2 * trits + 4 * quints) * lshift(1, bits)
end

local function unquantTable(range: number): { number }
	local cached = UNQ_CACHE[range]
	if cached then
		return cached
	end
	local bits = BISE_RANGES[range * 3 + 1]
	local trits = BISE_RANGES[range * 3 + 2]
	local quints = BISE_RANGES[range * 3 + 3]
	local levels = astcLevels(range)
	local t = table.create(levels, 0)
	for packed = 0, levels - 1 do
		local val: number
		if trits == 0 and quints == 0 then
			-- bit replication to 8 bits
			local v = 0
			local bitsLeft = 8
			while bitsLeft > 0 do
				local x = packed
				local n = math.min(bitsLeft, bits)
				if n < bits then
					x = rshift(x, bits - n)
				end
				v += band(x, lshift(1, n) - 1) * lshift(1, bitsLeft - n)
				bitsLeft -= n
			end
			val = v
		else
			local packedBits = band(packed, lshift(1, bits) - 1)
			local packedTQ = rshift(packed, bits)
			local A = if band(packedBits, 1) ~= 0 then 511 else 0
			local C = UNQUANT_C[range + 1]
			local D = packedTQ
			local Bstr = UNQUANT_B[range + 1]
			local B = 0
			for i = 1, 9 do
				B *= 2
				local ch = string.sub(Bstr, i, i)
				if ch ~= "0" then
					local bitIdx = string.byte(ch) - 97 -- 'a'
					B += band(rshift(packedBits, bitIdx), 1)
				end
			end
			local v = D * C + B
			v = bxor(v, A)
			val = band(A, 0x80) + rshift(v, 2)
		end
		t[packed + 1] = val
	end
	UNQ_CACHE[range] = t
	return t
end

local WEIGHT_TABLES: { { number } } = {}

local function weightTable(bits: number): { number }
	local t = WEIGHT_TABLES[bits]
	if t then
		return t
	end
	if bits == 1 then
		t = BC7_WEIGHTS1
	elseif bits == 2 then
		t = BC7_WEIGHTS2
	elseif bits == 3 then
		t = BC7_WEIGHTS3
	elseif bits == 4 then
		t = ASTC_WEIGHTS4
	else
		t = ASTC_WEIGHTS5
	end
	WEIGHT_TABLES[bits] = t
	return t
end

local function astcInterpolate(l: number, h: number, w: number): number
	l = l * 256 + l
	h = h * 256 + h
	local k = rshift(l * (64 - w) + h * w + 32, 6)
	return rshift(k, 8)
end

-- read n bits (<=9) from the 16-byte block at bitOfs (LSB-first)
local function blockBits(blk: buffer, base: number, bitOfs: number, n: number): number
	if n == 0 then
		return 0
	end
	local byteOfs = rshift(bitOfs, 3)
	local v = buffer.readu8(blk, base + byteOfs)
	if byteOfs + 1 < 16 then
		v += buffer.readu8(blk, base + byteOfs + 1) * 256
	end
	if byteOfs + 2 < 16 then
		v += buffer.readu8(blk, base + byteOfs + 2) * 65536
	end
	return band(rshift(v, band(bitOfs, 7)), lshift(1, n) - 1)
end

-- Decode one 16-byte UASTC block into out (RGBA at outPos, rowStride bytes),
-- writing maxX x maxY pixels.
local function decodeUASTCBlock(
	blk: buffer,
	base: number,
	out: buffer,
	outPos: number,
	rowStride: number,
	maxX: number,
	maxY: number
): string?
	local mode = HUFF_MODES[band(buffer.readu8(blk, base), 127) + 1]
	if mode >= 19 then
		return "uastc: invalid mode"
	end
	local bitOfs = MODE_HUFF_CODES[mode * 2 + 2]

	if mode == 8 then
		-- solid color
		local r = blockBits(blk, base, bitOfs, 8)
		local g = blockBits(blk, base, bitOfs + 8, 8)
		local bb = blockBits(blk, base, bitOfs + 16, 8)
		local a = blockBits(blk, base, bitOfs + 24, 8)
		for y = 0, maxY - 1 do
			local p = outPos + y * rowStride
			for x = 0, maxX - 1 do
				buffer.writeu8(out, p, r)
				buffer.writeu8(out, p + 1, g)
				buffer.writeu8(out, p + 2, bb)
				buffer.writeu8(out, p + 3, a)
				p += 4
			end
		end
		return nil
	end

	bitOfs += MODE_HINT_BITS[mode + 1]

	local subsets = 1
	local commonPattern = 0
	if mode == 2 or mode == 4 or mode == 7 or mode == 9 or mode == 16 then
		commonPattern = blockBits(blk, base, bitOfs, 5)
		bitOfs += 5
		subsets = 2
	elseif mode == 3 then
		commonPattern = blockBits(blk, base, bitOfs, 4)
		bitOfs += 4
		subsets = 3
	end
	if subsets == 2 then
		if mode == 7 then
			if commonPattern >= 19 then
				return "uastc: invalid common pattern (mode 7)"
			end
		elseif commonPattern >= 30 then
			return "uastc: invalid common pattern"
		end
	elseif subsets == 3 and commonPattern >= 11 then
		return "uastc: invalid common pattern (3 subsets)"
	end

	local totalPlanes = 1
	local ccs = -1
	if mode == 6 or mode == 11 or mode == 13 then
		ccs = blockBits(blk, base, bitOfs, 2)
		bitOfs += 2
		totalPlanes = 2
	elseif mode == 17 then
		ccs = 3
		totalPlanes = 2
	end

	local totalComps = MODE_COMPS[mode + 1]
	local weightBits = MODE_WEIGHT_BITS[mode + 1]
	local totalValues = totalComps * 2 * subsets
	local endpointRange = MODE_EP_RANGES[mode + 1]

	local epBits = BISE_RANGES[endpointRange * 3 + 1]
	local epTrits = BISE_RANGES[endpointRange * 3 + 2]
	local epQuints = BISE_RANGES[endpointRange * 3 + 3]

	local totalTqs = 0
	local bundleSize = 0
	local mul = 0
	if epTrits ~= 0 then
		totalTqs = math.floor((totalValues + 4) / 5)
		bundleSize = 5
		mul = 3
	elseif epQuints ~= 0 then
		totalTqs = math.floor((totalValues + 2) / 3)
		bundleSize = 3
		mul = 5
	end

	local tqValues = table.create(8, 0)
	for i = 1, totalTqs do
		local numBits = if epTrits ~= 0 then 8 else 7
		if i == totalTqs then
			local numRemaining = totalValues - (totalTqs - 1) * bundleSize
			if epTrits ~= 0 then
				if numRemaining == 1 then
					numBits = 2
				elseif numRemaining == 2 then
					numBits = 4
				elseif numRemaining == 3 then
					numBits = 5
				elseif numRemaining == 4 then
					numBits = 7
				end
			else
				if numRemaining == 1 then
					numBits = 3
				elseif numRemaining == 2 then
					numBits = 5
				end
			end
		end
		tqValues[i] = blockBits(blk, base, bitOfs, numBits)
		bitOfs += numBits
	end

	local endpoints = table.create(totalValues, 0)
	local accum = 0
	local accumRemaining = 0
	local nextTq = 1
	for i = 1, totalValues do
		local value = blockBits(blk, base, bitOfs, epBits)
		bitOfs += epBits
		if totalTqs > 0 then
			if accumRemaining == 0 then
				accum = tqValues[nextTq]
				nextTq += 1
				accumRemaining = bundleSize
			end
			local v = accum % mul
			accum = math.floor(accum / mul)
			accumRemaining -= 1
			value += v * lshift(1, epBits)
		end
		endpoints[i] = value
	end

	-- partition pattern + anchors
	local patBase = 0
	local pat: { number }? = nil
	local anchors: { number }? = nil
	if subsets >= 2 then
		if subsets == 3 then
			pat = PATTERNS3
			anchors = ANCHORS3
			patBase = commonPattern * 16
		elseif mode == 7 then
			pat = PATTERNS2_M7
			anchors = ANCHORS2_M7
			patBase = commonPattern * 16
		else
			pat = PATTERNS2
			anchors = ANCHORS2
			patBase = commonPattern * 16
		end
	end
	local anchorBase = commonPattern * 3

	-- weights
	local weights = table.create(16 * totalPlanes, 0)
	if totalPlanes == 2 then
		-- single subset; first two weights (both planes of texel 0) are anchors
		weights[1] = blockBits(blk, base, bitOfs, weightBits - 1)
		bitOfs += weightBits - 1
		weights[2] = blockBits(blk, base, bitOfs, weightBits - 1)
		bitOfs += weightBits - 1
		for i = 3, 32 do
			weights[i] = blockBits(blk, base, bitOfs, weightBits)
			bitOfs += weightBits
		end
	elseif subsets == 1 then
		weights[1] = blockBits(blk, base, bitOfs, weightBits - 1)
		bitOfs += weightBits - 1
		for i = 2, 16 do
			weights[i] = blockBits(blk, base, bitOfs, weightBits)
			bitOfs += weightBits
		end
	else
		local a = anchors :: { number }
		local a0 = a[anchorBase + 1]
		local a1 = a[anchorBase + 2]
		local a2 = if subsets == 3 then a[anchorBase + 3] else -1
		for i = 0, 15 do
			local nb = weightBits
			if i == a0 or i == a1 or i == a2 then
				nb = weightBits - 1
			end
			weights[i + 1] = blockBits(blk, base, bitOfs, nb)
			bitOfs += nb
		end
	end

	-- endpoints -> RGBA pairs per subset
	local unq = unquantTable(endpointRange)
	local epR0 = {}
	local epG0 = {}
	local epB0 = {}
	local epA0 = {}
	local epR1 = {}
	local epG1 = {}
	local epB1 = {}
	local epA1 = {}
	for s = 0, subsets - 1 do
		local o = s * totalComps * 2
		if totalComps == 2 then
			local ll = unq[endpoints[o + 1] + 1]
			local lh = unq[endpoints[o + 2] + 1]
			local al = unq[endpoints[o + 3] + 1]
			local ah = unq[endpoints[o + 4] + 1]
			epR0[s + 1] = ll
			epG0[s + 1] = ll
			epB0[s + 1] = ll
			epA0[s + 1] = al
			epR1[s + 1] = lh
			epG1[s + 1] = lh
			epB1[s + 1] = lh
			epA1[s + 1] = ah
		else
			epR0[s + 1] = unq[endpoints[o + 1] + 1]
			epR1[s + 1] = unq[endpoints[o + 2] + 1]
			epG0[s + 1] = unq[endpoints[o + 3] + 1]
			epG1[s + 1] = unq[endpoints[o + 4] + 1]
			epB0[s + 1] = unq[endpoints[o + 5] + 1]
			epB1[s + 1] = unq[endpoints[o + 6] + 1]
			if totalComps == 4 then
				epA0[s + 1] = unq[endpoints[o + 7] + 1]
				epA1[s + 1] = unq[endpoints[o + 8] + 1]
			else
				epA0[s + 1] = 255
				epA1[s + 1] = 255
			end
		end
	end

	local wt = weightTable(weightBits)

	-- decode pixels
	for i = 0, 15 do
		local x = band(i, 3)
		local y = rshift(i, 2)
		if x >= maxX or y >= maxY then
			continue
		end
		local subset = 0
		if pat then
			subset = (pat :: { number })[patBase + i + 1]
		end
		local r: number, g: number, bch: number, a: number
		if totalPlanes == 2 then
			local w0 = wt[weights[i * 2 + 1] + 1]
			local w1 = wt[weights[i * 2 + 2] + 1]
			local wr = if ccs == 0 then w1 else w0
			local wg = if ccs == 1 then w1 else w0
			local wb = if ccs == 2 then w1 else w0
			local wa = if ccs == 3 then w1 else w0
			r = astcInterpolate(epR0[1], epR1[1], wr)
			g = astcInterpolate(epG0[1], epG1[1], wg)
			bch = astcInterpolate(epB0[1], epB1[1], wb)
			a = if epA0[1] == 255 and epA1[1] == 255 and ccs ~= 3
				then astcInterpolate(epA0[1], epA1[1], wa)
				else astcInterpolate(epA0[1], epA1[1], wa)
		else
			local w = wt[weights[i + 1] + 1]
			local si = subset + 1
			r = astcInterpolate(epR0[si], epR1[si], w)
			g = astcInterpolate(epG0[si], epG1[si], w)
			bch = astcInterpolate(epB0[si], epB1[si], w)
			a = astcInterpolate(epA0[si], epA1[si], w)
		end
		local p = outPos + y * rowStride + x * 4
		buffer.writeu8(out, p, r)
		buffer.writeu8(out, p + 1, g)
		buffer.writeu8(out, p + 2, bch)
		buffer.writeu8(out, p + 3, a)
	end
	return nil
end

-- ---------------------------------------------------------------------------
-- KTX2 container
-- ---------------------------------------------------------------------------
local KTX2_IDENTIFIER = "\xABKTX 20\xBB\r\n\x1A\n"

local function isKTX2(bytes: buffer): boolean
	if buffer.len(bytes) < 12 then
		return false
	end
	return buffer.readstring(bytes, 0, 12) == KTX2_IDENTIFIER
end

local function decode(bytes: buffer): (DecodedTexture?, string?)
	local ok, result, err = pcall(function(): (DecodedTexture?, string?)
		if not isKTX2(bytes) then
			return nil, "not a KTX2 file"
		end
		local vkFormat = buffer.readu32(bytes, 12)
		local width = buffer.readu32(bytes, 20)
		local height = buffer.readu32(bytes, 24)
		local layerCount = buffer.readu32(bytes, 32)
		local faceCount = buffer.readu32(bytes, 36)
		local levelCount = math.max(1, buffer.readu32(bytes, 40))
		local scheme = buffer.readu32(bytes, 44)
		-- sgd offset/length are u64; low 32 bits suffice for sane files
		local sgdOffset = buffer.readu32(bytes, 64)
		local sgdLength = buffer.readu32(bytes, 72)

		if vkFormat ~= 0 then
			return nil, "KTX2: only Basis Universal (vkFormat 0) is supported, got " .. vkFormat
		end
		if layerCount > 1 or faceCount > 1 then
			return nil, "KTX2: array/cubemap textures are not supported"
		end
		if width == 0 or height == 0 then
			return nil, "KTX2: invalid dimensions"
		end

		-- level 0 from the level index (each entry: u64 offset/length/uncompressed)
		local lvlOffset = buffer.readu32(bytes, 80)
		local lvlLength = buffer.readu32(bytes, 88)
		local lvlUncompressed = buffer.readu32(bytes, 96)

		local numBlocksX = rshift(width + 3, 2)
		local numBlocksY = rshift(height + 3, 2)
		local rgba = buffer.create(width * height * 4)
		local rowStride = width * 4

		if scheme == 1 then
			-- BasisLZ / ETC1S
			if sgdLength < 20 then
				return nil, "KTX2: missing BasisLZ global data"
			end
			local numEndpoints = buffer.readu16(bytes, sgdOffset)
			local numSelectors = buffer.readu16(bytes, sgdOffset + 2)
			local endpointsLen = buffer.readu32(bytes, sgdOffset + 4)
			local selectorsLen = buffer.readu32(bytes, sgdOffset + 8)
			local tablesLen = buffer.readu32(bytes, sgdOffset + 12)
			-- extendedLen at +16 (unused)
			local numImages = 0
			do
				-- one imageDesc per level (single layer/face)
				numImages = levelCount
			end
			local descBase = sgdOffset + 20
			local dataBase = descBase + numImages * 20
			local endpointsBase = dataBase
			local selectorsBase = endpointsBase + endpointsLen
			local tablesBase = selectorsBase + selectorsLen

			local cbres, cerr = decodePalettes(
				numEndpoints,
				bytes,
				endpointsBase,
				endpointsLen,
				numSelectors,
				bytes,
				selectorsBase,
				selectorsLen
			)
			if not cbres then
				return nil, cerr
			end
			local cb = cbres :: Etc1sCodebooks
			local tbres, terr = decodeTables(bytes, tablesBase, tablesLen)
			if not tbres then
				return nil, terr
			end
			local tb = tbres :: Etc1sTables

			-- level 0 image desc
			local imageFlags = buffer.readu32(bytes, descBase)
			local rgbOff = buffer.readu32(bytes, descBase + 4)
			local rgbLen = buffer.readu32(bytes, descBase + 8)
			local alphaOff = buffer.readu32(bytes, descBase + 12)
			local alphaLen = buffer.readu32(bytes, descBase + 16)
			local hasAlpha = alphaLen > 0
			local _ = imageFlags

			-- decode the alpha slice first (alpha goes to the A channel from
			-- the green channel of the decoded blocks), then the color slice.
			if hasAlpha then
				local aerr = transcodeSlice(
					cb,
					tb,
					bytes,
					lvlOffset + alphaOff,
					alphaLen,
					numBlocksX,
					numBlocksY,
					function(bx: number, by: number, epIdx: number, selIdx: number)
						local g5 = cb.endG[epIdx + 1]
						local inten = ETC1_INTEN[cb.endI[epIdx + 1] + 1]
						local gBase = bor(lshift(g5, 3), rshift(g5, 2))
						local rows = cb.selRows[selIdx + 1]
						local maxX = math.min(4, width - bx * 4)
						local maxY = math.min(4, height - by * 4)
						for y = 0, maxY - 1 do
							local s = rows[y + 1]
							local p = (by * 4 + y) * rowStride + bx * 16
							for x = 0, maxX - 1 do
								local sel = band(rshift(s, x * 2), 3)
								buffer.writeu8(rgba, p + x * 4 + 3, clamp255(gBase + inten[sel + 1]))
							end
						end
					end
				)
				if aerr then
					return nil, aerr
				end
			end

			local serr = transcodeSlice(
				cb,
				tb,
				bytes,
				lvlOffset + rgbOff,
				rgbLen,
				numBlocksX,
				numBlocksY,
				function(bx: number, by: number, epIdx: number, selIdx: number)
					local r5 = cb.endR[epIdx + 1]
					local g5 = cb.endG[epIdx + 1]
					local b5 = cb.endB[epIdx + 1]
					local inten = ETC1_INTEN[cb.endI[epIdx + 1] + 1]
					local rBase = bor(lshift(r5, 3), rshift(r5, 2))
					local gBase = bor(lshift(g5, 3), rshift(g5, 2))
					local bBase = bor(lshift(b5, 3), rshift(b5, 2))
					local rows = cb.selRows[selIdx + 1]
					local maxX = math.min(4, width - bx * 4)
					local maxY = math.min(4, height - by * 4)
					for y = 0, maxY - 1 do
						local s = rows[y + 1]
						local p = (by * 4 + y) * rowStride + bx * 16
						for x = 0, maxX - 1 do
							local sel = band(rshift(s, x * 2), 3)
							local m = inten[sel + 1]
							local px = p + x * 4
							buffer.writeu8(rgba, px, clamp255(rBase + m))
							buffer.writeu8(rgba, px + 1, clamp255(gBase + m))
							buffer.writeu8(rgba, px + 2, clamp255(bBase + m))
							if not hasAlpha then
								buffer.writeu8(rgba, px + 3, 255)
							end
						end
					end
				end
			)
			if serr then
				return nil, serr
			end
		else
			-- UASTC: scheme 0 (none) or 2 (zstd)
			local levelData: buffer
			local levelBase: number
			if scheme == 2 then
				local comp = buffer.create(lvlLength)
				buffer.copy(comp, 0, bytes, lvlOffset, lvlLength)
				local dec, zerr = ZstdDecoder.decompress(comp)
				if not dec then
					return nil, "KTX2: zstd: " .. tostring(zerr)
				end
				levelData = dec :: buffer
				levelBase = 0
				if buffer.len(levelData) < lvlUncompressed then
					return nil, "KTX2: zstd output shorter than expected"
				end
			elseif scheme == 0 then
				levelData = bytes
				levelBase = lvlOffset
			else
				return nil, "KTX2: unsupported supercompression scheme " .. scheme
			end

			local needed = numBlocksX * numBlocksY * 16
			if buffer.len(levelData) - levelBase < needed then
				return nil, "KTX2: level data too short for UASTC blocks"
			end

			for by = 0, numBlocksY - 1 do
				for bx = 0, numBlocksX - 1 do
					local blockBase = levelBase + (by * numBlocksX + bx) * 16
					local maxX = math.min(4, width - bx * 4)
					local maxY = math.min(4, height - by * 4)
					local berr = decodeUASTCBlock(
						levelData,
						blockBase,
						rgba,
						(by * 4) * rowStride + bx * 16,
						rowStride,
						maxX,
						maxY
					)
					if berr then
						return nil, berr
					end
				end
			end
		end

		return { width = width, height = height, rgba = rgba }, nil
	end)
	if not ok then
		return nil, "ktx2 decode error: " .. tostring(result)
	end
	return result, err
end

return {
	decode = decode,
	isKTX2 = isKTX2,
}
