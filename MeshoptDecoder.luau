--!strict
--[[
  MeshoptDecoder -- pure-Luau decoder for EXT_meshopt_compression.
  Port of meshoptimizer's decoders (index codec v0/v1, index sequence,
  vertex codec v0/v1, octahedral/quaternion/exponential filters).
  https://github.com/zeux/meshoptimizer (MIT)

  Entry point: decodeBufferView(source, count, byteStride, mode, filter)
    mode   = "ATTRIBUTES" | "TRIANGLES" | "INDICES"
    filter = nil | "NONE" | "OCTAHEDRAL" | "QUATERNION" | "EXPONENTIAL"
  Returns a buffer of count*byteStride bytes, or nil + error string.
]]

local band = bit32.band
local bxor = bit32.bxor
local rshift = bit32.rshift
local lshift = bit32.lshift
local lrotate = bit32.lrotate

-- ---------------------------------------------------------------------------
-- Index buffer codec (mode TRIANGLES): edge fifo + codeaux table.
-- ---------------------------------------------------------------------------

-- decodeVByte: up to 5 groups of 7 bits, LSB first. Returns (value, newPos).
local function decodeVByte(data: buffer, pos: number): (number, number)
	local lead = buffer.readu8(data, pos)
	pos += 1
	if lead < 128 then
		return lead, pos
	end
	local result = band(lead, 127)
	local shift = 7
	for _ = 1, 4 do
		local group = buffer.readu8(data, pos)
		pos += 1
		result += band(group, 127) * lshift(1, shift)
		shift += 7
		if group < 128 then
			break
		end
	end
	-- keep the value in uint32 range like the reference decoder
	return result % 4294967296, pos
end

-- zigzag-decoded index delta applied to `last`, wrapped to uint32.
local function decodeIndex(data: buffer, pos: number, last: number): (number, number)
	local v, np = decodeVByte(data, pos)
	local index: number
	if v % 2 == 1 then
		index = last - (rshift(v, 1) + 1)
	else
		index = last + rshift(v, 1)
	end
	return index % 4294967296, np
end

local function decodeIndexBuffer(src: buffer, indexCount: number, indexSize: number): (buffer?, string?)
	local srcLen = buffer.len(src)
	local triCount = math.floor(indexCount / 3)
	if srcLen < 1 + triCount + 16 then
		return nil, "index data too short"
	end
	local header = buffer.readu8(src, 0)
	if band(header, 0xF0) ~= 0xE0 then
		return nil, string.format("bad index header 0x%02X", header)
	end
	local version = band(header, 0x0F)
	if version > 1 then
		return nil, "unsupported index codec version " .. version
	end
	local fecmax = if version >= 1 then 13 else 15

	-- fifos: edge fifo holds pairs, vertex fifo holds indices
	local edgeA = table.create(16, 4294967295)
	local edgeB = table.create(16, 4294967295)
	local vfifo = table.create(16, 4294967295)
	local edgeOff = 0 -- 0-based offsets like the C code
	local vertOff = 0

	local nextV = 0
	local last = 0

	local codePos = 1
	local dataPos = 1 + triCount
	local dataSafeEnd = srcLen - 16 -- codeaux table = last 16 bytes
	local codeauxBase = dataSafeEnd

	local dst = buffer.create(indexCount * indexSize)
	local writeIndex: (number, number) -> ()
	if indexSize == 2 then
		writeIndex = function(slot: number, v: number)
			buffer.writeu16(dst, slot * 2, band(v, 0xFFFF))
		end
	else
		writeIndex = function(slot: number, v: number)
			buffer.writeu32(dst, slot * 4, v % 4294967296)
		end
	end

	for i = 0, indexCount - 1, 3 do
		if dataPos > dataSafeEnd then
			return nil, "index data overrun"
		end
		local codetri = buffer.readu8(src, codePos)
		codePos += 1

		if codetri < 0xF0 then
			local fe = rshift(codetri, 4)
			local slot = band(edgeOff - 1 - fe, 15) + 1
			local a = edgeA[slot]
			local b = edgeB[slot]
			local c: number
			local fec = band(codetri, 15)

			if fec < fecmax then
				local cf = vfifo[band(vertOff - 1 - fec, 15) + 1]
				if fec == 0 then
					c = nextV
					nextV += 1
					vfifo[vertOff + 1] = c
					vertOff = band(vertOff + 1, 15)
				else
					c = cf
					vfifo[vertOff + 1] = c -- pushed with cond=0: written, not advanced
				end
			else
				if fec ~= 15 then
					-- fec - (fec XOR 3) maps 13 -> -1, 14 -> +1
					c = (last + (fec - bxor(fec, 3))) % 4294967296
					last = c
				else
					c, dataPos = decodeIndex(src, dataPos, last)
					last = c
				end
				vfifo[vertOff + 1] = c
				vertOff = band(vertOff + 1, 15)
			end

			edgeA[edgeOff + 1] = c
			edgeB[edgeOff + 1] = b
			edgeOff = band(edgeOff + 1, 15)
			edgeA[edgeOff + 1] = a
			edgeB[edgeOff + 1] = c
			edgeOff = band(edgeOff + 1, 15)

			writeIndex(i, a)
			writeIndex(i + 1, b)
			writeIndex(i + 2, c)
		else
			local codeaux: number
			local fea: number
			if codetri < 0xFE then
				codeaux = buffer.readu8(src, codeauxBase + band(codetri, 15))
				fea = 0
			else
				codeaux = buffer.readu8(src, dataPos)
				dataPos += 1
				fea = if codetri == 0xFE then 0 else 15
				if codeaux == 0 then
					nextV = 0
				end
			end
			local feb = rshift(codeaux, 4)
			local fec = band(codeaux, 15)

			local a: number
			if fea == 0 then
				a = nextV
				nextV += 1
			else
				a = 0
			end

			local b: number
			if feb == 0 then
				b = nextV
				nextV += 1
			else
				b = vfifo[band(vertOff - feb, 15) + 1]
			end

			local c: number
			if fec == 0 then
				c = nextV
				nextV += 1
			else
				c = vfifo[band(vertOff - fec, 15) + 1]
			end

			if fea == 15 then
				a, dataPos = decodeIndex(src, dataPos, last)
				last = a
			end
			if feb == 15 then
				b, dataPos = decodeIndex(src, dataPos, last)
				last = b
			end
			if fec == 15 then
				c, dataPos = decodeIndex(src, dataPos, last)
				last = c
			end

			writeIndex(i, a)
			writeIndex(i + 1, b)
			writeIndex(i + 2, c)

			vfifo[vertOff + 1] = a
			vertOff = band(vertOff + 1, 15)
			vfifo[vertOff + 1] = b
			if feb == 0 or feb == 15 then
				vertOff = band(vertOff + 1, 15)
			end
			vfifo[vertOff + 1] = c
			if fec == 0 or fec == 15 then
				vertOff = band(vertOff + 1, 15)
			end

			edgeA[edgeOff + 1] = b
			edgeB[edgeOff + 1] = a
			edgeOff = band(edgeOff + 1, 15)
			edgeA[edgeOff + 1] = c
			edgeB[edgeOff + 1] = b
			edgeOff = band(edgeOff + 1, 15)
			edgeA[edgeOff + 1] = a
			edgeB[edgeOff + 1] = c
			edgeOff = band(edgeOff + 1, 15)
		end
	end

	if dataPos ~= dataSafeEnd then
		return nil, "index data not fully consumed"
	end
	return dst, nil
end

-- ---------------------------------------------------------------------------
-- Index sequence codec (mode INDICES): vbyte deltas against two baselines.
-- ---------------------------------------------------------------------------

local function decodeIndexSequence(src: buffer, indexCount: number, indexSize: number): (buffer?, string?)
	local srcLen = buffer.len(src)
	if srcLen < 1 + indexCount + 4 then
		return nil, "index sequence too short"
	end
	local header = buffer.readu8(src, 0)
	if band(header, 0xF0) ~= 0xD0 then
		return nil, string.format("bad index sequence header 0x%02X", header)
	end
	if band(header, 0x0F) > 1 then
		return nil, "unsupported index sequence version"
	end

	local pos = 1
	local dataSafeEnd = srcLen - 4
	local last0, last1 = 0, 0
	local dst = buffer.create(indexCount * indexSize)

	for i = 0, indexCount - 1 do
		if pos >= dataSafeEnd then
			return nil, "index sequence overrun"
		end
		local v: number
		v, pos = decodeVByte(src, pos)
		local current = band(v, 1)
		v = rshift(v, 1)
		local index: number
		local base = if current == 0 then last0 else last1
		if v % 2 == 1 then
			index = base - (rshift(v, 1) + 1)
		else
			index = base + rshift(v, 1)
		end
		index %= 4294967296
		if current == 0 then
			last0 = index
		else
			last1 = index
		end
		if indexSize == 2 then
			buffer.writeu16(dst, i * 2, band(index, 0xFFFF))
		else
			buffer.writeu32(dst, i * 4, index)
		end
	end

	if pos ~= dataSafeEnd then
		return nil, "index sequence not fully consumed"
	end
	return dst, nil
end

-- ---------------------------------------------------------------------------
-- Vertex buffer codec (mode ATTRIBUTES).
-- Blocks of <=256 vertices; per byte-plane delta streams in 16-byte groups
-- with 2-bit headers; v1 adds per-channel control bytes + 16/32-bit channels.
-- ---------------------------------------------------------------------------

local BITS_V0 = { 0, 2, 4, 8 }
local BITS_V1 = { 0, 1, 2, 4, 8 } -- ctrl 0 uses entries 1..4, ctrl 1 entries 2..5

local function getVertexBlockSize(vertexSize: number): number
	local result = band(math.floor(8192 / vertexSize), bit32.bnot(15))
	return if result < 256 then result else 256
end

-- Decodes one 16-byte group with the given bit width into plane[outBase..outBase+15].
-- Returns new data position. Sentinel values ((1<<bits)-1) pull a full byte
-- from the trailing var-length section.
local function decodeBytesGroup(src: buffer, pos: number, plane: buffer, outBase: number, bits: number): number
	if bits == 0 then
		buffer.fill(plane, outBase, 0, 16)
		return pos
	elseif bits == 8 then
		buffer.copy(plane, outBase, src, pos, 16)
		return pos + 16
	end

	local sentinel = lshift(1, bits) - 1
	local varPos = pos + (if bits == 1 then 2 elseif bits == 2 then 4 else 8)

	if bits == 1 then
		for byteI = 0, 1 do
			local byte = buffer.readu8(src, pos + byteI)
			-- 1-bit values are stored LSB-first within each byte
			for k = 0, 7 do
				local enc = band(rshift(byte, k), 1)
				if enc == sentinel then
					buffer.writeu8(plane, outBase, buffer.readu8(src, varPos))
					varPos += 1
				else
					buffer.writeu8(plane, outBase, enc)
				end
				outBase += 1
			end
		end
	elseif bits == 2 then
		for byteI = 0, 3 do
			local byte = buffer.readu8(src, pos + byteI)
			for shift = 6, 0, -2 do
				local enc = band(rshift(byte, shift), 3)
				if enc == sentinel then
					buffer.writeu8(plane, outBase, buffer.readu8(src, varPos))
					varPos += 1
				else
					buffer.writeu8(plane, outBase, enc)
				end
				outBase += 1
			end
		end
	else -- bits == 4
		for byteI = 0, 7 do
			local byte = buffer.readu8(src, pos + byteI)
			local hi = rshift(byte, 4)
			if hi == sentinel then
				buffer.writeu8(plane, outBase, buffer.readu8(src, varPos))
				varPos += 1
			else
				buffer.writeu8(plane, outBase, hi)
			end
			outBase += 1
			local lo = band(byte, 15)
			if lo == sentinel then
				buffer.writeu8(plane, outBase, buffer.readu8(src, varPos))
				varPos += 1
			else
				buffer.writeu8(plane, outBase, lo)
			end
			outBase += 1
		end
	end
	return varPos
end

-- Decodes vertexCountAligned bytes (16-aligned) into plane at planeBase.
-- bitsTable selects the 4 possible widths for the 2-bit group headers.
local function decodeBytes(
	src: buffer,
	pos: number,
	srcEnd: number,
	plane: buffer,
	planeBase: number,
	alignedCount: number,
	bitsTable: { number },
	bitsOffset: number
): (number?, string?)
	local groups = math.floor(alignedCount / 16)
	local headerSize = math.floor((groups + 3) / 4)
	if srcEnd - pos < headerSize then
		return nil, "vertex group header overrun"
	end
	local headerPos = pos
	pos += headerSize
	for g = 0, groups - 1 do
		if srcEnd - pos < 24 then
			return nil, "vertex group data overrun"
		end
		local headerByte = buffer.readu8(src, headerPos + rshift(g, 2))
		local bitsk = band(rshift(headerByte, band(g, 3) * 2), 3)
		pos = decodeBytesGroup(src, pos, plane, planeBase + g * 16, bitsTable[bitsOffset + bitsk + 1])
	end
	return pos, nil
end

local function decodeVertexBlock(
	src: buffer,
	pos: number,
	srcEnd: number,
	dst: buffer,
	dstBase: number,
	vertexCount: number,
	vertexSize: number,
	lastVertex: buffer, -- vertexSize bytes, updated in place
	channels: buffer?, -- vertexSize/4 channel bytes (v1) or nil (v0)
	version: number
): (number?, string?)
	local alignedCount = band(vertexCount + 15, bit32.bnot(15))
	-- planes buffer: 4 byte-planes of vertexCount bytes each; aligned decode of
	-- plane j may spill into plane j+1's space, matching the reference layout.
	local planes = buffer.create(vertexCount * 4 + alignedCount)

	local controlPos = pos
	local controlSize = if version == 0 then 0 else math.floor(vertexSize / 4)
	if srcEnd - pos < controlSize then
		return nil, "control bytes overrun"
	end
	pos += controlSize

	for k = 0, vertexSize - 1, 4 do
		local ctrlByte = if version == 0 then 0 else buffer.readu8(src, controlPos + rshift(k, 2))

		for j = 0, 3 do
			local ctrl = band(rshift(ctrlByte, j * 2), 3)
			local planeBase = j * vertexCount
			if ctrl == 3 then
				-- literal: vertexCount raw bytes
				if srcEnd - pos < vertexCount then
					return nil, "literal overrun"
				end
				buffer.copy(planes, planeBase, src, pos, vertexCount)
				pos += vertexCount
			elseif ctrl == 2 then
				buffer.fill(planes, planeBase, 0, vertexCount)
			else
				local bitsTable = if version == 0 then BITS_V0 else BITS_V1
				local bitsOffset = if version == 0 then 0 else ctrl
				local np, err =
					decodeBytes(src, pos, srcEnd, planes, planeBase, alignedCount, bitsTable, bitsOffset)
				if not np then
					return nil, err
				end
				pos = np :: number
			end
		end

		local channel = if version == 0 then 0 else buffer.readu8(channels :: buffer, rshift(k, 2))
		local ctype = band(channel, 3)

		if ctype == 0 then
			-- 4 independent byte deltas (zigzag)
			for j = 0, 3 do
				local p = buffer.readu8(lastVertex, k + j)
				local off = dstBase + k + j
				local planeBase = j * vertexCount
				for i = 0, vertexCount - 1 do
					local v = buffer.readu8(planes, planeBase + i)
					-- unzigzag8 + p (mod 256)
					local d = if v % 2 == 1 then bxor(255, rshift(v, 1)) else rshift(v, 1)
					p = band(d + p, 255)
					buffer.writeu8(dst, off, p)
					off += vertexSize
				end
				buffer.writeu8(lastVertex, k + j, p)
			end
		elseif ctype == 1 then
			-- 2 x 16-bit deltas (zigzag), planes (0,1) and (2,3)
			for h = 0, 1 do
				local lo = h * 2
				local p = buffer.readu16(lastVertex, k + lo)
				local off = dstBase + k + lo
				local plane0 = lo * vertexCount
				local plane1 = (lo + 1) * vertexCount
				for i = 0, vertexCount - 1 do
					local v = buffer.readu8(planes, plane0 + i) + buffer.readu8(planes, plane1 + i) * 256
					local d = if v % 2 == 1 then bxor(65535, rshift(v, 1)) else rshift(v, 1)
					p = band(d + p, 65535)
					buffer.writeu16(dst, off, p)
					off += vertexSize
				end
				buffer.writeu16(lastVertex, k + lo, p)
			end
		elseif ctype == 2 then
			-- 32-bit rotate+xor channel
			local rot = band(32 - rshift(channel, 4), 31)
			local p = buffer.readu32(lastVertex, k)
			local off = dstBase + k
			local plane0 = 0
			local plane1 = vertexCount
			local plane2 = 2 * vertexCount
			local plane3 = 3 * vertexCount
			for i = 0, vertexCount - 1 do
				local v = buffer.readu8(planes, plane0 + i)
					+ buffer.readu8(planes, plane1 + i) * 256
					+ buffer.readu8(planes, plane2 + i) * 65536
					+ buffer.readu8(planes, plane3 + i) * 16777216
				p = bxor(lrotate(v, rot), p)
				buffer.writeu32(dst, off, p)
				off += vertexSize
			end
			buffer.writeu32(lastVertex, k, p)
		else
			return nil, "invalid channel type"
		end
	end

	return pos, nil
end

local function decodeVertexBuffer(src: buffer, vertexCount: number, vertexSize: number): (buffer?, string?)
	if vertexSize <= 0 or vertexSize > 256 or vertexSize % 4 ~= 0 then
		return nil, "invalid vertex size " .. vertexSize
	end
	local srcLen = buffer.len(src)
	if srcLen < 1 then
		return nil, "vertex data empty"
	end
	local header = buffer.readu8(src, 0)
	if band(header, 0xF0) ~= 0xA0 then
		return nil, string.format("bad vertex header 0x%02X", header)
	end
	local version = band(header, 0x0F)
	if version > 1 then
		return nil, "unsupported vertex codec version " .. version
	end

	local pos = 1
	local tailSize = vertexSize + (if version == 0 then 0 else math.floor(vertexSize / 4))
	local tailMin = if version == 0 then 32 else 24
	local tailPad = math.max(tailSize, tailMin)
	if srcLen - pos < tailPad then
		return nil, "vertex data too short for tail"
	end

	local tailBase = srcLen - tailSize
	local lastVertex = buffer.create(vertexSize)
	buffer.copy(lastVertex, 0, src, tailBase, vertexSize)

	local channels: buffer? = nil
	if version >= 1 then
		local cs = math.floor(vertexSize / 4)
		local cb = buffer.create(cs)
		buffer.copy(cb, 0, src, tailBase + vertexSize, cs)
		channels = cb
	end

	local dst = buffer.create(vertexCount * vertexSize)
	local blockSize = getVertexBlockSize(vertexSize)
	local vertexOffset = 0

	while vertexOffset < vertexCount do
		local blockCount = math.min(blockSize, vertexCount - vertexOffset)
		local np, err = decodeVertexBlock(
			src,
			pos,
			srcLen,
			dst,
			vertexOffset * vertexSize,
			blockCount,
			vertexSize,
			lastVertex,
			channels,
			version
		)
		if not np then
			return nil, err
		end
		pos = np :: number
		vertexOffset += blockCount
	end

	if srcLen - pos ~= tailPad then
		return nil, "vertex data not fully consumed"
	end
	return dst, nil
end

-- ---------------------------------------------------------------------------
-- Filters (applied in place after the vertex codec).
-- ---------------------------------------------------------------------------

local function readI8(b: buffer, off: number): number
	local v = buffer.readu8(b, off)
	return if v >= 128 then v - 256 else v
end

local function readI16(b: buffer, off: number): number
	local v = buffer.readu16(b, off)
	return if v >= 32768 then v - 65536 else v
end

-- Round a double to float32 precision (the reference decoder computes the
-- filters in float32; matching it keeps outputs bit-exact).
local f32buf = buffer.create(4)
local function f32(x: number): number
	buffer.writef32(f32buf, 0, x)
	return buffer.readf32(f32buf, 0)
end

local function truncToInt(v: number): number
	return if v >= 0 then math.floor(v) else math.ceil(v)
end

-- OCTAHEDRAL: stride 4 (snorm8) or 8 (snorm16). Components 0..2 rebuilt,
-- component 3 (tangent w sign) preserved.
local function decodeFilterOct(data: buffer, count: number, stride: number)
	local is16 = stride == 8
	local maxv = if is16 then 32767 else 127
	for i = 0, count - 1 do
		local base = i * stride
		local x: number, y: number, z: number
		if is16 then
			x = readI16(data, base)
			y = readI16(data, base + 2)
			z = readI16(data, base + 4)
		else
			x = readI8(data, base)
			y = readI8(data, base + 1)
			z = readI8(data, base + 2)
		end
		z = z - math.abs(x) - math.abs(y)
		local t = if z >= 0 then 0 else z
		x += if x >= 0 then t else -t
		y += if y >= 0 then t else -t
		-- float32 arithmetic to match the reference bit-for-bit
		local l = f32(math.sqrt(f32(f32(f32(x * x) + f32(y * y)) + f32(z * z))))
		local s = f32(maxv / l)
		local xf = truncToInt(f32(f32(x * s) + (if x >= 0 then 0.5 else -0.5)))
		local yf = truncToInt(f32(f32(y * s) + (if y >= 0 then 0.5 else -0.5)))
		local zf = truncToInt(f32(f32(z * s) + (if z >= 0 then 0.5 else -0.5)))
		if is16 then
			buffer.writeu16(data, base, band(xf, 0xFFFF))
			buffer.writeu16(data, base + 2, band(yf, 0xFFFF))
			buffer.writeu16(data, base + 4, band(zf, 0xFFFF))
		else
			buffer.writeu8(data, base, band(xf, 0xFF))
			buffer.writeu8(data, base + 1, band(yf, 0xFF))
			buffer.writeu8(data, base + 2, band(zf, 0xFF))
		end
	end
end

-- QUATERNION: stride 8 (4 x snorm16); component order restored from the
-- 2-bit index packed in the last component.
local QUAT_SCALE = f32(32767 / f32(math.sqrt(2)))
local function decodeFilterQuat(data: buffer, count: number, stride: number)
	for i = 0, count - 1 do
		local base = i * stride
		local cw = readI16(data, base + 6)
		local sf = bit32.bor(band(cw, 0xFFFF), 3)
		-- bor on the unsigned representation; recover the signed value
		if sf >= 32768 then
			sf -= 65536
		end
		local s = sf
		local x = readI16(data, base)
		local y = readI16(data, base + 2)
		local z = readI16(data, base + 4)
		-- float32 arithmetic to match the reference bit-for-bit
		local ww = f32(f32(f32(f32(f32(s * s) * 2) - f32(x * x)) - f32(y * y)) - f32(z * z))
		local w = f32(math.sqrt(if ww >= 0 then ww else 0))
		local ss = f32(QUAT_SCALE / s)
		local xf = truncToInt(f32(f32(x * ss) + (if x >= 0 then 0.5 else -0.5)))
		local yf = truncToInt(f32(f32(y * ss) + (if y >= 0 then 0.5 else -0.5)))
		local zf = truncToInt(f32(f32(z * ss) + (if z >= 0 then 0.5 else -0.5)))
		local wf = math.floor(f32(f32(w * ss) + 0.5))
		local qc = band(cw, 3)
		buffer.writeu16(data, base + band(qc + 1, 3) * 2, band(xf, 0xFFFF))
		buffer.writeu16(data, base + band(qc + 2, 3) * 2, band(yf, 0xFFFF))
		buffer.writeu16(data, base + band(qc + 3, 3) * 2, band(zf, 0xFFFF))
		buffer.writeu16(data, base + band(qc, 3) * 2, band(wf, 0xFFFF))
	end
end

-- EXPONENTIAL: every 4 bytes hold (exponent << 24) | 24-bit signed mantissa;
-- output is mantissa * 2^exponent as float32.
local function decodeFilterExp(data: buffer, count: number, stride: number)
	local total = math.floor(count * stride / 4)
	for i = 0, total - 1 do
		local off = i * 4
		local v = buffer.readu32(data, off)
		local m = band(v, 0xFFFFFF)
		if m >= 0x800000 then
			m -= 0x1000000
		end
		local e = rshift(v, 24)
		if e >= 128 then
			e -= 256
		end
		buffer.writef32(data, off, m * math.pow(2, e))
	end
end

-- ---------------------------------------------------------------------------
-- Public API
-- ---------------------------------------------------------------------------

local function decodeBufferView(
	src: buffer,
	count: number,
	byteStride: number,
	mode: string,
	filter: string?
): (buffer?, string?)
	local decoded: buffer?
	local err: string?
	if mode == "ATTRIBUTES" then
		decoded, err = decodeVertexBuffer(src, count, byteStride)
	elseif mode == "TRIANGLES" then
		decoded, err = decodeIndexBuffer(src, count, byteStride)
	elseif mode == "INDICES" then
		decoded, err = decodeIndexSequence(src, count, byteStride)
	else
		return nil, "unknown meshopt mode '" .. tostring(mode) .. "'"
	end
	if not decoded then
		return nil, err
	end
	local d = decoded :: buffer

	if filter == "OCTAHEDRAL" then
		decodeFilterOct(d, count, byteStride)
	elseif filter == "QUATERNION" then
		decodeFilterQuat(d, count, byteStride)
	elseif filter == "EXPONENTIAL" then
		decodeFilterExp(d, count, byteStride)
	end
	return d, nil
end

return {
	decodeBufferView = decodeBufferView,
	decodeVertexBuffer = decodeVertexBuffer,
	decodeIndexBuffer = decodeIndexBuffer,
	decodeIndexSequence = decodeIndexSequence,
}
