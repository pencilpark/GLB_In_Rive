--!strict
--[[
  ZstdDecoder -- pure-Luau Zstandard decompressor (RFC 8878).
  Used for KTX2 supercompression scheme 2 (UASTC textures are usually
  zstd-compressed). Supports standard frames without dictionaries, all
  block types, FSE + Huffman literals (1 and 4 streams), predefined and
  FSE-coded sequence tables, repeat offsets, and skippable frames.

  decompress(src) -> (buffer?, string?)
]]

local band = bit32.band
local rshift = bit32.rshift
local lshift = bit32.lshift

-- ---------------------------------------------------------------------------
-- backward bit reader: zstd entropy payloads are read from the LAST byte
-- backwards. The final byte carries a 1-bit sentinel above the data bits.
-- ---------------------------------------------------------------------------
type BackBits = {
	buf: buffer,
	-- absolute bit position (counting bits 0.. from the START of the region);
	-- reads consume downwards.
	bitPos: number,
	base: number, -- byte offset of region start
}

local function backBitsInit(buf: buffer, base: number, len: number): (BackBits?, string?)
	if len == 0 then
		return nil, "empty bitstream"
	end
	local last = buffer.readu8(buf, base + len - 1)
	if last == 0 then
		return nil, "corrupted bitstream (zero last byte)"
	end
	-- position of the sentinel bit (highest set bit of the last byte)
	local sentinel = 7
	while band(last, lshift(1, sentinel)) == 0 do
		sentinel -= 1
	end
	return {
		buf = buf,
		base = base,
		bitPos = (len - 1) * 8 + sentinel,
	}, nil
end

-- Read `n` bits (n <= 25) below the current position. The position may go
-- negative (reads past the stream start yield zero bits); this matches the
-- reference decoder's zero-fill when the bit container drains, and keeps
-- "give back unused bits" arithmetic exact at stream tails.
local function backBitsRead(bb: BackBits, n: number): number
	if n == 0 then
		return 0
	end
	local newPos = bb.bitPos - n
	bb.bitPos = newPos
	-- gather bytes covering bits [newPos, newPos+n)
	local firstByte = math.floor(newPos / 8)
	local shift = newPos - firstByte * 8
	local buf = bb.buf
	local base = bb.base
	local bufLen = buffer.len(buf)
	-- read 4 bytes little-endian starting at firstByte; bytes outside the
	-- region (before its start or past the buffer) read as zero
	local v = 0
	local m = 1
	for i = 0, 3 do
		local rel = firstByte + i
		local idx = base + rel
		if rel >= 0 and idx < bufLen then
			v += buffer.readu8(buf, idx) * m
		end
		m *= 256
	end
	return band(math.floor(v / lshift(1, shift)), lshift(1, n) - 1) % 4294967296
end

-- alias kept for call sites that may read past the start by design
local backBitsReadSafe = backBitsRead

-- ---------------------------------------------------------------------------
-- FSE table building (RFC 8878 4.1)
-- ---------------------------------------------------------------------------
type FSETable = {
	symbol: { number },
	numBits: { number },
	newState: { number },
	accuracyLog: number,
}

local function highBit(v: number): number
	local h = 0
	while v > 1 do
		h += 1
		v = rshift(v, 1)
	end
	return h
end

local function buildFSETable(normCounts: { number }, maxSymbol: number, accuracyLog: number): FSETable
	local tableSize = lshift(1, accuracyLog)
	local symbol = table.create(tableSize, 0)
	local numBits = table.create(tableSize, 0)
	local newState = table.create(tableSize, 0)

	local highThreshold = tableSize - 1
	-- next-state counter per symbol; starts at the symbol's probability
	local symbolNext: { number } = {}

	-- symbols with "less than 1" probability (-1) get one cell at the end
	for s = 0, maxSymbol do
		local c = normCounts[s + 1] or 0
		if c == -1 then
			symbol[highThreshold + 1] = s
			highThreshold -= 1
			symbolNext[s + 1] = 1
		else
			symbolNext[s + 1] = c
		end
	end

	-- spread symbols
	local step = rshift(tableSize, 1) + rshift(tableSize, 3) + 3
	local mask = tableSize - 1
	local pos = 0
	for s = 0, maxSymbol do
		local c = normCounts[s + 1]
		if c and c > 0 then
			for _ = 1, c do
				symbol[pos + 1] = s
				repeat
					pos = band(pos + step, mask)
				until pos <= highThreshold
			end
		end
	end

	-- per-cell decode info (FSE_buildDTable): nextState counters increment in
	-- table order; nbBits = tableLog - highbit(nextState);
	-- newState = (nextState << nbBits) - tableSize.
	for i = 0, tableSize - 1 do
		local s = symbol[i + 1]
		local nextState = symbolNext[s + 1]
		symbolNext[s + 1] = nextState + 1
		local nb = accuracyLog - highBit(nextState)
		numBits[i + 1] = nb
		newState[i + 1] = nextState * lshift(1, nb) - tableSize
	end

	return {
		symbol = symbol,
		numBits = numBits,
		newState = newState,
		accuracyLog = accuracyLog,
	}
end

-- Parse FSE normalized counts from a FORWARD bit stream (RFC 4.1.1).
type FwdBits = { buf: buffer, base: number, bitPos: number, len: number }

local function fwdRead(fb: FwdBits, n: number): number
	if n == 0 then
		return 0
	end
	local firstByte = math.floor(fb.bitPos / 8)
	local shift = fb.bitPos - firstByte * 8
	local v = 0
	local m = 1
	for i = 0, 3 do
		local idx = firstByte + i
		if idx < fb.len then
			v += buffer.readu8(fb.buf, fb.base + idx) * m
		end
		m *= 256
	end
	fb.bitPos += n
	return band(math.floor(v / lshift(1, shift)), lshift(1, n) - 1)
end

-- Port of FSE_readNCount: reads the normalized count distribution from a
-- forward bit stream, byte-aligned at the end.
local function parseFSEDistribution(fb: FwdBits, maxAllowedLog: number, maxSymbol: number): ({ number }?, number, number, string?)
	local accuracyLog = fwdRead(fb, 4) + 5
	if accuracyLog > maxAllowedLog then
		return nil, 0, 0, "FSE accuracy log too large"
	end
	local remaining = lshift(1, accuracyLog) + 1
	local threshold = lshift(1, accuracyLog)
	local nbBits = accuracyLog + 1
	local counts: { number } = {}
	local sym = 0
	local previous0 = false
	while remaining > 1 and sym <= maxSymbol do
		if previous0 then
			-- runs of zero-probability symbols: 2-bit repeat flags
			while true do
				local rep = fwdRead(fb, 2)
				for _ = 1, rep do
					counts[sym + 1] = 0
					sym += 1
				end
				if rep ~= 3 then
					break
				end
			end
			previous0 = false
			if sym > maxSymbol then
				break
			end
		end
		local maxV = 2 * threshold - 1 - remaining
		local count: number
		local low = fwdRead(fb, nbBits - 1)
		if low < maxV then
			count = low
		else
			local high = fwdRead(fb, 1)
			local value = low + high * lshift(1, nbBits - 1)
			if value >= threshold then
				count = value - maxV
			else
				count = value
			end
		end
		count -= 1 -- -1 means "less than 1" probability
		counts[sym + 1] = count
		sym += 1
		remaining -= if count < 0 then -count else count
		if count == 0 then
			previous0 = true
		end
		while remaining < threshold and threshold > 1 do
			nbBits -= 1
			threshold = rshift(threshold, 1)
		end
	end
	if remaining ~= 1 then
		return nil, 0, 0, "FSE distribution does not sum correctly"
	end
	-- align to byte
	fb.bitPos = math.floor((fb.bitPos + 7) / 8) * 8
	return counts, sym - 1, accuracyLog, nil
end

-- ---------------------------------------------------------------------------
-- Huffman literals
-- ---------------------------------------------------------------------------
type HuffTable = {
	-- fast lookup: maxBits-wide index -> symbol + length
	symbols: buffer,
	lengths: buffer,
	maxBits: number,
}

local function buildHuffTable(weights: { number }, numSymbols: number): (HuffTable?, string?)
	-- weights are 0..11; total = sum(2^(w-1) for w>0) must reach a power of 2
	local total = 0
	for i = 1, numSymbols do
		local w = weights[i]
		if w > 0 then
			total += lshift(1, w - 1)
		end
	end
	if total == 0 then
		return nil, "empty huffman table"
	end
	local maxBits = 0
	while lshift(1, maxBits) < total do
		maxBits += 1
	end
	if lshift(1, maxBits) ~= total then
		return nil, "huffman weights not a power of two"
	end
	local tableSize = lshift(1, maxBits)
	local symbols = buffer.create(tableSize)
	local lengths = buffer.create(tableSize)
	-- per RFC: symbols are sorted by weight then natural order; codes assigned
	-- from low weights upward
	local rankStart: { number } = table.create(maxBits + 2, 0)
	for i = 1, numSymbols do
		local w = weights[i]
		if w > 0 then
			rankStart[w + 1] += lshift(1, w - 1)
		end
	end
	-- prefix positions: weight 1 first (smallest codes... actually zstd assigns
	-- lower weights to lower code values); build by iterating weights ascending
	local pos = 0
	local startOfWeight: { number } = {}
	for w = 1, maxBits do
		startOfWeight[w] = pos
		-- count symbols of weight w
		for i = 1, numSymbols do
			if weights[i] == w then
				pos += lshift(1, w - 1)
			end
		end
	end
	for i = 1, numSymbols do
		local w = weights[i]
		if w > 0 then
			local len = maxBits + 1 - w
			local count = lshift(1, w - 1)
			local p = startOfWeight[w]
			for k = 0, count - 1 do
				buffer.writeu8(symbols, p + k, i - 1)
				buffer.writeu8(lengths, p + k, len)
			end
			startOfWeight[w] = p + count
		end
	end
	return { symbols = symbols, lengths = lengths, maxBits = maxBits }, nil
end

-- Decode huffman weights (RFC 4.2.1): either FSE-compressed or direct.
local function decodeHuffWeights(buf: buffer, base: number, avail: number): ({ number }?, number, number, string?)
	local headerByte = buffer.readu8(buf, base)
	if headerByte >= 128 then
		-- direct: (headerByte - 127) weights, 4 bits each
		local num = headerByte - 127
		local nbytes = math.floor((num + 1) / 2)
		if 1 + nbytes > avail then
			return nil, 0, 0, "huffman weights overflow"
		end
		local weights: { number } = {}
		for i = 0, num - 1 do
			local b = buffer.readu8(buf, base + 1 + rshift(i, 1))
			weights[i + 1] = if i % 2 == 0 then rshift(b, 4) else band(b, 15)
		end
		return weights, num, 1 + nbytes, nil
	else
		-- FSE-compressed weights: headerByte = compressed size
		local csize = headerByte
		if 1 + csize > avail then
			return nil, 0, 0, "huffman fse weights overflow"
		end
		local fb: FwdBits = { buf = buf, base = base + 1, bitPos = 0, len = csize }
		local counts, maxSym, accLog, err = parseFSEDistribution(fb, 6, 255)
		if not counts then
			return nil, 0, 0, err
		end
		local fse = buildFSETable(counts :: { number }, maxSym, accLog)
		local headerBytes = math.floor(fb.bitPos / 8)
		local bb, berr = backBitsInit(buf, base + 1 + headerBytes, csize - headerBytes)
		if not bb then
			return nil, 0, 0, berr
		end
		local bbs = bb :: BackBits
		-- two interleaved states; a symbol is emitted as soon as its state is
		-- formed, and decoding stops when the next state read would underflow
		-- the stream (fzstd-style, equivalent to the reference behaviour).
		local weights: { number } = {}
		local n = 0
		local btr1, btr2 = accLog, accLog
		local base1, base2 = 0, 0
		while n < 255 do
			if bbs.bitPos < btr1 then
				break
			end
			local s1 = base1 + backBitsRead(bbs, btr1)
			weights[n + 1] = fse.symbol[s1 + 1]
			n += 1
			if n >= 255 or bbs.bitPos < btr2 then
				break
			end
			local s2 = base2 + backBitsRead(bbs, btr2)
			weights[n + 1] = fse.symbol[s2 + 1]
			n += 1
			btr1 = fse.numBits[s1 + 1]
			base1 = fse.newState[s1 + 1]
			btr2 = fse.numBits[s2 + 1]
			base2 = fse.newState[s2 + 1]
		end
		return weights, n, 1 + csize, nil
	end
end

-- decode huffman-coded literal stream into out (forward writing)
local function huffDecodeStream(
	ht: HuffTable,
	buf: buffer,
	base: number,
	len: number,
	out: buffer,
	outPos: number,
	outCount: number
): string?
	local bb, err = backBitsInit(buf, base, len)
	if not bb then
		return err
	end
	local bbs = bb :: BackBits
	local maxBits = ht.maxBits
	local symbols = ht.symbols
	local lengths = ht.lengths
	for i = 1, outCount do
		local idx = backBitsReadSafe(bbs, maxBits)
		local sym = buffer.readu8(symbols, idx)
		local len2 = buffer.readu8(lengths, idx)
		-- return unused bits
		bbs.bitPos += maxBits - len2
		buffer.writeu8(out, outPos, sym)
		outPos += 1
	end
	return nil
end

-- ---------------------------------------------------------------------------
-- sequence tables (RFC 3.1.1.3.2)
-- ---------------------------------------------------------------------------
local LL_BASE = {
	0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
	16, 18, 20, 22, 24, 28, 32, 40, 48, 64, 128, 256, 512, 1024, 2048, 4096,
	8192, 16384, 32768, 65536,
}
local LL_BITS = {
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12,
	13, 14, 15, 16,
}
local ML_BASE = {
	3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
	19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
	35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027, 2051,
	4099, 8195, 16387, 32771, 65539,
}
local ML_BITS = {
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11,
	12, 13, 14, 15, 16,
}

local LL_DEFAULT = {
	4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1,
	2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
	-1, -1, -1, -1,
}
local ML_DEFAULT = {
	1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1,
	-1, -1, -1, -1, -1,
}
local OF_DEFAULT = {
	1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
}

-- ---------------------------------------------------------------------------
-- main frame decoding
-- ---------------------------------------------------------------------------
type DecState = {
	-- sticky entropy state across blocks
	huff: HuffTable?,
	llTable: FSETable?,
	mlTable: FSETable?,
	ofTable: FSETable?,
	rep1: number,
	rep2: number,
	rep3: number,
}

local function decodeSeqTable(
	r: { buf: buffer, pos: number, len: number },
	mode: number,
	maxSymbol: number,
	maxLog: number,
	defaultDist: { number },
	defaultLog: number,
	prev: FSETable?
): (FSETable?, string?)
	if mode == 0 then
		-- predefined
		return buildFSETable(defaultDist, #defaultDist - 1, defaultLog), nil
	elseif mode == 1 then
		-- RLE: single symbol byte
		local s = buffer.readu8(r.buf, r.pos)
		r.pos += 1
		return {
			symbol = { s },
			numBits = { 0 },
			newState = { 0 },
			accuracyLog = 0,
		}, nil
	elseif mode == 2 then
		-- FSE-compressed distribution
		local fb: FwdBits = { buf = r.buf, base = r.pos, bitPos = 0, len = r.len - r.pos }
		local counts, maxSym, accLog, err = parseFSEDistribution(fb, maxLog, maxSymbol)
		if not counts then
			return nil, err
		end
		r.pos += math.floor(fb.bitPos / 8)
		return buildFSETable(counts :: { number }, maxSym, accLog), nil
	else
		-- repeat
		if not prev then
			return nil, "repeat mode without previous table"
		end
		return prev, nil
	end
end

local function decompressFrame(src: buffer, pos: number, srcLen: number, out: { buffer }, st: DecState): (number?, string?)
	-- frame header
	local fhd = buffer.readu8(src, pos)
	pos += 1
	local dictIdFlag = band(fhd, 3)
	local contentChecksum = band(rshift(fhd, 2), 1)
	local singleSegment = band(rshift(fhd, 5), 1)
	local fcsFlag = rshift(fhd, 6)
	if singleSegment == 0 then
		pos += 1 -- window descriptor (we buffer the whole output anyway)
	end
	if dictIdFlag == 1 then
		pos += 1
	elseif dictIdFlag == 2 then
		pos += 2
	elseif dictIdFlag == 3 then
		pos += 4
	end
	local contentSize = -1
	if fcsFlag == 0 then
		if singleSegment == 1 then
			contentSize = buffer.readu8(src, pos)
			pos += 1
		end
	elseif fcsFlag == 1 then
		contentSize = buffer.readu16(src, pos) + 256
		pos += 2
	elseif fcsFlag == 2 then
		contentSize = buffer.readu32(src, pos)
		pos += 4
	else
		local lo = buffer.readu32(src, pos)
		local hi = buffer.readu32(src, pos + 4)
		contentSize = hi * 4294967296 + lo
		pos += 8
	end

	-- output accumulator: chunked buffers, with a flat window for matches
	-- (we just allocate the full content if known, else grow)
	local cap = if contentSize >= 0 then contentSize else 65536
	local dst = buffer.create(math.max(cap, 16))
	local dstLen = 0
	local function ensure(extra: number)
		local need = dstLen + extra
		if need > buffer.len(dst) then
			local newCap = math.max(need, buffer.len(dst) * 2)
			local nb = buffer.create(newCap)
			buffer.copy(nb, 0, dst, 0, dstLen)
			dst = nb
		end
	end

	-- literals scratch
	local litBuf = buffer.create(65536)
	local litLen = 0

	while true do
		local bh = buffer.readu8(src, pos)
			+ buffer.readu8(src, pos + 1) * 256
			+ buffer.readu8(src, pos + 2) * 65536
		pos += 3
		local lastBlock = band(bh, 1)
		local blockType = band(rshift(bh, 1), 3)
		local blockSize = rshift(bh, 3)

		if blockType == 0 then
			-- raw
			ensure(blockSize)
			buffer.copy(dst, dstLen, src, pos, blockSize)
			dstLen += blockSize
			pos += blockSize
		elseif blockType == 1 then
			-- RLE
			ensure(blockSize)
			buffer.fill(dst, dstLen, buffer.readu8(src, pos), blockSize)
			dstLen += blockSize
			pos += 1
		elseif blockType == 2 then
			-- compressed block
			local blockEnd = pos + blockSize
			-- ---------------- literals section ----------------
			local lsh = buffer.readu8(src, pos)
			local litType = band(lsh, 3)
			local sizeFormat = band(rshift(lsh, 2), 3)
			local regenSize = 0
			local compSize = 0
			local fourStreams = false
			if litType == 0 or litType == 1 then
				-- raw / RLE literals
				if sizeFormat == 0 or sizeFormat == 2 then
					regenSize = rshift(lsh, 3)
					pos += 1
				elseif sizeFormat == 1 then
					regenSize = rshift(lsh, 4) + buffer.readu8(src, pos + 1) * 16
					pos += 2
				else
					regenSize = rshift(lsh, 4)
						+ buffer.readu8(src, pos + 1) * 16
						+ buffer.readu8(src, pos + 2) * 4096
					pos += 3
				end
				if regenSize > buffer.len(litBuf) then
					litBuf = buffer.create(regenSize)
				end
				if litType == 0 then
					buffer.copy(litBuf, 0, src, pos, regenSize)
					pos += regenSize
				else
					buffer.fill(litBuf, 0, buffer.readu8(src, pos), regenSize)
					pos += 1
				end
				litLen = regenSize
			else
				-- compressed / treeless literals
				if sizeFormat == 0 then
					fourStreams = false
					regenSize = rshift(lsh, 4) + band(buffer.readu8(src, pos + 1), 63) * 16
					compSize = rshift(buffer.readu8(src, pos + 1), 6)
						+ buffer.readu8(src, pos + 2) * 4
					pos += 3
				elseif sizeFormat == 1 then
					fourStreams = true
					regenSize = rshift(lsh, 4) + band(buffer.readu8(src, pos + 1), 63) * 16
					compSize = rshift(buffer.readu8(src, pos + 1), 6)
						+ buffer.readu8(src, pos + 2) * 4
					pos += 3
				elseif sizeFormat == 2 then
					fourStreams = true
					regenSize = rshift(lsh, 4)
						+ buffer.readu8(src, pos + 1) * 16
						+ band(buffer.readu8(src, pos + 2), 3) * 4096
					compSize = rshift(buffer.readu8(src, pos + 2), 2)
						+ buffer.readu8(src, pos + 3) * 64
					pos += 4
				else
					fourStreams = true
					regenSize = rshift(lsh, 4)
						+ buffer.readu8(src, pos + 1) * 16
						+ band(buffer.readu8(src, pos + 2), 63) * 4096
					compSize = rshift(buffer.readu8(src, pos + 2), 6)
						+ buffer.readu8(src, pos + 3) * 4
						+ buffer.readu8(src, pos + 4) * 1024
					pos += 5
				end
				local litStart = pos
				local origCompSize = compSize
				if litType == 2 then
					-- new huffman table; the last symbol's weight is implicit
					-- (HUF_readStats): tableLog = highbit(total) + 1, and the
					-- remainder up to 1<<tableLog must be a power of two.
					local weights, numW, consumed, werr = decodeHuffWeights(src, pos, compSize)
					if not weights then
						return nil, werr
					end
					local ws = weights :: { number }
					local total = 0
					for i = 1, numW do
						if ws[i] > 0 then
							total += lshift(1, ws[i] - 1)
						end
					end
					if total == 0 then
						return nil, "huffman: zero weight total"
					end
					local tableLog = highBit(total) + 1
					local rest = lshift(1, tableLog) - total
					local lw = highBit(rest) + 1
					if lshift(1, lw - 1) ~= rest then
						return nil, "huffman: invalid implicit last weight"
					end
					ws[numW + 1] = lw
					local ht, herr = buildHuffTable(ws, numW + 1)
					if not ht then
						return nil, herr
					end
					st.huff = ht
					pos += consumed
					compSize -= consumed
				end
				if not st.huff then
					return nil, "treeless literals without huffman table"
				end
				if regenSize > buffer.len(litBuf) then
					litBuf = buffer.create(regenSize)
				end
				local ht = st.huff :: HuffTable
				if not fourStreams then
					local herr = huffDecodeStream(ht, src, pos, compSize, litBuf, 0, regenSize)
					if herr then
						return nil, herr
					end
				else
					-- jump table: 3 u16 sizes
					local s1 = buffer.readu16(src, pos)
					local s2 = buffer.readu16(src, pos + 2)
					local s3 = buffer.readu16(src, pos + 4)
					local dataStart = pos + 6
					local totalData = compSize - 6
					local s4 = totalData - s1 - s2 - s3
					if s4 < 0 then
						return nil, "bad literal jump table"
					end
					local part = math.floor((regenSize + 3) / 4)
					local sizes = { s1, s2, s3, s4 }
					local outOff = 0
					local off = dataStart
					for s = 1, 4 do
						local count = if s == 4 then regenSize - part * 3 else part
						local herr = huffDecodeStream(ht, src, off, sizes[s], litBuf, outOff, count)
						if herr then
							return nil, herr
						end
						off += sizes[s]
						outOff += count
					end
				end
				litLen = regenSize
				pos = litStart + origCompSize
			end

			-- ---------------- sequences section ----------------
			local nbSeq = 0
			local b0 = buffer.readu8(src, pos)
			pos += 1
			if b0 < 128 then
				nbSeq = b0
			elseif b0 < 255 then
				nbSeq = (b0 - 128) * 256 + buffer.readu8(src, pos)
				pos += 1
			else
				nbSeq = buffer.readu16(src, pos) + 32512
				pos += 2
			end

			if nbSeq == 0 then
				-- literals only
				ensure(litLen)
				buffer.copy(dst, dstLen, litBuf, 0, litLen)
				dstLen += litLen
				if litLen ~= 0 then
					litLen = 0
				end
			else
				local modesByte = buffer.readu8(src, pos)
				pos += 1
				local llMode = rshift(modesByte, 6)
				local ofMode = band(rshift(modesByte, 4), 3)
				local mlMode = band(rshift(modesByte, 2), 3)
				local rr = { buf = src, pos = pos, len = srcLen }
				local llT, e1 = decodeSeqTable(rr, llMode, 35, 9, LL_DEFAULT, 6, st.llTable)
				if not llT then
					return nil, e1
				end
				local ofT, e2 = decodeSeqTable(rr, ofMode, 31, 8, OF_DEFAULT, 5, st.ofTable)
				if not ofT then
					return nil, e2
				end
				local mlT, e3 = decodeSeqTable(rr, mlMode, 52, 9, ML_DEFAULT, 6, st.mlTable)
				if not mlT then
					return nil, e3
				end
				st.llTable = llT
				st.ofTable = ofT
				st.mlTable = mlT
				pos = rr.pos

				local bb, berr = backBitsInit(src, pos, blockEnd - pos)
				if not bb then
					return nil, berr
				end
				local bbs = bb :: BackBits
				local llS = backBitsRead(bbs, (llT :: FSETable).accuracyLog)
				local ofS = backBitsRead(bbs, (ofT :: FSETable).accuracyLog)
				local mlS = backBitsRead(bbs, (mlT :: FSETable).accuracyLog)

				local llTab = llT :: FSETable
				local ofTab = ofT :: FSETable
				local mlTab = mlT :: FSETable

				local litPos = 0
				for seqIdx = 1, nbSeq do
					local llCode = llTab.symbol[llS + 1]
					local ofCode = ofTab.symbol[ofS + 1]
					local mlCode = mlTab.symbol[mlS + 1]

					-- read extra bits: offset, then match, then literal
					local ofValue: number
					if ofCode == 0 then
						ofValue = 1
					elseif ofCode <= 25 then
						ofValue = lshift(1, ofCode) + backBitsReadSafe(bbs, ofCode)
					else
						-- wide offsets: split the read (the reader window is 32 bits)
						local hi = backBitsReadSafe(bbs, ofCode - 16)
						local lo = backBitsReadSafe(bbs, 16)
						ofValue = 2 ^ ofCode + hi * 65536 + lo
					end
					local ml = ML_BASE[mlCode + 1] + backBitsReadSafe(bbs, ML_BITS[mlCode + 1])
					local ll = LL_BASE[llCode + 1] + backBitsReadSafe(bbs, LL_BITS[llCode + 1])

					-- resolve offset with repeat history
					local offset: number
					if ofCode == 0 then
						-- ofValue == 1
						offset = if ll == 0 then st.rep2 else st.rep1
						if ll == 0 then
							st.rep2 = st.rep1
							st.rep1 = offset
						end
					else
						local ov = ofValue
						if ov > 3 then
							offset = ov - 3
							st.rep3 = st.rep2
							st.rep2 = st.rep1
							st.rep1 = offset
						else
							if ll == 0 then
								ov += 1
							end
							if ov == 1 then
								offset = st.rep1
							elseif ov == 2 then
								offset = st.rep2
								st.rep2 = st.rep1
								st.rep1 = offset
							elseif ov == 3 then
								offset = st.rep3
								st.rep3 = st.rep2
								st.rep2 = st.rep1
								st.rep1 = offset
							else
								offset = st.rep1 - 1
								if offset == 0 then
									return nil, "zero offset"
								end
								st.rep3 = st.rep2
								st.rep2 = st.rep1
								st.rep1 = offset
							end
						end
					end

					-- copy literals
					if ll > 0 then
						ensure(ll)
						buffer.copy(dst, dstLen, litBuf, litPos, ll)
						dstLen += ll
						litPos += ll
					end
					-- copy match
					if ml > 0 then
						if offset > dstLen then
							return nil, "match offset beyond output"
						end
						ensure(ml)
						local from = dstLen - offset
						if offset >= ml then
							buffer.copy(dst, dstLen, dst, from, ml)
							dstLen += ml
						else
							-- overlapping copy
							for k = 0, ml - 1 do
								buffer.writeu8(dst, dstLen, buffer.readu8(dst, from + k))
								dstLen += 1
							end
						end
					end

					-- update states (unless last sequence)
					if seqIdx < nbSeq then
						llS = llTab.newState[llS + 1] + backBitsReadSafe(bbs, llTab.numBits[llS + 1])
						mlS = mlTab.newState[mlS + 1] + backBitsReadSafe(bbs, mlTab.numBits[mlS + 1])
						ofS = ofTab.newState[ofS + 1] + backBitsReadSafe(bbs, ofTab.numBits[ofS + 1])
					end
				end
				-- trailing literals
				local rest = litLen - litPos
				if rest > 0 then
					ensure(rest)
					buffer.copy(dst, dstLen, litBuf, litPos, rest)
					dstLen += rest
				end
			end
			pos = blockEnd
		else
			return nil, "reserved block type"
		end

		if lastBlock == 1 then
			break
		end
	end

	if contentChecksum == 1 then
		pos += 4
	end

	-- trim
	local final = buffer.create(dstLen)
	buffer.copy(final, 0, dst, 0, dstLen)
	table.insert(out, final)
	return pos, nil
end

local function decompress(src: buffer): (buffer?, string?)
	local ok, result, err = pcall(function(): (buffer?, string?)
		local srcLen = buffer.len(src)
		local pos = 0
		local parts: { buffer } = {}
		while pos < srcLen do
			local magic = buffer.readu32(src, pos)
			pos += 4
			if magic == 0xFD2FB528 then
				local st: DecState = {
					huff = nil,
					llTable = nil,
					mlTable = nil,
					ofTable = nil,
					rep1 = 1,
					rep2 = 4,
					rep3 = 8,
				}
				local np, ferr = decompressFrame(src, pos, srcLen, parts, st)
				if not np then
					return nil, ferr
				end
				pos = np :: number
			elseif magic >= 0x184D2A50 and magic <= 0x184D2A5F then
				-- skippable frame
				local size = buffer.readu32(src, pos)
				pos += 4 + size
			else
				return nil, string.format("bad zstd magic 0x%08X", magic)
			end
		end
		if #parts == 1 then
			return parts[1], nil
		end
		local total = 0
		for _, p in ipairs(parts) do
			total += buffer.len(p)
		end
		local outb = buffer.create(total)
		local o = 0
		for _, p in ipairs(parts) do
			buffer.copy(outb, o, p, 0, buffer.len(p))
			o += buffer.len(p)
		end
		return outb, nil
	end)
	if not ok then
		return nil, "zstd decode error: " .. tostring(result)
	end
	return result, err
end

return {
	decompress = decompress,
}
