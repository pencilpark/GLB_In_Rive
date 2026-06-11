--!strict
--[[
  TextureOps -- CPU operations on RGBA8 images stored in Luau buffers.
  Companion to the texture decoders (KTX2Decoder, WebPDecoder): builds mip
  chains with a 2x2 box filter and blurred chains for equirectangular
  environment maps (a progressive approximation of roughness lobes).

  Pixel layout: tightly packed RGBA8, row major, offset = (y*width + x) * 4.
  Alpha is filtered exactly like the colour channels; values are NOT
  premultiplied before averaging, so colour can bleed from fully
  transparent texels. Premultiply upstream if that matters for the asset.

  Rounding: the box filter computes floor((p00+p01+p10+p11 + 2) / 4) per
  channel (half rounds up). The [1,2,1]/4 blur rounds half up on one axis
  and half down on the other (odd passes: horizontal up, vertical down;
  even passes: swapped), so blur passes stay mean neutral even on
  degenerate Nx1 / 1xN levels where one axis is an identity.

  Level 0 of a returned chain references the caller's buffer (no copy).
  Invalid dimensions raise an error.

  buildMipChain(rgba, w, h)                 -> { MipLevel }
  prefilterEnvChain(rgba, w, h, maxLevels?) -> { MipLevel }
  averageColor(rgba, w, h)                  -> (r, g, b, a)  -- floats 0..255
]]

export type MipLevel = {
	data: buffer,
	width: number,
	height: number,
}

local TextureOps = {}

local function checkImage(rgba: buffer, width: number, height: number)
	assert(
		width >= 1 and height >= 1 and width % 1 == 0 and height % 1 == 0,
		"TextureOps: width and height must be positive integers"
	)
	assert(
		buffer.len(rgba) >= width * height * 4,
		"TextureOps: buffer too small for width*height RGBA8 texels"
	)
end

-- ---------------------------------------------------------------------------
-- kernels. Hot loops read whole texels with readu32 and split channels with
-- plain // and % arithmetic: channel sums stay far below 2^32 and there are
-- no per-pixel closures or library calls besides the buffer fastcalls.
-- ---------------------------------------------------------------------------

-- 2x2 box reduction of src (sw x sh) into dst (dw x dh) where
-- dw = max(1, floor(sw/2)), dh = max(1, floor(sh/2)). Source samples at
-- 2x+1 / 2y+1 clamp to the last texel (only reachable when sw or sh is 1;
-- for odd sizes >= 3 the trailing row/column is dropped, floor convention).
local function boxHalve(src: buffer, sw: number, sh: number, dst: buffer, dw: number, dh: number)
	local srcStride = sw * 4
	local lastXOff = (sw - 1) * 4
	local lastY = sh - 1
	local di = 0
	for y = 0, dh - 1 do
		local sy = 2 * y
		local sy1 = sy + 1
		if sy1 > lastY then
			sy1 = lastY
		end
		local row0 = sy * srcStride
		local row1 = sy1 * srcStride
		for x = 0, dw - 1 do
			local b0 = 8 * x -- (2x) * 4 bytes
			local b1 = b0 + 4
			if b1 > lastXOff then
				b1 = lastXOff
			end
			local v00 = buffer.readu32(src, row0 + b0)
			local v01 = buffer.readu32(src, row0 + b1)
			local v10 = buffer.readu32(src, row1 + b0)
			local v11 = buffer.readu32(src, row1 + b1)
			local r = ((v00 % 256) + (v01 % 256) + (v10 % 256) + (v11 % 256) + 2) // 4
			local g = ((v00 // 256 % 256) + (v01 // 256 % 256) + (v10 // 256 % 256) + (v11 // 256 % 256) + 2) // 4
			local b = ((v00 // 65536 % 256) + (v01 // 65536 % 256) + (v10 // 65536 % 256) + (v11 // 65536 % 256) + 2) // 4
			local a = ((v00 // 16777216) + (v01 // 16777216) + (v10 // 16777216) + (v11 // 16777216) + 2) // 4
			buffer.writeu32(dst, di, r + g * 256 + b * 65536 + a * 16777216)
			di += 4
		end
	end
end

-- Horizontal [1,2,1]/4 pass. X wraps: equirectangular maps are continuous
-- across the left/right seam. rnd is the rounding offset, floor((sum+rnd)/4):
-- 2 rounds half up, 1 rounds half down.
local function blurRowsWrapX(src: buffer, dst: buffer, w: number, h: number, rnd: number)
	if w == 1 then
		buffer.copy(dst, 0, src, 0, h * 4) -- (4p + rnd) // 4 == p for rnd in 1..3
		return
	end
	local stride = w * 4
	for y = 0, h - 1 do
		local rowBase = y * stride
		local lastOff = rowBase + stride - 4
		local vl = buffer.readu32(src, lastOff) -- left neighbour of x = 0 wraps
		local vc = buffer.readu32(src, rowBase)
		for off = rowBase, lastOff, 4 do
			local vr = if off == lastOff
				then buffer.readu32(src, rowBase) -- right neighbour of x = w-1 wraps
				else buffer.readu32(src, off + 4)
			local r = ((vl % 256) + 2 * (vc % 256) + (vr % 256) + rnd) // 4
			local g = ((vl // 256 % 256) + 2 * (vc // 256 % 256) + (vr // 256 % 256) + rnd) // 4
			local b = ((vl // 65536 % 256) + 2 * (vc // 65536 % 256) + (vr // 65536 % 256) + rnd) // 4
			local a = ((vl // 16777216) + 2 * (vc // 16777216) + (vr // 16777216) + rnd) // 4
			buffer.writeu32(dst, off, r + g * 256 + b * 65536 + a * 16777216)
			vl = vc
			vc = vr
		end
	end
end

-- Vertical [1,2,1]/4 pass. Y clamps: the poles of an equirectangular map
-- are not continuous top to bottom. rnd as in blurRowsWrapX.
local function blurColsClampY(src: buffer, dst: buffer, w: number, h: number, rnd: number)
	if h == 1 then
		buffer.copy(dst, 0, src, 0, w * 4) -- (4p + rnd) // 4 == p for rnd in 1..3
		return
	end
	local stride = w * 4
	for y = 0, h - 1 do
		local rowC = y * stride
		local rowU = if y == 0 then 0 else rowC - stride
		local rowD = if y == h - 1 then rowC else rowC + stride
		for xb = 0, stride - 4, 4 do
			local vu = buffer.readu32(src, rowU + xb)
			local vc = buffer.readu32(src, rowC + xb)
			local vd = buffer.readu32(src, rowD + xb)
			local r = ((vu % 256) + 2 * (vc % 256) + (vd % 256) + rnd) // 4
			local g = ((vu // 256 % 256) + 2 * (vc // 256 % 256) + (vd // 256 % 256) + rnd) // 4
			local b = ((vu // 65536 % 256) + 2 * (vc // 65536 % 256) + (vd // 65536 % 256) + rnd) // 4
			local a = ((vu // 16777216) + 2 * (vc // 16777216) + (vd // 16777216) + rnd) // 4
			buffer.writeu32(dst, rowC + xb, r + g * 256 + b * 65536 + a * 16777216)
		end
	end
end

-- ---------------------------------------------------------------------------
-- public API
-- ---------------------------------------------------------------------------

-- Full mip chain: level 0 is the input as-is, then each level halves both
-- dimensions (floor, min 1) with the 2x2 box filter until 1x1 inclusive.
function TextureOps.buildMipChain(rgba: buffer, width: number, height: number): { MipLevel }
	checkImage(rgba, width, height)
	local levels: { MipLevel } = { { data = rgba, width = width, height = height } }
	local src, w, h = rgba, width, height
	while w > 1 or h > 1 do
		local dw = if w > 1 then w // 2 else 1
		local dh = if h > 1 then h // 2 else 1
		local dst = buffer.create(dw * dh * 4)
		boxHalve(src, w, h, dst, dw, dh)
		table.insert(levels, { data = dst, width = dw, height = dh })
		src, w, h = dst, dw, dh
	end
	return levels
end

-- Prefiltered chain for an equirectangular environment map. Level 0 is the
-- input as-is; level n is the box-halved previous level followed by
-- min(n, 4) separable [1,2,1]/4 blur passes (X wraps, Y clamps), a cheap
-- progressive stand-in for increasing roughness lobes. maxLevels caps the
-- total chain length including level 0 (default: down to 1x1).
function TextureOps.prefilterEnvChain(rgba: buffer, width: number, height: number, maxLevels: number?): { MipLevel }
	checkImage(rgba, width, height)
	assert(maxLevels == nil or maxLevels >= 1, "TextureOps.prefilterEnvChain: maxLevels must be >= 1")
	local cap = maxLevels or math.huge
	local levels: { MipLevel } = { { data = rgba, width = width, height = height } }
	local src, w, h = rgba, width, height
	while (w > 1 or h > 1) and #levels < cap do
		local dw = if w > 1 then w // 2 else 1
		local dh = if h > 1 then h // 2 else 1
		local dst = buffer.create(dw * dh * 4)
		boxHalve(src, w, h, dst, dw, dh)
		local passes = math.clamp(#levels, 1, 4) -- #levels == index of the new level
		local tmp = buffer.create(dw * dh * 4)
		for pass = 1, passes do
			-- alternate which axis rounds half up so quantisation bias
			-- cancels across passes (see header)
			if pass % 2 == 1 then
				blurRowsWrapX(dst, tmp, dw, dh, 2)
				blurColsClampY(tmp, dst, dw, dh, 1)
			else
				blurRowsWrapX(dst, tmp, dw, dh, 1)
				blurColsClampY(tmp, dst, dw, dh, 2)
			end
		end
		table.insert(levels, { data = dst, width = dw, height = dh })
		src, w, h = dst, dw, dh
	end
	return levels
end

-- Mean of every channel over the whole image, as floats in 0..255
-- (exact integer sums divided by the texel count).
function TextureOps.averageColor(rgba: buffer, width: number, height: number): (number, number, number, number)
	checkImage(rgba, width, height)
	local n = width * height
	local sumR, sumG, sumB, sumA = 0, 0, 0, 0
	for off = 0, n * 4 - 4, 4 do
		local v = buffer.readu32(rgba, off)
		sumR += v % 256
		sumG += v // 256 % 256
		sumB += v // 65536 % 256
		sumA += v // 16777216
	end
	return sumR / n, sumG / n, sumB / n, sumA / n
end

return TextureOps
