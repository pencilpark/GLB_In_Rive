--!strict
--[[
  WebPDecoder -- pure-Luau WebP decoder for EXT_texture_webp.
  Ports the libwebp decoding paths (BSD-3) needed for textures:
   - VP8L (lossless): huffman/meta-huffman, LZ77 + color cache, all four
     transforms (predictor, cross-color, subtract-green, color-indexing
     with pixel bundling)
   - VP8 (lossy keyframes): boolean decoder, segment/filter/quant headers,
     intra modes, token/coefficient decoding, IDCT/WHT, all intra
     predictors, normal + simple loop filters, fancy chroma upsampling
     and the exact libwebp YUV->RGB conversion
   - VP8X container with ALPH chunk (raw or lossless alpha, all filters)

  decode(bytes) -> ({ width, height, rgba: buffer }?, string?)
]]

local band = bit32.band
local bor = bit32.bor
local bxor = bit32.bxor
local rshift = bit32.rshift
local lshift = bit32.lshift

export type DecodedImage = {
	width: number,
	height: number,
	rgba: buffer,
}

local COEFFS_PROBA0 = {
	128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
	128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 253, 136, 254, 255, 228, 219, 128, 128, 128, 128, 128,
	189, 129, 242, 255, 227, 213, 255, 219, 128, 128, 128, 106, 126, 227, 252, 214, 209, 255, 255, 128, 128, 128,
	1, 98, 248, 255, 236, 226, 255, 255, 128, 128, 128, 181, 133, 238, 254, 221, 234, 255, 154, 128, 128, 128,
	78, 134, 202, 247, 198, 180, 255, 219, 128, 128, 128, 1, 185, 249, 255, 243, 255, 128, 128, 128, 128, 128,
	184, 150, 247, 255, 236, 224, 128, 128, 128, 128, 128, 77, 110, 216, 255, 236, 230, 128, 128, 128, 128, 128,
	1, 101, 251, 255, 241, 255, 128, 128, 128, 128, 128, 170, 139, 241, 252, 236, 209, 255, 255, 128, 128, 128,
	37, 116, 196, 243, 228, 255, 255, 255, 128, 128, 128, 1, 204, 254, 255, 245, 255, 128, 128, 128, 128, 128,
	207, 160, 250, 255, 238, 128, 128, 128, 128, 128, 128, 102, 103, 231, 255, 211, 171, 128, 128, 128, 128, 128,
	1, 152, 252, 255, 240, 255, 128, 128, 128, 128, 128, 177, 135, 243, 255, 234, 225, 128, 128, 128, 128, 128,
	80, 129, 211, 255, 194, 224, 128, 128, 128, 128, 128, 1, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128,
	246, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128, 255, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
	198, 35, 237, 223, 193, 187, 162, 160, 145, 155, 62, 131, 45, 198, 221, 172, 176, 220, 157, 252, 221, 1,
	68, 47, 146, 208, 149, 167, 221, 162, 255, 223, 128, 1, 149, 241, 255, 221, 224, 255, 255, 128, 128, 128,
	184, 141, 234, 253, 222, 220, 255, 199, 128, 128, 128, 81, 99, 181, 242, 176, 190, 249, 202, 255, 255, 128,
	1, 129, 232, 253, 214, 197, 242, 196, 255, 255, 128, 99, 121, 210, 250, 201, 198, 255, 202, 128, 128, 128,
	23, 91, 163, 242, 170, 187, 247, 210, 255, 255, 128, 1, 200, 246, 255, 234, 255, 128, 128, 128, 128, 128,
	109, 178, 241, 255, 231, 245, 255, 255, 128, 128, 128, 44, 130, 201, 253, 205, 192, 255, 255, 128, 128, 128,
	1, 132, 239, 251, 219, 209, 255, 165, 128, 128, 128, 94, 136, 225, 251, 218, 190, 255, 255, 128, 128, 128,
	22, 100, 174, 245, 186, 161, 255, 199, 128, 128, 128, 1, 182, 249, 255, 232, 235, 128, 128, 128, 128, 128,
	124, 143, 241, 255, 227, 234, 128, 128, 128, 128, 128, 35, 77, 181, 251, 193, 211, 255, 205, 128, 128, 128,
	1, 157, 247, 255, 236, 231, 255, 255, 128, 128, 128, 121, 141, 235, 255, 225, 227, 255, 255, 128, 128, 128,
	45, 99, 188, 251, 195, 217, 255, 224, 128, 128, 128, 1, 1, 251, 255, 213, 255, 128, 128, 128, 128, 128,
	203, 1, 248, 255, 255, 128, 128, 128, 128, 128, 128, 137, 1, 177, 255, 224, 255, 128, 128, 128, 128, 128,
	253, 9, 248, 251, 207, 208, 255, 192, 128, 128, 128, 175, 13, 224, 243, 193, 185, 249, 198, 255, 255, 128,
	73, 17, 171, 221, 161, 179, 236, 167, 255, 234, 128, 1, 95, 247, 253, 212, 183, 255, 255, 128, 128, 128,
	239, 90, 244, 250, 211, 209, 255, 255, 128, 128, 128, 155, 77, 195, 248, 188, 195, 255, 255, 128, 128, 128,
	1, 24, 239, 251, 218, 219, 255, 205, 128, 128, 128, 201, 51, 219, 255, 196, 186, 128, 128, 128, 128, 128,
	69, 46, 190, 239, 201, 218, 255, 228, 128, 128, 128, 1, 191, 251, 255, 255, 128, 128, 128, 128, 128, 128,
	223, 165, 249, 255, 213, 255, 128, 128, 128, 128, 128, 141, 124, 248, 255, 255, 128, 128, 128, 128, 128, 128,
	1, 16, 248, 255, 255, 128, 128, 128, 128, 128, 128, 190, 36, 230, 255, 236, 255, 128, 128, 128, 128, 128,
	149, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128, 1, 226, 255, 128, 128, 128, 128, 128, 128, 128, 128,
	247, 192, 255, 128, 128, 128, 128, 128, 128, 128, 128, 240, 128, 255, 128, 128, 128, 128, 128, 128, 128, 128,
	1, 134, 252, 255, 255, 128, 128, 128, 128, 128, 128, 213, 62, 250, 255, 255, 128, 128, 128, 128, 128, 128,
	55, 93, 255, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
	128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
	202, 24, 213, 235, 186, 191, 220, 160, 240, 175, 255, 126, 38, 182, 232, 169, 184, 228, 174, 255, 187, 128,
	61, 46, 138, 219, 151, 178, 240, 170, 255, 216, 128, 1, 112, 230, 250, 199, 191, 247, 159, 255, 255, 128,
	166, 109, 228, 252, 211, 215, 255, 174, 128, 128, 128, 39, 77, 162, 232, 172, 180, 245, 178, 255, 255, 128,
	1, 52, 220, 246, 198, 199, 249, 220, 255, 255, 128, 124, 74, 191, 243, 183, 193, 250, 221, 255, 255, 128,
	24, 71, 130, 219, 154, 170, 243, 182, 255, 255, 128, 1, 182, 225, 249, 219, 240, 255, 224, 128, 128, 128,
	149, 150, 226, 252, 216, 205, 255, 171, 128, 128, 128, 28, 108, 170, 242, 183, 194, 254, 223, 255, 255, 128,
	1, 81, 230, 252, 204, 203, 255, 192, 128, 128, 128, 123, 102, 209, 247, 188, 196, 255, 233, 128, 128, 128,
	20, 95, 153, 243, 164, 173, 255, 203, 128, 128, 128, 1, 222, 248, 255, 216, 213, 128, 128, 128, 128, 128,
	168, 175, 246, 252, 235, 205, 255, 255, 128, 128, 128, 47, 116, 215, 255, 211, 212, 255, 255, 128, 128, 128,
	1, 121, 236, 253, 212, 214, 255, 255, 128, 128, 128, 141, 84, 213, 252, 201, 202, 255, 219, 128, 128, 128,
	42, 80, 160, 240, 162, 185, 255, 205, 128, 128, 128, 1, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128,
	244, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128, 238, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128,
}

local COEFFS_UPDATE_PROBA = {
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 176, 246, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	223, 241, 252, 255, 255, 255, 255, 255, 255, 255, 255, 249, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 244, 252, 255, 255, 255, 255, 255, 255, 255, 255, 234, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255,
	253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 246, 254, 255, 255, 255, 255, 255, 255, 255, 255,
	239, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255, 254, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 248, 254, 255, 255, 255, 255, 255, 255, 255, 255, 251, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255,
	251, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 254, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 254, 253, 255, 254, 255, 255, 255, 255, 255, 255, 250, 255, 254, 255, 254, 255, 255, 255, 255, 255, 255,
	254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	217, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 225, 252, 241, 253, 255, 255, 254, 255, 255, 255, 255,
	234, 250, 241, 250, 253, 255, 253, 254, 255, 255, 255, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	223, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 238, 253, 254, 254, 255, 255, 255, 255, 255, 255, 255,
	255, 248, 254, 255, 255, 255, 255, 255, 255, 255, 255, 249, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 253, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	247, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255, 252, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255,
	253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 254, 253, 255, 255, 255, 255, 255, 255, 255, 255, 250, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	186, 251, 250, 255, 255, 255, 255, 255, 255, 255, 255, 234, 251, 244, 254, 255, 255, 255, 255, 255, 255, 255,
	251, 251, 243, 253, 254, 255, 254, 255, 255, 255, 255, 255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255,
	236, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255, 251, 253, 253, 254, 254, 255, 255, 255, 255, 255, 255,
	255, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	248, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 250, 254, 252, 254, 255, 255, 255, 255, 255, 255, 255,
	248, 254, 249, 253, 255, 255, 255, 255, 255, 255, 255, 255, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255,
	246, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255, 252, 254, 251, 254, 254, 255, 255, 255, 255, 255, 255,
	255, 254, 252, 255, 255, 255, 255, 255, 255, 255, 255, 248, 254, 253, 255, 255, 255, 255, 255, 255, 255, 255,
	253, 255, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 251, 254, 255, 255, 255, 255, 255, 255, 255, 255,
	245, 251, 254, 255, 255, 255, 255, 255, 255, 255, 255, 253, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 251, 253, 255, 255, 255, 255, 255, 255, 255, 255, 252, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 252, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	249, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 253, 255, 255, 255, 255, 255, 255, 255, 255, 250, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
}

local BMODES_PROBA = {
	231, 120, 48, 89, 115, 113, 120, 152, 112, 152, 179, 64, 126, 170, 118, 46, 70, 95,
	175, 69, 143, 80, 85, 82, 72, 155, 103, 56, 58, 10, 171, 218, 189, 17, 13, 152,
	114, 26, 17, 163, 44, 195, 21, 10, 173, 121, 24, 80, 195, 26, 62, 44, 64, 85,
	144, 71, 10, 38, 171, 213, 144, 34, 26, 170, 46, 55, 19, 136, 160, 33, 206, 71,
	63, 20, 8, 114, 114, 208, 12, 9, 226, 81, 40, 11, 96, 182, 84, 29, 16, 36,
	134, 183, 89, 137, 98, 101, 106, 165, 148, 72, 187, 100, 130, 157, 111, 32, 75, 80,
	66, 102, 167, 99, 74, 62, 40, 234, 128, 41, 53, 9, 178, 241, 141, 26, 8, 107,
	74, 43, 26, 146, 73, 166, 49, 23, 157, 65, 38, 105, 160, 51, 52, 31, 115, 128,
	104, 79, 12, 27, 217, 255, 87, 17, 7, 87, 68, 71, 44, 114, 51, 15, 186, 23,
	47, 41, 14, 110, 182, 183, 21, 17, 194, 66, 45, 25, 102, 197, 189, 23, 18, 22,
	88, 88, 147, 150, 42, 46, 45, 196, 205, 43, 97, 183, 117, 85, 38, 35, 179, 61,
	39, 53, 200, 87, 26, 21, 43, 232, 171, 56, 34, 51, 104, 114, 102, 29, 93, 77,
	39, 28, 85, 171, 58, 165, 90, 98, 64, 34, 22, 116, 206, 23, 34, 43, 166, 73,
	107, 54, 32, 26, 51, 1, 81, 43, 31, 68, 25, 106, 22, 64, 171, 36, 225, 114,
	34, 19, 21, 102, 132, 188, 16, 76, 124, 62, 18, 78, 95, 85, 57, 50, 48, 51,
	193, 101, 35, 159, 215, 111, 89, 46, 111, 60, 148, 31, 172, 219, 228, 21, 18, 111,
	112, 113, 77, 85, 179, 255, 38, 120, 114, 40, 42, 1, 196, 245, 209, 10, 25, 109,
	88, 43, 29, 140, 166, 213, 37, 43, 154, 61, 63, 30, 155, 67, 45, 68, 1, 209,
	100, 80, 8, 43, 154, 1, 51, 26, 71, 142, 78, 78, 16, 255, 128, 34, 197, 171,
	41, 40, 5, 102, 211, 183, 4, 1, 221, 51, 50, 17, 168, 209, 192, 23, 25, 82,
	138, 31, 36, 171, 27, 166, 38, 44, 229, 67, 87, 58, 169, 82, 115, 26, 59, 179,
	63, 59, 90, 180, 59, 166, 93, 73, 154, 40, 40, 21, 116, 143, 209, 34, 39, 175,
	47, 15, 16, 183, 34, 223, 49, 45, 183, 46, 17, 33, 183, 6, 98, 15, 32, 183,
	57, 46, 22, 24, 128, 1, 54, 17, 37, 65, 32, 73, 115, 28, 128, 23, 128, 205,
	40, 3, 9, 115, 51, 192, 18, 6, 223, 87, 37, 9, 115, 59, 77, 64, 21, 47,
	104, 55, 44, 218, 9, 54, 53, 130, 226, 64, 90, 70, 205, 40, 41, 23, 26, 57,
	54, 57, 112, 184, 5, 41, 38, 166, 213, 30, 34, 26, 133, 152, 116, 10, 32, 134,
	39, 19, 53, 221, 26, 114, 32, 73, 255, 31, 9, 65, 234, 2, 15, 1, 118, 73,
	75, 32, 12, 51, 192, 255, 160, 43, 51, 88, 31, 35, 67, 102, 85, 55, 186, 85,
	56, 21, 23, 111, 59, 205, 45, 37, 192, 55, 38, 70, 124, 73, 102, 1, 34, 98,
	125, 98, 42, 88, 104, 85, 117, 175, 82, 95, 84, 53, 89, 128, 100, 113, 101, 45,
	75, 79, 123, 47, 51, 128, 81, 171, 1, 57, 17, 5, 71, 102, 57, 53, 41, 49,
	38, 33, 13, 121, 57, 73, 26, 1, 85, 41, 10, 67, 138, 77, 110, 90, 47, 114,
	115, 21, 2, 10, 102, 255, 166, 23, 6, 101, 29, 16, 10, 85, 128, 101, 196, 26,
	57, 18, 10, 102, 102, 213, 34, 20, 43, 117, 20, 15, 36, 163, 128, 68, 1, 26,
	102, 61, 71, 37, 34, 53, 31, 243, 192, 69, 60, 71, 38, 73, 119, 28, 222, 37,
	68, 45, 128, 34, 1, 47, 11, 245, 171, 62, 17, 19, 70, 146, 85, 55, 62, 70,
	37, 43, 37, 154, 100, 163, 85, 160, 1, 63, 9, 92, 136, 28, 64, 32, 201, 85,
	75, 15, 9, 9, 64, 255, 184, 119, 16, 86, 6, 28, 5, 64, 255, 25, 248, 1,
	56, 8, 17, 132, 137, 255, 55, 116, 128, 58, 15, 20, 82, 135, 57, 26, 121, 40,
	164, 50, 31, 137, 154, 133, 25, 35, 218, 51, 103, 44, 131, 131, 123, 31, 6, 158,
	86, 40, 64, 135, 148, 224, 45, 183, 128, 22, 26, 17, 131, 240, 154, 14, 1, 209,
	45, 16, 21, 91, 64, 222, 7, 1, 197, 56, 21, 39, 155, 60, 138, 23, 102, 213,
	83, 12, 13, 54, 192, 255, 68, 47, 28, 85, 26, 85, 85, 128, 128, 32, 146, 171,
	18, 11, 7, 63, 144, 171, 4, 4, 246, 35, 27, 10, 146, 174, 171, 12, 26, 128,
	190, 80, 35, 99, 180, 80, 126, 54, 45, 85, 126, 47, 87, 176, 51, 41, 20, 32,
	101, 75, 128, 139, 118, 146, 116, 128, 85, 56, 41, 15, 176, 236, 85, 37, 9, 62,
	71, 30, 17, 119, 118, 255, 17, 18, 138, 101, 38, 60, 138, 55, 70, 43, 26, 142,
	146, 36, 19, 30, 171, 255, 97, 27, 20, 138, 45, 61, 62, 219, 1, 81, 188, 64,
	32, 41, 20, 117, 151, 142, 20, 21, 163, 112, 19, 12, 61, 195, 128, 48, 4, 24,
}

local DC_TABLE = {
	4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 17,
	18, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 25, 25, 26, 27, 28,
	29, 30, 31, 32, 33, 34, 35, 36, 37, 37, 38, 39, 40, 41, 42, 43,
	44, 45, 46, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
	59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
	75, 76, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
	91, 93, 95, 96, 98, 100, 101, 102, 104, 106, 108, 110, 112, 114, 116, 118,
	122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 143, 145, 148, 151, 154, 157,
}

local AC_TABLE = {
	4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
	20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
	36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
	52, 53, 54, 55, 56, 57, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76,
	78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108,
	110, 112, 114, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149, 152,
	155, 158, 161, 164, 167, 170, 173, 177, 181, 185, 189, 193, 197, 201, 205, 209,
	213, 217, 221, 225, 229, 234, 239, 245, 249, 254, 259, 264, 269, 274, 279, 284,
}


local KBANDS = { 0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0 }
local KCAT3 = { 173, 148, 140 }
local KCAT4 = { 176, 155, 140, 135 }
local KCAT5 = { 180, 157, 141, 134, 130 }
local KCAT6 = { 254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129 }
local KCAT3456 = { KCAT3, KCAT4, KCAT5, KCAT6 }
local KZIGZAG = { 0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15 }

local function clip255(v: number): number
	return if v < 0 then 0 elseif v > 255 then 255 else v
end

local function sclip1(v: number): number
	return if v < -128 then -128 elseif v > 127 then 127 else v
end

local function sclip2(v: number): number
	return if v < -16 then -16 elseif v > 15 then 15 else v
end

-- ===========================================================================
-- VP8 boolean decoder (RFC 6386 / dixie style)
-- ===========================================================================
type BoolDec = {
	buf: buffer,
	pos: number,
	endPos: number,
	range: number,
	value: number,
	bitCount: number,
}

local function bdInit(buf: buffer, base: number, len: number): BoolDec
	local b0 = if len > 0 then buffer.readu8(buf, base) else 0
	local b1 = if len > 1 then buffer.readu8(buf, base + 1) else 0
	return {
		buf = buf,
		pos = base + 2,
		endPos = base + len,
		range = 255,
		value = b0 * 256 + b1,
		bitCount = 0,
	}
end

local function bdBit(bd: BoolDec, prob: number): number
	local split = 1 + rshift((bd.range - 1) * prob, 8)
	local bigSplit = split * 256
	local bit: number
	if bd.value >= bigSplit then
		bit = 1
		bd.range -= split
		bd.value -= bigSplit
	else
		bit = 0
		bd.range = split
	end
	while bd.range < 128 do
		bd.value *= 2
		bd.range *= 2
		bd.bitCount += 1
		if bd.bitCount == 8 then
			bd.bitCount = 0
			if bd.pos < bd.endPos then
				bd.value += buffer.readu8(bd.buf, bd.pos)
			end
			bd.pos += 1
		end
	end
	return bit
end

local function bdVal(bd: BoolDec, bits: number): number
	local v = 0
	for _ = 1, bits do
		v = v * 2 + bdBit(bd, 128)
	end
	return v
end

local function bdSignedValue(bd: BoolDec, bits: number): number
	local v = bdVal(bd, bits)
	return if bdBit(bd, 128) == 1 then -v else v
end

-- ===========================================================================
-- VP8L bit reader (LSB-first)
-- ===========================================================================
type LBits = {
	buf: buffer,
	base: number,
	len: number,
	bitPos: number,
}

local function lBits(lb: LBits, n: number): number
	if n == 0 then
		return 0
	end
	local bitPos = lb.bitPos
	lb.bitPos = bitPos + n
	local byteOfs = rshift(bitPos, 3)
	local v = 0
	local m = 1
	for i = 0, 3 do
		local idx = byteOfs + i
		if idx < lb.len then
			v += buffer.readu8(lb.buf, lb.base + idx) * m
		end
		m *= 256
	end
	return band(math.floor(v / lshift(1, band(bitPos, 7))), lshift(1, n) - 1)
end

-- canonical huffman for VP8L (codes read LSB-first, lengths <= 15)
type LHuff = { map: { [number]: number }, maxLen: number }

local function lHuffBuild(codeLengths: { number }, numSymbols: number): (LHuff?, string?)
	local counts = table.create(16, 0)
	local total = 0
	local maxLen = 0
	local numUsed = 0
	local lastSym = 0
	for i = 1, numSymbols do
		local l = codeLengths[i]
		if l > 15 then
			return nil, "vp8l: code length too large"
		end
		if l > 0 then
			counts[l] += 1
			numUsed += 1
			lastSym = i - 1
			if l > maxLen then
				maxLen = l
			end
		end
	end
	if numUsed == 0 then
		return nil, "vp8l: empty huffman code"
	end
	if numUsed == 1 then
		-- degenerate: 0-bit code
		return { map = { [0] = lastSym }, maxLen = 0 }, nil
	end
	local nextCode = table.create(16, 0)
	local code = 0
	for l = 1, 15 do
		nextCode[l] = code
		code = (code + counts[l]) * 2
	end
	local map: { [number]: number } = {}
	for i = 1, numSymbols do
		local l = codeLengths[i]
		if l > 0 then
			local c = nextCode[l]
			nextCode[l] = c + 1
			-- reverse bits for LSB-first reading
			local rev = 0
			local cc = c
			for _ = 1, l do
				rev = rev * 2 + band(cc, 1)
				cc = rshift(cc, 1)
			end
			map[l * 65536 + rev] = i - 1
		end
	end
	return { map = map, maxLen = maxLen }, nil
end

local function lHuffDecode(lb: LBits, h: LHuff): number
	if h.maxLen == 0 then
		return h.map[0]
	end
	local v = 0
	local m = 1
	for len = 1, h.maxLen do
		v += lBits(lb, 1) * m
		m *= 2
		local sym = h.map[len * 65536 + v]
		if sym ~= nil then
			return sym
		end
	end
	return -1
end

-- ===========================================================================
-- VP8L decoder
-- ===========================================================================
local CODE_LENGTH_ORDER = { 17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }
local CODE_TO_PLANE = {
	0x18, 0x07, 0x17, 0x19, 0x28, 0x06, 0x27, 0x29, 0x16, 0x1a, 0x26, 0x2a,
	0x38, 0x05, 0x37, 0x39, 0x15, 0x1b, 0x36, 0x3a, 0x25, 0x2b, 0x48, 0x04,
	0x47, 0x49, 0x14, 0x1c, 0x35, 0x3b, 0x46, 0x4a, 0x24, 0x2c, 0x58, 0x45,
	0x4b, 0x34, 0x3c, 0x03, 0x57, 0x59, 0x13, 0x1d, 0x56, 0x5a, 0x23, 0x2d,
	0x44, 0x4c, 0x55, 0x5b, 0x33, 0x3d, 0x68, 0x02, 0x67, 0x69, 0x12, 0x1e,
	0x66, 0x6a, 0x22, 0x2e, 0x54, 0x5c, 0x43, 0x4d, 0x65, 0x6b, 0x32, 0x3e,
	0x78, 0x01, 0x77, 0x79, 0x53, 0x5d, 0x11, 0x1f, 0x64, 0x6c, 0x42, 0x4e,
	0x76, 0x7a, 0x21, 0x2f, 0x75, 0x7b, 0x31, 0x3f, 0x63, 0x6d, 0x52, 0x5e,
	0x00, 0x74, 0x7c, 0x41, 0x4f, 0x10, 0x20, 0x62, 0x6e, 0x30, 0x73, 0x7d,
	0x51, 0x5f, 0x40, 0x72, 0x7e, 0x61, 0x6f, 0x50, 0x71, 0x7f, 0x60, 0x70,
}

local function subSampleSize(size: number, bits: number): number
	return rshift(size + lshift(1, bits) - 1, bits)
end

-- color cache hash: (argb * 0x1e35a7bd) >> (32 - bits), exact mod 2^32
local function cacheHash(argb: number, shiftAmt: number): number
	local lo = band(argb, 0xFFFF)
	local hi = math.floor(argb / 65536)
	local m = 0x1e35a7bd
	local p = (lo * m) % 4294967296
	p = (p + ((hi * m) % 65536) * 65536) % 4294967296
	return math.floor(p / lshift(1, shiftAmt))
end

type LTransform = {
	ttype: number,
	bits: number,
	xsize: number, -- ORIGINAL xsize at the time this transform was read
	ysize: number,
	data: { number }?, -- subimage pixels (ARGB)
}

type LGroup = {
	-- 5 huffman codes: green+len+cache, red, blue, alpha, dist
	g: LHuff,
	r: LHuff,
	b: LHuff,
	a: LHuff,
	d: LHuff,
}

type LDec = {
	lb: LBits,
	transforms: { LTransform },
	-- huffman metadata
	huffImage: { number }?,
	huffBits: number,
	huffXsize: number,
	groups: { LGroup },
	cacheBits: number,
	cache: { number },
}

local function lReadHuffmanCode(dec: LDec, alphabetSize: number): (LHuff?, string?)
	local lb = dec.lb
	local simple = lBits(lb, 1)
	local codeLengths = table.create(alphabetSize, 0)
	if simple == 1 then
		local numSymbols = lBits(lb, 1) + 1
		local firstLenCode = lBits(lb, 1)
		local symbol = lBits(lb, if firstLenCode == 0 then 1 else 8)
		if symbol >= alphabetSize then
			return nil, "vp8l: bad simple code symbol"
		end
		codeLengths[symbol + 1] = 1
		if numSymbols == 2 then
			local symbol2 = lBits(lb, 8)
			if symbol2 >= alphabetSize then
				return nil, "vp8l: bad simple code symbol 2"
			end
			codeLengths[symbol2 + 1] = 1
		end
	else
		local clcLengths = table.create(19, 0)
		local numCodes = lBits(lb, 4) + 4
		if numCodes > 19 then
			return nil, "vp8l: bad code length count"
		end
		for i = 1, numCodes do
			clcLengths[CODE_LENGTH_ORDER[i] + 1] = lBits(lb, 3)
		end
		local clcTable, cerr = lHuffBuild(clcLengths, 19)
		if not clcTable then
			return nil, cerr
		end
		-- read code lengths with the code-length code
		local maxSymbol: number
		if lBits(lb, 1) == 1 then
			local lengthNbits = 2 + 2 * lBits(lb, 3)
			maxSymbol = 2 + lBits(lb, lengthNbits)
			if maxSymbol > alphabetSize then
				return nil, "vp8l: bad max symbol"
			end
		else
			maxSymbol = alphabetSize
		end
		local symbol = 0
		local prevCodeLen = 8
		while symbol < alphabetSize do
			if maxSymbol == 0 then
				break
			end
			maxSymbol -= 1
			local codeLen = lHuffDecode(lb, clcTable :: LHuff)
			if codeLen < 0 then
				return nil, "vp8l: bad code length symbol"
			end
			if codeLen < 16 then
				codeLengths[symbol + 1] = codeLen
				symbol += 1
				if codeLen ~= 0 then
					prevCodeLen = codeLen
				end
			else
				local usePrev = codeLen == 16
				local slot = codeLen - 16
				local extraBits = if slot == 0 then 2 elseif slot == 1 then 3 else 7
				local repeatOffset = if slot == 0 then 3 elseif slot == 1 then 3 else 11
				local repeatCount = lBits(lb, extraBits) + repeatOffset
				if symbol + repeatCount > alphabetSize then
					return nil, "vp8l: code length repeat overflow"
				end
				local fillLen = if usePrev then prevCodeLen else 0
				for _ = 1, repeatCount do
					codeLengths[symbol + 1] = fillLen
					symbol += 1
				end
			end
		end
	end
	return lHuffBuild(codeLengths, alphabetSize)
end

local lDecodeImageStream: (dec: LDec, xsize: number, ysize: number, isLevel0: boolean) -> ({ number }?, number, string?)

local function lReadHuffmanCodes(dec: LDec, xsize: number, ysize: number, cacheBits: number, allowRecursion: boolean): string?
	local lb = dec.lb
	local numGroups = 1
	dec.huffImage = nil
	dec.huffBits = 0
	if allowRecursion and lBits(lb, 1) == 1 then
		local huffBits = lBits(lb, 3) + 2
		local hx = subSampleSize(xsize, huffBits)
		local hy = subSampleSize(ysize, huffBits)
		local img, _, err = lDecodeImageStream(dec, hx, hy, false)
		if not img then
			return err or "vp8l: bad huffman image"
		end
		local image = img :: { number }
		dec.huffBits = huffBits
		dec.huffXsize = hx
		for i = 1, hx * hy do
			local group = band(math.floor(image[i] / 256), 0xFFFF)
			image[i] = group
			if group + 1 > numGroups then
				numGroups = group + 1
			end
		end
		dec.huffImage = image
	end
	dec.cacheBits = cacheBits
	local groups: { LGroup } = table.create(numGroups)
	local greenAlphabet = 256 + 24 + (if cacheBits > 0 then lshift(1, cacheBits) else 0)
	for i = 1, numGroups do
		local g, e1 = lReadHuffmanCode(dec, greenAlphabet)
		if not g then
			return e1
		end
		local r, e2 = lReadHuffmanCode(dec, 256)
		if not r then
			return e2
		end
		local b, e3 = lReadHuffmanCode(dec, 256)
		if not b then
			return e3
		end
		local a, e4 = lReadHuffmanCode(dec, 256)
		if not a then
			return e4
		end
		local d, e5 = lReadHuffmanCode(dec, 40)
		if not d then
			return e5
		end
		groups[i] = {
			g = g :: LHuff,
			r = r :: LHuff,
			b = b :: LHuff,
			a = a :: LHuff,
			d = d :: LHuff,
		}
	end
	dec.groups = groups
	return nil
end

local function lGetCopyDistance(lb: LBits, symbol: number): number
	if symbol < 4 then
		return symbol + 1
	end
	local extraBits = rshift(symbol - 2, 1)
	local offset = lshift(2 + band(symbol, 1), extraBits)
	return offset + lBits(lb, extraBits) + 1
end

local function lPlaneCodeToDistance(xsize: number, planeCode: number): number
	if planeCode > 120 then
		return planeCode - 120
	end
	local code = CODE_TO_PLANE[planeCode]
	local yoffset = rshift(code, 4)
	local xoffset = 8 - band(code, 15)
	local dist = yoffset * xsize + xoffset
	return if dist >= 1 then dist else 1
end

-- decode an LZ77+huffman coded ARGB image of xsize*ysize into a flat array
local function lDecodeImageData(dec: LDec, xsize: number, ysize: number): ({ number }?, string?)
	local lb = dec.lb
	local total = xsize * ysize
	local data = table.create(total, 0)
	local cacheBits = dec.cacheBits
	local cacheSize = if cacheBits > 0 then lshift(1, cacheBits) else 0
	local cache: { number } = if cacheSize > 0 then table.create(cacheSize, 0) else {}
	local cacheShift = 32 - cacheBits
	local lastCached = 1
	local huffImage = dec.huffImage
	local huffMask = if dec.huffBits == 0 then -1 else lshift(1, dec.huffBits) - 1
	local groups = dec.groups
	local group = groups[1]
	local lenCodeLimit = 256 + 24
	local cacheLimit = lenCodeLimit + cacheSize

	local pos = 1
	local col = 0
	local row = 0
	while pos <= total do
		if huffImage and band(col, huffMask) == 0 then
			local hIdx = rshift(row, dec.huffBits) * dec.huffXsize + rshift(col, dec.huffBits)
			group = groups[(huffImage :: { number })[hIdx + 1] + 1]
		end
		local code = lHuffDecode(lb, group.g)
		if code < 0 then
			return nil, "vp8l: bad green code"
		end
		if code < 256 then
			local red = lHuffDecode(lb, group.r)
			local blue = lHuffDecode(lb, group.b)
			local alpha = lHuffDecode(lb, group.a)
			if red < 0 or blue < 0 or alpha < 0 then
				return nil, "vp8l: bad literal"
			end
			data[pos] = alpha * 16777216 + red * 65536 + code * 256 + blue
			pos += 1
			col += 1
			if col >= xsize then
				col = 0
				row += 1
				if cacheSize > 0 then
					while lastCached < pos do
						local px = data[lastCached]
						cache[cacheHash(px, cacheShift) + 1] = px
						lastCached += 1
					end
				end
			end
		elseif code < lenCodeLimit then
			local lengthSym = code - 256
			local length = lGetCopyDistance(lb, lengthSym)
			local distSymbol = lHuffDecode(lb, group.d)
			if distSymbol < 0 then
				return nil, "vp8l: bad distance code"
			end
			local distCode = lGetCopyDistance(lb, distSymbol)
			local dist = lPlaneCodeToDistance(xsize, distCode)
			if pos - dist < 1 or pos + length - 1 > total then
				return nil, "vp8l: bad backward reference"
			end
			for k = 0, length - 1 do
				data[pos + k] = data[pos + k - dist]
			end
			pos += length
			col += length
			while col >= xsize do
				col -= xsize
				row += 1
			end
			if huffImage and band(col, huffMask) ~= 0 then
				local hIdx = rshift(row, dec.huffBits) * dec.huffXsize + rshift(col, dec.huffBits)
				group = groups[(huffImage :: { number })[hIdx + 1] + 1]
			end
			if cacheSize > 0 then
				while lastCached < pos do
					local px = data[lastCached]
					cache[cacheHash(px, cacheShift) + 1] = px
					lastCached += 1
				end
			end
		elseif code < cacheLimit then
			local key = code - lenCodeLimit
			if cacheSize > 0 then
				while lastCached < pos do
					local px = data[lastCached]
					cache[cacheHash(px, cacheShift) + 1] = px
					lastCached += 1
				end
			end
			data[pos] = cache[key + 1]
			pos += 1
			col += 1
			if col >= xsize then
				col = 0
				row += 1
				if cacheSize > 0 then
					while lastCached < pos do
						local px = data[lastCached]
						cache[cacheHash(px, cacheShift) + 1] = px
						lastCached += 1
					end
				end
			end
		else
			return nil, "vp8l: invalid code"
		end
	end
	return data, nil
end

-- inverse transforms ---------------------------------------------------------

local function avg2c(a: number, b: number): number
	return rshift(a + b, 1)
end

-- per-channel average of two ARGB pixels
local function lAverage2(a: number, b: number): number
	local aa = math.floor(a / 16777216)
	local ar = band(math.floor(a / 65536), 255)
	local ag = band(math.floor(a / 256), 255)
	local ab = band(a, 255)
	local ba = math.floor(b / 16777216)
	local br2 = band(math.floor(b / 65536), 255)
	local bg = band(math.floor(b / 256), 255)
	local bb = band(b, 255)
	return avg2c(aa, ba) * 16777216 + avg2c(ar, br2) * 65536 + avg2c(ag, bg) * 256 + avg2c(ab, bb)
end

local function lAddPixels(a: number, b: number): number
	local aa = math.floor(a / 16777216)
	local ar = band(math.floor(a / 65536), 255)
	local ag = band(math.floor(a / 256), 255)
	local ab = band(a, 255)
	local ba = math.floor(b / 16777216)
	local br2 = band(math.floor(b / 65536), 255)
	local bg = band(math.floor(b / 256), 255)
	local bb = band(b, 255)
	return band(aa + ba, 255) * 16777216 + band(ar + br2, 255) * 65536 + band(ag + bg, 255) * 256 + band(ab + bb, 255)
end

local function lSelect(a: number, b: number, c: number): number
	-- a = top, b = left, c = top-left
	local function sub3(x: number, y: number, z: number): number
		return math.abs(y - z) - math.abs(x - z)
	end
	local paMinusPb = sub3(math.floor(a / 16777216), math.floor(b / 16777216), math.floor(c / 16777216))
		+ sub3(band(math.floor(a / 65536), 255), band(math.floor(b / 65536), 255), band(math.floor(c / 65536), 255))
		+ sub3(band(math.floor(a / 256), 255), band(math.floor(b / 256), 255), band(math.floor(c / 256), 255))
		+ sub3(band(a, 255), band(b, 255), band(c, 255))
	return if paMinusPb <= 0 then a else b
end

local function lClampedAddSubtractFull(c0: number, c1: number, c2: number): number
	local function ch(sh: number): number
		local x0 = band(math.floor(c0 / sh), 255)
		local x1 = band(math.floor(c1 / sh), 255)
		local x2 = band(math.floor(c2 / sh), 255)
		return clip255(x0 + x1 - x2)
	end
	return ch(16777216) * 16777216 + ch(65536) * 65536 + ch(256) * 256 + ch(1)
end

local function lClampedAddSubtractHalf(c0: number, c1: number, c2: number): number
	local ave = lAverage2(c0, c1)
	local function ch(sh: number): number
		local a = band(math.floor(ave / sh), 255)
		local b = band(math.floor(c2 / sh), 255)
		-- a + (a - b) / 2, C division truncates toward zero
		local d = a - b
		local half = if d >= 0 then math.floor(d / 2) else -math.floor(-d / 2)
		return clip255(a + half)
	end
	return ch(16777216) * 16777216 + ch(65536) * 65536 + ch(256) * 256 + ch(1)
end

local function lPredictor(mode: number, left: number, data: { number }, topIdx: number): number
	if mode == 0 then
		return 0xff000000
	elseif mode == 1 then
		return left
	elseif mode == 2 then
		return data[topIdx]
	elseif mode == 3 then
		return data[topIdx + 1]
	elseif mode == 4 then
		return data[topIdx - 1]
	elseif mode == 5 then
		return lAverage2(lAverage2(left, data[topIdx + 1]), data[topIdx])
	elseif mode == 6 then
		return lAverage2(left, data[topIdx - 1])
	elseif mode == 7 then
		return lAverage2(left, data[topIdx])
	elseif mode == 8 then
		return lAverage2(data[topIdx - 1], data[topIdx])
	elseif mode == 9 then
		return lAverage2(data[topIdx], data[topIdx + 1])
	elseif mode == 10 then
		return lAverage2(lAverage2(left, data[topIdx - 1]), lAverage2(data[topIdx], data[topIdx + 1]))
	elseif mode == 11 then
		return lSelect(data[topIdx], left, data[topIdx - 1])
	elseif mode == 12 then
		return lClampedAddSubtractFull(left, data[topIdx], data[topIdx - 1])
	else
		return lClampedAddSubtractHalf(left, data[topIdx], data[topIdx - 1])
	end
end

local function lApplyInverseTransforms(
	transforms: { LTransform },
	data: { number },
	finalXsize: number
): ({ number }, number)
	local cur = data
	local xsize = finalXsize
	-- transforms were stored in stream order; apply in reverse
	for t = #transforms, 1, -1 do
		local tr = transforms[t]
		local width = tr.xsize
		local height = tr.ysize
		if tr.ttype == 0 then
			-- PREDICTOR
			local bits = tr.bits
			local mask = lshift(1, bits) - 1
			local tilesPerRow = subSampleSize(width, bits)
			local modes = tr.data :: { number }
			-- first pixel: black (add), rest of first row: left
			cur[1] = lAddPixels(cur[1], 0xff000000)
			for x = 2, width do
				cur[x] = lAddPixels(cur[x], cur[x - 1])
			end
			local idx = width + 1
			for y = 1, height - 1 do
				local modeRowBase = rshift(y, bits) * tilesPerRow
				-- first pixel of the row: top
				cur[idx] = lAddPixels(cur[idx], cur[idx - width])
				idx += 1
				for x = 1, width - 1 do
					local mode = band(rshift(modes[modeRowBase + rshift(x, bits) + 1], 8), 0xF)
					local pred = lPredictor(mode, cur[idx - 1], cur, idx - width)
					cur[idx] = lAddPixels(cur[idx], pred)
					idx += 1
				end
			end
		elseif tr.ttype == 1 then
			-- CROSS COLOR
			local bits = tr.bits
			local tilesPerRow = subSampleSize(width, bits)
			local cdata = tr.data :: { number }
			local idx = 1
			for y = 0, height - 1 do
				local rowBase = rshift(y, bits) * tilesPerRow
				for x = 0, width - 1 do
					local cc = cdata[rowBase + rshift(x, bits) + 1]
					local g2r = band(cc, 255)
					local g2b = band(math.floor(cc / 256), 255)
					local r2b = band(math.floor(cc / 65536), 255)
					-- signed multipliers
					if g2r >= 128 then
						g2r -= 256
					end
					if g2b >= 128 then
						g2b -= 256
					end
					if r2b >= 128 then
						r2b -= 256
					end
					local argb = cur[idx]
					local green = band(math.floor(argb / 256), 255)
					local sg = if green >= 128 then green - 256 else green
					local red = band(math.floor(argb / 65536), 255)
					local blue = band(argb, 255)
					local newRed = band(red + math.floor(g2r * sg / 32), 255)
					local snr = if newRed >= 128 then newRed - 256 else newRed
					local newBlue = blue + math.floor(g2b * sg / 32)
					newBlue += math.floor(r2b * snr / 32)
					newBlue = band(newBlue, 255)
					cur[idx] = argb - band(math.floor(argb / 65536), 255) * 65536 - band(argb, 255)
						+ newRed * 65536
						+ newBlue
					idx += 1
				end
			end
		elseif tr.ttype == 2 then
			-- SUBTRACT GREEN
			for i = 1, width * height do
				local argb = cur[i]
				local green = band(math.floor(argb / 256), 255)
				local red = band(math.floor(argb / 65536) + green, 255)
				local blue = band(argb + green, 255)
				cur[i] = argb - band(math.floor(argb / 65536), 255) * 65536 - band(argb, 255) + red * 65536 + blue
			end
		else
			-- COLOR INDEXING (with pixel bundling); cur has the REDUCED width
			local bits = tr.bits
			local cmap = tr.data :: { number }
			local out = table.create(width * height, 0)
			if bits > 0 then
				local bitsPerPixel = rshift(8, bits)
				local pixelsPerByte = lshift(1, bits)
				local countMask = pixelsPerByte - 1
				local bitMask = lshift(1, bitsPerPixel) - 1
				local packedWidth = subSampleSize(width, bits)
				local o = 1
				for y = 0, height - 1 do
					local packed = 0
					for x = 0, width - 1 do
						if band(x, countMask) == 0 then
							packed = band(math.floor(cur[y * packedWidth + rshift(x, bits) + 1] / 256), 255)
						end
						out[o] = cmap[band(packed, bitMask) + 1] or 0
						packed = rshift(packed, bitsPerPixel)
						o += 1
					end
				end
			else
				for i = 1, width * height do
					local idx2 = band(math.floor(cur[i] / 256), 255)
					out[i] = cmap[idx2 + 1] or 0
				end
			end
			cur = out
		end
		xsize = width
	end
	return cur, xsize
end

lDecodeImageStream = function(dec: LDec, xsize: number, ysize: number, isLevel0: boolean): ({ number }?, number, string?)
	local lb = dec.lb
	local transformXsize = xsize
	local transforms: { LTransform } = {}

	if isLevel0 then
		while lBits(lb, 1) == 1 do
			local ttype = lBits(lb, 2)
			for _, tr in ipairs(transforms) do
				if tr.ttype == ttype then
					return nil, 0, "vp8l: duplicate transform"
				end
			end
			local tr: LTransform = { ttype = ttype, bits = 0, xsize = transformXsize, ysize = ysize, data = nil }
			if ttype == 0 or ttype == 1 then
				tr.bits = lBits(lb, 3) + 2
				local img, _, err = lDecodeImageStream(
					dec,
					subSampleSize(transformXsize, tr.bits),
					subSampleSize(ysize, tr.bits),
					false
				)
				if not img then
					return nil, 0, err
				end
				tr.data = img
			elseif ttype == 3 then
				local numColors = lBits(lb, 8) + 1
				local bits = if numColors > 16 then 0 elseif numColors > 4 then 1 elseif numColors > 2 then 2 else 3
				transformXsize = subSampleSize(transformXsize, bits)
				tr.bits = bits
				local img, _, err = lDecodeImageStream(dec, numColors, 1, false)
				if not img then
					return nil, 0, err
				end
				-- expand color map (delta coded per byte channel)
				local finalNumColors = lshift(1, rshift(8, bits))
				local cmap = table.create(finalNumColors, 0)
				local image = img :: { number }
				cmap[1] = image[1]
				for i = 2, numColors do
					cmap[i] = lAddPixels(image[i], cmap[i - 1])
				end
				for i = numColors + 1, finalNumColors do
					cmap[i] = 0
				end
				tr.data = cmap
			end
			-- SUBTRACT_GREEN (2): nothing to read
			table.insert(transforms, tr)
		end
	end

	local cacheBits = 0
	if lBits(lb, 1) == 1 then
		cacheBits = lBits(lb, 4)
		if cacheBits < 1 or cacheBits > 11 then
			return nil, 0, "vp8l: bad cache bits"
		end
	end

	-- huffman codes (saving/restoring the metadata around recursion)
	local savedImage = dec.huffImage
	local savedBits = dec.huffBits
	local savedXsize = dec.huffXsize
	local savedGroups = dec.groups
	local savedCache = dec.cacheBits
	local herr = lReadHuffmanCodes(dec, transformXsize, ysize, cacheBits, isLevel0)
	if herr then
		return nil, 0, herr
	end

	local data, derr = lDecodeImageData(dec, transformXsize, ysize)
	if not data then
		return nil, 0, derr
	end

	local result = data :: { number }
	local outXsize = transformXsize
	if isLevel0 and #transforms > 0 then
		result, outXsize = lApplyInverseTransforms(transforms, result, transformXsize)
	end

	if not isLevel0 then
		-- restore parent metadata
		dec.huffImage = savedImage
		dec.huffBits = savedBits
		dec.huffXsize = savedXsize
		dec.groups = savedGroups
		dec.cacheBits = savedCache
	end

	return result, outXsize, nil
end

-- Decode a VP8L stream (after the 0x2F signature byte if present).
-- When `noHeader` is true (ALPH chunk), width/height come from the caller.
local function decodeVP8L(
	buf: buffer,
	base: number,
	len: number,
	noHeader: boolean,
	knownW: number,
	knownH: number
): ({ number }?, number, number, string?)
	local lb: LBits = { buf = buf, base = base, len = len, bitPos = 0 }
	local width: number
	local height: number
	if noHeader then
		width = knownW
		height = knownH
	else
		local sig = lBits(lb, 8)
		if sig ~= 0x2F then
			return nil, 0, 0, "vp8l: bad signature"
		end
		width = lBits(lb, 14) + 1
		height = lBits(lb, 14) + 1
		lBits(lb, 1) -- alpha hint
		if lBits(lb, 3) ~= 0 then
			return nil, 0, 0, "vp8l: bad version"
		end
	end
	local dec: LDec = {
		lb = lb,
		transforms = {},
		huffImage = nil,
		huffBits = 0,
		huffXsize = 0,
		groups = {},
		cacheBits = 0,
		cache = {},
	}
	local data, _, err = lDecodeImageStream(dec, width, height, true)
	if not data then
		return nil, 0, 0, err
	end
	return data, width, height, nil
end

-- ===========================================================================
-- VP8 (lossy) decoder
-- ===========================================================================
local BPS = 32
local YOFF = BPS * 1 + 8
local UOFF = YOFF + BPS * 16 + BPS
local VOFF = UOFF + 16

-- AVG conventions
local function avg2(a: number, b: number): number
	return rshift(a + b + 1, 1)
end

local function avg3(a: number, b: number, c: number): number
	return rshift(a + 2 * b + c + 2, 2)
end

type VP8State = {
	-- headers
	mbW: number,
	mbH: number,
	width: number,
	height: number,
	useSegment: boolean,
	updateMap: boolean,
	absoluteDelta: boolean,
	segQuant: { number },
	segFilter: { number },
	segProbs: { number },
	filterSimple: boolean,
	filterLevel: number,
	filterSharpness: number,
	useLfDelta: boolean,
	refLfDelta: { number },
	modeLfDelta: { number },
	filterType: number, -- 0 none, 1 simple, 2 complex
	useSkipProba: boolean,
	skipP: number,
	-- quant matrices per segment: y1dc,y1ac,y2dc,y2ac,uvdc,uvac
	dqm: { { number } },
	-- probabilities
	coeffs: { number }, -- flattened [type][band][ctx][proba] 4*8*3*11
	-- filter strengths per segment & i4x4: {limit, ilevel, hev, inner}
	fstrengths: { { { number } } },
}

local function vp8ParseQuant(bd: BoolDec, s: VP8State)
	local baseQ0 = bdVal(bd, 7)
	local dqy1dc = if bdBit(bd, 128) == 1 then bdSignedValue(bd, 4) else 0
	local dqy2dc = if bdBit(bd, 128) == 1 then bdSignedValue(bd, 4) else 0
	local dqy2ac = if bdBit(bd, 128) == 1 then bdSignedValue(bd, 4) else 0
	local dquvdc = if bdBit(bd, 128) == 1 then bdSignedValue(bd, 4) else 0
	local dquvac = if bdBit(bd, 128) == 1 then bdSignedValue(bd, 4) else 0
	local function clipQ(q: number, mx: number): number
		return if q < 0 then 0 elseif q > mx then mx else q
	end
	for i = 0, 3 do
		local q: number
		if s.useSegment then
			q = s.segQuant[i + 1]
			if not s.absoluteDelta then
				q += baseQ0
			end
		else
			if i > 0 then
				s.dqm[i + 1] = s.dqm[1]
				continue
			else
				q = baseQ0
			end
		end
		local y2ac = math.floor(AC_TABLE[clipQ(q + dqy2ac, 127) + 1] * 101581 / 65536)
		if y2ac < 8 then
			y2ac = 8
		end
		s.dqm[i + 1] = {
			DC_TABLE[clipQ(q + dqy1dc, 127) + 1],
			AC_TABLE[clipQ(q, 127) + 1],
			DC_TABLE[clipQ(q + dqy2dc, 127) + 1] * 2,
			y2ac,
			DC_TABLE[clipQ(q + dquvdc, 117) + 1],
			AC_TABLE[clipQ(q + dquvac, 127) + 1],
		}
	end
end

local function coeffIdx(t: number, b: number, c: number, p: number): number
	return ((t * 8 + b) * 3 + c) * 11 + p + 1
end

-- residual decoding
local function getLargeValue(bd: BoolDec, probs: { number }, pBase: number): number
	-- pBase indexes the 11-proba array (0-based) within probs
	local function P(i: number): number
		return probs[pBase + i + 1]
	end
	local v: number
	if bdBit(bd, P(3)) == 0 then
		if bdBit(bd, P(4)) == 0 then
			v = 2
		else
			v = 3 + bdBit(bd, P(5))
		end
	else
		if bdBit(bd, P(6)) == 0 then
			if bdBit(bd, P(7)) == 0 then
				v = 5 + bdBit(bd, 159)
			else
				v = 7 + 2 * bdBit(bd, 165)
				v += bdBit(bd, 145)
			end
		else
			local bit1 = bdBit(bd, P(8))
			local bit0 = bdBit(bd, P(9 + bit1))
			local cat = 2 * bit1 + bit0
			v = 0
			for _, tp in ipairs(KCAT3456[cat + 1]) do
				v += v + bdBit(bd, tp)
			end
			v += 3 + lshift(8, cat)
		end
	end
	return v
end

-- Returns position of last non-zero coeff plus one. out is 0-based 16 coeffs
-- (flat array slot base outBase).
local function getCoeffs(
	bd: BoolDec,
	coeffs: { number },
	ctype: number,
	ctx: number,
	dqDC: number,
	dqAC: number,
	n: number,
	out: { number },
	outBase: number
): number
	-- probs base for band/ctx
	local function probBase(nn: number, cc: number): number
		return ((ctype * 8 + KBANDS[nn + 1]) * 3 + cc) * 11
	end
	local pBase = probBase(n, ctx)
	while n < 16 do
		if bdBit(bd, coeffs[pBase + 1]) == 0 then
			return n
		end
		while bdBit(bd, coeffs[pBase + 2]) == 0 do
			n += 1
			if n == 16 then
				return 16
			end
			pBase = probBase(n, 0)
		end
		local v: number
		local pCtxBase = probBase(n + 1, 0) -- will adjust ctx below
		if bdBit(bd, coeffs[pBase + 3]) == 0 then
			v = 1
			pBase = ((ctype * 8 + KBANDS[n + 2]) * 3 + 1) * 11
		else
			v = getLargeValue(bd, coeffs, pBase)
			pBase = ((ctype * 8 + KBANDS[n + 2]) * 3 + 2) * 11
		end
		local _ = pCtxBase
		local sign = bdBit(bd, 128)
		local dq = if n > 0 then dqAC else dqDC
		out[outBase + KZIGZAG[n + 1] + 1] = (if sign == 1 then -v else v) * dq
		n += 1
	end
	return 16
end

-- transforms
local function mul1(a: number): number
	return math.floor(a * 20091 / 65536) + a
end

local function mul2(a: number): number
	return math.floor(a * 35468 / 65536)
end

-- dst = clip(dst + (v >> 3)) with arithmetic shift semantics
local function addClipShift3(dst: buffer, off: number, v: number)
	local sh = math.floor(v / 8)
	buffer.writeu8(dst, off, clip255(buffer.readu8(dst, off) + sh))
end

local function transformOne(coeffs: { number }, base: number, dst: buffer, dstOff: number)
	local C = table.create(16, 0)
	for i = 0, 3 do
		local a = coeffs[base + i + 1] + coeffs[base + 8 + i + 1]
		local b = coeffs[base + i + 1] - coeffs[base + 8 + i + 1]
		local c = mul2(coeffs[base + 4 + i + 1]) - mul1(coeffs[base + 12 + i + 1])
		local d = mul1(coeffs[base + 4 + i + 1]) + mul2(coeffs[base + 12 + i + 1])
		C[i * 4 + 1] = a + d
		C[i * 4 + 2] = b + c
		C[i * 4 + 3] = b - c
		C[i * 4 + 4] = a - d
	end
	for i = 0, 3 do
		local dc = C[i + 1] + 4
		local a = dc + C[8 + i + 1]
		local b = dc - C[8 + i + 1]
		local c = mul2(C[4 + i + 1]) - mul1(C[12 + i + 1])
		local d = mul1(C[4 + i + 1]) + mul2(C[12 + i + 1])
		local o = dstOff + i * BPS
		addClipShift3(dst, o, a + d)
		addClipShift3(dst, o + 1, b + c)
		addClipShift3(dst, o + 2, b - c)
		addClipShift3(dst, o + 3, a - d)
	end
end

local function transformDC(coeffs: { number }, base: number, dst: buffer, dstOff: number)
	local DC = coeffs[base + 1] + 4
	local sh = math.floor(DC / 8)
	for j = 0, 3 do
		for i = 0, 3 do
			local off = dstOff + j * BPS + i
			buffer.writeu8(dst, off, clip255(buffer.readu8(dst, off) + sh))
		end
	end
end

local function transformAC3(coeffs: { number }, base: number, dst: buffer, dstOff: number)
	local a = coeffs[base + 1] + 4
	local c4 = mul2(coeffs[base + 5])
	local d4 = mul1(coeffs[base + 5])
	local c1 = mul2(coeffs[base + 2])
	local d1 = mul1(coeffs[base + 2])
	local function store2(y: number, dc: number)
		local o = dstOff + y * BPS
		addClipShift3(dst, o, dc + d1)
		addClipShift3(dst, o + 1, dc + c1)
		addClipShift3(dst, o + 2, dc - c1)
		addClipShift3(dst, o + 3, dc - d1)
	end
	store2(0, a + d4)
	store2(1, a + c4)
	store2(2, a - c4)
	store2(3, a - d4)
end

local function transformWHT(inC: { number }, out: { number })
	local tmp = table.create(16, 0)
	for i = 0, 3 do
		local a0 = inC[i + 1] + inC[12 + i + 1]
		local a1 = inC[4 + i + 1] + inC[8 + i + 1]
		local a2 = inC[4 + i + 1] - inC[8 + i + 1]
		local a3 = inC[i + 1] - inC[12 + i + 1]
		tmp[i + 1] = a0 + a1
		tmp[8 + i + 1] = a0 - a1
		tmp[4 + i + 1] = a3 + a2
		tmp[12 + i + 1] = a3 - a2
	end
	local outIdx = 0
	for i = 0, 3 do
		local dc = tmp[i * 4 + 1] + 3
		local a0 = dc + tmp[3 + i * 4 + 1]
		local a1 = tmp[1 + i * 4 + 1] + tmp[2 + i * 4 + 1]
		local a2 = tmp[1 + i * 4 + 1] - tmp[2 + i * 4 + 1]
		local a3 = dc - tmp[3 + i * 4 + 1]
		out[outIdx + 1] = math.floor((a0 + a1) / 8)
		out[outIdx + 16 + 1] = math.floor((a3 + a2) / 8)
		out[outIdx + 32 + 1] = math.floor((a0 - a1) / 8)
		out[outIdx + 48 + 1] = math.floor((a3 - a2) / 8)
		outIdx += 64
	end
end

-- intra predictors on the BPS-strided scratch buffer ------------------------

local function put(dst: buffer, off: number, size: number, v: number)
	for j = 0, size - 1 do
		buffer.fill(dst, off + j * BPS, v, size)
	end
end

local function predDC(dst: buffer, off: number, size: number, haveTop: boolean, haveLeft: boolean)
	local dc = 0
	local count = 0
	if haveTop then
		for i = 0, size - 1 do
			dc += buffer.readu8(dst, off - BPS + i)
		end
		count += size
	end
	if haveLeft then
		for j = 0, size - 1 do
			dc += buffer.readu8(dst, off - 1 + j * BPS)
		end
		count += size
	end
	if count == 0 then
		dc = 0x80
	else
		if haveTop and haveLeft then
			-- (dc + size) >> (log2(size)+1)
			dc = 0
			for i = 0, size - 1 do
				dc += buffer.readu8(dst, off - BPS + i) + buffer.readu8(dst, off - 1 + i * BPS)
			end
			dc = rshift(dc + size, if size == 16 then 5 elseif size == 8 then 4 else 3)
		else
			dc = 0
			if haveTop then
				for i = 0, size - 1 do
					dc += buffer.readu8(dst, off - BPS + i)
				end
			else
				for j = 0, size - 1 do
					dc += buffer.readu8(dst, off - 1 + j * BPS)
				end
			end
			dc = rshift(dc + rshift(size, 1), if size == 16 then 4 elseif size == 8 then 3 else 2)
		end
	end
	put(dst, off, size, dc)
end

local function predTM(dst: buffer, off: number, size: number)
	local tl = buffer.readu8(dst, off - BPS - 1)
	for y = 0, size - 1 do
		local l = buffer.readu8(dst, off - 1 + y * BPS)
		local base = l - tl
		for x = 0, size - 1 do
			buffer.writeu8(dst, off + y * BPS + x, clip255(base + buffer.readu8(dst, off - BPS + x)))
		end
	end
end

local function predVE(dst: buffer, off: number, size: number)
	for j = 0, size - 1 do
		buffer.copy(dst, off + j * BPS, dst, off - BPS, size)
	end
end

local function predHE(dst: buffer, off: number, size: number)
	for j = 0, size - 1 do
		buffer.fill(dst, off + j * BPS, buffer.readu8(dst, off - 1 + j * BPS), size)
	end
end

-- 4x4 luma predictors (mode 0..9) on scratch at off
local function pred4(dst: buffer, off: number, mode: number)
	local function T(i: number): number
		return buffer.readu8(dst, off - BPS + i)
	end
	local function L(j: number): number
		return buffer.readu8(dst, off - 1 + j * BPS)
	end
	local function D(x: number, y: number, v: number)
		buffer.writeu8(dst, off + y * BPS + x, v)
	end
	if mode == 0 then
		-- B_DC_PRED
		local dc = 4
		for i = 0, 3 do
			dc += T(i) + L(i)
		end
		dc = rshift(dc, 3)
		for j = 0, 3 do
			buffer.fill(dst, off + j * BPS, dc, 4)
		end
	elseif mode == 1 then
		predTM(dst, off, 4)
	elseif mode == 2 then
		-- B_VE_PRED (note the smoothing with top-left/top-right)
		local xm1 = buffer.readu8(dst, off - BPS - 1)
		local vals = {
			avg3(xm1, T(0), T(1)),
			avg3(T(0), T(1), T(2)),
			avg3(T(1), T(2), T(3)),
			avg3(T(2), T(3), T(4)),
		}
		for j = 0, 3 do
			for i = 0, 3 do
				D(i, j, vals[i + 1])
			end
		end
	elseif mode == 3 then
		-- B_HE_PRED
		local A = buffer.readu8(dst, off - BPS - 1)
		local B = L(0)
		local C = L(1)
		local Dd = L(2)
		local E = L(3)
		local function fillRow(j: number, v: number)
			for i = 0, 3 do
				D(i, j, v)
			end
		end
		fillRow(0, avg3(A, B, C))
		fillRow(1, avg3(B, C, Dd))
		fillRow(2, avg3(C, Dd, E))
		fillRow(3, avg3(Dd, E, E))
	elseif mode == 4 then
		-- B_RD_PRED
		local I, J, K, Ll = L(0), L(1), L(2), L(3)
		local X = buffer.readu8(dst, off - BPS - 1)
		local A, B, C, Dd = T(0), T(1), T(2), T(3)
		D(0, 3, avg3(J, K, Ll))
		local v = avg3(I, J, K)
		D(1, 3, v)
		D(0, 2, v)
		v = avg3(X, I, J)
		D(2, 3, v)
		D(1, 2, v)
		D(0, 1, v)
		v = avg3(A, X, I)
		D(3, 3, v)
		D(2, 2, v)
		D(1, 1, v)
		D(0, 0, v)
		v = avg3(B, A, X)
		D(3, 2, v)
		D(2, 1, v)
		D(1, 0, v)
		v = avg3(C, B, A)
		D(3, 1, v)
		D(2, 0, v)
		D(3, 0, avg3(Dd, C, B))
	elseif mode == 5 then
		-- B_VR_PRED
		local I, J, K = L(0), L(1), L(2)
		local X = buffer.readu8(dst, off - BPS - 1)
		local A, B, C, Dd = T(0), T(1), T(2), T(3)
		local v = avg2(X, A)
		D(0, 0, v)
		D(1, 2, v)
		v = avg2(A, B)
		D(1, 0, v)
		D(2, 2, v)
		v = avg2(B, C)
		D(2, 0, v)
		D(3, 2, v)
		D(3, 0, avg2(C, Dd))
		D(0, 3, avg3(K, J, I))
		D(0, 2, avg3(J, I, X))
		v = avg3(I, X, A)
		D(0, 1, v)
		D(1, 3, v)
		v = avg3(X, A, B)
		D(1, 1, v)
		D(2, 3, v)
		v = avg3(A, B, C)
		D(2, 1, v)
		D(3, 3, v)
		D(3, 1, avg3(B, C, Dd))
	elseif mode == 6 then
		-- B_LD_PRED
		local A, B, C, Dd = T(0), T(1), T(2), T(3)
		local E, F, G, H = T(4), T(5), T(6), T(7)
		D(0, 0, avg3(A, B, C))
		local v = avg3(B, C, Dd)
		D(1, 0, v)
		D(0, 1, v)
		v = avg3(C, Dd, E)
		D(2, 0, v)
		D(1, 1, v)
		D(0, 2, v)
		v = avg3(Dd, E, F)
		D(3, 0, v)
		D(2, 1, v)
		D(1, 2, v)
		D(0, 3, v)
		v = avg3(E, F, G)
		D(3, 1, v)
		D(2, 2, v)
		D(1, 3, v)
		v = avg3(F, G, H)
		D(3, 2, v)
		D(2, 3, v)
		D(3, 3, avg3(G, H, H))
	elseif mode == 7 then
		-- B_VL_PRED
		local A, B, C, Dd = T(0), T(1), T(2), T(3)
		local E, F, G, H = T(4), T(5), T(6), T(7)
		D(0, 0, avg2(A, B))
		local v = avg2(B, C)
		D(1, 0, v)
		D(0, 2, v)
		v = avg2(C, Dd)
		D(2, 0, v)
		D(1, 2, v)
		v = avg2(Dd, E)
		D(3, 0, v)
		D(2, 2, v)
		D(0, 1, avg3(A, B, C))
		v = avg3(B, C, Dd)
		D(1, 1, v)
		D(0, 3, v)
		v = avg3(C, Dd, E)
		D(2, 1, v)
		D(1, 3, v)
		v = avg3(Dd, E, F)
		D(3, 1, v)
		D(2, 3, v)
		D(3, 2, avg3(E, F, G))
		D(3, 3, avg3(F, G, H))
	elseif mode == 8 then
		-- B_HD_PRED
		local I, J, K, Ll = L(0), L(1), L(2), L(3)
		local X = buffer.readu8(dst, off - BPS - 1)
		local A, B, C = T(0), T(1), T(2)
		local v = avg2(I, X)
		D(0, 0, v)
		D(2, 1, v)
		v = avg2(J, I)
		D(0, 1, v)
		D(2, 2, v)
		v = avg2(K, J)
		D(0, 2, v)
		D(2, 3, v)
		D(0, 3, avg2(Ll, K))
		D(3, 0, avg3(A, B, C))
		D(2, 0, avg3(X, A, B))
		v = avg3(I, X, A)
		D(1, 0, v)
		D(3, 1, v)
		v = avg3(J, I, X)
		D(1, 1, v)
		D(3, 2, v)
		v = avg3(K, J, I)
		D(1, 2, v)
		D(3, 3, v)
		D(1, 3, avg3(Ll, K, J))
	else
		-- B_HU_PRED
		local I, J, K, Ll = L(0), L(1), L(2), L(3)
		D(0, 0, avg2(I, J))
		local v = avg2(J, K)
		D(2, 0, v)
		D(0, 1, v)
		v = avg2(K, Ll)
		D(2, 1, v)
		D(0, 2, v)
		D(1, 0, avg3(I, J, K))
		v = avg3(J, K, Ll)
		D(3, 0, v)
		D(1, 1, v)
		v = avg3(K, Ll, Ll)
		D(3, 1, v)
		D(1, 2, v)
		D(3, 2, Ll)
		D(2, 2, Ll)
		D(0, 3, Ll)
		D(1, 3, Ll)
		D(2, 3, Ll)
		D(3, 3, Ll)
	end
end

-- 16x16 / chroma predictors with border availability
local function predBig(dst: buffer, off: number, size: number, mode: number, mbX: number, mbY: number)
	-- CheckMode mapping for DC
	if mode == 0 then
		predDC(dst, off, size, mbY > 0, mbX > 0)
	elseif mode == 1 then
		predTM(dst, off, size)
	elseif mode == 2 then
		predVE(dst, off, size)
	else
		predHE(dst, off, size)
	end
end

-- loop filters over plane buffers --------------------------------------------

local function doFilter2(buf: buffer, off: number, step: number)
	local p1 = buffer.readu8(buf, off - 2 * step)
	local p0 = buffer.readu8(buf, off - step)
	local q0 = buffer.readu8(buf, off)
	local q1 = buffer.readu8(buf, off + step)
	local a = 3 * (q0 - p0) + sclip1(p1 - q1)
	local a1 = sclip2(math.floor((a + 4) / 8))
	local a2 = sclip2(math.floor((a + 3) / 8))
	buffer.writeu8(buf, off - step, clip255(p0 + a2))
	buffer.writeu8(buf, off, clip255(q0 - a1))
end

local function doFilter4(buf: buffer, off: number, step: number)
	local p1 = buffer.readu8(buf, off - 2 * step)
	local p0 = buffer.readu8(buf, off - step)
	local q0 = buffer.readu8(buf, off)
	local q1 = buffer.readu8(buf, off + step)
	local a = 3 * (q0 - p0)
	local a1 = sclip2(math.floor((a + 4) / 8))
	local a2 = sclip2(math.floor((a + 3) / 8))
	local a3 = math.floor((a1 + 1) / 2) -- (a1 + 1) >> 1, arithmetic
	buffer.writeu8(buf, off - 2 * step, clip255(p1 + a3))
	buffer.writeu8(buf, off - step, clip255(p0 + a2))
	buffer.writeu8(buf, off, clip255(q0 - a1))
	buffer.writeu8(buf, off + step, clip255(q1 - a3))
end

local function doFilter6(buf: buffer, off: number, step: number)
	local p2 = buffer.readu8(buf, off - 3 * step)
	local p1 = buffer.readu8(buf, off - 2 * step)
	local p0 = buffer.readu8(buf, off - step)
	local q0 = buffer.readu8(buf, off)
	local q1 = buffer.readu8(buf, off + step)
	local q2 = buffer.readu8(buf, off + 2 * step)
	local a = sclip1(3 * (q0 - p0) + sclip1(p1 - q1))
	local a1 = math.floor((27 * a + 63) / 128)
	local a2 = math.floor((18 * a + 63) / 128)
	local a3 = math.floor((9 * a + 63) / 128)
	buffer.writeu8(buf, off - 3 * step, clip255(p2 + a3))
	buffer.writeu8(buf, off - 2 * step, clip255(p1 + a2))
	buffer.writeu8(buf, off - step, clip255(p0 + a1))
	buffer.writeu8(buf, off, clip255(q0 - a1))
	buffer.writeu8(buf, off + step, clip255(q1 - a2))
	buffer.writeu8(buf, off + 2 * step, clip255(q2 - a3))
end

local function hev(buf: buffer, off: number, step: number, thresh: number): boolean
	local p1 = buffer.readu8(buf, off - 2 * step)
	local p0 = buffer.readu8(buf, off - step)
	local q0 = buffer.readu8(buf, off)
	local q1 = buffer.readu8(buf, off + step)
	return math.abs(p1 - p0) > thresh or math.abs(q1 - q0) > thresh
end

local function needsFilter(buf: buffer, off: number, step: number, t: number): boolean
	local p1 = buffer.readu8(buf, off - 2 * step)
	local p0 = buffer.readu8(buf, off - step)
	local q0 = buffer.readu8(buf, off)
	local q1 = buffer.readu8(buf, off + step)
	return 4 * math.abs(p0 - q0) + math.abs(p1 - q1) <= t
end

local function needsFilter2(buf: buffer, off: number, step: number, t: number, it: number): boolean
	local p3 = buffer.readu8(buf, off - 4 * step)
	local p2 = buffer.readu8(buf, off - 3 * step)
	local p1 = buffer.readu8(buf, off - 2 * step)
	local p0 = buffer.readu8(buf, off - step)
	local q0 = buffer.readu8(buf, off)
	local q1 = buffer.readu8(buf, off + step)
	local q2 = buffer.readu8(buf, off + 2 * step)
	local q3 = buffer.readu8(buf, off + 3 * step)
	if 4 * math.abs(p0 - q0) + math.abs(p1 - q1) > t then
		return false
	end
	return math.abs(p3 - p2) <= it
		and math.abs(p2 - p1) <= it
		and math.abs(p1 - p0) <= it
		and math.abs(q3 - q2) <= it
		and math.abs(q2 - q1) <= it
		and math.abs(q1 - q0) <= it
end

local function filterLoop26(
	buf: buffer,
	off: number,
	hstride: number,
	vstride: number,
	size: number,
	thresh: number,
	ithresh: number,
	hevThresh: number
)
	local thresh2 = 2 * thresh + 1
	for _ = 1, size do
		if needsFilter2(buf, off, hstride, thresh2, ithresh) then
			if hev(buf, off, hstride, hevThresh) then
				doFilter2(buf, off, hstride)
			else
				doFilter6(buf, off, hstride)
			end
		end
		off += vstride
	end
end

local function filterLoop24(
	buf: buffer,
	off: number,
	hstride: number,
	vstride: number,
	size: number,
	thresh: number,
	ithresh: number,
	hevThresh: number
)
	local thresh2 = 2 * thresh + 1
	for _ = 1, size do
		if needsFilter2(buf, off, hstride, thresh2, ithresh) then
			if hev(buf, off, hstride, hevThresh) then
				doFilter2(buf, off, hstride)
			else
				doFilter4(buf, off, hstride)
			end
		end
		off += vstride
	end
end

-- ===========================================================================
-- decodeVP8: full lossy frame decode -> Y/U/V planes
-- ===========================================================================
local function decodeVP8(buf: buffer, base: number, len: number): (buffer?, buffer?, buffer?, number, number, number, number, string?)
	if len < 10 then
		return nil, nil, nil, 0, 0, 0, 0, "vp8: too short"
	end
	local b0 = buffer.readu8(buf, base)
	local b1 = buffer.readu8(buf, base + 1)
	local b2 = buffer.readu8(buf, base + 2)
	local bits = b0 + b1 * 256 + b2 * 65536
	local keyFrame = band(bits, 1) == 0
	local partitionLength = math.floor(bits / 32)
	if not keyFrame then
		return nil, nil, nil, 0, 0, 0, 0, "vp8: not a keyframe"
	end
	if buffer.readu8(buf, base + 3) ~= 0x9D
		or buffer.readu8(buf, base + 4) ~= 0x01
		or buffer.readu8(buf, base + 5) ~= 0x2A
	then
		return nil, nil, nil, 0, 0, 0, 0, "vp8: bad start code"
	end
	local width = band(buffer.readu8(buf, base + 6) + buffer.readu8(buf, base + 7) * 256, 0x3FFF)
	local height = band(buffer.readu8(buf, base + 8) + buffer.readu8(buf, base + 9) * 256, 0x3FFF)
	if width == 0 or height == 0 then
		return nil, nil, nil, 0, 0, 0, 0, "vp8: bad dimensions"
	end

	local hdrBase = base + 10
	local hdrSize = len - 10
	if partitionLength > hdrSize then
		return nil, nil, nil, 0, 0, 0, 0, "vp8: bad partition length"
	end
	local bd = bdInit(buf, hdrBase, partitionLength)

	local s: VP8State = {
		mbW = rshift(width + 15, 4),
		mbH = rshift(height + 15, 4),
		width = width,
		height = height,
		useSegment = false,
		updateMap = false,
		absoluteDelta = false,
		segQuant = { 0, 0, 0, 0 },
		segFilter = { 0, 0, 0, 0 },
		segProbs = { 255, 255, 255 },
		filterSimple = false,
		filterLevel = 0,
		filterSharpness = 0,
		useLfDelta = false,
		refLfDelta = { 0, 0, 0, 0 },
		modeLfDelta = { 0, 0, 0, 0 },
		filterType = 0,
		useSkipProba = false,
		skipP = 0,
		dqm = {},
		coeffs = table.create(4 * 8 * 3 * 11, 0),
		fstrengths = {},
	}

	bdBit(bd, 128) -- color space
	bdBit(bd, 128) -- clamp type

	-- segment header
	s.useSegment = bdBit(bd, 128) == 1
	if s.useSegment then
		s.updateMap = bdBit(bd, 128) == 1
		if bdBit(bd, 128) == 1 then
			s.absoluteDelta = bdBit(bd, 128) == 1
			for i = 1, 4 do
				s.segQuant[i] = if bdBit(bd, 128) == 1 then bdSignedValue(bd, 7) else 0
			end
			for i = 1, 4 do
				s.segFilter[i] = if bdBit(bd, 128) == 1 then bdSignedValue(bd, 6) else 0
			end
		end
		if s.updateMap then
			for i = 1, 3 do
				s.segProbs[i] = if bdBit(bd, 128) == 1 then bdVal(bd, 8) else 255
			end
		end
	end

	-- filter header
	s.filterSimple = bdBit(bd, 128) == 1
	s.filterLevel = bdVal(bd, 6)
	s.filterSharpness = bdVal(bd, 3)
	s.useLfDelta = bdBit(bd, 128) == 1
	if s.useLfDelta then
		if bdBit(bd, 128) == 1 then
			for i = 1, 4 do
				if bdBit(bd, 128) == 1 then
					s.refLfDelta[i] = bdSignedValue(bd, 6)
				end
			end
			for i = 1, 4 do
				if bdBit(bd, 128) == 1 then
					s.modeLfDelta[i] = bdSignedValue(bd, 6)
				end
			end
		end
	end
	s.filterType = if s.filterLevel == 0 then 0 elseif s.filterSimple then 1 else 2

	-- partitions
	local numParts = lshift(1, bdVal(bd, 2))
	local partsBase = hdrBase + partitionLength
	local partsSize = hdrSize - partitionLength
	if partsSize < 3 * (numParts - 1) then
		return nil, nil, nil, 0, 0, 0, 0, "vp8: truncated partitions"
	end
	local parts: { BoolDec } = {}
	do
		local szBase = partsBase
		local partStart = partsBase + (numParts - 1) * 3
		local sizeLeft = partsSize - (numParts - 1) * 3
		for p = 1, numParts - 1 do
			local psize = buffer.readu8(buf, szBase)
				+ buffer.readu8(buf, szBase + 1) * 256
				+ buffer.readu8(buf, szBase + 2) * 65536
			if psize > sizeLeft then
				psize = sizeLeft
			end
			parts[p] = bdInit(buf, partStart, psize)
			partStart += psize
			sizeLeft -= psize
			szBase += 3
		end
		parts[numParts] = bdInit(buf, partStart, sizeLeft)
	end

	-- quant
	vp8ParseQuant(bd, s)

	-- update proba flag (keyframe: ignore)
	bdBit(bd, 128)

	-- coefficient probabilities
	for t = 0, 3 do
		for b = 0, 7 do
			for c = 0, 2 do
				for p = 0, 10 do
					local idx = coeffIdx(t, b, c, p)
					if bdBit(bd, COEFFS_UPDATE_PROBA[idx]) == 1 then
						s.coeffs[idx] = bdVal(bd, 8)
					else
						s.coeffs[idx] = COEFFS_PROBA0[idx]
					end
				end
			end
		end
	end

	s.useSkipProba = bdBit(bd, 128) == 1
	if s.useSkipProba then
		s.skipP = bdVal(bd, 8)
	end

	-- precompute filter strengths
	for seg = 0, 3 do
		local rows: { { number } } = {}
		for i4x4 = 0, 1 do
			local baseLevel: number
			if s.useSegment then
				baseLevel = s.segFilter[seg + 1]
				if not s.absoluteDelta then
					baseLevel += s.filterLevel
				end
			else
				baseLevel = s.filterLevel
			end
			local level = baseLevel
			if s.useLfDelta then
				level += s.refLfDelta[1]
				if i4x4 == 1 then
					level += s.modeLfDelta[1]
				end
			end
			level = if level < 0 then 0 elseif level > 63 then 63 else level
			if level > 0 then
				local ilevel = level
				if s.filterSharpness > 0 then
					if s.filterSharpness > 4 then
						ilevel = rshift(ilevel, 2)
					else
						ilevel = rshift(ilevel, 1)
					end
					if ilevel > 9 - s.filterSharpness then
						ilevel = 9 - s.filterSharpness
					end
				end
				if ilevel < 1 then
					ilevel = 1
				end
				rows[i4x4 + 1] = { 2 * level + ilevel, ilevel, if level >= 40 then 2 elseif level >= 15 then 1 else 0, i4x4 }
			else
				rows[i4x4 + 1] = { 0, 0, 0, i4x4 }
			end
		end
		s.fstrengths[seg + 1] = rows
	end

	-- plane buffers
	local mbW = s.mbW
	local mbH = s.mbH
	local yStride = mbW * 16
	local uvStride = mbW * 8
	local yPlane = buffer.create(yStride * mbH * 16)
	local uPlane = buffer.create(uvStride * mbH * 8)
	local vPlane = buffer.create(uvStride * mbH * 8)

	-- scratch with borders (like libwebp yuv_b)
	local scratch = buffer.create(BPS * 28)

	-- per-MB row state
	local mbNz = table.create(mbW + 1, 0) -- [1] = left, [x+2] = column x
	local mbNzDc = table.create(mbW + 1, 0)
	local intraT = table.create(4 * mbW, 0) -- B_DC_PRED = 0
	local intraL = { 0, 0, 0, 0 }
	-- top samples per MB: y (16+4 incl top-right), u(8), v(8)
	local topY: { buffer } = {}
	local topU: { buffer } = {}
	local topV: { buffer } = {}
	for x = 1, mbW do
		topY[x] = buffer.create(20)
		topU[x] = buffer.create(8)
		topV[x] = buffer.create(8)
	end

	-- per-MB data for the current row
	type MBData = {
		isI4x4: boolean,
		imodes: { number },
		uvmode: number,
		segment: number,
		skip: boolean,
		nonZeroY: number,
		nonZeroUV: number,
		coeffs: { number }, -- 384 values
	}
	local rowMB: { MBData } = table.create(mbW)
	-- filter info per MB (whole frame)
	local fInfo: { { number } } = table.create(mbW * mbH)

	for mbY = 0, mbH - 1 do
		-- parse intra modes for the row (from partition 0 / header bd)
		intraL[1], intraL[2], intraL[3], intraL[4] = 0, 0, 0, 0
		for mbX = 0, mbW - 1 do
			local block: MBData = {
				isI4x4 = false,
				imodes = table.create(16, 0),
				uvmode = 0,
				segment = 0,
				skip = false,
				nonZeroY = 0,
				nonZeroUV = 0,
				coeffs = table.create(384, 0),
			}
			if s.updateMap then
				block.segment = if bdBit(bd, s.segProbs[1]) == 0
					then bdBit(bd, s.segProbs[2])
					else bdBit(bd, s.segProbs[3]) + 2
			end
			if s.useSkipProba then
				block.skip = bdBit(bd, s.skipP) == 1
			end
			block.isI4x4 = bdBit(bd, 145) == 0
			if not block.isI4x4 then
				local ymode = if bdBit(bd, 156) == 1
					then (if bdBit(bd, 128) == 1 then 1 else 3)
					else (if bdBit(bd, 163) == 1 then 2 else 0)
				block.imodes[1] = ymode
				for k = 0, 3 do
					intraT[4 * mbX + k + 1] = ymode
					intraL[k + 1] = ymode
				end
			else
				for y = 0, 3 do
					local ymode = intraL[y + 1]
					for x = 0, 3 do
						local pb = (intraT[4 * mbX + x + 1] * 10 + ymode) * 9
						ymode = if bdBit(bd, BMODES_PROBA[pb + 1]) == 0
							then 0
							elseif bdBit(bd, BMODES_PROBA[pb + 2]) == 0 then 1
							elseif bdBit(bd, BMODES_PROBA[pb + 3]) == 0 then 2
							elseif bdBit(bd, BMODES_PROBA[pb + 4]) == 0 then (
								if bdBit(bd, BMODES_PROBA[pb + 5]) == 0
									then 3
									else (if bdBit(bd, BMODES_PROBA[pb + 6]) == 0 then 4 else 5)
							)
							else (
								if bdBit(bd, BMODES_PROBA[pb + 7]) == 0
									then 6
									else (
										if bdBit(bd, BMODES_PROBA[pb + 8]) == 0
											then 7
											else (if bdBit(bd, BMODES_PROBA[pb + 9]) == 0 then 8 else 9)
									)
							)
						intraT[4 * mbX + x + 1] = ymode
						block.imodes[y * 4 + x + 1] = ymode
					end
					intraL[y + 1] = block.imodes[y * 4 + 4]
				end
			end
			block.uvmode = if bdBit(bd, 142) == 0
				then 0
				elseif bdBit(bd, 114) == 0 then 2
				elseif bdBit(bd, 183) == 1 then 1
				else 3
			rowMB[mbX + 1] = block
		end

		-- parse residuals for the row from the token partition
		local tokenBd = parts[band(mbY, numParts - 1) + 1]
		mbNz[1] = 0
		mbNzDc[1] = 0
		for mbX = 0, mbW - 1 do
			local block = rowMB[mbX + 1]
			local q = s.dqm[block.segment + 1]
			local skip = if s.useSkipProba then block.skip else false
			if not skip then
				-- parse residuals
				local coeffsArr = block.coeffs
				local first: number
				local acType: number -- 0 = type0 (with Y2), 3 = type 3
				if not block.isI4x4 then
					local dc = table.create(16, 0)
					local ctx = mbNzDc[1] + mbNzDc[mbX + 2]
					local nzDc = getCoeffs(tokenBd, s.coeffs, 1, ctx, q[3], q[4], 0, dc, 0)
					local dcNz = if nzDc > 0 then 1 else 0
					mbNzDc[1] = dcNz
					mbNzDc[mbX + 2] = dcNz
					if nzDc > 1 then
						transformWHT(dc, coeffsArr)
					else
						local dc0 = math.floor((dc[1] + 3) / 8)
						for i = 0, 15 do
							coeffsArr[i * 16 + 1] = dc0
						end
					end
					first = 1
					acType = 0
				else
					first = 0
					acType = 3
				end

				local tnz = band(mbNz[mbX + 2], 0x0F)
				local lnz = band(mbNz[1], 0x0F)
				local nonZeroY = 0
				for y = 0, 3 do
					local l = band(lnz, 1)
					local nzCoeffs = 0
					for x = 0, 3 do
						local ctx = l + band(tnz, 1)
						local blockIdx = y * 4 + x
						local nz = getCoeffs(
							tokenBd,
							s.coeffs,
							acType,
							ctx,
							q[1],
							q[2],
							first,
							coeffsArr,
							blockIdx * 16
						)
						l = if nz > first then 1 else 0
						tnz = bor(rshift(tnz, 1), lshift(l, 7))
						local dcNzVal = if coeffsArr[blockIdx * 16 + 1] ~= 0 then 1 else 0
						nzCoeffs = nzCoeffs * 4
							+ (if nz > 3 then 3 elseif nz > 1 then 2 else dcNzVal)
						-- NzCodeBits
					end
					tnz = rshift(tnz, 4)
					lnz = bor(rshift(lnz, 1), lshift(l, 7))
					nonZeroY = (nonZeroY * 256 + nzCoeffs) % 4294967296
				end
				local outTnz = tnz
				local outLnz = rshift(lnz, 4)

				local nonZeroUV = 0
				for ch = 0, 2, 2 do
					local nzCoeffs = 0
					tnz = rshift(mbNz[mbX + 2], 4 + ch)
					lnz = rshift(mbNz[1], 4 + ch)
					for y = 0, 1 do
						local l = band(lnz, 1)
						for x = 0, 1 do
							local ctx = l + band(tnz, 1)
							local blockIdx = 16 + ch * 2 + y * 2 + x
							local nz = getCoeffs(
								tokenBd,
								s.coeffs,
								2,
								ctx,
								q[5],
								q[6],
								0,
								coeffsArr,
								blockIdx * 16
							)
							l = if nz > 0 then 1 else 0
							tnz = bor(rshift(tnz, 1), lshift(l, 3))
							local dcNzVal = if coeffsArr[blockIdx * 16 + 1] ~= 0 then 1 else 0
							nzCoeffs = nzCoeffs * 4
								+ (if nz > 3 then 3 elseif nz > 1 then 2 else dcNzVal)
						end
						tnz = rshift(tnz, 2)
						lnz = bor(rshift(lnz, 1), lshift(l, 5))
					end
					nonZeroUV = bor(nonZeroUV, lshift(nzCoeffs, 4 * ch))
					outTnz = bor(outTnz, lshift(lshift(tnz, 4), ch))
					outLnz = bor(outLnz, lshift(band(lnz, 0xF0), ch))
				end
				mbNz[mbX + 2] = outTnz
				mbNz[1] = outLnz

				block.nonZeroY = nonZeroY
				block.nonZeroUV = nonZeroUV
				block.skip = (nonZeroY == 0 and nonZeroUV == 0)
			else
				mbNz[1] = 0
				mbNz[mbX + 2] = 0
				if not block.isI4x4 then
					mbNzDc[1] = 0
					mbNzDc[mbX + 2] = 0
				end
				block.nonZeroY = 0
				block.nonZeroUV = 0
				block.skip = true
			end

			-- filter info
			if s.filterType > 0 then
				local fs = s.fstrengths[block.segment + 1][(if block.isI4x4 then 1 else 0) + 1]
				local inner = if block.isI4x4 then 1 else 0
				if not block.skip then
					inner = 1
				end
				fInfo[mbY * mbW + mbX + 1] = { fs[1], fs[2], fs[3], inner }
			end
		end

		-- reconstruct the row
		for j = 0, 15 do
			buffer.writeu8(scratch, YOFF + j * BPS - 1, 129)
		end
		for j = 0, 7 do
			buffer.writeu8(scratch, UOFF + j * BPS - 1, 129)
			buffer.writeu8(scratch, VOFF + j * BPS - 1, 129)
		end
		if mbY > 0 then
			buffer.writeu8(scratch, YOFF - 1 - BPS, 129)
			buffer.writeu8(scratch, UOFF - 1 - BPS, 129)
			buffer.writeu8(scratch, VOFF - 1 - BPS, 129)
		else
			buffer.fill(scratch, YOFF - BPS - 1, 127, 21)
			buffer.fill(scratch, UOFF - BPS - 1, 127, 9)
			buffer.fill(scratch, VOFF - BPS - 1, 127, 9)
		end

		for mbX = 0, mbW - 1 do
			local block = rowMB[mbX + 1]
			-- rotate in the left samples
			if mbX > 0 then
				for j = -1, 15 do
					buffer.copy(scratch, YOFF + j * BPS - 4, scratch, YOFF + j * BPS + 12, 4)
				end
				for j = -1, 7 do
					buffer.copy(scratch, UOFF + j * BPS - 4, scratch, UOFF + j * BPS + 4, 4)
					buffer.copy(scratch, VOFF + j * BPS - 4, scratch, VOFF + j * BPS + 4, 4)
				end
			end
			-- top samples
			if mbY > 0 then
				buffer.copy(scratch, YOFF - BPS, topY[mbX + 1], 0, 16)
				buffer.copy(scratch, UOFF - BPS, topU[mbX + 1], 0, 8)
				buffer.copy(scratch, VOFF - BPS, topV[mbX + 1], 0, 8)
			end
			local coeffsArr = block.coeffs
			local bits2 = block.nonZeroY
			if block.isI4x4 then
				if mbY > 0 then
					if mbX >= mbW - 1 then
						local v = buffer.readu8(topY[mbX + 1], 15)
						for k = 0, 3 do
							buffer.writeu8(scratch, YOFF - BPS + 16 + k, v)
						end
					else
						buffer.copy(scratch, YOFF - BPS + 16, topY[mbX + 2], 0, 4)
					end
				end
				-- replicate the top-right pixels for the lower block rows
				-- (rows 3, 7 and 11 of the scratch, i.e. just above each row
				-- of 4x4 blocks; matches top_right[BPS,2*BPS,3*BPS] in C with
				-- uint32 pointer arithmetic)
				for k = 0, 3 do
					local v = buffer.readu8(scratch, YOFF - BPS + 16 + k)
					buffer.writeu8(scratch, YOFF - BPS + 16 + k + 4 * BPS, v)
					buffer.writeu8(scratch, YOFF - BPS + 16 + k + 8 * BPS, v)
					buffer.writeu8(scratch, YOFF - BPS + 16 + k + 12 * BPS, v)
				end
				for n = 0, 15 do
					local dstOff = YOFF + band(n, 3) * 4 + rshift(n, 2) * 4 * BPS
					pred4(scratch, dstOff, block.imodes[n + 1])
					local code = band(math.floor(bits2 / 2 ^ 30), 3)
					if code == 3 then
						transformOne(coeffsArr, n * 16, scratch, dstOff)
					elseif code == 2 then
						transformAC3(coeffsArr, n * 16, scratch, dstOff)
					elseif code == 1 then
						transformDC(coeffsArr, n * 16, scratch, dstOff)
					end
					bits2 = (bits2 * 4) % 4294967296
				end
			else
				local mode = block.imodes[1]
				-- CheckMode for DC handled inside predBig
				predBig(scratch, YOFF, 16, mode, mbX, mbY)
				if bits2 ~= 0 then
					for n = 0, 15 do
						local dstOff = YOFF + band(n, 3) * 4 + rshift(n, 2) * 4 * BPS
						local code = band(math.floor(bits2 / 2 ^ 30), 3)
						if code == 3 then
							transformOne(coeffsArr, n * 16, scratch, dstOff)
						elseif code == 2 then
							transformAC3(coeffsArr, n * 16, scratch, dstOff)
						elseif code == 1 then
							transformDC(coeffsArr, n * 16, scratch, dstOff)
						end
						bits2 = (bits2 * 4) % 4294967296
					end
				end
			end
			-- chroma
			local bitsUV = block.nonZeroUV
			predBig(scratch, UOFF, 8, block.uvmode, mbX, mbY)
			predBig(scratch, VOFF, 8, block.uvmode, mbX, mbY)
			-- U
			local function doUV(bitsCh: number, baseOff: number, coefBase: number)
				if band(bitsCh, 0xFF) == 0 then
					return
				end
				if band(bitsCh, 0xAA) ~= 0 then
					for n = 0, 3 do
						local dstOff = baseOff + band(n, 1) * 4 + rshift(n, 1) * 4 * BPS
						transformOne(coeffsArr, (coefBase + n) * 16, scratch, dstOff)
					end
				else
					for n = 0, 3 do
						if coeffsArr[(coefBase + n) * 16 + 1] ~= 0 then
							local dstOff = baseOff + band(n, 1) * 4 + rshift(n, 1) * 4 * BPS
							transformDC(coeffsArr, (coefBase + n) * 16, scratch, dstOff)
						end
					end
				end
			end
			doUV(bitsUV, UOFF, 16)
			doUV(rshift(bitsUV, 8), VOFF, 20)

			-- stash top samples for next row
			if mbY < mbH - 1 then
				buffer.copy(topY[mbX + 1], 0, scratch, YOFF + 15 * BPS, 16)
				buffer.copy(topU[mbX + 1], 0, scratch, UOFF + 7 * BPS, 8)
				buffer.copy(topV[mbX + 1], 0, scratch, VOFF + 7 * BPS, 8)
			end

			-- copy to planes
			local yOut = mbY * 16 * yStride + mbX * 16
			for j = 0, 15 do
				buffer.copy(yPlane, yOut + j * yStride, scratch, YOFF + j * BPS, 16)
			end
			local uvOut = mbY * 8 * uvStride + mbX * 8
			for j = 0, 7 do
				buffer.copy(uPlane, uvOut + j * uvStride, scratch, UOFF + j * BPS, 8)
				buffer.copy(vPlane, uvOut + j * uvStride, scratch, VOFF + j * BPS, 8)
			end
		end
	end

	-- loop filter (raster order over the full planes)
	if s.filterType > 0 then
		for mbY = 0, mbH - 1 do
			for mbX = 0, mbW - 1 do
				local fi = fInfo[mbY * mbW + mbX + 1]
				local limit = fi[1]
				if limit > 0 then
					local ilevel = fi[2]
					local hevT = fi[3]
					local inner = fi[4] == 1
					local yOff = mbY * 16 * yStride + mbX * 16
					if s.filterType == 1 then
						if mbX > 0 then
							for j = 0, 15 do
								if needsFilter(yPlane, yOff + j * yStride, 1, 2 * (limit + 4) + 1) then
									doFilter2(yPlane, yOff + j * yStride, 1)
								end
							end
						end
						if inner then
							for k = 4, 12, 4 do
								for j = 0, 15 do
									if needsFilter(yPlane, yOff + j * yStride + k, 1, 2 * limit + 1) then
										doFilter2(yPlane, yOff + j * yStride + k, 1)
									end
								end
							end
						end
						if mbY > 0 then
							for i = 0, 15 do
								if needsFilter(yPlane, yOff + i, yStride, 2 * (limit + 4) + 1) then
									doFilter2(yPlane, yOff + i, yStride)
								end
							end
						end
						if inner then
							for k = 4, 12, 4 do
								for i = 0, 15 do
									if needsFilter(yPlane, yOff + k * yStride + i, yStride, 2 * limit + 1) then
										doFilter2(yPlane, yOff + k * yStride + i, yStride)
									end
								end
							end
						end
					else
						local uvOff = mbY * 8 * uvStride + mbX * 8
						if mbX > 0 then
							filterLoop26(yPlane, yOff, 1, yStride, 16, limit + 4, ilevel, hevT)
							filterLoop26(uPlane, uvOff, 1, uvStride, 8, limit + 4, ilevel, hevT)
							filterLoop26(vPlane, uvOff, 1, uvStride, 8, limit + 4, ilevel, hevT)
						end
						if inner then
							for k = 4, 12, 4 do
								filterLoop24(yPlane, yOff + k, 1, yStride, 16, limit, ilevel, hevT)
							end
							filterLoop24(uPlane, uvOff + 4, 1, uvStride, 8, limit, ilevel, hevT)
							filterLoop24(vPlane, uvOff + 4, 1, uvStride, 8, limit, ilevel, hevT)
						end
						if mbY > 0 then
							filterLoop26(yPlane, yOff, yStride, 1, 16, limit + 4, ilevel, hevT)
							filterLoop26(uPlane, uvOff, uvStride, 1, 8, limit + 4, ilevel, hevT)
							filterLoop26(vPlane, uvOff, uvStride, 1, 8, limit + 4, ilevel, hevT)
						end
						if inner then
							for k = 4, 12, 4 do
								filterLoop24(yPlane, yOff + k * yStride, yStride, 1, 16, limit, ilevel, hevT)
							end
							filterLoop24(uPlane, uvOff + 4 * uvStride, uvStride, 1, 8, limit, ilevel, hevT)
							filterLoop24(vPlane, uvOff + 4 * uvStride, uvStride, 1, 8, limit, ilevel, hevT)
						end
					end
				end
			end
		end
	end

	return yPlane, uPlane, vPlane, width, height, yStride, uvStride, nil
end

-- ===========================================================================
-- YUV -> RGBA with fancy upsampling (libwebp-exact)
-- ===========================================================================
local function yuvToRgba(y: number, u: number, v: number, out: buffer, off: number)
	local function multHi(a: number, b: number): number
		return math.floor(a * b / 256)
	end
	local function clip8(x: number): number
		return if x < 0 then 0 elseif x >= 16384 then 255 else rshift(x, 6)
	end
	buffer.writeu8(out, off, clip8(multHi(y, 19077) + multHi(v, 26149) - 14234))
	buffer.writeu8(out, off + 1, clip8(multHi(y, 19077) - multHi(u, 6419) - multHi(v, 13320) + 8708))
	buffer.writeu8(out, off + 2, clip8(multHi(y, 19077) + multHi(u, 33050) - 17685))
	buffer.writeu8(out, off + 3, 255)
end

local function upsampleFrame(
	yPlane: buffer,
	uPlane: buffer,
	vPlane: buffer,
	width: number,
	height: number,
	yStride: number,
	uvStride: number
): buffer
	local rgba = buffer.create(width * height * 4)
	local uvW = rshift(width + 1, 1)

	-- one row pair; bottomRow < 0 means absent
	local function rowPair(topRow: number, bottomRow: number, topUVRow: number, curUVRow: number)
		local topYBase = topRow * yStride
		local botYBase = if bottomRow >= 0 then bottomRow * yStride else 0
		local tuB = topUVRow * uvStride
		local cuB = curUVRow * uvStride
		local topDst = topRow * width * 4
		local botDst = if bottomRow >= 0 then bottomRow * width * 4 else 0
		local lastPixelPair = rshift(width - 1, 1)
		local tlU = buffer.readu8(uPlane, tuB)
		local tlV = buffer.readu8(vPlane, tuB)
		local lU = buffer.readu8(uPlane, cuB)
		local lV = buffer.readu8(vPlane, cuB)
		do
			local u0 = rshift(3 * tlU + lU + 2, 2)
			local v0 = rshift(3 * tlV + lV + 2, 2)
			yuvToRgba(buffer.readu8(yPlane, topYBase), u0, v0, rgba, topDst)
			if bottomRow >= 0 then
				local u1 = rshift(3 * lU + tlU + 2, 2)
				local v1 = rshift(3 * lV + tlV + 2, 2)
				yuvToRgba(buffer.readu8(yPlane, botYBase), u1, v1, rgba, botDst)
			end
		end
		for x = 1, lastPixelPair do
			local tU = buffer.readu8(uPlane, tuB + x)
			local tV = buffer.readu8(vPlane, tuB + x)
			local cU = buffer.readu8(uPlane, cuB + x)
			local cV = buffer.readu8(vPlane, cuB + x)
			local avgU = tlU + tU + lU + cU + 8
			local avgV = tlV + tV + lV + cV + 8
			local diag12U = rshift(avgU + 2 * (tU + lU), 3)
			local diag12V = rshift(avgV + 2 * (tV + lV), 3)
			local diag03U = rshift(avgU + 2 * (tlU + cU), 3)
			local diag03V = rshift(avgV + 2 * (tlV + cV), 3)
			do
				local u0 = rshift(diag12U + tlU, 1)
				local v0 = rshift(diag12V + tlV, 1)
				local u1 = rshift(diag03U + tU, 1)
				local v1 = rshift(diag03V + tV, 1)
				yuvToRgba(buffer.readu8(yPlane, topYBase + 2 * x - 1), u0, v0, rgba, topDst + (2 * x - 1) * 4)
				if 2 * x < width then
					yuvToRgba(buffer.readu8(yPlane, topYBase + 2 * x), u1, v1, rgba, topDst + 2 * x * 4)
				end
			end
			if bottomRow >= 0 then
				local u0 = rshift(diag03U + lU, 1)
				local v0 = rshift(diag03V + lV, 1)
				local u1 = rshift(diag12U + cU, 1)
				local v1 = rshift(diag12V + cV, 1)
				yuvToRgba(buffer.readu8(yPlane, botYBase + 2 * x - 1), u0, v0, rgba, botDst + (2 * x - 1) * 4)
				if 2 * x < width then
					yuvToRgba(buffer.readu8(yPlane, botYBase + 2 * x), u1, v1, rgba, botDst + 2 * x * 4)
				end
			end
			tlU = tU
			tlV = tV
			lU = cU
			lV = cV
		end
		if band(width, 1) == 0 then
			local u0 = rshift(3 * tlU + lU + 2, 2)
			local v0 = rshift(3 * tlV + lV + 2, 2)
			yuvToRgba(buffer.readu8(yPlane, topYBase + width - 1), u0, v0, rgba, topDst + (width - 1) * 4)
			if bottomRow >= 0 then
				local u1 = rshift(3 * lU + tlU + 2, 2)
				local v1 = rshift(3 * lV + tlV + 2, 2)
				yuvToRgba(buffer.readu8(yPlane, botYBase + width - 1), u1, v1, rgba, botDst + (width - 1) * 4)
			end
		end
		local _ = uvW
	end

	local uvH = rshift(height + 1, 1)
	-- first row: chroma rows (0, 0)
	rowPair(0, -1, 0, 0)
	for j = 0, uvH - 2 do
		rowPair(2 * j + 1, 2 * j + 2, j, j + 1)
	end
	if band(height, 1) == 0 then
		rowPair(height - 1, -1, uvH - 1, uvH - 1)
	end
	return rgba
end

-- ===========================================================================
-- ALPH chunk
-- ===========================================================================
local function decodeAlpha(buf: buffer, base: number, len: number, width: number, height: number, rgba: buffer): string?
	if len < 1 then
		return "alpha: empty chunk"
	end
	local h = buffer.readu8(buf, base)
	local method = band(h, 3)
	local filter = band(rshift(h, 2), 3)
	local pre = band(rshift(h, 4), 3)
	local alpha: buffer
	if method == 0 then
		if len - 1 < width * height then
			return "alpha: truncated raw data"
		end
		alpha = buffer.create(width * height)
		buffer.copy(alpha, 0, buf, base + 1, width * height)
	elseif method == 1 then
		local data, _, _, err = decodeVP8L(buf, base + 1, len - 1, true, width, height)
		if not data then
			return "alpha: " .. tostring(err)
		end
		local pixels = data :: { number }
		alpha = buffer.create(width * height)
		for i = 1, width * height do
			buffer.writeu8(alpha, i - 1, band(math.floor(pixels[i] / 256), 255))
		end
	else
		return "alpha: unknown method " .. method
	end
	-- unfilter (per libwebp dsp/filters.c semantics, row by row)
	if filter ~= 0 then
		for y = 0, height - 1 do
			local rowBase = y * width
			local prevBase = rowBase - width
			if filter == 1 then
				-- horizontal: first pixel predicted from prev row's first pixel
				local pred = if y > 0 then buffer.readu8(alpha, prevBase) else 0
				for x = 0, width - 1 do
					local v = band(pred + buffer.readu8(alpha, rowBase + x), 255)
					buffer.writeu8(alpha, rowBase + x, v)
					pred = v
				end
			elseif filter == 2 then
				-- vertical (first row decays to horizontal)
				if y == 0 then
					local pred = 0
					for x = 0, width - 1 do
						local v = band(pred + buffer.readu8(alpha, x), 255)
						buffer.writeu8(alpha, x, v)
						pred = v
					end
				else
					for x = 0, width - 1 do
						buffer.writeu8(
							alpha,
							rowBase + x,
							band(buffer.readu8(alpha, rowBase + x) + buffer.readu8(alpha, prevBase + x), 255)
						)
					end
				end
			else
				-- gradient (first row decays to horizontal)
				if y == 0 then
					local pred = 0
					for x = 0, width - 1 do
						local v = band(pred + buffer.readu8(alpha, x), 255)
						buffer.writeu8(alpha, x, v)
						pred = v
					end
				else
					local top = buffer.readu8(alpha, prevBase)
					local topLeft = top
					local left = top
					for x = 0, width - 1 do
						top = buffer.readu8(alpha, prevBase + x)
						local g = clip255(left + top - topLeft)
						left = band(buffer.readu8(alpha, rowBase + x) + g, 255)
						topLeft = top
						buffer.writeu8(alpha, rowBase + x, left)
					end
				end
			end
		end
	end
	if pre == 1 then
		print("[WebPDecoder] note: preprocessed alpha levels not smoothed (cosmetic only).")
	end
	-- write into RGBA
	for i = 0, width * height - 1 do
		buffer.writeu8(rgba, i * 4 + 3, buffer.readu8(alpha, i))
	end
	return nil
end

-- ===========================================================================
-- container
-- ===========================================================================
local function isWebP(bytes: buffer): boolean
	if buffer.len(bytes) < 12 then
		return false
	end
	return buffer.readstring(bytes, 0, 4) == "RIFF" and buffer.readstring(bytes, 8, 4) == "WEBP"
end

local function argbToRgbaBuffer(data: { number }, width: number, height: number): buffer
	local rgba = buffer.create(width * height * 4)
	for i = 0, width * height - 1 do
		local px = data[i + 1]
		buffer.writeu8(rgba, i * 4, band(math.floor(px / 65536), 255))
		buffer.writeu8(rgba, i * 4 + 1, band(math.floor(px / 256), 255))
		buffer.writeu8(rgba, i * 4 + 2, band(px, 255))
		buffer.writeu8(rgba, i * 4 + 3, math.floor(px / 16777216))
	end
	return rgba
end

local function decode(bytes: buffer): (DecodedImage?, string?)
	local ok, result, err = pcall(function(): (DecodedImage?, string?)
		if not isWebP(bytes) then
			return nil, "not a WebP file"
		end
		local total = buffer.len(bytes)
		local pos = 12
		local alphaBase = -1
		local alphaLen = 0
		local vp8Base = -1
		local vp8Len = 0
		local vp8lBase = -1
		local vp8lLen = 0
		while pos + 8 <= total do
			local fourcc = buffer.readstring(bytes, pos, 4)
			local size = buffer.readu32(bytes, pos + 4)
			local dataPos = pos + 8
			if fourcc == "VP8 " then
				vp8Base = dataPos
				vp8Len = size
			elseif fourcc == "VP8L" then
				vp8lBase = dataPos
				vp8lLen = size
			elseif fourcc == "ALPH" then
				alphaBase = dataPos
				alphaLen = size
			end
			-- VP8X and metadata chunks are skipped (we read dimensions from
			-- the image chunks themselves)
			pos = dataPos + size + band(size, 1)
		end

		if vp8lBase >= 0 then
			local data, w, h, derr = decodeVP8L(bytes, vp8lBase, vp8lLen, false, 0, 0)
			if not data then
				return nil, derr
			end
			return {
				width = w,
				height = h,
				rgba = argbToRgbaBuffer(data :: { number }, w, h),
			}, nil
		end

		if vp8Base >= 0 then
			local yP, uP, vP, w, h, ys, uvs, verr = decodeVP8(bytes, vp8Base, vp8Len)
			if not yP then
				return nil, verr
			end
			local rgba = upsampleFrame(yP :: buffer, uP :: buffer, vP :: buffer, w, h, ys, uvs)
			if alphaBase >= 0 then
				local aerr = decodeAlpha(bytes, alphaBase, alphaLen, w, h, rgba)
				if aerr then
					return nil, aerr
				end
			end
			return { width = w, height = h, rgba = rgba }, nil
		end

		return nil, "WebP: no VP8/VP8L chunk found"
	end)
	if not ok then
		return nil, "webp decode error: " .. tostring(result)
	end
	return result, err
end

return {
	decode = decode,
	isWebP = isWebP,
}
