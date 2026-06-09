--!strict
-- Math3D
-- Pure-Luau 4x4 matrix maths on flat, COLUMN-MAJOR {number}[16] arrays
-- (same layout as Rive's Mat4.values, glTF node.matrix, and WGSL mat4x4<f32>).
-- This exists because Rive provides no quaternion type, no Mat4.lookAt and no
-- Mat4.ortho; and keeping the maths pure makes it unit-testable without Rive.
--
-- Flat index for column c (0..3), row r (0..3): c*4 + r + 1  (1-based Lua).
-- A point p is transformed as M * p (column vector), i.e. apply this matrix.

export type Mat = { number } -- length 16, column-major

local Math3D = {}

function Math3D.identity(): Mat
  return { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 }
end

-- C = A * B  (apply B, then A)
function Math3D.multiply(a: Mat, b: Mat): Mat
  local out = table.create(16, 0) :: Mat
  for col = 0, 3 do
    for row = 0, 3 do
      local s = 0
      for k = 0, 3 do
        s += a[k * 4 + row + 1] * b[col * 4 + k + 1]
      end
      out[col * 4 + row + 1] = s
    end
  end
  return out
end

function Math3D.fromTranslation(x: number, y: number, z: number): Mat
  return { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1 }
end

function Math3D.fromScale(x: number, y: number, z: number): Mat
  return { x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1 }
end

function Math3D.fromRotationX(a: number): Mat
  local c, s = math.cos(a), math.sin(a)
  return { 1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1 }
end

function Math3D.fromRotationY(a: number): Mat
  local c, s = math.cos(a), math.sin(a)
  return { c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1 }
end

function Math3D.fromRotationZ(a: number): Mat
  local c, s = math.cos(a), math.sin(a)
  return { c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 }
end

-- Quaternion (x,y,z,w) -> rotation matrix (column-major). Normalises input.
function Math3D.quatToMat4(x: number, y: number, z: number, w: number): Mat
  local n = math.sqrt(x * x + y * y + z * z + w * w)
  if n > 1e-8 then
    x, y, z, w = x / n, y / n, z / n, w / n
  end
  local xx, yy, zz = x * x, y * y, z * z
  local xy, xz, yz = x * y, x * z, y * z
  local wx, wy, wz = w * x, w * y, w * z
  -- row-major entries r[row][col]
  local r00, r01, r02 = 1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)
  local r10, r11, r12 = 2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)
  local r20, r21, r22 = 2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)
  -- store column-major: flat[c*4+r+1] = r[r][c]
  return { r00, r10, r20, 0, r01, r11, r21, 0, r02, r12, r22, 0, 0, 0, 0, 1 }
end

-- glTF node TRS -> local matrix. Uses node.matrix verbatim if present.
function Math3D.composeTRS(
  translation: { number }?,
  rotation: { number }?,
  scale: { number }?
): Mat
  local t = translation or { 0, 0, 0 }
  local r = rotation or { 0, 0, 0, 1 }
  local sc = scale or { 1, 1, 1 }
  local m = Math3D.multiply(
    Math3D.fromTranslation(t[1], t[2], t[3]),
    Math3D.quatToMat4(r[1], r[2], r[3], r[4])
  )
  return Math3D.multiply(m, Math3D.fromScale(sc[1], sc[2], sc[3]))
end

function Math3D.transpose(m: Mat): Mat
  local out = table.create(16, 0) :: Mat
  for col = 0, 3 do
    for row = 0, 3 do
      out[col * 4 + row + 1] = m[row * 4 + col + 1]
    end
  end
  return out
end

-- General 4x4 inverse (column-major). Returns nil if singular. Port of the
-- well-known MESA gluInvertMatrix (column-major in, column-major out).
function Math3D.invert(m: Mat): Mat?
  local function M(i: number): number
    return m[i + 1]
  end
  local inv = table.create(16, 0)
  inv[0] = M(5) * M(10) * M(15)
    - M(5) * M(11) * M(14)
    - M(9) * M(6) * M(15)
    + M(9) * M(7) * M(14)
    + M(13) * M(6) * M(11)
    - M(13) * M(7) * M(10)
  inv[4] = -M(4) * M(10) * M(15)
    + M(4) * M(11) * M(14)
    + M(8) * M(6) * M(15)
    - M(8) * M(7) * M(14)
    - M(12) * M(6) * M(11)
    + M(12) * M(7) * M(10)
  inv[8] = M(4) * M(9) * M(15)
    - M(4) * M(11) * M(13)
    - M(8) * M(5) * M(15)
    + M(8) * M(7) * M(13)
    + M(12) * M(5) * M(11)
    - M(12) * M(7) * M(9)
  inv[12] = -M(4) * M(9) * M(14)
    + M(4) * M(10) * M(13)
    + M(8) * M(5) * M(14)
    - M(8) * M(6) * M(13)
    - M(12) * M(5) * M(10)
    + M(12) * M(6) * M(9)
  inv[1] = -M(1) * M(10) * M(15)
    + M(1) * M(11) * M(14)
    + M(9) * M(2) * M(15)
    - M(9) * M(3) * M(14)
    - M(13) * M(2) * M(11)
    + M(13) * M(3) * M(10)
  inv[5] = M(0) * M(10) * M(15)
    - M(0) * M(11) * M(14)
    - M(8) * M(2) * M(15)
    + M(8) * M(3) * M(14)
    + M(12) * M(2) * M(11)
    - M(12) * M(3) * M(10)
  inv[9] = -M(0) * M(9) * M(15)
    + M(0) * M(11) * M(13)
    + M(8) * M(1) * M(15)
    - M(8) * M(3) * M(13)
    - M(12) * M(1) * M(11)
    + M(12) * M(3) * M(9)
  inv[13] = M(0) * M(9) * M(14)
    - M(0) * M(10) * M(13)
    - M(8) * M(1) * M(14)
    + M(8) * M(2) * M(13)
    + M(12) * M(1) * M(10)
    - M(12) * M(2) * M(9)
  inv[2] = M(1) * M(6) * M(15)
    - M(1) * M(7) * M(14)
    - M(5) * M(2) * M(15)
    + M(5) * M(3) * M(14)
    + M(13) * M(2) * M(7)
    - M(13) * M(3) * M(6)
  inv[6] = -M(0) * M(6) * M(15)
    + M(0) * M(7) * M(14)
    + M(4) * M(2) * M(15)
    - M(4) * M(3) * M(14)
    - M(12) * M(2) * M(7)
    + M(12) * M(3) * M(6)
  inv[10] = M(0) * M(5) * M(15)
    - M(0) * M(7) * M(13)
    - M(4) * M(1) * M(15)
    + M(4) * M(3) * M(13)
    + M(12) * M(1) * M(7)
    - M(12) * M(3) * M(5)
  inv[14] = -M(0) * M(5) * M(14)
    + M(0) * M(6) * M(13)
    + M(4) * M(1) * M(14)
    - M(4) * M(2) * M(13)
    - M(12) * M(1) * M(6)
    + M(12) * M(2) * M(5)
  inv[3] = -M(1) * M(6) * M(11)
    + M(1) * M(7) * M(10)
    + M(5) * M(2) * M(11)
    - M(5) * M(3) * M(10)
    - M(9) * M(2) * M(7)
    + M(9) * M(3) * M(6)
  inv[7] = M(0) * M(6) * M(11)
    - M(0) * M(7) * M(10)
    - M(4) * M(2) * M(11)
    + M(4) * M(3) * M(10)
    + M(8) * M(2) * M(7)
    - M(8) * M(3) * M(6)
  inv[11] = -M(0) * M(5) * M(11)
    + M(0) * M(7) * M(9)
    + M(4) * M(1) * M(11)
    - M(4) * M(3) * M(9)
    - M(8) * M(1) * M(7)
    + M(8) * M(3) * M(5)
  inv[15] = M(0) * M(5) * M(10)
    - M(0) * M(6) * M(9)
    - M(4) * M(1) * M(10)
    + M(4) * M(2) * M(9)
    + M(8) * M(1) * M(6)
    - M(8) * M(2) * M(5)
  local det = M(0) * inv[0] + M(1) * inv[4] + M(2) * inv[8] + M(3) * inv[12]
  if math.abs(det) < 1e-12 then
    return nil
  end
  det = 1 / det
  local out = table.create(16, 0) :: Mat
  for i = 0, 15 do
    out[i + 1] = inv[i] * det
  end
  return out
end

-- Inverse-transpose, for transforming normals. (Translation is irrelevant when
-- transforming directions with w=0, but we keep the full 4x4.)
function Math3D.normalMatrix(model: Mat): Mat
  local i = Math3D.invert(model)
  if not i then
    return Math3D.identity()
  end
  return Math3D.transpose(i)
end

-- Transform a point (w=1). Returns x,y,z (perspective divide applied if w~=1).
function Math3D.transformPoint(
  m: Mat,
  x: number,
  y: number,
  z: number
): (number, number, number)
  local rx = m[1] * x + m[5] * y + m[9] * z + m[13]
  local ry = m[2] * x + m[6] * y + m[10] * z + m[14]
  local rz = m[3] * x + m[7] * y + m[11] * z + m[15]
  local rw = m[4] * x + m[8] * y + m[12] * z + m[16]
  if rw ~= 0 and math.abs(rw - 1) > 1e-9 then
    return rx / rw, ry / rw, rz / rw
  end
  return rx, ry, rz
end

-- Transform a direction (w=0): rotation/scale only, translation ignored.
function Math3D.transformDir(
  m: Mat,
  x: number,
  y: number,
  z: number
): (number, number, number)
  return m[1] * x + m[5] * y + m[9] * z,
    m[2] * x + m[6] * y + m[10] * z,
    m[3] * x + m[7] * y + m[11] * z
end

-- WebGPU-style perspective (right-handed, looks down -Z, NDC z in [0,1]).
function Math3D.perspective(
  fovY: number,
  aspect: number,
  near: number,
  far: number
): Mat
  local f = 1 / math.tan(fovY / 2)
  local nf = 1 / (near - far)
  return {
    f / aspect,
    0,
    0,
    0,
    0,
    f,
    0,
    0,
    0,
    0,
    far * nf,
    -1,
    0,
    0,
    far * near * nf,
    0,
  }
end

-- Orbit camera view matrix = inverse(translate(target) * rotY(yaw) * rotX(pitch) * translate(0,0,distance)).
-- Reverse-Z perspective (right-handed, looks down -Z). Maps near -> 1, far -> 0
-- in NDC. Combined with a depth buffer cleared to 0 and a 'greater' depth test,
-- this distributes float depth precision far more uniformly across the frustum
-- and all but eliminates z-fighting on stacked/coplanar geometry.
function Math3D.perspectiveReverseZ(
  fovY: number,
  aspect: number,
  near: number,
  far: number
): Mat
  local f = 1 / math.tan(fovY / 2)
  local nf = 1 / (far - near)
  return {
    f / aspect,
    0,
    0,
    0,
    0,
    f,
    0,
    0,
    0,
    0,
    near * nf, -- (vs standard far*nf) -> reverses the depth range
    -1,
    0,
    0,
    far * near * nf,
    0,
  }
end

function Math3D.orbitView(
  tx: number,
  ty: number,
  tz: number,
  yaw: number,
  pitch: number,
  distance: number
): Mat
  local world = Math3D.multiply(
    Math3D.fromTranslation(tx, ty, tz),
    Math3D.fromRotationY(yaw)
  )
  world = Math3D.multiply(world, Math3D.fromRotationX(pitch))
  world = Math3D.multiply(world, Math3D.fromTranslation(0, 0, distance))
  return Math3D.invert(world) or Math3D.identity()
end

-- Like orbitView but also returns the camera world position (eye), for specular.
function Math3D.orbitCamera(
  tx: number,
  ty: number,
  tz: number,
  yaw: number,
  pitch: number,
  distance: number
): (Mat, number, number, number)
  local world = Math3D.multiply(
    Math3D.fromTranslation(tx, ty, tz),
    Math3D.fromRotationY(yaw)
  )
  world = Math3D.multiply(world, Math3D.fromRotationX(pitch))
  world = Math3D.multiply(world, Math3D.fromTranslation(0, 0, distance))
  local ex, ey, ez = Math3D.transformPoint(world, 0, 0, 0)
  local view = Math3D.invert(world) or Math3D.identity()
  return view, ex, ey, ez
end

-- Distance so a bounding sphere of `radius` fits a vertical field of view.
function Math3D.fitDistance(
  radius: number,
  fovY: number,
  margin: number?
): number
  local m = margin or 1.1
  return (radius / math.sin(fovY / 2)) * m
end

-- Build a world-space picking ray for a point on screen, given the CAMERA
-- WORLD matrix (the inverse of the view matrix), the vertical FOV, the aspect
-- ratio, and normalised device coords ndcX/ndcY in [-1, 1] (ndcY up = +1).
-- Returns the ray origin (camera eye) and a normalised direction, as 6 numbers:
-- ox, oy, oz, dx, dy, dz.
function Math3D.screenRay(
  camWorld: Mat,
  fovY: number,
  aspect: number,
  ndcX: number,
  ndcY: number
): (number, number, number, number, number, number)
  -- Camera-space ray direction (camera looks down -Z).
  local th = math.tan(fovY / 2)
  local cx = ndcX * th * aspect
  local cy = ndcY * th
  local cz = -1
  -- Rotate the direction into world space (ignore translation, w = 0).
  local dx, dy, dz = Math3D.transformDir(camWorld, cx, cy, cz)
  local len = math.sqrt(dx * dx + dy * dy + dz * dz)
  if len > 1e-8 then
    dx, dy, dz = dx / len, dy / len, dz / len
  end
  -- Ray origin is the camera position (the translation column of camWorld).
  local ox, oy, oz = camWorld[13], camWorld[14], camWorld[15]
  return ox, oy, oz, dx, dy, dz
end

-- Write a flat matrix into a buffer as 16 little-endian f32 (column-major),
-- ready for a WGSL mat4x4<f32> uniform. No Rive Mat4 needed.
function Math3D.writeToBuffer(buf: buffer, byteOffset: number, m: Mat)
  for i = 0, 15 do
    buffer.writef32(buf, byteOffset + i * 4, m[i + 1])
  end
end

return Math3D
