// model.wgsl -- PBR shader for the GLB Loader node (Rive WGSL Shader asset).
//
// Replace the content of the "model" WGSL Shader Script asset with this file.
// The bind group layout and UBO byte offsets are the contract shared with
// GLBModel.luau -- existing fields keep their offsets, new features append
// to the end of each uniform block.
//
// Group 0 (per frame):
//   @binding(0) Frame uniform (512 bytes)
//   @binding(1) shadow map (rgba8-packed depth, light1's view)
//   @binding(2) shadow sampler
//   @binding(3) environment map (equirectangular, CPU-prefiltered mips)
//   @binding(4) environment sampler (trilinear)
// Group 1 (per material, 256-byte dynamic UBO):
//   @binding(0) Obj uniform   @binding(1) base colour texture
//   @binding(2) sampler       @binding(3) metallic-roughness texture
//   @binding(4) emissive      @binding(5) normal map
//   @binding(6) occlusion
//
// Entry points: vs_main/fs_main (scene), vs_shadow/fs_shadow (depth-only
// pass into an rgba8 target, packed depth).

struct Light {
    dir: vec4f,    // xyz direction TOWARDS the light, w intensity (0 = off)
    color: vec4f,  // rgb colour
}

struct Frame {
    view: mat4x4f,         // @0
    proj: mat4x4f,         // @64
    camPos: vec4f,         // @128 xyz camera position
    params: vec4f,         // @144 x ambient, y exposure, z debug, w reflectivity
    surf: vec4f,           // @160 x specularStrength, y rimStrength, z metallicMul, w roughnessMul
    tint: vec4f,           // @176 rgba multiplies base colour
    misc: vec4f,           // @192 x flipV, y alphaBlend, z width, w height
    texControl: vec4f,     // @208 x showBase, y textureMix, z showEmissive, w showMR
    replaceColor: vec4f,   // @224 solid colour when the base texture is hidden
    lights: array<Light, 3>, // @240 (3 x 32 bytes)
    hilite: vec4f,         // @336 rgb highlight colour, w strength
    env0: vec4f,           // @352 x envIntensity, y envMaxLod, z envYaw (radians), w envMode (0 gradient, 1 equirect)
    skyColor: vec4f,       // @368 gradient sky colour (rgb)
    horizonColor: vec4f,   // @384 gradient horizon colour (rgb)
    groundColor: vec4f,    // @400 gradient ground colour (rgb)
    shadow0: vec4f,        // @416 x strength (0 = off), y bias, z map size, w pcf radius (texels)
    lightVP: mat4x4f,      // @432 light view-projection (depth 0..1)
    mode0: vec4f,          // @496 x shadingMode (0 pbr, 1 unlit, 2 toon, 3 wireframe), y toonSteps, z wireWidth, w dofDepth (0 off, else 1/far)
}

struct Obj {
    model: mat4x4f,        // @0
    normalMat: mat4x4f,    // @64
    baseColor: vec4f,      // @128
    emissive: vec4f,       // @144 rgb emissive factor
    mrFlags: vec4f,        // @160 x metallic, y roughness, z hasBaseTex, w alphaCutoff
    flags2: vec4f,         // @176 x hasMR, y hasEmissive, z emissiveBoost, w highlight
    flags3: vec4f,         // @192 x hasNormal, y hasOcclusion, z normalScale, w occlusionStrength
    flags4: vec4f,         // @208 x emissiveStrength, y occlusionUV, z clearcoat, w clearcoatRoughness
    ext1: vec4f,           // @224 x transmission, y ior, z sheenRoughness, w alphaMode (0 opaque, 1 mask, 2 blend)
    sheenColor: vec4f,     // @240 rgb sheen colour
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var shadowTex: texture_2d<f32>;
@group(0) @binding(2) var shadowSamp: sampler;
@group(0) @binding(3) var envTex: texture_2d<f32>;
@group(0) @binding(4) var envSamp: sampler;

@group(1) @binding(0) var<uniform> obj: Obj;
@group(1) @binding(1) var baseTex: texture_2d<f32>;
@group(1) @binding(2) var samp: sampler;
@group(1) @binding(3) var mrTex: texture_2d<f32>;
@group(1) @binding(4) var emisTex: texture_2d<f32>;
@group(1) @binding(5) var normalTex: texture_2d<f32>;
@group(1) @binding(6) var occlusionTex: texture_2d<f32>;

struct VSIn {
    @location(0) pos: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv0: vec2f,
    @location(3) tangent: vec4f,
    @location(4) uv1: vec2f,
    @location(5) color: vec4f,
}

struct VSOut {
    @builtin(position) clip: vec4f,
    @location(0) worldPos: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv0: vec2f,
    @location(3) tangent: vec4f,
    @location(4) uv1: vec2f,
    @location(5) color: vec4f,
    @location(6) viewZ: f32,
}

@vertex
fn vs_main(in: VSIn) -> VSOut {
    var out: VSOut;
    let world = obj.model * vec4f(in.pos, 1.0);
    let viewPos = frame.view * world;
    out.clip = frame.proj * viewPos;
    out.worldPos = world.xyz;
    out.normal = normalize((obj.normalMat * vec4f(in.normal, 0.0)).xyz);
    let tan = (obj.model * vec4f(in.tangent.xyz, 0.0)).xyz;
    out.tangent = vec4f(tan, in.tangent.w);
    out.uv0 = in.uv0;
    out.uv1 = in.uv1;
    out.color = in.color;
    out.viewZ = -viewPos.z;
    return out;
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn flipUV(uv: vec2f) -> vec2f {
    if (frame.misc.x > 0.5) {
        return vec2f(uv.x, 1.0 - uv.y);
    }
    return uv;
}

const PI: f32 = 3.141592653589793;

fn dGGX(nDotH: f32, rough: f32) -> f32 {
    let a = rough * rough;
    let a2 = a * a;
    let d = nDotH * nDotH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-6);
}

fn gSmith(nDotV: f32, nDotL: f32, rough: f32) -> f32 {
    let r = rough + 1.0;
    let k = (r * r) / 8.0;
    let gv = nDotV / (nDotV * (1.0 - k) + k);
    let gl = nDotL / (nDotL * (1.0 - k) + k);
    return gv * gl;
}

fn fresnelSchlick(cosT: f32, f0: vec3f) -> vec3f {
    return f0 + (vec3f(1.0) - f0) * pow(clamp(1.0 - cosT, 0.0, 1.0), 5.0);
}

// Charlie sheen distribution (approximation)
fn dCharlie(nDotH: f32, rough: f32) -> f32 {
    let r = max(rough, 1e-3);
    let invR = 1.0 / r;
    let cos2 = nDotH * nDotH;
    let sin2 = 1.0 - cos2;
    return (2.0 + invR) * pow(sin2, invR * 0.5) / (2.0 * PI);
}

// Direction -> equirectangular UV, with an adjustable yaw so the environment
// can be rotated around the model.
fn dirToEquirect(d: vec3f) -> vec2f {
    let yaw = frame.env0.z;
    let u = (atan2(d.x, d.z) + yaw) / (2.0 * PI) + 0.5;
    let v = acos(clamp(d.y, -1.0, 1.0)) / PI;
    return vec2f(u, v);
}

// Environment radiance for a direction at a given roughness (0..1).
fn envRadiance(dir: vec3f, rough: f32) -> vec3f {
    if (frame.env0.w > 0.5) {
        let lod = rough * frame.env0.y;
        return textureSampleLevel(envTex, envSamp, dirToEquirect(dir), lod).rgb;
    }
    // procedural three-colour gradient: ground -> horizon -> sky
    let y = clamp(dir.y, -1.0, 1.0);
    var col: vec3f;
    if (y >= 0.0) {
        col = mix(frame.horizonColor.rgb, frame.skyColor.rgb, pow(y, 0.6));
    } else {
        col = mix(frame.horizonColor.rgb, frame.groundColor.rgb, pow(-y, 0.6));
    }
    return col;
}

// unpack rgba8-packed depth written by fs_shadow
fn unpackDepth(c: vec4f) -> f32 {
    return dot(c, vec4f(1.0, 1.0 / 255.0, 1.0 / 65025.0, 1.0 / 16581375.0));
}

// Shadow factor for a world position: 1 = lit, 0 = fully shadowed.
fn shadowFactor(worldPos: vec3f, nDotL: f32) -> f32 {
    let strength = frame.shadow0.x;
    if (strength <= 0.0) {
        return 1.0;
    }
    let lp = frame.lightVP * vec4f(worldPos, 1.0);
    let ndc = lp.xyz / max(lp.w, 1e-6);
    let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z <= 0.0 || ndc.z >= 1.0) {
        return 1.0;
    }
    // slope-scaled bias
    let bias = frame.shadow0.y * (1.0 + (1.0 - nDotL) * 2.0);
    let depth = ndc.z - bias;
    let texel = frame.shadow0.w / max(frame.shadow0.z, 1.0);
    var lit: f32 = 0.0;
    // 5-tap PCF (centre + diagonals)
    for (var i: i32 = 0; i < 5; i = i + 1) {
        var o = vec2f(0.0, 0.0);
        if (i == 1) { o = vec2f(-1.0, -1.0); }
        if (i == 2) { o = vec2f(1.0, -1.0); }
        if (i == 3) { o = vec2f(-1.0, 1.0); }
        if (i == 4) { o = vec2f(1.0, 1.0); }
        let s = unpackDepth(textureSampleLevel(shadowTex, shadowSamp, uv + o * texel, 0.0));
        if (depth <= s) {
            lit = lit + 1.0;
        }
    }
    lit = lit / 5.0;
    return mix(1.0, lit, clamp(strength, 0.0, 1.0));
}

// ---------------------------------------------------------------------------
// fragment
// ---------------------------------------------------------------------------

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4f {
    let uv = flipUV(in.uv0);
    let uvOcc = select(flipUV(in.uv0), flipUV(in.uv1), obj.flags3.y > 0.5 && obj.flags4.y > 0.5);

    // ---- base colour ----
    // showBase off -> replaceColor; otherwise blend replaceColor -> texture
    // by textureMix (1 = pure texture).
    var baseSample = vec4f(frame.replaceColor.rgb, 1.0);
    if (obj.mrFlags.z > 0.5) {
        let tc = textureSample(baseTex, samp, uv);
        let mixed = mix(frame.replaceColor.rgb, tc.rgb, clamp(frame.texControl.y, 0.0, 1.0));
        let rgb = select(frame.replaceColor.rgb, mixed, frame.texControl.x > 0.5);
        baseSample = vec4f(rgb, tc.a);
    }
    var base = baseSample * obj.baseColor * in.color * frame.tint;

    // ---- alpha mode ----
    let alphaMode = obj.ext1.w;
    if (alphaMode > 0.5 && alphaMode < 1.5) {
        // MASK
        if (base.a < obj.mrFlags.w) {
            discard;
        }
        base.a = 1.0;
    }

    // ---- normal ----
    var n = normalize(in.normal);
    if (obj.flags3.x > 0.5) {
        let t = normalize(in.tangent.xyz - n * dot(in.tangent.xyz, n));
        let b = cross(n, t) * in.tangent.w;
        var nm = textureSample(normalTex, samp, uv).xyz * 2.0 - 1.0;
        nm = vec3f(nm.xy * obj.flags3.z, nm.z);
        n = normalize(t * nm.x + b * nm.y + n * nm.z);
    }
    let v = normalize(frame.camPos.xyz - in.worldPos);
    if (dot(n, v) < 0.0) {
        n = -n; // double-sided: flip towards the viewer
    }
    let nDotV = max(dot(n, v), 1e-4);

    // ---- metallic / roughness ----
    var metal = obj.mrFlags.x * frame.surf.z;
    var rough = obj.mrFlags.y * frame.surf.w;
    if (obj.flags2.x > 0.5 && frame.texControl.w > 0.5) {
        let mr = textureSample(mrTex, samp, uv);
        metal = metal * mr.b;
        rough = rough * mr.g;
    }
    metal = clamp(metal, 0.0, 1.0);
    rough = clamp(rough, 0.04, 1.0);

    // ---- occlusion ----
    var occ: f32 = 1.0;
    if (obj.flags3.y > 0.5) {
        let o = textureSample(occlusionTex, samp, uvOcc).r;
        occ = 1.0 + obj.flags3.w * (o - 1.0);
    }

    // ---- shading mode shortcuts ----
    let mode = frame.mode0.x;
    let exposure = frame.params.y;
    if (frame.params.z > 0.5 && frame.params.z < 1.5) { return vec4f(base.rgb, 1.0); }       // debug 1: raw base
    if (frame.params.z > 1.5 && frame.params.z < 2.5) { return vec4f(uv, 0.0, 1.0); }        // debug 2: uv
    if (frame.params.z > 2.5 && frame.params.z < 3.5) { return vec4f(metal, rough, 0.0, 1.0); } // debug 3: metal/rough
    if (frame.params.z > 3.5) {                                                              // debug 4: emissive
        var em = obj.emissive.rgb;
        if (obj.flags2.y > 0.5) { em = em * textureSample(emisTex, samp, uv).rgb; }
        return vec4f(em, 1.0);
    }

    if (mode > 2.5) {
        // wireframe: vertex colour carries barycentrics (set up by the Luau)
        let bary = in.color.rgb;
        let d = fwidth(bary);
        let a = smoothstep(vec3f(0.0), d * max(frame.mode0.z, 0.5), bary);
        let edge = 1.0 - min(min(a.x, a.y), a.z);
        let wcol = mix(vec3f(0.02, 0.02, 0.025), frame.hilite.rgb, edge);
        return vec4f(wcol * exposure, 1.0);
    }
    if (mode > 0.5 && mode < 1.5) {
        // unlit
        var col = base.rgb * occ * exposure;
        col = applyHighlight(col);
        return premultiply(vec4f(col, base.a));
    }

    // ---- lighting ----
    let f0 = mix(vec3f(0.04) * obj.ext1.y / 1.5, base.rgb, metal); // ior scales dielectric F0
    let diffuseColor = base.rgb * (1.0 - metal);
    var direct = vec3f(0.0);
    let toon = mode > 1.5 && mode < 2.5;
    let toonSteps = max(frame.mode0.y, 2.0);

    for (var i: i32 = 0; i < 3; i = i + 1) {
        let li = frame.lights[i];
        let intensity = li.dir.w;
        if (intensity <= 0.0) {
            continue;
        }
        let l = normalize(li.dir.xyz);
        var nDotL = max(dot(n, l), 0.0);
        if (nDotL <= 0.0) {
            continue;
        }
        var shadow: f32 = 1.0;
        if (i == 0) {
            shadow = shadowFactor(in.worldPos, nDotL);
        }
        if (toon) {
            nDotL = floor(nDotL * toonSteps + 0.5) / toonSteps;
        }
        let h = normalize(l + v);
        let nDotH = max(dot(n, h), 0.0);
        let d = dGGX(nDotH, rough);
        let g = gSmith(nDotV, nDotL, rough);
        let f = fresnelSchlick(max(dot(h, v), 0.0), f0);
        var spec = (d * g / max(4.0 * nDotV * nDotL, 1e-4)) * f * frame.surf.x;
        var kd = (vec3f(1.0) - f) * diffuseColor / PI;
        if (toon) {
            spec = step(vec3f(0.6), spec) * f0 * frame.surf.x * 0.5;
        }
        // sheen lobe
        if (obj.sheenColor.r + obj.sheenColor.g + obj.sheenColor.b > 0.0) {
            let ds = dCharlie(nDotH, obj.ext1.z);
            spec = spec + obj.sheenColor.rgb * ds * 0.25;
        }
        // clearcoat lobe (fixed F0 0.04, its own roughness)
        if (obj.flags4.z > 0.0) {
            let dc = dGGX(nDotH, max(obj.flags4.w, 0.04));
            let fc = 0.04 + 0.96 * pow(1.0 - max(dot(h, v), 0.0), 5.0);
            let cc = dc * fc * obj.flags4.z * 0.25;
            spec = spec * (1.0 - fc * obj.flags4.z) + vec3f(cc);
        }
        direct = direct + (kd + spec) * li.color.rgb * intensity * nDotL * PI * shadow;
    }

    // ---- environment / ambient ----
    let envI = frame.env0.x;
    let irr = envRadiance(n, 1.0) * envI;
    var ambientCol = (frame.params.x + irr) * diffuseColor * occ;
    let r = reflect(-v, n);
    let envSpec = envRadiance(r, rough) * envI;
    let fEnv = fresnelSchlick(nDotV, f0);
    var indirectSpec = envSpec * fEnv * frame.params.w * occ;
    // transmission approximation: see through the surface into the environment
    let trans = obj.ext1.x;
    if (trans > 0.0) {
        let eta = 1.0 / max(obj.ext1.y, 1.0);
        let rd = refract(-v, n, eta);
        var td = rd;
        if (dot(rd, rd) < 1e-6) {
            td = -v;
        }
        let through = envRadiance(td, rough) * max(envI, 0.35) * base.rgb;
        ambientCol = mix(ambientCol, through, trans);
        base.a = max(base.a * (1.0 - trans * 0.85), 0.05);
    }

    // ---- rim ----
    let rim = pow(1.0 - nDotV, 3.0) * frame.surf.y;

    var col = (direct + ambientCol + indirectSpec + vec3f(rim)) * exposure;

    // ---- emissive ----
    if (frame.texControl.z > 0.5) {
        var em = obj.emissive.rgb * obj.flags4.x * max(obj.flags2.z, 0.0);
        if (obj.flags2.y > 0.5) {
            em = em * textureSample(emisTex, samp, uv).rgb;
        }
        col = col + em;
    }

    col = applyHighlight(col);

    var outA = base.a;
    if (frame.mode0.w > 0.0 && frame.misc.y < 0.5 && alphaMode < 1.5) {
        // DOF support: pack normalized linear view depth into alpha
        // (opaque/mask draws only -- BLEND draws keep coverage alpha)
        outA = clamp(in.viewZ * frame.mode0.w, 0.0, 1.0);
        return vec4f(col, outA);
    }
    return premultiply(vec4f(col, outA));
}

fn applyHighlight(col: vec3f) -> vec3f {
    if (obj.flags2.w > 0.5) {
        return mix(col, frame.hilite.rgb, frame.hilite.w);
    }
    return col;
}

fn premultiply(c: vec4f) -> vec4f {
    // Per-frame blending (misc.y) or a per-material BLEND alpha mode
    // (ext1.w == 2) both need premultiplied output for the srcOver pass.
    if (frame.misc.y > 0.5 || obj.ext1.w > 1.5) {
        return vec4f(c.rgb * c.a, c.a);
    }
    return vec4f(c.rgb, 1.0);
}

// ---------------------------------------------------------------------------
// shadow pass: depth-only render from the light, packed into rgba8
// ---------------------------------------------------------------------------

struct ShadowOut {
    @builtin(position) clip: vec4f,
    @location(0) depth: f32,
}

@vertex
fn vs_shadow(in: VSIn) -> ShadowOut {
    var out: ShadowOut;
    let world = obj.model * vec4f(in.pos, 1.0);
    let clip = frame.lightVP * world;
    out.clip = clip;
    out.depth = clip.z / max(clip.w, 1e-6);
    return out;
}

fn packDepth(d: f32) -> vec4f {
    let dc = clamp(d, 0.0, 0.999999);
    var enc = fract(dc * vec4f(1.0, 255.0, 65025.0, 16581375.0));
    enc = enc - enc.yzww * vec4f(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0, 0.0);
    return enc;
}

@fragment
fn fs_shadow(in: ShadowOut) -> @location(0) vec4f {
    return packDepth(in.depth);
}
