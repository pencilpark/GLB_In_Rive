// Post-processing module for the GLB importer node.
// Separate module from model.wgsl so bind group layouts stay independent.
//
// All passes draw a fullscreen triangle (3 vertices, no vertex buffer) and
// share one bind group layout:
//   group(0) binding(0)  uniform Post
//   group(0) binding(1)  srcTex   (input of the pass)
//   group(0) binding(2)  samp     (linear clamp sampler)
//   group(0) binding(3)  auxTex   (composite only: bloom/blur chain result;
//                                  bind a 1x1 dummy for the other passes)
//
// Pass chain driven from Luau:
//   scene (offscreen, alpha = packed linear depth when DOF on)
//     -> fs_bright   (half res, threshold + knee)
//     -> fs_blur     (horizontal, dir in post.texel.zw)
//     -> fs_blur     (vertical)
//     -> fs_composite(src = scene, aux = blurred bright) -> final target
//
// Scene colours are premultiplied; every operation here keeps them
// premultiplied (pivots and offsets are scaled by alpha).

struct Post {
    // 1/srcWidth, 1/srcHeight, blurDir.x, blurDir.y (texel units)
    texel: vec4f,
    // threshold, knee, intensity, enabled
    bloom: vec4f,
    // amount, radius, softness, enabled
    vignette: vec4f,
    // exposure, contrast, saturation, enabled
    grade: vec4f,
    // focusDepth (0..1, linear viewZ/far), range, maxRadiusPx, enabled
    dof: vec4f,
};

@group(0) @binding(0) var<uniform> post: Post;
@group(0) @binding(1) var srcTex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var auxTex: texture_2d<f32>;

struct VSOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VSOut {
    // Oversized triangle covering the viewport.
    var out: VSOut;
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    out.pos = vec4f(x, y, 0.0, 1.0);
    out.uv = vec2f(x, -y) * 0.5 + vec2f(0.5, 0.5);
    return out;
}

fn luma(c: vec3f) -> f32 {
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// ---------------------------------------------------------------------------
// Bright pass: soft-knee threshold, used as the bloom source.
// ---------------------------------------------------------------------------
@fragment
fn fs_bright(in: VSOut) -> @location(0) vec4f {
    let c = textureSampleLevel(srcTex, samp, in.uv, 0.0);
    let threshold = post.bloom.x;
    let knee = max(post.bloom.y, 1.0e-4);
    let l = luma(c.rgb);
    // Soft knee: quadratic ramp between threshold-knee and threshold+knee.
    let soft = clamp(l - threshold + knee, 0.0, 2.0 * knee);
    let contrib = max(soft * soft / (4.0 * knee), l - threshold);
    let w = contrib / max(l, 1.0e-4);
    return vec4f(c.rgb * w, 0.0);
}

// ---------------------------------------------------------------------------
// Separable 9-tap Gaussian blur. Direction (in texels) in post.texel.zw.
// ---------------------------------------------------------------------------
@fragment
fn fs_blur(in: VSOut) -> @location(0) vec4f {
    let step = post.texel.zw * post.texel.xy;
    var acc = textureSampleLevel(srcTex, samp, in.uv, 0.0) * 0.227027;
    let w = array<f32, 4>(0.1945946, 0.1216216, 0.054054, 0.016216);
    let o = array<f32, 4>(1.0, 2.0, 3.0, 4.0);
    for (var i = 0; i < 4; i = i + 1) {
        let d = step * o[i];
        acc = acc + textureSampleLevel(srcTex, samp, in.uv + d, 0.0) * w[i];
        acc = acc + textureSampleLevel(srcTex, samp, in.uv - d, 0.0) * w[i];
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Composite: DOF (depth from scene alpha) + bloom add + grade + vignette.
// ---------------------------------------------------------------------------
@fragment
fn fs_composite(in: VSOut) -> @location(0) vec4f {
    var scene = textureSampleLevel(srcTex, samp, in.uv, 0.0);
    var alpha = scene.a;

    // Depth of field: scene alpha holds linear depth (model.wgsl packs it
    // when frame.mode0.w > 0). Poisson-ish 8-tap disc, radius from CoC.
    if (post.dof.w > 0.5) {
        let depth = scene.a;
        let coc = clamp(abs(depth - post.dof.x) / max(post.dof.y, 1.0e-4), 0.0, 1.0);
        let radiusPx = coc * post.dof.z;
        if (radiusPx > 0.5) {
            let r = radiusPx * post.texel.xy;
            var acc = scene.rgb;
            var wsum = 1.0;
            let taps = array<vec2f, 8>(
                vec2f(1.0, 0.0), vec2f(-1.0, 0.0),
                vec2f(0.0, 1.0), vec2f(0.0, -1.0),
                vec2f(0.707, 0.707), vec2f(-0.707, 0.707),
                vec2f(0.707, -0.707), vec2f(-0.707, -0.707));
            for (var i = 0; i < 8; i = i + 1) {
                let s = textureSampleLevel(srcTex, samp, in.uv + taps[i] * r, 0.0);
                // Reject taps much closer than this pixel to limit bleed.
                let sCoc = clamp(abs(s.a - post.dof.x) / max(post.dof.y, 1.0e-4), 0.0, 1.0);
                let w = 0.5 + 0.5 * min(sCoc / max(coc, 1.0e-4), 1.0);
                acc = acc + s.rgb * w;
                wsum = wsum + w;
            }
            scene = vec4f(acc / wsum, scene.a);
        }
        // Alpha carried depth, not coverage: the composited result is opaque.
        alpha = 1.0;
    }

    var rgb = scene.rgb;

    // Bloom (premultiplied additive: alpha unchanged).
    if (post.bloom.w > 0.5) {
        let b = textureSampleLevel(auxTex, samp, in.uv, 0.0);
        rgb = rgb + b.rgb * post.bloom.z;
    }

    // Grade: exposure, contrast (pivot scaled by alpha to stay
    // premultiplied-correct), saturation around premultiplied luma.
    if (post.grade.w > 0.5) {
        rgb = rgb * post.grade.x;
        let pivot = 0.5 * alpha;
        rgb = (rgb - vec3f(pivot)) * post.grade.y + vec3f(pivot);
        rgb = max(rgb, vec3f(0.0));
        let g = vec3f(luma(rgb));
        rgb = mix(g, rgb, post.grade.z);
    }

    // Vignette: darken rgb only (premultiplied-safe).
    if (post.vignette.w > 0.5) {
        let d = distance(in.uv, vec2f(0.5, 0.5)) * 1.4142136;
        let v = 1.0 - post.vignette.x * smoothstep(post.vignette.y, post.vignette.y + max(post.vignette.z, 1.0e-3), d);
        rgb = rgb * clamp(v, 0.0, 1.0);
    }

    rgb = min(rgb, vec3f(alpha));
    return vec4f(rgb, alpha);
}
