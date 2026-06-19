# GLB_In_Rive


https://github.com/user-attachments/assets/1a25f3b5-0a5b-49c5-b5e4-0937413074aa


File embedding all scripts to add GLB file with all embedded properties.

## Overview

This repository provides a GLB loader node for Rive artboards, designed to import `.glb` files with their embedded properties already handled.

Compressed GLBs are decoded in pure Luau, so files processed by gltfpack, gltf-transform, toktx or exported from Blender with compression enabled load directly:

| Format | Extension | Decoder |
| --- | --- | --- |
| meshopt geometry (codec v0 + v1, all filters) | `EXT_meshopt_compression` | `MeshoptDecoder.luau` |
| Draco geometry (edgebreaker standard/valence + sequential) | `KHR_draco_mesh_compression` | `DracoDecoder.luau` |
| Basis Universal textures (ETC1S/BasisLZ + UASTC, zstd) | `KHR_texture_basisu` | `KTX2Decoder.luau` (+ `ZstdDecoder.luau`) |
| WebP textures (lossy VP8, lossless VP8L, alpha) | `EXT_texture_webp` | `WebPDecoder.luau` |

Quantized files are also handled: `KHR_mesh_quantization` (normalized accessors) and `KHR_texture_transform` (baked into the UVs at mesh build time), which gltfpack emits alongside meshopt.

Material extensions with factor support: `KHR_materials_clearcoat`, `KHR_materials_transmission`, `KHR_materials_sheen`, `KHR_materials_ior`, `KHR_materials_emissive_strength` (scalar/colour factors feed the shader; their texture maps are not sampled). Alpha modes OPAQUE / MASK (cutoff) / BLEND are honoured per material.

## How to use

1. Add a fresh **GLB Loader** node to your artboard.
2. Import a `.glb` file and rename it however you like.
3. In the node properties, set the **textBlob** field to exactly match the asset name used in your Rive assets.
4. You do not need to manage textures manually — the script handles them for you, including KTX2 and WebP.
5. All animations baked into the GLB are indexed in the exposed properties.
6. At the very bottom of the properties, choose the animation index you want to use (`0`, `1`, `2`, `3`, etc.), depending on how many animations are included in your GLB.

### Required assets

Each `.luau` file must be added as a script asset with its exact name:
`GLBModel` (the node script), `GLBParser`, `GLTFMesh`, `Math3D`, `SceneGraph`,
`TextureOps`, `MeshoptDecoder`, `DracoDecoder`, `ZstdDecoder`, `KTX2Decoder`,
`WebPDecoder`.

Shaders: the WGSL Shader asset named **model** must contain `shaders/model.wgsl`
from this repo (it carries the extended frame layout, IBL, shadows and the
shading modes — the Luau script keeps working with an older copy but those
features stay off). For post FX, add a second WGSL Shader asset named **post**
containing `shaders/post.wgsl`.

## Rendering

- Lighting: 3 directional lights (key/fill/back) with yaw/pitch/colour/intensity, ambient, exposure, reflectivity, specular and rim controls.
- Image-based lighting: set **envName** to a blob asset containing an equirectangular panorama (PNG/JPEG/WebP/KTX2). It is prefiltered on the CPU into a mip chain, so rough surfaces see a blurred environment. **envIntensity** scales it, **envYaw** rotates it. Without an envName, a procedural gradient (skyColor / horizonColor / groundColor) is used instead.
- Shadows: enable **shadowEnabled** (allocates the map at init, resolution **shadowSize**) and light 1 casts a soft PCF shadow. Live controls: **shadowStrength** (% darkness, 0 = off), **shadowBias**, **shadowSoftness** (blur radius in texels).
- Shading modes via **shadingMode**: 0 = PBR, 1 = unlit, 2 = toon (**toonSteps** bands), 3 = wireframe (**wireWidth** px). Wireframe rebuilds the geometry at init, so set it before loading; the other modes switch live.
- Textures upload with full mip chains and sample trilinearly (anisotropic when the device supports it); minified textures no longer shimmer.
- Transparency: materials with alphaMode BLEND draw after the opaque pass, sorted back-to-front, without depth writes; MASK materials use their cutoff. The global **alphaBlend** input forces every draw through the blended path.

## Post FX

Enable **postEnabled** (needs the **post** shader asset; read at init). The scene renders offscreen and fullscreen passes composite it back, all live:

- Bloom: **bloomEnabled**, **bloomThreshold**, **bloomIntensity**.
- Vignette: **vignetteAmount** (0 = off), **vignetteRadius**, **vignetteSoftness**.
- Colour grade: **gradeEnabled**, **gradeExposure**, **gradeContrast**, **gradeSaturation**.
- Depth of field: **dofEnabled**, with **dofAutoFocus** keeping the model centre sharp (or **dofFocus** 0..1 manual), **dofRange**, **dofRadius** (px). DOF needs an opaque scene: it is bypassed while the global alphaBlend is on, and the composited node becomes opaque while active.

## Camera

- Orbit: drag rotates (interactive + dragSpeed as before), with **pitchMin**/**pitchMax** limits and release inertia (**inertia** 0..1).
- Touch gestures: two fingers pinch-zoom (clamped to **zoomMin**..**zoomMax**) and pan the camera target.
- **focusOnSelect**: when a part is picked, the camera glides onto it (and back on deselect).
- GLB cameras: set **cameraIndex** >= 0 to render through a camera authored in the file (perspective or orthographic, follows its animated node). -1 returns to the orbit camera.

## Component as interactive texture

Up to three components (nested artboards) can each drive a material, via **componentArtboard**/**componentMatIndex** and the B/C pairs (**componentArtboardB**/**componentMatIndexB**, **componentArtboardC**/**componentMatIndexC**). Each is rendered every frame into an offscreen canvas (resolution **componentSize**) and used as the base-colour texture of its material. The components' own animations, state machines and data bindings keep running.

With **componentInteractive** enabled (default), pointer events on the 3D surface are ray-cast against the mesh, converted to the surface's UV coordinates and forwarded into the component under the pointer (nearest surface wins), so buttons and hover states inside the components respond where you touch the model. Notes: forwarding uses the rest-pose geometry, presses landing on a component-textured surface take priority over orbit/selection, and the databound image override is ignored for materials driven by a component.

## Part picking

With **pickEnabled**, clicking a part selects and highlights it. Picking is triangle-accurate against the current (animated/skinned/morphed) pose by default; set **pickPrecise** off for the faster bounding-box test. The script writes to optional ViewModel properties (create the ones you need):

- String named by **pickPropName**: the part (node) name
- Number named by **pickIndexPropName**: the part-list ordinal, 0-based, -1 when nothing is selected
- Number named by **pickNodePropName**: the glTF node index of the part, stable for a given file, -1 when nothing is selected

Selection also flows the other way: write a glTF node index into the Number property named by **selectPropName** (from data binding or another script) and that part selects as if clicked (-1 clears).

Triggers: **pickTriggerName** fires a ViewModel Trigger whenever a new part is picked, **hoverTriggerName** when the hovered part changes. On runtimes without trigger support the script increments a Number property of the same name instead.

## Animation data

- TRS node animations and CPU skinning as before (animationIndex / animationSpeed / playAnimation / animationTime).
- Morph targets: POSITION/NORMAL targets blend on the CPU, driven by animated weights (LINEAR/STEP/CUBICSPLINE), node weights or mesh defaults — including sparse-encoded targets and morph+skin combinations.

## Notes

- You have many exposed properties available for testing; ask the agent to expose more if needed.
- It is strongly recommended to **always add a new GLB Loader node** into your artboard for each model, to avoid mismatches between models.
- `shaders/model.wgsl` is a reconstruction of the shader contract documented in GLBModel (the original WGSL asset was never in the repo). Replacing your **model** asset with it may shift the look slightly; every new feature degrades cleanly when something is missing (old shader asset = old behaviour, missing post asset = no post FX, missing env blob = gradient sky).
- KTX2/WebP textures are transcoded to RGBA on the CPU at load time. A 1024x1024 ETC1S texture takes around 100 ms, a 4096x4096 one a couple of seconds — prefer `gltfpack -tl 2048` (or similar) to cap texture sizes for snappy loads. Mip chains add roughly 0.1 s per 1024x1024 image; environment prefiltering about 0.25 s.
- WebP decoding first tries the runtime's native image decoder and falls back to the pure-Luau decoder when the build lacks WebP support.
- Decoders are validated byte-exact against the reference implementations (meshoptimizer, draco3d, zstd, basis_universal transcoder, libwebp/sharp) on fixtures generated with gltfpack and gltf-transform; both WGSL modules validate with naga (wgpu).

And voilà.
