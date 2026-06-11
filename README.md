# GLB_In_Rive

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
`MeshoptDecoder`, `DracoDecoder`, `ZstdDecoder`, `KTX2Decoder`, `WebPDecoder`.

## Component as interactive texture

Assign a component (nested artboard) to the **componentArtboard** input and it is rendered every frame into an offscreen canvas, then used as the base-colour texture of the material selected by **componentMatIndex** (resolution: **componentSize**, default 512). The component's own animations, state machines and data bindings keep running.

With **componentInteractive** enabled (default), pointer events on the 3D surface are ray-cast against the mesh, converted to the surface's UV coordinates and forwarded into the component, so buttons and hover states inside the component respond where you touch the model. Notes: forwarding uses the rest-pose geometry (same as part picking), presses landing on the component-textured surface take priority over orbit/selection, and the databound image override is ignored for that material while the component drives it.

## Part picking

With **pickEnabled**, clicking a part selects and highlights it, and the script writes to optional ViewModel properties (create the ones you need):

- String named by **pickPropName**: the part (node) name
- Number named by **pickIndexPropName**: the part-list ordinal, 0-based, -1 when nothing is selected
- Number named by **pickNodePropName**: the glTF node index of the part, stable for a given file, -1 when nothing is selected

## Notes

- This `.rev` is an example using a relatively heavy GLB file with no animation.
- You have many exposed properties available for testing.
- If needed, you can ask the agent to expose or add more properties.
- It is strongly recommended to **always add a new GLB Loader node** into your artboard for each model, to avoid mismatches between models.
- KTX2/WebP textures are transcoded to RGBA on the CPU at load time; only mip level 0 is used. A 1024x1024 ETC1S texture takes around 100 ms, a 4096x4096 one a couple of seconds — prefer `gltfpack -tl 2048` (or similar) to cap texture sizes for snappy loads.
- WebP decoding first tries the runtime's native image decoder and falls back to the pure-Luau decoder when the build lacks WebP support.
- Decoders are validated byte-exact against the reference implementations (meshoptimizer, draco3d, zstd, basis_universal transcoder, libwebp/sharp) on fixtures generated with gltfpack and gltf-transform.

And voilà.
