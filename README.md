# GLB_In_Rive

File embedding all scripts to add GLB file with all embedded properties.

## Overview

This repository provides a GLB loader node for Rive artboards, designed to import `.glb` files with their embedded properties already handled.

## How to use

1. Add a fresh **GLB Loader** node to your artboard.
2. Import a `.glb` file and rename it however you like.
3. In the node properties, set the **textBlob** field to exactly match the asset name used in your Rive assets.
4. You do not need to manage textures manually — the script handles them for you.
5. All animations baked into the GLB are indexed in the exposed properties.
6. At the very bottom of the properties, choose the animation index you want to use (`0`, `1`, `2`, `3`, etc.), depending on how many animations are included in your GLB.

## Notes

- This `.riv` is an example using a relatively heavy GLB file (~9 MB) with 5 animations.
- You have many exposed properties available for testing.
- If needed, you can ask the agent to expose or add more properties.
- It is strongly recommended to **always add a new GLB Loader node** into your artboard for each model, to avoid mismatches between models.

And voilà.
