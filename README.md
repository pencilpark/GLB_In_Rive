# GLB_In_Rive

This `.rev` setup contains everything needed to import a GLB into Rive.

## How to use

1. Add a **new GLB Loader node** on your artboard (recommended every time to avoid model mismatch).
2. Import your `.glb` file and rename it however you want (just keep the exact same name in `textBlob`).
3. In the GLB Loader properties, set **`textBlob`** to the exact asset name used in Rive.
4. Textures/material handling is already managed by the script.
5. All baked animations are exposed by index in the properties panel.
6. In the animation field at the bottom, enter a **single index**: `0` for the first animation, `1` for the second, `2` for the third, and so on.

This repository example uses a heavy GLB (~9 MB) with 5 animations, and you can expose/add extra properties as needed.
