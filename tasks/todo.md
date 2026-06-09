# Support Draco / meshopt / Basis-KTX2 / WebP en pur Luau

Problème: GLBParser bloque les GLB compressés (KHR_draco_mesh_compression,
EXT_meshopt_compression, KHR_texture_basisu, EXT_texture_webp). Les outils de
compression (gltfpack, gltf-transform, export Blender) produisent des GLB
inutilisables dans Rive. Objectif: décoder les 4 formats en pur Luau.

## Plan

- [x] 1. Environnement de test: luau CLI + gltf-transform + gltfpack + fixtures
- [x] 2. MeshoptDecoder.luau (EXT_meshopt_compression: index codec, vertex codec v0/v1, filtres oct/quat/exp)
- [x] 3. Intégration meshopt dans GLBParser (bufferViews décodés à la volée) + test vs gltfpack
- [x] 4. DracoDecoder.luau (rANS, edgebreaker standard+valence, séquentiel, prédictions, déquantification)
- [x] 5. Intégration Draco dans GLBParser (overrides d'accessors par primitive) + test vs gltf-transform/Blender
- [ ] 6. ZstdDecoder.luau (RFC 8878: FSE, huffman, séquences) — requis pour KTX2/UASTC
- [ ] 7. KTX2Decoder.luau (conteneur KTX2, BasisLZ→ETC1S→RGBA, UASTC→RGBA, zstd)
- [ ] 8. WebPDecoder.luau (VP8L lossless, VP8 lossy intra, canal alpha) — fallback si decodeImage natif échoue
- [ ] 9. Intégration textures dans GLBModel (KTX2/WebP → upload GPUTexture direct)
- [ ] 10. Tests bout-en-bout: GLB compressés par gltfpack/gltf-transform décodés = référence (PSNR / exact)
- [ ] 11. README + commit + push + PR draft

## Décisions

- Décodage à l'import (parse), pas de dépendance externe: tout en pur Luau (buffer + bit32).
- meshopt: bufferViews décodés stockés dans glb.viewData[index]; les lecteurs passent par viewBuffer().
- Draco: décode → tableaux plats par accessor (glb.accessorOverrides), pas de re-packing binaire.
- WebP: on tente context:decodeImage (le runtime Rive peut savoir décoder WebP), sinon décodeur Luau.
- KTX2: transcodage CPU vers rgba8unorm puis upload direct (pas de formats GPU compressés exposés).

## Review

(à compléter en fin de session)
