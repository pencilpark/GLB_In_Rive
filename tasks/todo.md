# Support Draco / meshopt / Basis-KTX2 / WebP en pur Luau

Problème: GLBParser bloquait les GLB compressés (KHR_draco_mesh_compression,
EXT_meshopt_compression, KHR_texture_basisu, EXT_texture_webp). Les outils de
compression (gltfpack, gltf-transform, export Blender) produisaient des GLB
inutilisables dans Rive. Objectif: décoder les 4 formats en pur Luau.

## Plan

- [x] 1. Environnement de test: luau CLI + gltf-transform + gltfpack + fixtures
- [x] 2. MeshoptDecoder.luau (EXT_meshopt_compression: index codec, vertex codec v0/v1, filtres oct/quat/exp)
- [x] 3. Intégration meshopt dans GLBParser (bufferViews décodés à la volée) + test vs gltfpack
- [x] 4. DracoDecoder.luau (rANS, edgebreaker standard+valence, séquentiel, prédictions, déquantification)
- [x] 5. Intégration Draco dans GLBParser (overrides d'accessors par primitive) + test vs gltf-transform/Blender
- [x] 6. ZstdDecoder.luau (RFC 8878: FSE, huffman, séquences) — requis pour KTX2/UASTC
- [x] 7. KTX2Decoder.luau (conteneur KTX2, BasisLZ→ETC1S→RGBA, UASTC→RGBA, zstd)
- [x] 8. WebPDecoder.luau (VP8L lossless, VP8 lossy intra, canal alpha) — fallback si decodeImage natif échoue
- [x] 9. Intégration textures dans GLBModel (KTX2/WebP → upload GPUTexture direct)
- [x] 10. Tests bout-en-bout: GLB compressés par gltfpack/gltf-transform décodés = référence (PSNR / exact)
- [x] 11. README + commit + push + PR draft

## Décisions

- Décodage à l'import (parse), pas de dépendance externe: tout en pur Luau (buffer + bit32).
- meshopt: bufferViews décodés stockés dans glb.viewData[index]; les lecteurs passent par viewSource().
- Draco: décode → tableaux plats par accessor (glb.accessorOverrides), pas de re-packing binaire.
- WebP: on tente context:decodeImage (le runtime Rive peut savoir décoder WebP), sinon décodeur Luau.
- KTX2: transcodage CPU vers rgba8unorm puis upload direct (pas de formats GPU compressés exposés).
- En plus (nécessaire pour un rendu correct des fichiers gltfpack): accessors
  `normalized` (KHR_mesh_quantization) convertis dans readAccessorFloats, et
  KHR_texture_transform parsé puis baké dans les UV par GLTFMesh.

## Review

Tous les décodeurs sont validés byte-exact contre les implémentations de
référence, via un harnais luau CLI (fixtures base64) :

- meshopt: 51/51 comparaisons d'accessors exactes sur gltfpack -c/-cc, flux
  bruts v0+v1 byte-exact (filtres à 1 LSB près du chemin SIMD wasm, conformes
  au chemin scalaire C de référence).
- Draco: 57/57 exactes sur gltf-transform (edgebreaker standard, valence,
  séquentiel) + fixtures unitaires multi-attributs avec seams UV; 60 vecteurs
  rabs.
- zstd: 10/10 byte-exact (niveaux 1/3/19, huffman 255 poids, payload UASTC réel).
- KTX2: ETC1S et UASTC byte-exact vs transcodeur basis officiel, avec et sans
  tranche alpha; UASTC supercompressé zstd inclus.
- WebP: 10/10 byte-exact vs sharp/libwebp (lossy q30–q95, alpha, lossless
  photo/palette, dimensions impaires, 320x200).
- Bout-en-bout: starship.glb complet compressé gltfpack -cc -tc (meshopt +
  ETC1S 4096²) parse en ~26 ms; géométrie identique à la référence; mixed
  draco+ktx2 OK. ETC1S 4096² ≈ 1.5–2.3 s/texture (CPU, niveau 0 seulement).
- luau-analyze sans erreur sur les 9 modules (mode strict).

# Geometry extensions: cameras, morph targets, pick mesh, glTF projections

Branch agent-a-geometry.

- [x] GLBParser: glTF cameras (flattened Camera type, GLBData.cameras, Node.camera)
- [x] GLBParser: morph targets (Primitive.targets, Mesh.weights, Node.weights)
- [x] GLBParser: material extension factors (clearcoat, transmission, sheen, specular, ior)
- [x] GLBParser fix: accessor.bufferView keeps nil (was `or 0`), so sparse-only
      accessors (zero base) no longer alias bufferView 0 -- required for sparse
      morph deltas; Draco accessors unaffected (overrides short-circuit first)
- [x] SceneGraph: sampleClip second return = per-node morph weights
      (LINEAR/STEP/CUBICSPLINE), first return unchanged
- [x] SceneGraph: applyMorphs (base + sum w_i * delta_i, POSITION/NORMAL)
- [x] SceneGraph: PickMesh / buildPickMesh / rayPickMesh (triangle-precise,
      per-triangle node id, optional world-space position overrides)
- [x] Math3D: perspectiveGltfReverseZ (zfar nil = infinite far), orthographicReverseZ
- [x] Tests in /tmp/t/luau_A: hand-built fixture GLB (sparse morph target,
      weights anims, cameras, extension materials); 145 checks + full
      regression suite green; luau-analyze clean on all touched modules
