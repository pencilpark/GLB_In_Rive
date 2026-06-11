# Couverture intégrale interactif design (PR #4)

Objectif: couvrir la liste de manques identifiée (interaction, rendu, shaders,
données) avec livrables fiables (tests CLI byte/valeur-exacts quand possible,
analyse stricte, compile checks; chemins runtime Rive gardés par pcall).

## Phases

- [x] P0 Doc API (mips/cube/aniso/compare/triggers) -- tout est exposé
- [x] P1a Agent A: GLBParser (caméras, morph targets, weights anim, factors
      clearcoat/transmission/sheen/specular/ior) + SceneGraph (pick mesh
      triangle-précis avec node par triangle, morph/skin aware, sampleClip
      weights) + Math3D (proj caméra glTF) + tests CLI (145 checks)
- [x] P1b Agent B: TextureOps.luau (chaîne de mips box-filter, préfiltrage
      env progressif, helpers rgba) + tests CLI
- [x] P2 Shader WGSL complet (contrat existant reconstruit depuis GLBModel +
      IBL équirect préfiltrée, ombres dir. PCF (depth packé), clearcoat/
      sheen/transmission approx, modes unlit/toon/wireframe, alphaCutoff) +
      entry points shadow + module post séparé (bloom/vignette/grade/DOF
      alpha-depth) + validation naga (les deux modules)
- [x] P3 GLBModel integration (un seul rédacteur = session principale):
      - [x] picking triangle-précis + pose animée/skinnée/morphée
            (pickPrecise, cache par animationIndex:animTime, overrides par
            node single-prim; fallback AABB)
      - [x] sélection entrante (selectPropName + listener, anti-écho),
            triggers sortants (pickTriggerName/hoverTriggerName + fallback
            compteur Number)
      - [x] caméra UX: pinch zoom (zoomMin/zoomMax), pan 2 doigts (centroïde
            -> offsets monde via colonnes camWorld), inertie (flick + decay),
            limites pitchMin/pitchMax, focusOnSelect (glide exponentiel)
      - [x] mips à l'upload (chaîne CPU TextureOps pour tous les décodés) +
            sampler trilinéaire + aniso si features().anisotropicFiltering
      - [x] IBL: blob equirect préfiltré CPU (7 niveaux) + gradient
            sky/horizon/ground fallback (envMode)
      - [x] ombre directionnelle light1: passe vs_shadow/fs_shadow depth
            packé rgba8, ortho fitté sur la sphère du modèle, PCF 5 taps,
            bind group frame séparé (dummy shadow tex)
      - [x] tri de transparence: alphaMode BLEND -> pipeline blend (depth
            write off) trié back-to-front; MASK cutoff dans le shader
      - [x] component-texture slots A/B/C (forward au plus proche hit)
      - [x] caméras GLB (cameraIndex, perspective + ortho reverse-Z, suit la
            pose animée) + morph targets (non-skinné: réécriture pos/normal
            in-place; skinné: morph puis skin)
      - [x] post FX: scène offscreen (+ resolve MSAA dedans), bright/blurH/
            blurV/composite, DOF par profondeur en alpha, pcall + désactivation
            propre en cas d'échec
      - frame UBO 352 -> 512 (env0, sky/horizon/ground, shadow0, lightVP,
        mode0); records étendus (alphaCutoff, clearcoat, ext1, sheenColor)
- [x] P4 README + suite complète (run_all + geomext + textureops + analyze
      10 modules + compile GLBModel) + PR draft

## Notes de fiabilité

- Le WGSL original n'est pas dans le repo: shaders/model.wgsl est une
  reconstruction du contrat documenté dans GLBModel (offsets UBO, slots,
  vertex layout) -- la PR le signale et l'apparence peut bouger légèrement.
- naga (wgpu) valide syntaxe + layouts des deux modules WGSL hors GPU.
- Les chemins exécutables uniquement dans Rive restent gardés par pcall.
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
