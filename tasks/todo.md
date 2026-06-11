# Couverture intégrale interactif design (PR #4)

Objectif: couvrir la liste de manques identifiée (interaction, rendu, shaders,
données) avec livrables fiables (tests CLI byte/valeur-exacts quand possible,
analyse stricte, compile checks; chemins runtime Rive gardés par pcall).

## Phases

- [ ] P0 Doc API (mips/cube/aniso/compare/triggers) -- FAIT, tout est exposé
- [ ] P1a Agent A: GLBParser (caméras, morph targets, weights anim, factors
      clearcoat/transmission/sheen/specular/ior) + SceneGraph (pick mesh
      triangle-précis avec node par triangle, morph/skin aware, sampleClip
      weights) + Math3D (proj caméra glTF) + tests CLI
- [ ] P1b Agent B: TextureOps.luau (chaîne de mips box-filter, préfiltrage
      env progressif, helpers rgba) + tests CLI
- [ ] P2 Shader WGSL complet (contrat existant reconstruit depuis GLBModel +
      IBL équirect préfiltrée, ombres dir. PCF (depth packé), clearcoat/
      sheen/transmission approx, modes unlit/toon/wireframe, alphaCutoff) +
      entry points shadow + post (bloom/vignette/grade/DOF alpha-depth) +
      validation wgsl_reflect (layout des bind groups == Luau)
- [ ] P3 GLBModel integration (un seul rédacteur = session principale):
      - picking triangle-précis + picking sur pose animée/skinnée
      - sélection entrante (propriété Number + listener), triggers sortants
      - caméra UX: pinch zoom, pan 2 doigts, inertie, limites orbit,
        zoom-sur-partie
      - mips à l'upload (KTX2 tous niveaux + chaîne CPU pour PNG/WebP),
        sampler trilinéaire + aniso (feature-gated)
      - IBL: texture env (blob) préfiltrée CPU + gradient procédural fallback
      - ombre directionnelle light1 (passe depth-only + depthBias)
      - tri de transparence back-to-front + alphaCutoff MASK
      - component-texture multi-slots (A/B/C)
      - caméras GLB (cameraIndex) + morph targets branchés
      - post FX: scène offscreen + passe plein écran
- [ ] P4 README + suite complète + PR draft

## Notes de fiabilité

- Le WGSL original n'est pas dans le repo: shaders/model.wgsl est une
  reconstruction du contrat documenté dans GLBModel (offsets UBO, slots,
  vertex layout) -- la PR le signale et l'apparence peut bouger légèrement.
- wgsl_reflect (npm) valide syntaxe + layout des bindings hors GPU.
- Les chemins exécutables uniquement dans Rive restent gardés par pcall.
