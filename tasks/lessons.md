# Lessons (GLB_In_Rive)

Notes techniques accumulées en implémentant les décodeurs purs Luau, pour ne
pas retomber dans les mêmes pièges.

## Méthode

- Porter depuis le code C/C++ de référence, jamais depuis une spec
  pseudo-codée seule: la spec Draco officielle contient des écarts réels avec
  draco C++ (octet `compressed` toujours lu même sans prédiction, init rANS
  des décodeurs binaires AVEC ajout de la base 4096 — un extrait tronqué du
  source m'avait fait croire le contraire).
- Extraire les grosses tables (probas VP8, partitions UASTC...)
  mécaniquement avec un script depuis le source de référence. Attention aux
  littéraux hexadécimaux (`0x35`) cassés par une regex décimale, et aux
  lignes `{a, b}` avec padding implicite dans des tableaux `[N][3]`.
- Toujours valider contre l'implémentation officielle (wasm npm: draco3d,
  meshoptimizer, basis via three.js, sharp/libwebp, node zstd) avec des
  comparaisons byte-exactes; un PSNR "correct" cache des bugs réels.
- luau CLI: pas d'io — embarquer les fixtures en base64 dans des modules.
  `require('X')` style Rive doit être réécrit en `require('./X')` pour le CLI
  (script de sync avec sed).

## Pièges précis rencontrés

- Luau buffer: les fenêtres de lecture multi-octets débordent vite —
  toujours borner par `buffer.len` ET le début de région (bits < 0 lus comme
  zéro pour les bit-readers backward zstd).
- zstd: après consommation des poids huffman, repositionner avec la taille
  compressée ORIGINALE (pas décrémentée) → sinon désynchronisation différée.
  La boucle FSE des poids émet le symbole DÈS formation de l'état et stoppe
  quand la lecture suivante passerait sous le début (fzstd), max 255 poids.
- Draco: les prédicteurs (texcoords, normales) lisent l'attribut parent
  PORTABLE (entiers quantifiés), pas les valeurs déquantifiées. IntSqrt sur
  produits 64 bits: les doubles ne suffisent pas, comparer via limbes 26 bits.
- Float32-exactitude: émuler avec
  `buffer.writef32/readf32` quand la référence calcule en float
  (déquantification draco, filtres meshopt, normales octaédriques).
  Divisions C sur entiers négatifs: troncature vers zéro, pas floor.
- VP8: la réplication top-right des blocs 4x4 utilise une arithmétique de
  pointeur uint32 (top_right[BPS] = +4 lignes d'octets). Le dispatch UV:
  si un coefficient AC existe dans le canal, TransformOne sur les QUATRE
  blocs; sinon TransformDC par bloc si DC non nul.
- ETC1S 4096²: precomputer une LUT de 4 couleurs packées par endpoint et
  écrire en u32 → ~2x. Les écritures par octet dominent sinon.

## WGSL / GPU Rive (PR #4)

- wgsl_reflect (npm) a des exports cassés (WgslReflect non constructible en
  require ET import): valider les WGSL avec naga-cli (cargo install naga-cli)
  — c'est le validateur wgpu de référence.
- naga refuse `let _ = expr` (identifiant `_` invalide): écrire le bloc
  proprement avec select/mix plutôt que de "consommer" une binding.
- Layouts de bind group dérivés du shader (GPUBindGroupLayout.new({shader,
  groupIndex})): quand on étend le groupe 0, créer le bind group riche sous
  pcall et retomber sur la version UBO-seule → un asset shader ancien garde
  le comportement legacy au lieu de tuer le node.
- Passe d'ombre sans comparison sampler: packer la profondeur en rgba8
  (fract cascade), pipeline séparé en z standard (clear 1, less) même si la
  passe principale est reverse-Z — seule la cohérence interne compte.
- Accessors glTF sparse: bufferView ABSENT est légal (base zéro); `or 0`
  aliasait silencieusement la view 0 (corrigé par agent A — piège classique
  de défaut trop optimiste sur un champ optionnel).
- Tri de transparence sans re-upload: lire la translation (octets 48/52/56)
  du record déjà packé et la projeter sur la rangée z de la view.
