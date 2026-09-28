# Design Notes — Toward a General Embedding Workbench

*Working design document. Status: exploratory / not committed. Written 2026-09-28 against plugin version 0.10.0 (experimental).*

This note captures the ideas discussed for taking the AlphaEarth Toolbox beyond its
current AlphaEarth-only scope, and the accompanying decision about whether that work
belongs in the existing plugin or in a new one. It is a design and decision record,
not release documentation — it is deliberately kept out of the distributed combined
`.docx` (like `SMOKE_TEST.md` and `RELEASE_NOTES.md`).

Three longer-horizon initiatives are in scope:

1. **Pluggable embedding backends** (Clay, Prithvi, SatlasPretrain) — turn an
   AlphaEarth-only tool into a general embedding workbench.
2. **Active-learning labelling loop** — an iterative train → query → label → retrain
   cycle that deepens the analytical side.
3. **Targeted preset profiles** — WUI fuel / land cover, built-up fraction, and canopy
   — domain-ready starting configurations.

The bottom-line recommendation, argued below, is to **extend the existing plugin in
place** for all three, gating any heavy machine-learning inference behind an optional
dependency tier exactly as scikit-learn is gated today, and to reserve a genuinely
separate plugin for the single case where inference becomes a first-class, actively
maintained capability with its own audience — at which point the clean split is a
shared `aecore` library consumed by thin plugins, not a duplicated codebase.

---

## 1. Where the plugin stands today

The toolbox is a QGIS Processing provider (`alphaearth`) with twelve algorithms and a
guided tabbed wizard. Its architecture is deliberately layered:

- **`aecore/`** — pure compute cores (NumPy + a thin GDAL glue layer). Import-light and
  unit-testable *without* QGIS or a full geospatial stack. This is where the real work
  lives: intake/tile discovery, RGB reduction, similarity, change, the ML core, presets.
- **`algorithms/`** — thin `QgsProcessingAlgorithm` wrappers that marshal parameters and
  call the cores. They add no compute of their own.
- **`gui/dialog.py`** — the tabbed wizard (Load/Visualise, Classify, Similarity, Change),
  which runs algorithms off the GUI thread via `QgsProcessingAlgRunnerTask` and simply
  builds parameter dictionaries for existing algorithms.

Two properties of the current design are load-bearing for everything below:

- **Zero-dependency core.** The headline promise, stated in the README, is "no Earth
  Engine account, API key or GPU, and no extra Python packages for the core workflow."
  scikit-learn is the *only* optional dependency, and it is never hard: it is detected at
  runtime (`algorithms/_sklearn.py`), the wizard offers a one-click install when it is
  missing, and every ML enhancement is opt-in and default-off with a guard test.
  matplotlib is treated the same way for the PNG/PDF report outputs.
- **The unit-vector property.** AlphaEarth pixels are L2-normalised 64-D vectors, so
  cosine similarity equals an exact dot product. `aecore/similarity.py` and the cosine
  path of `aecore/change.py` exploit this directly. It is an AlphaEarth-specific fact,
  and — as Section 2 explains — it is exactly the assumption a multi-backend design must
  stop taking for granted.

The data path is an *archive fetch*: a year (2017–2025) resolves to 64-band int8 COGs at
10 m in the relevant UTM zone on the public `gs://alphaearth_foundations` bucket, read
anonymously through `/vsicurl`, dequantised (÷127.5) to float, and mosaicked onto the
requested grid. `algorithms/_fetch.py` and `aecore/intake.py` own this path.

---

## 2. Initiative 1 — Pluggable embedding backends

### 2.1 The framing fact: archive vs inference

The single most important observation is that **AlphaEarth is unusual in shipping a
global, pre-computed, annual embedding *archive* on public cloud.** Clay, Prithvi and
SatlasPretrain are — at least as of current knowledge — primarily **models you run**: you
feed them your own Sentinel-2 / Landsat / HLS imagery and they emit embeddings from a
(usually GPU-friendly) forward pass. There is no equivalent global tile bucket to
`/vsicurl` against.

This splits "backends" into two families with very different acquisition paths:

- **Archive backends** obtain a cube by *fetching* pre-computed embeddings for an
  AOI + time. This is exactly today's `_fetch` / `intake` path (UTM-zone tile discovery,
  scene culling, caches, cross-CRS mosaic warp). AlphaEarth is the only one available.
- **Inference backends** obtain a cube by *ingesting source imagery* (Sentinel-2 via
  STAC — the same muscle used in the burnt-area toolbox) and running a model encoder over
  it. That needs `torch`, model weights, tiling, and memory management. Clay, Prithvi and
  Satlas all live here.

> **Verify before committing.** Whether each of Clay / Prithvi / Satlas is inference-only
> or offers any fetchable product moves over time and should be re-checked before Phase D
> below is scheduled — it materially changes that phase's cost.

### 2.2 The abstraction

Introduce a new `aecore/backends/` package with a protocol that separates *acquisition*
from *semantics*:

```python
# aecore/backends/_base.py  (pure: no QGIS, no torch)
@dataclass(frozen=True)
class EmbeddingDescriptor:
    n_bands: int
    is_unit_normalized: bool      # AlphaEarth True; Clay/Prithvi/Satlas almost certainly False
    native_dtype: str             # "int8" for AlphaEarth; "float32" for model outputs
    nodata: float                 # AlphaEarth fill -128 → sentinel
    dequant_scale: float | None   # AlphaEarth 127.5; None for float model outputs
    temporal: Literal["annual", "monthly", "scene"]
    available_times: tuple[str, ...]
    resolution_m: float

@dataclass(frozen=True)
class BackendCaps:
    needs_gpu: bool
    needs_auth: bool
    extra_deps: tuple[str, ...]   # e.g. ("torch", "<weights-package>")

class EmbeddingBackend(Protocol):
    id: str
    display_name: str
    def descriptor(self) -> EmbeddingDescriptor: ...
    def capabilities(self) -> BackendCaps: ...
    def fetch(self, aoi_wkt: str, target_crs: str, resolution: float,
              time: str, feedback) -> "np.ndarray": ...   # float32 cube on the requested grid
```

`backends/alphaearth.py` implements the protocol by delegating to the existing
`_fetch` / `intake` / `_raster` code — **a pure refactor with zero behaviour change.** The
fetch-based algorithms (`loadembedding`, `loadyears`, `changeyears`, `classifyyears`) gain
a `BACKEND` enum parameter defaulting to AlphaEarth; the wizard gains a backend selector
that defaults invisibly to AlphaEarth, so nothing changes for existing users.

### 2.3 The one real correctness risk

The unit-vector assumption must stop being implicit. A non-normalised backend would make
`similarity.py` and the cosine path of `change.py` produce *wrong maps with no error*.
The fix is to make the metrics descriptor-aware:

- A shared `aecore/backends/_embed.py::ensure_unit(cube, descriptor)` L2-normalises per
  pixel when `descriptor.is_unit_normalized` is `False`, and is a no-op when it is `True`.
- Cosine metrics normalise (via `ensure_unit`); Euclidean distance leaves magnitude
  intact, because normalising there would be wrong. The metric functions therefore take
  the descriptor and choose correctly.

RGB/PCA and the classify/cluster/regress cores are already band-count generic
(`A00…A{n-1}`), so they need no change beyond a dynamic band count. However
`aecore/model_io.py`, which today stamps a saved model with the scikit-learn version and
band count, must **also stamp the backend id**, so a Clay-trained model cannot be silently
reused on an AlphaEarth cube (the mismatch should fail loudly, as version/band mismatches
already do).

### 2.4 Dependency handling

`torch` and per-backend weights must never become hard dependencies — this is the same
principle that keeps scikit-learn optional. A `backends/_deps.py` mirrors
`algorithms/_sklearn.py`: it probes for a backend's `extra_deps`, offers a one-click
guided install into the QGIS Python environment, and makes an inference backend decline
cleanly (skip-with-message, never crash) when they are absent. Weights are opt-in
downloads, never bundled into the plugin ZIP.

### 2.5 Effort

- Protocol + AlphaEarth-behind-it + normalisation-aware metrics + model_io backend
  stamp: **M**. Mechanical refactor; the metric change is the only delicate part.
- Each inference backend (Clay first): **L / XL** — torch, weights hosting,
  Sentinel-2 / STAC imagery ingestion, tiled GPU-optional inference, memory management.
  This is where scope explodes, which is why the phasing quarantines it.

---

## 3. Initiative 2 — Active-learning labelling loop

### 3.1 Why it cannot be only an algorithm

Active learning is iterative and stateful: train → predict → surface the most informative
pixels → user labels them → retrain. That fights the stateless `QgsProcessingAlgorithm`
model, and a *modal* wizard blocks the user from editing the query layer. So the
interactive loop belongs in a **non-modal `QgsDockWidget`** — a new UI surface for the
plugin and the main UX cost of this initiative.

### 3.2 Pure core

`aecore/active.py` (NumPy, unit-testable, no QGIS) holds the acquisition maths over
per-pixel class probabilities:

- `margin` — `1 − (top1 − top2)`; smallest-margin pixels are most informative.
- `entropy` and `least_confidence` — alternative uncertainty measures.
- `vote_entropy` — query-by-committee, which a random forest gives for free from its
  per-tree votes.
- `select_query_points(score, n, mask, min_separation)` — picks the top-N uncertain
  pixels *with spatial diversity* (a minimum pixel separation and/or stratification by
  currently-predicted class), so queries do not all clump in one confusing corner.

### 3.3 Scriptable seam + the loop

A thin `algorithms/query_points.py` (`alphaearth:querypoints`) runs the classify predict
and emits the top-N uncertain points as a layer. This keeps the pure-core / thin-wrapper /
GUI layering intact and makes the acquisition step testable in the existing QGIS-backed
suite, independent of the interactive UI.

The dockwidget then drives the cycle: initial labels + cube → train (reuse the classify
core) → predict probabilities → run acquisition → add a styled "query points" layer with
an empty class field → user labels those points in QGIS → "Add to training & retrain"
merges the new labels and loops, showing a per-round accuracy / uncertainty trend. State
lives naturally in the project: the growing training layer is a real editable vector
layer, the model sits in a temp `.pkl` via `model_io`, and the panel tracks the round
count and history. A lighter *seed-refinement* mode reuses the same loop against
Similarity's seeds instead of class labels.

### 3.4 Effort

Pure core **S / M**; the `querypoints` algorithm **S**; the interactive dockwidget loop
**M / L** (the edit-and-retrain UX, and it introduces the plugin's first dockable panel).

---

## 4. Initiative 3 — Targeted preset profiles

This is the lightest lift and the highest immediate user value, building directly on
`aecore/presets.py` (frozen dataclasses of *semantic* values surfaced as wizard
dropdowns).

A **profile** is a higher-level bundle than a preset — a named, domain-targeted
configuration that may span an algorithm choice, a suggested class scheme, tuned
parameters, and (optionally) a shipped model or seed set:

```python
# aecore/profiles.py
@dataclass(frozen=True)
class Profile:
    id: str
    display_name: str
    target_algorithm: Literal["classify", "cluster", "regress", "similarity"]
    class_scheme: tuple[ClassDef, ...]        # ordered labels + palette colours
    suggested_params: Mapping[str, object]
    applicable_backends: tuple[str, ...]      # a profile trained on AlphaEarth shouldn't claim Clay
    bundled_model_ref: str | None = None      # opt-in download, never in the ZIP
    provenance: str = ""                       # + a "not a substitute for field validation" note
```

The three named profiles map cleanly onto the sibling packages' domain knowledge:

- **WUI fuel / land cover** — a classify profile with a suggested scheme (e.g.
  built / grass / shrub / forest / water / bare) and tuned RF parameters; reuses
  `alphaearth-classify` / `alphaearth-lulc`.
- **Built-up fraction** — a regress profile, or the cluster → label-the-built-cluster
  workflow from `alphaearth-urban-growth` (where cluster 6 was found to be built-up).
- **Canopy** — a regress/classify profile for tree cover.

Profiles surface as a "Profile" dropdown on the relevant wizard tab that fills the class
scheme and parameters exactly as presets do now. The only coupling to Initiative 1 is
`applicable_backends`, so the profile registry should be backend-aware — but it does not
*block* on Initiative 1; profiles can ship AlphaEarth-only first.

**Recommendation:** ship profiles as configuration-only first (schemes + params +
workflow guidance). Add bundled, validated, backend- and version-stamped pre-trained
models later as separate opt-in downloads, so the plugin ZIP does not bloat and every
shipped model is clearly a starting point that needs local validation.

### 4.1 Effort

Registry + wizard wiring **S**; each profile's config content **S**; a *validated*
bundled model per profile **M each** (needs training data, validation, and licensing/size
thought).

---

## 5. Extend the plugin, or build a new one?

The three initiatives do not split evenly, so "separate plugin" is only a live question
for one of them.

### 5.1 Initiatives 2 and 3 clearly belong in the existing plugin

Active learning reuses the classify/regress cores in `aecore/ml.py` and the
model-save/reuse plumbing; profiles are a richer layer over `aecore/presets.py` with the
same wizard-dropdown surfacing. Pulling either into its own plugin would force either
duplicating those cores (drift risk) or making one plugin import another — and **QGIS has
no clean inter-plugin import mechanism.** The `plugin_dependencies` metadata field only
nudges install order; it does not give a reliable `import`, so cross-plugin reuse means
fragile `sys.path` and load-order hacks.

### 5.2 The abstraction of Initiative 1 also belongs in the existing plugin

The `EmbeddingBackend` protocol with AlphaEarth refactored behind it (Phase A) is a
no-behaviour-change refactor that keeps AlphaEarth the silent default and makes the code
cleaner regardless. Splitting it out gains nothing.

### 5.3 The real tension: heavy inference backends

The separate-plugin question genuinely arises only for the heavy *inference* backends
(Clay, Prithvi, Satlas), because they drag in `torch`, model weights and imagery
ingestion — and that collides head-on with the plugin's stated ethos of a zero-dependency
core. A user who installs a zero-dependency tool and then finds it wants to pull a couple
of gigabytes of PyTorch is a real ethos violation. That brand/ethos protection is the
strongest argument for a separate artefact — stronger than any code-organisation
argument.

But note what "separate plugin" would actually require. The backends are only useful *in
combination with* the analysis algorithms (similarity, change, classify, cluster), which
are the bulk of the codebase. A standalone backends plugin would have to move all twelve
algorithms across or duplicate them. In other words, "separate plugin for backends"
collapses into "fork or rebrand the whole plugin" — a much bigger decision than it first
sounds.

### 5.4 The three options

1. **One plugin, inference gated as an optional tier.** Add the backends behind the same
   detect-and-guided-install pattern as scikit-learn (`_sklearn.py` → `backends/_deps.py`).
   Torch stays opt-in and default-off; the no-dependency AlphaEarth workflow is untouched;
   and all inter-plugin coupling pain is avoided. **Recommended for at least the first
   inference backend.**
2. **Rebrand the existing plugin** toward a general "Satellite Embedding Workbench" with
   AlphaEarth as one backend. Keep the provider id `alphaearth:*` (changing it breaks any
   saved Models or scripts that reference those ids); broaden only the display name and add
   a backend parameter. This is option 1 plus a rename, not a new codebase.
3. **Extract `aecore` into a pip-installable library**, then have thin plugins consume it.
   This is the only structure in which a genuinely separate plugin makes sense: a lean
   zero-dependency AlphaEarth plugin *and* a heavier ML-inference companion off one shared
   core, without duplication. The cost is real: install friction for QGIS users (the
   library must reach QGIS's Python) and a doubled packaging / CI / docs / version-bump
   burden on top of the existing four-gate + docx + ZIP pipeline. Defer until at least one
   inference backend has proven the abstraction earns its keep.

### 5.5 A note on the one-plugin-per-domain habit

The existing plugins (bal, burnt-area, alphaearth) are one-per-*domain* — BAL rating,
dNBR burn severity, embeddings. Backends, active learning and profiles are not a new
domain; they deepen the *same* embeddings domain. So the analogy that would normally push
toward a split does not really apply here.

### 5.6 Decision (leaning, not locked)

Put Initiatives 2 and 3 in the existing plugin without hesitation; do the Initiative 1
abstraction there too; gate the heavy inference backends as an optional torch tier rather
than as a new plugin; and only reach for a truly separate plugin — via a shared `aecore`
library — if and when inference becomes a first-class, actively maintained capability with
its own audience.

---

## 6. The flow (dependency-ordered phasing)

Dependencies drive the order. Initiative 1's abstraction touches the deepest layer (fetch
plus metric semantics), so doing it first means Initiatives 2 and 3 are born
backend-aware rather than retrofitted — but Initiative 1's *inference backends* are the
expensive, risky part and should be deferred until the abstraction is proven. Initiative 3
is cheapest and only lightly couples to Initiative 1. Initiative 2's core is independent
but adds a new UI surface.

```
Phase A ── Backend abstraction (keystone, no new backend)          [M]
   EmbeddingBackend protocol + EmbeddingDescriptor; AlphaEarth
   refactored behind it; similarity/change made normalisation-aware;
   model_io stamps backend id; backend selector defaulting to
   AlphaEarth. Zero behaviour change; 266 tests still green + new
   descriptor/normalisation tests.
        │
        ├─► Phase B ── Targeted profiles, config-only              [S/M]
        │      aecore/profiles.py + wizard "Profile" dropdown;
        │      applicable_backends guard. WUI / built-up / canopy as
        │      config + guidance, no bundled weights yet.
        │
        └─► Phase C ── Active-learning loop                        [M/L]
               aecore/active.py → alphaearth:querypoints (scriptable,
               testable) → non-modal dockwidget loop. Reuses B's class
               schemes as starting points. Independent of A's inference
               work.
                    │
Phase D ── First inference backend: Clay, new torch-gated tier     [L/XL]
   torch + weights + Sentinel-2/STAC ingestion behind the same
   detect-and-guided-install pattern as scikit-learn; declared via the
   Phase-A protocol as a non-unit-normalised inference backend. Proves
   the abstraction end-to-end.  ⚠ Verify Clay availability first.
        │
Phase E ── (optional) Bundled/validated profile models +           [L each]
   Prithvi & Satlas backends, once Clay validates the pattern.
```

This front-loads the cheap architectural keystone (A) and the cheapest user value (B),
slots active learning (C) where it can reuse B's schemes, and quarantines the
expensive/uncertain inference work (D/E) behind a proven abstraction.

Throughout, the house patterns are preserved: pure `aecore/` core + thin `algorithms/`
wrapper + GUI; every heavy dependency (scikit-learn, matplotlib, torch, weights) stays
opt-in and default-off with a guard test; the ZIP is built to a tempfile then copied via
`shutil.copyfile` (OneDrive locks block truncate-in-place); and a version bump spans
`metadata.txt`, `__init__.py` and `pyproject.toml`, with the six markdown docs plus the
combined `.docx` regenerated via `scripts/build_docs_docx.py`.

---

## 7. Open decisions before any of this starts

- **Verify current Clay / Prithvi / Satlas availability** (inference-only vs any fetchable
  product) — this gates Phase D's real cost.
- **Confirm the active-learning dockwidget** is acceptable as the plugin's first non-modal
  panel, or whether the loop should be constrained to stay inside the modal wizard.
- **Decide the rebrand question** (option 2) — whether/when to broaden the display name
  toward "Embedding Workbench" while keeping the `alphaearth:*` provider id.

---

## Appendix A — Candidate inference backends

*Specifications verified 2026-09-28 against each project's own sources (see below). Model
versions and licences move quickly — re-check before Phase D is scheduled.*

The single most important finding: **all three candidates are inference models you run
over your own imagery — none ships a global, pre-computed embedding archive the way
AlphaEarth does.** That is what places every one of them in the "inference backend" family
of Section 2.1, and it is why all three carry `EmbeddingDescriptor.is_unit_normalized =
False`: none produces the L2-normalised per-pixel vectors that the cosine-equals-dot-product
shortcut in `similarity.py` / `change.py` assumes, so the normalisation-aware metric change
in Phase A is a prerequisite, not an optional extra.

### Clay (v1.5)

An open-source Earth-observation foundation model: a Vision Transformer trained by
self-supervised learning with a Masked Autoencoder objective. It takes satellite imagery
plus metadata (location and acquisition time) and outputs embeddings. It is explicitly
**multi-sensor** — Sentinel-2, Landsat, NAIP, Sentinel-1 and more — because each input
carries that sensor's band centre-wavelengths, so the model is not hard-wired to one band
layout. Installed via `pip install git+https://github.com/Clay-foundation/model`, with
`clay-v1.5.ckpt` weights from Hugging Face (`made-with-clay/Clay`). Code and weights are
**Apache-2.0**; docs are CC-BY. Now a program of Renaissance Philanthropy (previously
fiscally sponsored by Radiant Earth). Clay publishes embeddings of its *training* data on
Source Cooperative (ODC-BY), but that is a fixed research dataset, not a queryable global
archive — so it remains an inference backend for our purposes. Of the three it is the
best-documented, most sensor-flexible, and produces the cleanest embedding output, which is
why it is the recommended first inference backend (Phase D).

### Prithvi-EO-2.0 (IBM + NASA)

A temporal-first foundation model: a multi-temporal ViT MAE built to reason over image time
series. The 2.0 family comes in 300M and 600M sizes, each also in a "-TL" variant adding
explicit temporal and location embeddings (smaller 100M/5M members exist too). Pre-trained
on NASA's Harmonized Landsat-Sentinel (HLS) V2 product at **30 m**, on 4.2M samples, on a
fixed six-band input in order — Blue, Green, Red, narrow-NIR, SWIR-1, SWIR-2 — in
reflectance units. Because band set and resolution are baked into pre-training, it is far
less input-flexible than Clay: you feed it HLS-style six-band geotiffs, chronologically for
the temporal modes. Run through IBM's TerraTorch framework (or the provided `inference.py`).
Encoder token dimensions are roughly 1024 (300M) / 1280 (600M). Licensing is permissive
(Apache-2.0 — confirm per weight file). Its 30 m HLS lineage suits coarser multi-date
analysis but is a poor match for AlphaEarth-like 10 m detail, and its STAC ingestion path
would target HLS rather than raw Sentinel-2.

### SatlasPretrain (Allen Institute for AI / AI2)

Structurally different: a pre-trained **backbone**, not an embedding generator per se. Each
model is a backbone (Swin-v2 Base/Tiny, or ResNet-50/152) plus a feature pyramid network
(FPN) plus a prediction head, trained on Sentinel-2, Sentinel-1 and NAIP. It comes in
single-image ("SI") and multi-image ("MI", max-pooled over images) flavours and in RGB or
9-band multispectral ("MS") input variants. Installed via `pip install
satlaspretrain-models`, loaded by checkpoint ID such as `Sentinel2_SwinB_SI_RGB`;
checkpoints are ODC-BY. The important catch: its output is a **multi-scale feature map**,
not a flat per-pixel embedding vector, so using it as an embedding backend means choosing a
feature level and pooling/resampling it into a per-pixel raster — the least natural fit of
the three for a per-pixel similarity/change workflow. Note also that SatlasPretrain (the
models) is distinct from **Satlas**, AI2's separately published global thematic data
products (solar farms, wind turbines, tree cover, …), which are finished pre-computed map
layers, not embeddings.

### Comparison

| | Clay v1.5 | Prithvi-EO-2.0 | SatlasPretrain |
|---|---|---|---|
| Maker | Clay Foundation / Renaissance Philanthropy | IBM + NASA | Allen Institute for AI |
| Architecture | ViT + MAE (SSL) | Multi-temporal ViT + MAE | Swin-v2 / ResNet + FPN + head |
| Input sensors | Multi-sensor (S2, Landsat, NAIP, S1, …) | HLS (6 fixed bands) | S2 (RGB or 9-band MS), S1, NAIP |
| Native resolution | Sensor-native (e.g. 10 m for S2) | 30 m | ~10 m (S2) / finer (NAIP) |
| Temporal handling | Per-scene + time metadata | Native multi-date stacks | Single- or multi-image |
| Output form | Patch/token embeddings | Token embeddings (~1024 / ~1280-D) | Multi-scale feature maps (needs pooling) |
| Unit-normalised? | No → L2-normalise for cosine | No → L2-normalise for cosine | No; not a flat vector at all |
| Acquisition | Inference (no global archive) | Inference (no global archive) | Inference; separate Satlas *products* exist |
| Licence | Apache-2.0 (code + weights) | Apache-2.0 (confirm per file) | ODC-BY (checkpoints) |
| How to run | pip + HF ckpt | TerraTorch / inference.py | `satlaspretrain-models` package |

### Implications for the backend protocol

Any of the three, added under the `EmbeddingBackend` protocol, requires the same heavy
machinery and therefore the same torch-gated optional tier: source-imagery acquisition via
STAC (Sentinel-2 for Clay/Satlas, HLS for Prithvi), `torch` plus downloaded weights, tiled
inference, and a conversion step that turns patch/token/feature-map outputs into a per-pixel
embedding raster comparable to AlphaEarth's cube — followed by an L2-normalise so the
similarity/change metrics stay correct. Clay is the cleanest fit (per-pixel-ish embeddings,
sensor-flexible, best docs); Prithvi suits coarser multi-temporal work at 30 m; Satlas is
the loosest fit because its native output is a feature map rather than an embedding vector.

### Appendix sources

- Clay Foundation Model documentation — <https://clay-foundation.github.io/model/index.html>
- Prithvi-EO-2.0-600M, Hugging Face — <https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M>
- SatlasPretrain models, GitHub (allenai/satlaspretrain_models) — <https://github.com/allenai/satlaspretrain_models>
