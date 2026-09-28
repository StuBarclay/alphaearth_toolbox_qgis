# AlphaEarth Toolbox for QGIS — Plugin Scope & Design

**Status:** Draft scope, v0 (2026-09-24)
**Working title:** AlphaEarth Toolbox
**Proposed provider id:** `alphaearth`
**Target:** QGIS 3.34 LTR (Qt5/PyQt5) **and** QGIS 4.x (Qt6/PyQt6) — dual-toolkit, same pattern as `bal_toolbox_qgis` / `burnt_area_toolbox_qgis`.

---

## 1. Purpose

A QGIS plugin that makes Google's **AlphaEarth Satellite Embedding V1** usable end-to-end inside QGIS — loading, visualising, classifying, clustering, similarity search, regression, and multi-year change/trajectory analysis — without an Earth Engine account, API key, pip install, or GPU.

The analytics already exist and are proven in the `alphaearth-classify`, `alphaearth-lulc`, and `alphaearth-urban-growth` packages. This plugin **packages that capability** as a QGIS Processing provider plus a dialog GUI, using the GDAL-native, no-auth-cloud-COG approach already validated in those projects and in `burnt_area_toolbox_qgis`.

### Why now
The only AlphaEarth plugin on the QGIS repository today is **EMBED-CD** — deliberately narrow (bi-temporal change detection, GUI-only, experimental, not yet on the official repo). It confirms the "read embeddings straight from the cloud, no account" path works inside a plugin, but leaves the rest of the embedding workflow (single-date classification, clustering, similarity, regression, trajectories, batchable/scriptable use) unserved. That is the white space this plugin fills.

---

## 2. Positioning vs EMBED-CD

| Capability | EMBED-CD | AlphaEarth Toolbox (this) |
|---|---|---|
| Bi-temporal change map | ✅ (its core) | ✅ (one of several algorithms) |
| Few-shot on-the-fly labelling | ✅ | ✅ (supervised classify) |
| Single-date land-cover / fuel classification | ❌ | ✅ |
| Unsupervised clustering (KMeans) | ❌ | ✅ |
| Similarity search ("find more like this") | ❌ | ✅ |
| Continuous regression (canopy %, built-up frac.) | ❌ | ✅ |
| Multi-year trajectory / anomaly | ❌ (2 years only) | ✅ |
| Embedding-as-a-layer loader + PCA→RGB | ❌ | ✅ |
| Processing provider (headless, batch, model builder) | ❌ (dialog only) | ✅ |
| Dialog GUI wizard | ✅ | ✅ |

Not a competitor so much as the general-purpose superset; change detection is one algorithm among many.

---

## 3. Data model & access

- **Dataset:** AlphaEarth / Satellite Embedding V1 — 64-band, 10 m, **annual** (2017–2025), global, licensed **CC BY 4.0**.
- **Band semantics:** 64 dimensions (`A00`…`A63`), unitless. Each pixel is a **unit-length vector** in 64-D, so a dot product **is** cosine similarity — the format is purpose-built for similarity/distance work (important for the similarity and change algorithms).
- **Access:** cloud-optimised GeoTIFFs read over `/vsicurl` with **no authentication** (same no-auth GCS route as the existing `alphaearth-*` packages). ⚠️ *Confirm the exact public endpoint and tiling scheme from the intake code in `alphaearth-classify`/`alphaearth-lulc` before implementation — do not hard-code a guessed URL.*
- **Robustness:** wrap `/vsicurl` reads with HTTP retry + timeout + GDAL-HTTP2 settings (reuse the `_util` HTTP-retry helper from `burnt_area_toolbox_qgis` v0.2.0).
- **Working CRS:** default to **EPSG:3577** (Australian Albers, matching the existing pipelines and giving equal-area stats), but expose the target CRS as a parameter so the plugin is usable outside Australia. Densify geometries on `transform_geom` (burnt-area lesson).
- **Tiling / AOI:** clip to a user AOI (drawn rectangle, layer extent, or selected polygon); process tile-by-tile so large areas stay practical (EMBED-CD does the same).
- **No-data honesty:** where a tile or year is missing, surface it explicitly (a distinct no-data class/mask) rather than silently emitting "no change" / "unclassified" — a deliberate EMBED-CD design principle worth copying.

---

## 4. Architecture

Mirror the structure that already works in `burnt_area_toolbox_qgis`:

```
alphaearth_toolbox_qgis/
├─ alphaearth_toolbox/
│  ├─ __init__.py                # classFactory
│  ├─ metadata.txt               # experimental=True initially; CC BY 4.0 note
│  ├─ provider.py                # AlphaEarthProvider (id="alphaearth")
│  ├─ plugin.py                  # menu/toolbar hooks, dialog launch
│  ├─ algorithms/                # one module per Processing algorithm (§5)
│  ├─ gui/                       # dialog(s), AOI draw tool, year pickers (§6)
│  ├─ aecore/                    # vendored analytics (from alphaearth-* pkgs)
│  │  ├─ intake.py               # no-auth COG access, tiling, CRS
│  │  ├─ classify.py             # RF/GB supervised + KMeans
│  │  ├─ similarity.py           # cosine similarity / seed aggregation
│  │  ├─ change.py               # multi-year distance / trajectory / anomaly
│  │  ├─ _rio.py / _gdal.py      # GDAL-native raster I/O shim (no rasterio)
│  │  └─ _fiona.py               # GDAL-native vector I/O shim (no fiona)
│  └─ resources/                 # icons, QML styles, presets
├─ tests/                        # pytest + GDAL-parity fixtures
├─ .github/workflows/            # ruff/format/mypy/pytest + GDAL-container job
├─ build_zip.py                  # reproducible stdlib ZIP builder
├─ SCOPE.md                      # this document
└─ README.md
```

Design principles carried over from your existing plugins:
- **GDAL-native I/O** via vendored `_rio`/`_fiona` shims — **no rasterio/fiona dependency**; numpy core unchanged.
- **Off-GUI-thread execution** (`QgsProcessingAlgRunnerTask` + progress/Cancel) so long STAC/COG runs never freeze the UI (burnt-area v0.2.0 pattern).
- **Auto-styling** of outputs (bundled QML) and progress/cancel on every algorithm.
- **Reproducible ZIP** built in `/tmp` with `cp -f` (OneDrive lock workaround noted in your build gotchas).

---

## 5. Processing algorithms (comprehensive v1)

Each algorithm is a standard `QgsProcessingAlgorithm` (headless, batchable, model-builder-ready). "Maps to" = the existing module that supplies the core logic.

**5.1 Load Embedding** — `alphaearth:loadembedding`
Inputs: AOI, year(s), target CRS, resolution. Output: 64-band embedding raster (COG) clipped to AOI, optional local cache. *Maps to:* `intake`.

**5.2 Embedding → RGB** — `alphaearth:torgb`
Inputs: embedding raster; method (PCA-to-3 or fixed band triplet). Output: styled 3-band visualisation raster. *Maps to:* new thin wrapper over numpy PCA. Dependency-free.

**5.3 Supervised Classify** — `alphaearth:classify`
Inputs: embedding raster/AOI+year, training layer (labelled points/polygons), classifier (RandomForest | GradientBoosting), class field. Outputs: class raster + per-pixel confidence/probability raster; optional saved model. *Maps to:* `alphaearth-lulc` RF/GB engines; `alphaearth-urban-growth` RF calibrate.

**5.4 Unsupervised Cluster** — `alphaearth:cluster`
Inputs: embedding, k, sample size. Outputs: cluster raster + optional cluster-centroid/stats table. *Maps to:* `alphaearth-classify` / `urban-growth` KMeans.

**5.5 Similarity Search** — `alphaearth:similarity`
Inputs: embedding, seed feature(s) (point/polygon), aggregation (mean/medoid), optional threshold. Outputs: cosine-similarity raster (0–1) + optional binary "find-more-like-this" mask. Pure numpy — no sklearn. *Maps to:* new `similarity` module (the unit-vector format makes this a dot product). **This is the standout feature no existing plugin offers.**

**5.6 Regression** — `alphaearth:regress`
Inputs: embedding, training layer with continuous target, regressor (RF | GB). Output: continuous raster (e.g. canopy %, built-up fraction) + optional model. *Maps to:* `alphaearth-lulc`/`urban-growth` (swap classifier for regressor).

**5.7 Change / Trajectory** — `alphaearth:change`
Inputs: two or N years, metric (cosine | euclidean), mode (pairwise change | trajectory magnitude | anomaly-vs-baseline). Outputs: change-magnitude raster (+ optional labelled-change objects via few-shot, EMBED-CD-style). Pure numpy for the distance step. *Maps to:* `alphaearth-lulc` change detection.

**5.8 Extract Training Samples** — `alphaearth:extract`
Inputs: embedding, feature layer. Output: table of 64-D embedding values per feature (for external ML / QA). Pure numpy. *Maps to:* `intake` + sampling.

### Dependency tiers (important design decision)
QGIS ships numpy but **not scikit-learn**. To avoid a hard blocker, split algorithms into two tiers:

- **No-dependency tier** (works on any stock QGIS): Load, →RGB, Similarity, Change/Trajectory, Extract Samples. All pure numpy + GDAL.
- **scikit-learn tier** (RF/GB/KMeans): Supervised Classify, Cluster, Regression. Detect sklearn at runtime; if absent, fail gracefully with a one-click guided install (into the QGIS Python env) and a clear message. Do **not** vendor sklearn (too heavy) and do **not** silently require it.

This means a fresh install is immediately useful (similarity + change + loader) even before anyone touches the ML tier.

---

## 6. Dialog (GUI) — v1

A tabbed dialog wrapping the most common workflows for non-scripters, delegating to the same Processing algorithms under the hood:

- **Load & Visualise** — draw AOI, pick year, load embedding, PCA→RGB preview, auto-style.
- **Classify (few-shot)** — pick/collect training features, choose RF/GB, run, review confidence.
- **Similarity** — click/draw a seed, set threshold, get a "more like this" layer live.
- **Change / Trajectory** — pick two (or N) years, get change map; optional label-and-classify objects.

Shared UX: draw-AOI map tool, year pickers, progress bar + Cancel (off-thread), explicit no-data reporting, auto-styled outputs, and a persistent attribution line for CC BY 4.0.

---

## 7. Reuse map (existing code → plugin)

| Plugin algorithm | Source package/module | Work required |
|---|---|---|
| Load / intake / tiling / CRS | `alphaearth-*` intake (no-auth GCS COG, EPSG:3577) | Wrap as algorithm; add HTTP retry/timeout |
| Supervised classify / regress | `alphaearth-lulc` (RF/GB), `alphaearth-urban-growth` (RF calibrate) | Expose params; add confidence output |
| KMeans cluster | `alphaearth-classify` / `urban-growth` | Wrap as algorithm |
| Change / trajectory | `alphaearth-lulc` change detection | Add N-year + anomaly modes |
| Similarity search | *(new)* | Small module; dot product on unit vectors |
| GDAL-native I/O shims | `burnt_area_toolbox_qgis` `_rio` | Copy/adapt (adds vector shim) |
| Off-thread run + progress | `burnt_area_toolbox_qgis` v0.2.0 | Copy pattern |
| Reproducible ZIP builder + CI | `burnt_area_toolbox_qgis` | Copy/adapt |

Most of v1 is **packaging and I/O plumbing**, not new analytics.

---

## 8. Dependencies & packaging

- **Runtime:** QGIS 3.34 LTR + QGIS 4.x; numpy (bundled); GDAL (bundled). scikit-learn **optional** (ML tier only, guided install). No rasterio/fiona.
- **mypy:** pin toolchain as per your other plugins (mypy 1.11.2 pin noted in gotchas).
- **Packaging:** stdlib reproducible ZIP builder; build in `/tmp` + `cp -f` (OneDrive lock workaround).
- **metadata.txt:** `experimental=True` for first releases; embed CC BY 4.0 attribution requirement; semantic version.

---

## 9. Licensing & attribution

- **Plugin code:** GPL-2.0-or-later (matches the QGIS ecosystem and EMBED-CD).
- **AlphaEarth data:** CC BY 4.0 — bake a required attribution string into outputs/metadata and the About dialog. **Commercially usable with attribution** (a real advantage over EMBED-CD, whose *Sentinel-2 reference imagery* is non-commercial). If any optical preview imagery is added later, isolate it and flag its licence — keep the core commercial-clean for a Geoscape/PSMA context.

---

## 10. Testing & CI (4 gates, your standard)

- **ruff** (lint) + **format** + **mypy** + **pytest** — all green before release.
- GDAL-shim parity tests (mirror the rasterio-parity suite from burnt-area).
- Algorithm unit tests on **tiny synthetic 64-band fixtures** (no network) — deterministic classify/cluster/similarity/change.
- Network-dependent intake tests marked/skippable; a GDAL-container CI job for the geo stack.

---

## 11. Milestones

- **M1 — no-dependency core:** Load, →RGB, Similarity, Change/Trajectory, Extract Samples (Processing provider). Installable and useful on stock QGIS.
- **M2 — ML tier:** Supervised Classify, Cluster, Regression + sklearn detection/guided install.
- **M3 — dialog GUI:** tabbed wizard, AOI draw tool, off-thread + progress/cancel, auto-styling.
- **M4 — hardening & release:** tests/CI, reproducible ZIP, docs, `experimental` → stable, **submit to plugins.qgis.org**.

---

## 12. Open questions (confirm before build)

1. **Exact no-auth COG endpoint & tiling scheme** for AlphaEarth V1 — pull from existing `alphaearth-*` intake code (don't guess).
2. **scikit-learn policy** — guided install vs. documented manual install; which minimum version.
3. **Default working CRS** — EPSG:3577 default vs. native 4326 (proposal: 3577 default, configurable).
4. **Provider id / display name** — `alphaearth` / "AlphaEarth Toolbox" (confirm no clash).
5. **Similarity defaults** — cosine metric, mean-of-seeds aggregation, default threshold.
6. **Coexistence with EMBED-CD** — fine to overlap on change; make sure algorithm ids/menus don't collide if both installed.

---

## 13. Post-v1 roadmap

Pluggable embedding backends (Clay, Prithvi, SatlasPretrain alongside AlphaEarth); model persistence & reuse across AOIs; active-learning labelling loop; cloud-native COG outputs; time-series dashboards; preset library for common targets (fuel/land-cover for WUI, built-up fraction, canopy).
