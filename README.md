# AlphaEarth Toolbox for QGIS

Load, visualise and analyse Google's **AlphaEarth Satellite Embedding V1** inside
QGIS — with no Earth Engine account, API key or GPU, and no extra Python packages
for the core workflow.

AlphaEarth V1 is a global, 10 m, annual (2017–2025) **64-band** learned embedding
of the Earth's surface, licensed **CC BY 4.0** and published as Cloud-Optimised
GeoTIFFs in the **public** `gs://alphaearth_foundations` bucket — readable over
`/vsicurl` with no sign-in. Each pixel is a unit-length vector in 64-D, which
makes "find more like this" similarity search an exact cosine dot product, and
multi-year change a direct pixel-to-pixel distance — the standout features of
this toolbox.

> **Status:** v0.10.0, experimental. The provider now ships **twelve** algorithms.
> The eight no-dependency algorithms (Load, Load multiple years, Embedding→RGB,
> Similarity, Change/trajectory, Change over years, Change report, Extract training
> samples) need only NumPy and GDAL, which ship with QGIS. An optional
> **scikit-learn** tier (Classify, Classify over years, Cluster, Regress) is detected
> at runtime and offered as a one-click install — it is never a hard dependency and
> the plugin always loads without it. Classify and Regress can **save a trained model
> and reuse it** on another scene, and optionally emit a **k-fold accuracy report**;
> Classify, Regress and Classify over years can opt in to a **hyper-parameter search**
> and Cluster to an **automatic-k** choice, both default-off. The dialog is a guided
> **tabbed wizard** with a draw-on-canvas extent, a year picker, curated **presets**
> and off-GUI-thread execution. See `SCOPE.md` for the full design.

## What's in this release

The toolbox registers a Processing provider, **AlphaEarth Toolbox**
(id `alphaearth`), with twelve algorithms.

### No-dependency tier (NumPy + GDAL only)

- **Load embedding** (`alphaearth:loadembedding`) — clip and warp a 64-band
  AlphaEarth source to an area of interest, CRS (default `EPSG:3577`) and
  resolution. Give it a raster layer, an explicit `/vsicurl` COG URL, **or just a
  year (2017–2025)**: the tiles are then found straight from the public GCS bucket
  (the AOI is reprojected to WGS84 to pick the UTM-zone folders it touches, each
  zone is listed, tile footprints are header-scanned to keep only those
  overlapping the AOI, and the survivors — possibly across several UTM-zone CRSs —
  are mosaicked in one warp and de-quantised back to unit-length vectors). GDAL
  windowed reads pull a small AOI from large remote scenes without downloading the
  whole thing; remote reads retry with back-off and time out on a stall.
  Nearest-neighbour resampling preserves the learned vectors.
- **Load embeddings (multiple years)** (`alphaearth:loadyears`) — the same fetch
  for a **set** of years chosen with tick-boxes over the *same* area of interest,
  CRS and resolution, writing one 64-band GeoTIFF per year
  (`alphaearth_<year>.tif`) into an output folder. Every year lands on the
  identical snapped grid, so the outputs are pixel-aligned and drop straight into
  **Change / trajectory** or **Similarity** with no re-projection. The AOI/grid
  setup is done once and reused, and each year shares the single-year loader's
  tile discovery, scene-grouped culling and per-(year, zone) caches.
- **Embedding → RGB** (`alphaearth:torgb`) — paint a 64-band embedding to a 3-band
  preview so it can be seen at a glance, either as the top three **PCA** components
  (fitted on a pixel subsample for speed) or a chosen **band triplet**, with a
  percentile contrast stretch. Pure NumPy; no scikit-learn.
- **Similarity search** (`alphaearth:similarity`) — score every pixel of an
  embedding by cosine similarity to one or more **seed** features (points or
  polygons), producing a 0–1 similarity raster and an optional threshold
  ("more like this") mask. Pure NumPy.
- **Change / trajectory** (`alphaearth:change`) — compare two or more embedding
  years (each already loaded onto the same grid) and map how much every pixel
  changed, by cosine or Euclidean distance. Three modes: **pairwise** (first vs
  last), **trajectory magnitude** (cumulative distance across the whole sequence)
  and **anomaly vs baseline** (latest year vs the multi-year mean). Output is
  auto-styled blue→red. Because AlphaEarth pixels are unit vectors, this is a
  direct distance — the capability the narrow EMBED-CD plugin only does for two
  dates.
- **Change over years** (`alphaearth:changeyears`) — the one-step convenience:
  tick two or more years, set one area of interest, CRS and resolution, pick a
  change mode and metric, and get the change raster directly — *Load embeddings
  (multiple years)* and *Change / trajectory* folded together. Because the years
  are fetched onto the same grid in one pass they are always pixel-aligned (no
  alignment check to fail), and the fetched per-year rasters can optionally be
  kept for reuse. Same shared fetch and change maths as the two algorithms it
  combines, so it cannot drift from either.
- **Change report** (`alphaearth:changereport`) — turn any single-band change
  raster into a small report: a change-magnitude distribution with summary
  statistics, the percentage of area changed at each threshold, an optional
  over-time series and an optional change-vs-embedding-band scatter, written as any
  combination of **HTML**, **PNG**, **CSV** and **PDF** (HTML and CSV need nothing
  extra; PNG and PDF use matplotlib when present and are skipped with a message when
  it is absent). The same report is offered as an opt-in extra on both change
  algorithms and in the wizard's Change tab.
- **Extract training samples** (`alphaearth:extract`) — sample the 64-D embedding
  vector at point/polygon features and write one point row per covered pixel,
  carrying the source attributes (class labels, ids) plus band columns
  `A00…A63`. Ready to export to CSV for external ML, or as a QA check on what a
  class looks like in embedding space.

### scikit-learn tier (optional, guided install)

- **Classify embedding** (`alphaearth:classify`) — train a random-forest or
  gradient-boosting classifier on labelled point/polygon features and map land
  cover across every valid pixel; optionally write a per-pixel **confidence**
  raster (the winning-class probability, 0–1). Outputs are auto-styled (paletted
  classes; a red→green confidence ramp).
- **Classify over years** (`alphaearth:classifyyears`) — the classification
  analogue of *Change over years*: tick two or more years, set one area of
  interest, CRS and resolution, and it fetches every year onto the same grid and
  maps land cover for each in one step. **One** model is trained on the most recent
  year (or reused from a saved file) and applied across the series, so a class code
  means the same thing every year. Writes a **folder** of per-year class rasters
  (`classification_<year>.tif`) plus an optional first-to-last **transition** raster
  and a **changed/unchanged mask**.
- **Cluster embedding** (`alphaearth:cluster`) — unsupervised K-means with a
  subsampled fit and chunked prediction; paletted output. Can **choose k
  automatically** by silhouette over a range.
- **Regress embedding** (`alphaearth:regress`) — predict a continuous target
  (e.g. canopy height, an index) from a numeric field with a random-forest or
  gradient-boosting regressor; pseudocolour output.

Classify and Regress can **save the fitted model** to a `.pkl` and **reuse it**
on another scene without retraining (skip the training layer entirely) — the
saved file is stamped with the scikit-learn version and the 64-band embedding
count, so a mismatched reuse fails loudly rather than producing garbage. They
also take an optional **accuracy report**: a stratified k-fold cross-validation
writing overall accuracy, per-class precision/recall/F1 and a confusion matrix
(Classify) or per-fold and overall R²/RMSE (Regress), plus a class-balance
summary that warns on a heavily imbalanced training set. Classify, Regress and
Classify over years can additionally **tune their hyper-parameters** (a
cross-validated randomized search over a small preset grid), all opt-in and
default-off, falling back to the defaults with a warning when the data is too
small to search.

scikit-learn is **not** bundled and is **never** a hard dependency: importing the
plugin, and all eight no-dependency algorithms, work without it. The four ML
algorithms detect scikit-learn at runtime and, when it is missing, stop with a
clear message. The wizard's **Install scikit-learn** button installs it once into
the QGIS Python environment.

All twelve are standard `QgsProcessingAlgorithm`s, so they run headless, in batch,
and in the Graphical Modeler.

## The dialog

**Plugins → AlphaEarth Toolbox** opens a guided **tabbed wizard** — Load /
Visualise, Classify, Similarity and Change — with a draw-on-canvas extent, an
AlphaEarth year picker, a target-CRS chooser, layer/field pickers, curated
**preset** dropdowns (sensible RGB / similarity / cluster / change starting
points), model save-and-reuse and report pickers, an opt-in **Tune
hyper-parameters** switch and a one-step **"fetch ticked years and classify"**
mode on the Classify tab, a one-step **"fetch ticked years and compare"** on the
Change tab, and an **Install scikit-learn** button. The Classify and Change tabs
reuse the years ticked and the extent set on the Load / Visualise tab for their
one-step fetch modes. Every run executes off the GUI thread as a background task
with a progress bar and a Cancel button, and results are added to the project
when the run finishes.

## Install

1. Build the plugin ZIP:

   ```bash
   python scripts/build_plugin_zip.py
   ```

   This writes a clean, reproducible `alphaearth_toolbox.zip`.

2. In QGIS: **Plugins → Manage and Install Plugins → Install from ZIP**, choose
   that file. Requires QGIS 3.34 LTR or newer (Qt5/PyQt5 **and** Qt6/PyQt6 are
   supported). Because the plugin is flagged experimental, tick *Show also
   experimental plugins* if it is hidden.

No extra Python packages are needed for the core — NumPy and GDAL ship with QGIS.
scikit-learn is only needed for the optional ML tier, and the wizard installs it
for you.

## Getting embedding data

AlphaEarth V1 is published as no-login Cloud-Optimised GeoTIFFs in the public
`gs://alphaearth_foundations` bucket, tiled by year and UTM zone:

```
satellite_embedding/v1/annual/{YEAR}/{UTM_ZONE}/{hash}-{offsetA}-{offsetB}.tiff
```

where `{UTM_ZONE}` is the zone number with a hemisphere suffix (e.g. `56S`) and
each tile is a 64-band int8 COG at 10 m in that zone's UTM CRS. The two trailing
integers are per-*scene* pixel offsets (each `{hash}` scene has its own origin),
not global row/col — which is why tile discovery groups tiles by scene.

The simplest workflow is to run **Load embedding**, draw or type an area of
interest, and set a **year** — the toolbox finds, filters, mosaics and
de-quantises the covering tiles for you, no URLs required. You can still point it
at an explicit COG URL (`http(s)://…` or `/vsicurl/…`) or a raster layer you
already have when you want a specific tile or a pre-clipped scene.

The endpoint is defined by the constants `GCS_BUCKET`, `GCS_PREFIX` and
`GCS_HOST` in `alphaearth_toolbox/aecore/intake.py`; the tile-selection maths and
GCS listing there are pure standard library and unit-tested with an injected
fetch (no network in tests).

## Architecture

```
alphaearth_toolbox/
├─ __init__.py          # classFactory (import-light: no PyQt/qgis at import)
├─ metadata.txt         # QGIS plugin metadata (experimental, CC BY 4.0 note)
├─ provider.py          # AlphaEarthProvider (id="alphaearth")
├─ plugin.py            # menu/toolbar wiring + provider registration
├─ algorithms/          # one module per Processing algorithm (+ QGIS I/O glue)
│  ├─ load_embedding.py · load_years.py · embedding_rgb.py · similarity_search.py
│  ├─ change_detection.py · change_years.py · change_report.py · extract_samples.py
│  ├─ classify.py · classify_years.py · cluster.py · regress.py   # scikit-learn tier
│  ├─ _fetch.py          # shared public-GCS fetch-by-year (single + multi-year) + opt-in cache dir
│  ├─ _sklearn.py        # PURE runtime detect + guided-install plumbing
│  ├─ _ml_shared.py      # shared model-reuse, accuracy-report + opt-in tuning params for the ML tier
│  ├─ _report.py         # shared change-report parameter plumbing (content + HTML/PNG/CSV/PDF)
│  ├─ _features.py       # QGIS feature → training-sample bridge
│  ├─ _styling.py        # best-effort renderers (paletted / ramps)
│  ├─ _qgis_io.py        # QGIS layer ⇄ core file I/O bridge
│  ├─ _raster_source.py  # PURE source-resolution policy + VRT XML builder
│  └─ _progress.py       # PURE duck-typed feedback adapters
├─ gui/dialog.py        # tabbed wizard (Load/Visualise, Classify, Similarity, Change)
└─ aecore/              # compute core
   ├─ similarity.py     # pure-NumPy cosine similarity — unit-tested, no GDAL/QGIS
   ├─ change.py         # pure-NumPy multi-year change / trajectory — unit-tested
   ├─ sampling.py       # pure per-pixel vector sampling helpers — unit-tested
   ├─ rgb.py            # pure-NumPy PCA / band-triplet RGB reduction — unit-tested
   ├─ ml.py             # pure-NumPy ML plumbing (matrix/mask/chunked predict, CV, class balance, transitions, param search) — unit-tested
   ├─ report.py         # pure-NumPy change-report core (distribution/thresholds/series/scatter + HTML/CSV render) — unit-tested
   ├─ model_io.py       # PURE save/load of a fitted model + version/band-count stamp — unit-tested
   ├─ presets.py        # PURE named RGB/similarity/cluster/change presets — unit-tested
   ├─ intake.py         # constants, UTM-zone maths, GCS tile listing + persistent cache — pure stdlib
   └─ _raster.py        # GDAL read / warp / mosaic / write (runtime only)
```

The top-level package and the pure compute core never import PyQt, `qgis`, GDAL
or scikit-learn, so they are importable and unit-testable in a plain Python
environment. QGIS-facing code — and scikit-learn — are imported lazily.

## Development

```bash
pip install -e ".[dev]"

ruff check .            # lint
ruff format --check .   # formatting
mypy                    # types (strict; QGIS/GDAL glue relaxed)
pytest                  # unit tests (pure-NumPy core, no network)
```

The pure tiers are covered by the sandbox `checks` CI job. A QGIS/GDAL container
job runs the suite with the geo stack available — including the GDAL round-trip /
rasterio-parity tests in `tests/test_raster_gdal.py` and the QGIS-backed
end-to-end Processing tests in `tests/test_algorithms_qgis.py` (which register the
provider and run each offline algorithm on synthetic inputs); both skip
automatically where GDAL / `qgis.core` are absent, so they are no-ops in the
sandbox and run for real in the container, against both the 3.34 LTR image
(gating) and the moving latest image (Qt6/QGIS 4, informational).

## Licensing & attribution

- **Plugin code:** GPL-2.0-or-later.
- **AlphaEarth data:** CC BY 4.0 — attribute Google DeepMind. The required
  attribution string is exposed as `aecore.intake.ATTRIBUTION` and shown in each
  algorithm's help. AlphaEarth is commercially usable with attribution.

## Roadmap

See `SCOPE.md` and `docs/ROADMAP.md`. The scoped no-dependency tier, the
scikit-learn tier, the guided wizard, model persistence, the preset library, the
one-step change-over-years algorithm, the change-report algorithm, the one-step
classify-over-years algorithm and opt-in hyper-parameter tuning / automatic-k are
all in place; the remaining work is broadening real-QGIS coverage (smoke tests in
3.34 LTR and Qt6/QGIS 4 — the QGIS-backed regression tests already run in the CI
container), cutting a GitHub Release, and submitting to plugins.qgis.org, after
which the `experimental` flag can be cleared. Longer-horizon ideas include pluggable
embedding backends and richer batch/model-management workflows.
