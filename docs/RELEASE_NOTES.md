# Release notes

## 0.10.0 — Wizard parity for tuning and Classify over years

This release closes the gap between the guided wizard and the Processing Toolbox for
the machine-learning tier. The hyper-parameter tuning and the *Classify over years*
one-step fetch-and-classify — both added to the Processing algorithms in 0.9.0 — are
now reachable straight from the wizard's **Classify** tab, so a non-scripter can use
them without opening the Toolbox. It is a GUI-only change: **no new algorithm and no
new compute** — the tab simply marshals parameters for the existing
`alphaearth:classify` and `alphaearth:classifyyears` algorithms and runs them off the
GUI thread, so behaviour, outputs and every existing parameter are unchanged.

### Tune hyper-parameters on the Classify tab

The Classify tab gains an opt-in **Tune hyper-parameters** switch and a tuning-
iterations spin-box (enabled only when the switch is on). Ticking it sets the same
`TUNE`/`TUNE_ITERS` parameters the Processing algorithm exposes, so the wizard runs
the identical cross-validated randomized search before the final fit. It applies to
both runs the tab can launch — the single-year *Classify* and the new fetch-years
flow — and, like the algorithm, is default-off and falls back to the defaults with a
warning when the training set is too small to search.

### Fetch ticked years and classify, from the wizard

Mirroring the *Change* tab's one-step compare, the Classify tab adds a **Fetch ticked
years and classify** button: tick two or more years on the *Load / Visualise* tab,
set the training layer and label field (or load a saved model) once, and it runs
`alphaearth:classifyyears` to fetch every year onto one shared grid and map land
cover for each in a single pass. Two extra tick-boxes optionally also request the
first-to-last **transition raster** and the **changed / unchanged mask**, both
auto-styled exactly as from the Processing algorithm. The run happens on the same
off-GUI-thread task with progress and Cancel as every other wizard action.

### Docs and testing

The user guide, architecture notes and the release smoke-test checklist are updated
for the new controls, and the smoke test now carries a fill-in **recorded run log**
plus explicit wizard-tune and wizard-fetch-years checks. The pure test suite is
unchanged at 266 passing tests (the wizard change is GUI-only and exercised by the
existing compile/GUI-design coverage); all four gates (ruff, ruff-format,
mypy-strict, pytest) stay green.

## 0.9.0 — A twelfth algorithm and optional hyper-parameter tuning

This release completes the "fetch and do it in one step" pattern for the
scikit-learn tier and lets that tier tune its own hyper-parameters. It adds a
twelfth algorithm and an opt-in tuning switch on every supervised/clustering
algorithm. It is additive — no breaking changes to the existing eleven algorithms
or their parameters.

### Classify over years (fetch + classify)

A new `alphaearth:classifyyears`, **Classify over years (fetch + classify)**, is
the classification analogue of *Change over years*. Tick two or more years, set one
area of interest, CRS and resolution, supply training features (or a saved model),
and it fetches every year onto the *same* snapped grid and maps land cover for each
in a single run — a classify run no longer needs its years pre-loaded and aligned
by hand. **One** classifier is trained on the most recent ticked year (or a saved
model is reused) and applied to every year, so a given class code means the same
thing across the whole series. The outputs are a **folder of per-year class
rasters** (`classification_<year>.tif`), an optional **first-to-last transition
raster** (one code per from/to class pair, auto-styled with a legend of
`"<from> -> <to>"` labels) and an optional **changed / unchanged mask**. It shares
the fetch path with *Change over years* and reuses *Classify*'s estimator-building,
cross-validation and tuning helpers, so it cannot drift from either.

### Opt-in hyper-parameter tuning across the ML tier

*Classify*, *Regress* and the new *Classify over years* gain a **Tune
hyper-parameters** switch: a cross-validated randomized search over a small preset
grid (for the random-forest or gradient-boosting estimator) that picks the best
configuration before the final fit and writes its search summary into the same
accuracy/report text. *Cluster* gains a **Choose k automatically** switch that
selects the number of clusters by the best mean silhouette over a range. Both are
**opt-in and default-off** — an existing run is unchanged unless you tick them — and
both **fall back to the defaults with a warning, never a failure**, when the
training data is too small to score. A QGIS-backed guard test asserts every tuning
switch is off by default.

### No drift, fully testable

All of the new logic is pure NumPy/stdlib in `aecore.ml` — transition encoding, the
changed mask, the `"<from> -> <to>"` transition labels, grid sampling, best-candidate
selection and the randomized-search driver (which takes a scoring closure that owns
scikit-learn, keeping the driver itself sklearn-free) — with its own unit tests,
plus a pure test of the shared tuning plumbing in `algorithms/_ml_shared.py` that
runs against a fake QGIS. The pure suite is now 266 passing tests; all four gates
(ruff, ruff-format, mypy-strict, pytest) stay green.

## 0.8.0 — An eleventh algorithm and reports / charts for change

This release turns a change raster into something you can read at a glance: a small
report with charts. It adds an eleventh algorithm and offers the same report as an
opt-in extra on both change algorithms and in the guided wizard. It is additive —
no breaking changes to the existing ten algorithms or their parameters.

### Change report (chart / summary)

A new `alphaearth:changereport`, **Change report (chart / summary)**, turns any
single-band change raster (from *Change / trajectory* or *Change over years*, or
any magnitude raster) into a small report. Four content types are independently
selectable: a **change-magnitude distribution** with summary statistics (count,
min/max, mean, median, standard deviation and percentiles) and a histogram; the
**percentage of the area changed** at each of several thresholds (with hectares
when the CRS is in metres); a **change-over-time series**, one point per period,
built from several change rasters supplied in order; and a **scatter** of change
against a chosen embedding band. Any combination of four output formats can be
written — an HTML page and a CSV of the underlying numbers need nothing beyond
NumPy, while PNG and PDF use matplotlib when it is present and are skipped with a
clear message (never a run failure) when it is not, exactly as the scikit-learn
tier treats its optional dependency.

### The report on both change algorithms and in the wizard

The same report is available as an opt-in extra directly on *Change / trajectory*
and *Change over years*: set one or more output paths (tucked away as advanced
parameters, so an existing run is unchanged unless you opt in) and a report is
written alongside the change raster — its over-time series taken from the
consecutive-year pairs, its scatter plotted against the most recent year. The
guided dialog's **Change** tab gains matching content tick-boxes and HTML / PNG /
CSV / PDF pickers that drive both the raster-based *Run change detection* and the
*Fetch ticked years and compare* buttons.

### No drift, fully testable

All of the report maths and every format's rendering are a new pure-NumPy/stdlib
core, `aecore.report` (HTML is emitted as a standalone page with inline SVG charts;
PNG/PDF use matplotlib only when present), with the Processing parameter plumbing
shared once in `algorithms/_report.py` so the standalone algorithm and both change
algorithms cannot drift. The core is unit-tested with no QGIS, GDAL or matplotlib
present; the pure suite is now 231 passing tests, and all four gates (ruff,
ruff-format, mypy-strict, pytest) stay green.

## 0.7.0 — A tenth algorithm, model reuse, accuracy reporting, presets and a durable cache

This release rounds out the analysis workflow. It adds a tenth algorithm, lets the
scikit-learn tier **save and reuse a fitted model** and **report its accuracy**,
surfaces a curated **preset** library in the wizard, and makes the tile-discovery
cache **survive across QGIS sessions**. It is additive — no breaking changes to the
existing nine algorithms or their parameters.

### Change over years (fetch + compare)

A new `alphaearth:changeyears`, **Change over years (fetch + compare)**, folds the
two-step *Load embeddings (multiple years)* → *Change / trajectory* workflow into a
single algorithm. Tick two or more years, set one area of interest, CRS and
resolution, pick a change mode (pairwise, trajectory magnitude, anomaly vs
baseline) and distance metric, and get the change raster directly — the per-year
rasters never have to be written out and re-selected by hand. Because every year is
fetched onto the *same* snapped grid in one pass, the years are inherently
pixel-aligned and there is no alignment check to fail. The fetched per-year
embedding rasters can optionally still be written to a folder for reuse. It reuses
the shared `_fetch.fetch_year_cube` and the same `compute_change` dispatch as the
raster-based *Change / trajectory*, so it cannot drift from either.

### Model save and reuse (Classify and Regress)

Classify and Regress can now **save the fitted estimator** to a `.pkl` and **reuse
it** on another scene without retraining — supply a saved model and the training
layer and label field become unnecessary. The saved file is stamped with the
scikit-learn version and the 64-band embedding count it was trained on, and a reuse
whose band count or scikit-learn version does not match **fails loudly** with a
clear message rather than silently producing meaningless output. The save/load
logic lives in a pure, unit-tested `aecore.model_io`, and the shared Processing
parameters (load model, save model, accuracy folds, report) are factored into
`algorithms/_ml_shared.py` so the two ML algorithms stay in step.

### Accuracy assessment and class balance

Both ML algorithms take an optional **accuracy report**. Set a number of folds and
a report path and the algorithm runs a stratified k-fold cross-validation: for
Classify it writes overall accuracy, per-class precision/recall/F1 and a confusion
matrix; for Regress it writes per-fold and overall R² and RMSE. The report also
includes a **class-balance summary** that warns when the training set is heavily
imbalanced (a common cause of misleading overall accuracy). The cross-validation,
confusion-matrix and class-balance maths are pure NumPy in `aecore.ml` and are
unit-tested with no scikit-learn, QGIS or GDAL present.

### Preset library

A new pure-stdlib `aecore.presets` holds small ordered registries of named
starting points for the tasks users ask most often — RGB rendering, similarity
"find more like this", clustering and change — storing *semantic* values
(`method="pca"`, `metric="cosine"`) rather than dropdown indices. The wizard
surfaces them as dropdowns on the relevant tabs so a sensible configuration is one
pick. Presets are guidance, not guarantees: because the embedding has no fixed
band semantics, each is a first guess to refine.

### Durable cross-session tile cache

The per-`(year, zone)` object listing, tile footprints and scene anchors were
already cached, but only under the volatile OS temp directory. An opt-in advanced
tick-box, **Remember the tile index across sessions**, now stores that cache under
your QGIS profile directory instead, so a later QGIS session that fetches over the
same zone reuses the discovery work rather than re-listing and re-scanning. The
durable path resolver is the only QGIS-dependent piece; the directory logic
(`intake.persistent_cache_dir`) is pure and unit-tested, and the default remains
the temp-dir cache.

### Tests

A new `tests/test_algorithms_qgis.py` registers the provider inside a headless
QGIS and runs each offline algorithm (RGB, similarity, change, extract — and, when
scikit-learn is present, classify) end-to-end through `processing.run` on tiny
synthetic 64-band inputs, asserting valid outputs and that the provider exposes all
ten algorithm ids. Like the GDAL parity tests, it **skips automatically** where
`qgis.core` is absent, so it is a no-op in the sandbox/`checks` job and runs for
real in the QGIS CI container and a live QGIS. New pure-stdlib/NumPy unit tests
cover the presets, the model-IO round-trip and version/band stamping, the
cross-validation and class-balance maths, the change-over-years year handling and
the persistent-cache directory. The pure suite is now 207 passing tests; all four
gates (ruff, ruff-format, mypy-strict, pytest) stay green.

## 0.6.0 — Multi-year fetch over one area of interest

This release adds a ninth algorithm, **Load embeddings (multiple years)**
(`alphaearth:loadyears`), for fetching several AlphaEarth years over the *same*
area of interest in a single run. It is additive — no breaking changes to the
existing eight algorithms.

### Load embeddings (multiple years)

Tick the years you want (any of 2017–2025; the most recent is ticked by default so
a stray run does not pull all nine), set one area of interest, target CRS and
resolution, and choose an output folder. The algorithm writes **one 64-band
GeoTIFF per year**, named `alphaearth_<year>.tif`. Every year is clipped and warped
onto the *same* snapped grid, so the outputs are pixel-aligned and feed straight
into *Change / trajectory* and *Similarity* with no re-projection or resampling in
between — which is the point of fetching the years together. The grid is sized once
and the large-area heads-up (if any) is given once; the progress bar is split evenly
across the ticked years and **Cancel** is honoured between them. Headless callers
get each raster back under a `YEAR_<year>` result key alongside the `OUTPUT` folder.

### Shared fetch path (no drift)

Rather than duplicate the ~250-line fetch-by-year routine, the public-GCS tile
discovery, scene-grouped culling, mosaicking and de-quantisation now live once in
`algorithms/_fetch.py`. Both the single-year *Load embedding* and the new
multi-year loader call the same `fetch_year_cube` / `dequantise` helpers (the fetch
maps its phases onto a caller-supplied slice of the progress bar), so the two paths
cannot drift, and every v0.5.x optimisation — the per-`(year, zone)` footprint and
listing/anchor caches, scene-grouped culling, float32 throughout, the large-AOI
guard — applies to multi-year fetches unchanged.

### Guided wizard

The wizard's *Load / Visualise* tab gains a tick-list of years and a **Load selected
years** button below the single-year Load; each fetched raster is added to the
project automatically.

### Tests

Three new pure-stdlib unit tests cover the tick-box-index → year mapping
(`years_from_indices`: sort, de-duplication and out-of-range rejection) and the
per-year output naming (`year_raster_name`). The pure suite is now 136 passing
tests (up from 133); all four gates (ruff, ruff-format, mypy-strict, pytest) stay
green.

## 0.5.1 — Fewer remote reads on a first fetch (scene-grouped culling)

This release attacks the one cost the 0.5.0 footprint cache could not: the *first*
fetch over a new year-and-region, whose 0–90 % tile-discovery phase still had to
header-scan every candidate tile. It does so without changing which tiles are
selected. No breaking changes.

### Scene-grouped culling

An AlphaEarth object name ends `{hash}-{offset_a}-{offset_b}.tiff`, and the two
trailing integers are that *scene's* per-tile pixel offsets — each `{hash}` is an
independent export with its own local origin, so the numbers are not positions on
a global grid and a tile's location still cannot be read from its name alone. But
tiles *can* be grouped by their scene hash. The fetch now reads a single **anchor**
tile per scene and, from its footprint and pixel size plus the (known) pixel
offsets of the scene's other tiles, computes a deliberately generous over-estimate
of the whole scene's extent. Any scene whose over-estimated extent cannot touch the
area of interest is dropped without reading its tiles; only the surviving scenes
are then scanned tile-by-tile for exact footprints. That turns the first-fetch scan
from roughly one read per *tile* into roughly one read per *scene* — commonly a few
times fewer reads — while remaining correctness-preserving: the estimate can only
ever over-include, so a scene that truly intersects is never culled, and the final
tile selection still uses exact scanned footprints. Unparseable names, unreadable
anchors and single-tile scenes all fall back to a plain per-tile scan.

### Listing and anchor caching

Alongside the existing footprint cache, each `(year, zone)` folder's full object
listing and its per-scene anchors are now cached (same temp-dir, schema-versioned,
bucket/prefix-guarded scheme). A later run over a *different* area of interest in a
zone already visited therefore skips the GCS listing entirely and re-culls its
scenes arithmetically from the cached anchors, with no new reads for scenes it has
already anchored.

### Visible skipped tiles and bounded concurrency

Tiles whose header cannot be read are counted across both scan phases and reported
as a single warning ("N tile header(s) could not be read and were skipped; the
mosaic may have small gaps there") instead of passing silently. The concurrent
header reads remain bounded to a small fixed worker pool, so the scan never opens
an unbounded number of remote connections at once.

### Testing

New pure-standard-library unit tests (no network, no GDAL) cover tile-offset
parsing, scene grouping, the conservative extent estimate (asserted to contain
every tile of a synthetic scene under *either* pixel-axis-to-lon/lat orientation,
and to cull a far AOI while keeping a near one), and round-trips of both the
footprint and the listing/anchor caches. All four quality gates are green
(133 passed, 1 GDAL-only test skipped where GDAL is absent).

## 0.5.0 — Performance and memory hardening for large areas

This release does not add algorithms; it makes the existing ones faster and
lighter on memory, especially for large areas of interest, and finishes the
progress/cancel work on the year fetch. No breaking changes.

### Faster repeat fetches (tile-footprint cache)

The 0–90 % tile-discovery phase of a fetch-by-year run header-scans every
candidate tile to find those covering the area of interest. Those footprints are
now cached to a small JSON file per `(year, zone)`, in WGS84 so the cache is
reused whatever the run's target CRS is. AlphaEarth's annual data is immutable, so
footprints never expire: a second run over the same year and zones skips the
network scan for every tile it has already seen and jumps almost straight to the
mosaic. The cache is stored under the system temp directory and is keyed by a
schema version plus the bucket/prefix, so a stale or foreign cache is ignored.

### Float32 end-to-end

`read_cube` now returns a float32 cube by default — lossless for AlphaEarth's
int8-quantised embeddings (values de-quantised by `÷127.5` are exactly
representable in float32) and half the RAM of float64. The similarity, change,
RGB, ML and sampling compute cores were updated to *preserve* that float32 through
their maths (via a shared `aecore._num.as_float` helper) instead of silently
upcasting back to float64, so the saving is real rather than undone by the first
arithmetic operation. Passing `dtype=np.float64`/`out_dtype=np.float64` still
forces double precision where wanted.

### Large-AOI guard and single-pass write

*Load embedding* now estimates the output size up front and warns (without
blocking) when a requested area would need roughly 2 GiB or more in memory, so a
too-large request is obvious rather than appearing to hang. When the chosen output
is a `.tif`/`.tiff` file the de-quantised cube is written straight to it in one
pass, skipping the scratch-copy step.

### Testing / CI

New GDAL-container tests exercise the warp progress callback (fires and finishes
at 100 %), the cancel path (`WarpCanceledError` when the sink returns `False`), and
the read/warp dtype controls (float32 default, float64 opt-in, native-dtype
passthrough). All four quality gates are green.

### Resilient tile scan and visible failures

The footprint scan now treats a single tile's failure as "skip it", not "abort the
run". Because a year fetch header-scans every tile in each touched UTM zone
(often thousands, concurrently), one transient `/vsicurl` timeout or one
untransformable edge tile used to be able to fail the whole run — and, under
GDAL 3.9+ where Python exceptions are enabled by default, such faults surface as
exceptions rather than a quiet `None`. `dataset_bounds_in_crs` now catches any
such error and drops just that tile, honouring its documented contract. Separately,
*Load embedding* now catches any unexpected error and writes the full traceback to
the Processing log before failing, so a problem shows its real cause instead of the
background task runner's opaque `Task failed: Executing 'Load embedding'`.

## 0.4.0 — Visualisation, an optional ML tier and a guided wizard

This release adds RGB visualisation, an optional scikit-learn machine-learning
tier, and a full tabbed wizard dialog, and hardens the test/CI setup. The
provider now ships eight algorithms.

### New algorithms

- **Embedding → RGB** (`alphaearth:torgb`, no extra dependencies) — paints a
  64-band embedding to a 3-band preview, either as the top three PCA components
  (fitted on a pixel subsample for speed) or a chosen band triplet, with a
  percentile contrast stretch.
- **Classify embedding** (`alphaearth:classify`, scikit-learn tier) — trains a
  random-forest or gradient-boosting classifier on labelled point/polygon
  features and maps land cover across every valid pixel; optionally writes a
  per-pixel confidence raster (the winning-class probability, 0–1). Outputs are
  auto-styled (paletted classes; a red→green confidence ramp).
- **Cluster embedding** (`alphaearth:cluster`, scikit-learn tier) —
  unsupervised K-means with a subsampled fit and chunked prediction; paletted
  output.
- **Regress embedding** (`alphaearth:regress`, scikit-learn tier) — predicts a
  continuous target (e.g. canopy height, an index) from a numeric field using a
  random-forest or gradient-boosting regressor; pseudocolour output.

### Guided wizard dialog

The toolbar button now opens a tabbed wizard (Load / Visualise, Classify,
Similarity, Change) instead of a launcher stub. It provides a draw-on-canvas
extent, an AlphaEarth year picker, a target-CRS chooser and layer/field pickers,
and an **Install scikit-learn** button. Every run executes off the GUI thread as
a background task with a progress bar and a Cancel button, and results are added
to the project when the run finishes.

### The scikit-learn tier is optional

scikit-learn is **not** bundled and is **never** a hard dependency: importing the
plugin, and all four no-dependency algorithms, work without it. The three ML
algorithms detect scikit-learn at runtime and, when it is missing, stop with a
clear message. The wizard's **Install scikit-learn** button installs it once into
the QGIS Python environment (`python -m pip install "scikit-learn>=1.0"`).

### Testing / CI

- New GDAL round-trip and rasterio-parity tests
  (`tests/test_raster_gdal.py`) verify `aecore._raster` write/read/warp against
  raw GDAL and (when installed) rasterio. They are skipped automatically where
  GDAL is absent, so they run in the QGIS/GDAL container job and inside QGIS but
  not in the pure lint/type job.
- The CI QGIS/GDAL job now runs against both the **3.34 LTR** image (gating) and
  the moving **latest** image (Qt6/QGIS 4, informational/non-blocking), prints
  interpreter/GDAL/QGIS versions, and installs pytest in a PEP 668-safe way.
- Pure-NumPy ML plumbing (`aecore.ml`) and scikit-learn detection
  (`algorithms._sklearn`) are unit-tested with no scikit-learn/QGIS/GDAL present.

### Upgrade notes

No breaking changes. Existing project files and the four original algorithms are
unaffected. The plugin remains marked `experimental=True` while real-QGIS
coverage broadens.

## Publishing checklist (maintainer)

Run these on Windows, where git and a real QGIS are available.

1. Confirm the four quality gates pass: `ruff check .`, `ruff format --check .`,
   `mypy`, `pytest` (the GDAL parity tests will skip unless you run them inside a
   GDAL-enabled interpreter).
2. Rebuild the plugin ZIP: `python scripts/build_plugin_zip.py`.
3. Work through `docs/SMOKE_TEST.md` in QGIS 3.34 LTR (and, if convenient, a
   QGIS 4 / Qt6 build).
4. Commit and push to `StuBarclay/alphaearth_toolbox_qgis`:
   `git add -A && git commit -m "Release 0.9.0" && git push origin main`.
5. Tag and create a GitHub Release (`v0.9.0`) and attach `alphaearth_toolbox.zip`
   as a release asset.
6. Submit/update the plugin on <https://plugins.qgis.org> (upload the ZIP;
   the listing stays flagged experimental).
