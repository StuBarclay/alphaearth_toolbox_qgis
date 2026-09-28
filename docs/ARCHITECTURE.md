# Architecture

*AlphaEarth Toolbox for QGIS — v0.10.0*

This document describes how the plugin is put together: the layering that keeps
most of the code testable without QGIS, how a Processing algorithm is discovered
and run, how the optional scikit-learn tier is detected and installed, how remote
tiles are read, and how the pieces are tested and packaged. It is written for
developers who want to extend the toolbox or review it, and for technically-minded
users who want to understand what runs when they press *Run*.

## 1. Design goals

The toolbox exists to make Google DeepMind's AlphaEarth Satellite Embedding V1
usable end-to-end inside QGIS with no Earth Engine account, no API key, no GPU
and no extra Python packages. Four principles shape the code, and every module
below can be traced back to one of them.

The first is an *import-light core*. Importing the top-level `alphaearth_toolbox`
package must never pull in PyQt, the `qgis` runtime or GDAL. That keeps the pure
compute and configuration code importable — and unit-testable — in a plain Python
environment, which is what lets the continuous-integration "checks" job run
without a geospatial stack installed.

The second is a *dependency-tiered* algorithm set. QGIS ships NumPy and GDAL but
not scikit-learn, so the algorithms are split into a no-dependency tier (Load,
Load multiple years, Embedding→RGB, Similarity, Change/trajectory, Change over
years, Change report and Extract training samples, which need only NumPy and
GDAL — the change report's PNG/PDF outputs use matplotlib when present but never
require it) and a scikit-learn tier (classification, classification over years,
clustering, regression). A
fresh install is
immediately useful without anyone touching the machine-learning tier, and
scikit-learn is detected at runtime and offered as a one-click guided install —
never a hard dependency, never vendored (§9).

The third is *reading straight from the cloud, no sign-in*. AlphaEarth V1 is
published as Cloud-Optimised GeoTIFFs in the public `gs://alphaearth_foundations`
bucket. The toolbox reads them anonymously over GDAL's `/vsicurl`, using windowed
reads so a small area of interest is pulled from large remote scenes without
downloading whole tiles.

The fourth is *house-style consistency* with the sibling `bal_toolbox_qgis` and
`burnt_area_toolbox_qgis` plugins: a Processing provider plus a dialog (here a
guided tabbed wizard), GDAL-native I/O with no rasterio/fiona dependency, four
green quality gates (ruff, ruff-format, mypy-strict, pytest), and a reproducible
standard-library ZIP builder.

## 2. Package layout

```
alphaearth_toolbox_qgis/
├─ alphaearth_toolbox/            # the installable plugin package
│  ├─ __init__.py                 # classFactory + Python-version guard (import-light)
│  ├─ metadata.txt                # QGIS plugin metadata (v0.10.0, experimental, CC BY 4.0)
│  ├─ icon.svg                    # provider/menu icon
│  ├─ provider.py                 # AlphaEarthProvider  (id = "alphaearth")
│  ├─ plugin.py                   # menu/toolbar wiring + provider registration
│  ├─ aecore/                     # compute core
│  │  ├─ similarity.py            # pure-NumPy cosine similarity (no GDAL/QGIS)
│  │  ├─ change.py                # pure-NumPy multi-year change / trajectory (no GDAL/QGIS)
│  │  ├─ sampling.py              # pure per-pixel vector sampling helpers (no GDAL/QGIS)
│  │  ├─ rgb.py                   # pure-NumPy PCA / band-triplet RGB reduction (no GDAL/QGIS)
│  │  ├─ ml.py                    # pure-NumPy ML plumbing: matrix/mask/chunked predict + CV/accuracy/class-balance (no GDAL/QGIS)
│  │  ├─ model_io.py              # pure save/load of a fitted estimator to .pkl, stamped + validated (no GDAL/QGIS)
│  │  ├─ presets.py               # pure-stdlib named starting points for RGB/similarity/cluster/change
│  │  ├─ report.py                 # pure-NumPy/stdlib change-report maths + HTML/CSV render (matplotlib optional for PNG/PDF)
│  │  ├─ intake.py                # constants, UTM-zone maths, GCS listing, persistent-cache dir (pure stdlib)
│  │  └─ _raster.py               # GDAL read / warp / mosaic / write (runtime only)
│  ├─ algorithms/                 # one module per Processing algorithm + glue
│  │  ├─ __init__.py              # lazy algorithm discovery (_algorithms)
│  │  ├─ load_embedding.py        # alphaearth:loadembedding
│  │  ├─ load_years.py            # alphaearth:loadyears
│  │  ├─ embedding_rgb.py         # alphaearth:torgb
│  │  ├─ similarity_search.py     # alphaearth:similarity
│  │  ├─ change_detection.py      # alphaearth:change
│  │  ├─ change_years.py          # alphaearth:changeyears (fetch ticked years + compare, no-dep)
│  │  ├─ change_report.py         # alphaearth:changereport (report/charts from a change raster, no-dep)
│  │  ├─ extract_samples.py       # alphaearth:extract
│  │  ├─ classify.py              # alphaearth:classify   (scikit-learn tier)
│  │  ├─ classify_years.py        # alphaearth:classifyyears (fetch ticked years + classify each, scikit-learn tier)
│  │  ├─ cluster.py               # alphaearth:cluster    (scikit-learn tier)
│  │  ├─ regress.py               # alphaearth:regress    (scikit-learn tier)
│  │  ├─ _fetch.py                # shared public-GCS fetch-by-year (single + multi-year, uses QGIS/GDAL)
│  │  ├─ _ml_shared.py            # shared scikit-learn-tier Processing params (LOAD/SAVE_MODEL, ACCURACY_FOLDS, REPORT, TUNE/TUNE_K)
│  │  ├─ _sklearn.py              # PURE scikit-learn detect + guided-install plumbing
│  │  ├─ _features.py             # QGIS feature → training-sample bridge (uses QGIS/GDAL)
│  │  ├─ _styling.py              # best-effort renderers: paletted / colour ramps (uses QGIS)
│  │  ├─ _qgis_io.py              # QGIS layer ⇄ core file I/O bridge (uses GDAL/QGIS)
│  │  ├─ _raster_source.py        # PURE source-resolution policy + VRT XML builder
│  │  └─ _progress.py             # PURE duck-typed feedback adapters
│  └─ gui/
│     └─ dialog.py                # guided tabbed wizard (Load/Visualise, Classify, Similarity, Change)
├─ tests/                         # pytest (pure core + pure policy; GDAL parity tests skip w/o GDAL)
├─ scripts/build_plugin_zip.py    # reproducible stdlib ZIP builder
├─ .github/workflows/ci.yml       # 4 gates + a QGIS/GDAL container job (3.34 LTR + latest)
├─ pyproject.toml                 # ruff / mypy / pytest config; dev extras
├─ SCOPE.md                       # full design & roadmap
├─ README.md
└─ docs/                          # this documentation set
```

## 3. The import-light boundary

The single most important structural rule is which modules may import `qgis`,
PyQt or `osgeo`, and which may not. Getting this wrong breaks the CI "checks"
job, which runs on a plain Ubuntu Python with no geospatial stack.

Several modules import **nothing** heavy and are therefore importable and
unit-testable anywhere: `alphaearth_toolbox/__init__.py` (only `sys` and
`typing`), the pure compute cores `aecore/similarity.py`, `aecore/change.py`,
`aecore/sampling.py`, `aecore/rgb.py` and `aecore/ml.py` (only NumPy — `ml.py`'s
cross-validation, confusion-matrix and class-balance maths included),
`aecore/model_io.py` (only the standard library — `pickle` plus a version/band
stamp), `aecore/presets.py` (only the standard library) and `aecore/report.py`
(only NumPy and the standard library for its maths and its HTML/CSV rendering —
matplotlib is imported lazily and defensively for the optional PNG/PDF output,
never at module top),
`aecore/intake.py` (only the standard library — `json`, `math`, `urllib`;
`persistent_cache_dir`'s directory logic is pure here, only the profile-path
lookup is done in QGIS-facing code), `algorithms/_raster_source.py`,
`algorithms/_progress.py` (only `typing` and `collections.abc`) and
`algorithms/_sklearn.py` (only the standard library — scikit-learn itself is
imported lazily and defensively, never at module top).

The remaining modules do import `qgis` and/or `osgeo` and only run inside QGIS:
`provider.py`, `plugin.py`, all twelve algorithm modules, `algorithms/_fetch.py`,
`algorithms/_ml_shared.py` (which declares the shared scikit-learn-tier Processing
parameters), `algorithms/_report.py` (which declares the shared change-report
Processing parameters and reads/writes them through the pure `aecore.report`),
`algorithms/_qgis_io.py`, `algorithms/_features.py`,
`algorithms/_styling.py`, `aecore/_raster.py` and `gui/dialog.py`. Even these defer the heaviest imports. Each algorithm class
imports `numpy`, `osgeo.gdal` and its `aecore` core (and `_raster`) *inside*
`processAlgorithm`, not at module top level, so that merely discovering an
algorithm never forces GDAL to load; the ML algorithms likewise import
scikit-learn only once a run actually begins.

The lazy-import discipline is deliberate and appears at every layer. `__init__.py`
imports `plugin` only inside `classFactory`. `plugin.py` imports the dialog only
inside `_open_dialog`. `algorithms/__init__.py` imports the algorithm classes
only inside `_algorithms()` and inside a module-level `__getattr__`, so importing
the `algorithms` package does not pull in `qgis`.

## 4. How QGIS loads the plugin

QGIS constructs the plugin by calling `classFactory(iface)` in
`alphaearth_toolbox/__init__.py`. Before doing anything else, `classFactory`
checks `sys.version_info` against `_MIN_PYTHON = (3, 10)` and raises a clear
`RuntimeError` on older interpreters, rather than letting the core fail cryptically
on 3.10-only syntax (the compute core uses `zip(..., strict=True)`). It then
lazily imports and returns an `AlphaEarthToolboxPlugin`.

`AlphaEarthToolboxPlugin` (`plugin.py`) owns two things: the Processing provider
and one GUI action. `initProcessing()` creates an `AlphaEarthProvider` and adds it
to `QgsApplication.processingRegistry()`. `initGui()` calls `initProcessing()` and
then adds a single toolbar/menu action that opens the wizard dialog. `unload()`
removes both the provider and the action, so the plugin unloads cleanly. Because
`initProcessing()` is separate, a headless QGIS (Processing only, no GUI) gets the
algorithms without ever constructing a Qt widget.

`AlphaEarthProvider` (`provider.py`) is intentionally thin. It advertises its
identity (`id() == "alphaearth"`, which prefixes every algorithm id such as
`alphaearth:similarity`), its display names and its icon, and `loadAlgorithms()`
registers each instance returned by `algorithms._algorithms()`. That function
returns fresh `LoadEmbeddingAlgorithm`, `LoadYearsAlgorithm`,
`EmbeddingToRgbAlgorithm`, `SimilaritySearchAlgorithm`, `ChangeDetectionAlgorithm`,
`ChangeOverYearsAlgorithm`, `ChangeReportAlgorithm`, `ExtractSamplesAlgorithm`,
`ClassifyEmbeddingAlgorithm`, `ClassifyOverYearsAlgorithm`,
`ClusterEmbeddingAlgorithm` and
`RegressEmbeddingAlgorithm` instances — twelve in all —
importing them lazily so registration does not drag in GDAL or scikit-learn.

## 5. Anatomy of an algorithm

All twelve algorithms are standard `QgsProcessingAlgorithm` subclasses, which is
what makes them runnable from the toolbox dialog, headless via `processing.run`,
in batch, and in the Graphical Modeler. Each declares `name`, `displayName`,
`group`/`groupId` ("AlphaEarth"), a `shortHelpString` that ends with the required
CC BY 4.0 attribution string, `initAlgorithm` (which declares typed parameters),
and `processAlgorithm` (the work).

The three raster-producing algorithms follow the same shape. Parameters are read
through the typed accessors (`parameterAsCrs`, `parameterAsExtent`,
`parameterAsSource`, `parameterAsLayerList`, and so on). A QGIS raster layer input
is turned into a path the compute core can open by `raster_source_path` (see §7).
The heavy work is delegated to the compute core (`aecore.similarity` or
`aecore.change` for the maths, `aecore._raster` for the pixels). Progress and
status are reported through the duck-typed adapters in `_progress.py`, and
cancellation is honoured by checking `feedback.isCanceled()` at loop boundaries.
Outputs are written to a temporary GeoTIFF and then mirrored into the user's
chosen Processing destination by `export_raster`, with the scratch directory
cleaned up in a `finally` block. Both `SimilaritySearchAlgorithm` and
`ChangeDetectionAlgorithm` implement `postProcessAlgorithm` to apply a best-effort
blue-to-red pseudocolour style once the raster is a real map layer.
`ChangeDetectionAlgorithm` additionally reads *two or more* rasters (via
`parameterAsLayerList`) and verifies that every input shares one grid — matching
shape and geotransform within a small tolerance — before comparing pixel-for-pixel.

`ChangeOverYearsAlgorithm` (`change_years.py`, id `alphaearth:changeyears`) folds
*Load embeddings (multiple years)* and *Change / trajectory* into one no-dependency
step. The user ticks two or more years and sets a single area of interest, target
CRS and resolution; the algorithm sizes the snapped grid once and fetches every
ticked year onto that same grid in one pass through the shared
`_fetch.fetch_year_cube` — so the years are inherently pixel-aligned and there is no
alignment check that can fail. It then applies exactly the same `aecore.change`
`compute_change` dispatch as the raster-based *Change / trajectory* (a change mode
of pairwise, trajectory magnitude or anomaly vs baseline, and a cosine or Euclidean
metric), writing the single-band change raster directly, and can optionally also
save the fetched per-year cubes to a folder. Because it reuses `fetch_year_cube` and
`compute_change` rather than reimplementing them, its fetch and its maths cannot
drift from the loaders and from *Change / trajectory*.

`ChangeReportAlgorithm` (`change_report.py`, id `alphaearth:changereport`) is the
one algorithm whose output is neither a raster nor a vector layer but a *report*.
It reads one single-band change raster (a chosen band, no-data mapped to NaN),
optionally a list of change rasters "over time" and optionally an embedding raster
for a scatter, and turns them into any combination of an HTML page, a PNG, a CSV
and a PDF. All the maths and the HTML/CSV rendering live in the pure
`aecore.report` core — `finite_values`, `summarise`, `histogram`, `area_fractions`,
`series_points` and `scatter_sample` build a `ReportModel`, and `render_html`
(standalone page, inline SVG charts, everything escaped) and `render_csv` emit the
dependency-free formats; `render_png`/`render_pdf` use matplotlib only when
`matplotlib_available()` is true and are otherwise skipped with a message, never a
run failure, exactly like the scikit-learn tier's optional dependency. The
Processing parameter plumbing (the four output-path parameters, the content
tick-list, the scatter-band number, and the `read_request` / `build_and_write`
helpers plus `pixel_area_m2` and `scatter_from_band`) is declared once in
`algorithms/_report.py`, so the standalone algorithm and the report wired into both
*Change* algorithms share one source of truth and cannot drift.

The report is also offered as an opt-in extra on `ChangeDetectionAlgorithm` and
`ChangeOverYearsAlgorithm`: `_report.add_report_parameters(self, advanced=True)`
adds the same four output paths and content controls as advanced parameters, so an
existing change run is unchanged unless the user sets a report path. When one is
set, the change algorithm feeds its own computed magnitudes to the same
`_report.build_and_write` — its over-time series taken from the consecutive-year
pairs and its scatter plotted against the most recent year — so the report matches
the raster exactly.

`ExtractSamplesAlgorithm` is the one vector-producing algorithm and so diverges:
it reads one embedding raster and a `QgsProcessingFeatureSource`, builds an output
field list (the source fields plus pixel metadata and one column per band) and
writes a `QgsProcessingFeatureSink` of points rather than a raster. Its per-pixel
sampling maths lives in the pure `aecore.sampling` core; the GDAL parts are limited
to reading the embedding cube and its georeferencing.

`EmbeddingToRgbAlgorithm` follows the raster shape but needs no seeds: it reads one
embedding cube, reduces its 64 bands to three via the pure `aecore.rgb` core (top-3
PCA fitted on a pixel subsample, or a user-chosen band triplet) with a percentile
contrast stretch, and writes a 3-band `uint8` RGB GeoTIFF. It is in the
no-dependency tier — PCA here is a small NumPy SVD, not scikit-learn.

`LoadEmbeddingAlgorithm` and `LoadYearsAlgorithm` share the public-GCS
fetch-by-year path. Rather than duplicate ~250 lines of tile discovery,
scene-grouped culling, mosaicking and de-quantisation, that logic lives once in
`algorithms/_fetch.py`: `fetch_year_cube(year, …, progress_start, progress_end)`
returns the raw int8 mosaic and its geotransform for a single year, mapping its
internal phases (anchor reads, survivor scans, warp) onto a caller-supplied slice
of the progress bar; `dequantise` turns that int8 cube into unit-length float32
vectors; and `warn_if_large` / `report_and_raise` are the shared large-AOI heads-up
and traceback-to-log error path. `LoadEmbedding` calls it once for its single
`YEAR`; `LoadYears` reads the tick-box `YEARS` enum (`intake.years_from_indices`
maps the selected indices to sorted, de-duplicated years), sizes the grid **once**
so every year is pixel-aligned, then loops the years — giving each an equal
progress slice and honouring **Cancel** between years — and writes one
`alphaearth_<year>.tif` (named by `intake.year_raster_name`) into the output
folder, returning a distinct `YEAR_<year>` result key per raster so the wizard
adds each to the project. Because both loaders go through the same `_fetch`
functions, the single-year and multi-year paths cannot drift.

The four scikit-learn algorithms (`ClassifyEmbeddingAlgorithm`,
`ClassifyOverYearsAlgorithm`, `ClusterEmbeddingAlgorithm`,
`RegressEmbeddingAlgorithm`) share a common flow.
Before any heavy work each calls `_sklearn.ensure_sklearn` and, if it is absent,
raises a `QgsProcessingException` whose message names the missing library and
points at the wizard's install button — so a missing dependency is a clean,
readable failure, never a traceback. `Classify` and `Regress` first turn labelled
point/polygon features into a training matrix via `_features` (which samples the
embedding at each feature, reusing the `aecore.sampling` maths) and `aecore.ml`
(`cube_to_matrix`, `flat_valid_mask`, `encode_labels`); they fit an estimator, then
predict across every valid pixel in bounded row-chunks (`ml.predict_in_chunks`) to
keep peak memory flat, and scatter the results back to the grid
(`ml.scatter_to_grid`). `Cluster` skips the training features and fits K-means on a
subsample. Outputs are auto-styled through `_styling` — a paletted renderer for
class/cluster rasters, a red→green ramp for the optional confidence raster, and a
pseudocolour ramp for regression predictions.

`Classify` and `Regress` also gain model persistence and an optional accuracy
report, whose shared Processing parameters (`LOAD_MODEL`, `SAVE_MODEL`,
`ACCURACY_FOLDS`, `REPORT`) are declared once in `algorithms/_ml_shared.py` so the
two algorithms cannot diverge on wording or defaults. When a `SAVE_MODEL` path is
given, the fitted estimator is written to a `.pkl` through the pure
`aecore/model_io.py`, which stamps the file with the installed scikit-learn version
and the 64-band embedding count. When a `LOAD_MODEL` path is supplied instead, the
saved estimator is loaded and applied to the current scene without retraining — so
the training layer and label field are not needed — and `model_io` fails loudly
with a clear message if the stamped band count or scikit-learn version does not
match, rather than predicting from a mismatched model. If `ACCURACY_FOLDS` and a
`REPORT` path are set, a stratified k-fold cross-validation is run and written to
the report: for *Classify*, overall accuracy plus per-class precision, recall and
F1 and a confusion matrix; for *Regress*, per-fold and overall R² and RMSE. Both
reports include a class-balance summary that warns on a heavily imbalanced training
set. All of that cross-validation, confusion-matrix and class-balance arithmetic is
pure NumPy in `aecore/ml.py`, unit-tested with no scikit-learn, QGIS or GDAL; the
`.pkl` round-trip and its version/band validation live in the equally pure
`aecore/model_io.py`.

`Classify`, `Regress` and `ClassifyOverYears` also gain **opt-in hyper-parameter
tuning** and `Cluster` an **opt-in automatic k**, whose shared parameters (`TUNE`,
`TUNE_ITERS` for the search budget; `TUNE_K`, `TUNE_K_MIN`, `TUNE_K_MAX` for the
silhouette k-search) are declared once in `algorithms/_ml_shared.py` alongside the
model/report parameters. When `TUNE` is ticked, the algorithm asks `_ml_shared` for
the preset search space of the chosen estimator (`tuning_grid("random_forest")` /
`("gradient_boosting")`) and how many folds to score with
(`resolve_tune_folds`, which reuses the accuracy-report fold count, defaults to
three, and — critically — returns zero and warns *"tuning skipped"* rather than
failing when there are fewer samples than folds), then runs a cross-validated
randomized search and fits the final estimator with the winning parameters, writing
the search summary into the same report text. The search **driver** is pure NumPy in
`aecore.ml` (`randomized_search`): it samples parameter combinations and picks the
best by a caller-supplied `score` closure that owns scikit-learn, so the driver
itself never imports sklearn and is unit-tested with a plain in-memory scorer;
`ml.format_search_report` renders the result table shared by both searches. For
`Cluster`, a local `_select_k` (in `cluster.py`) fits K-means for each k in
`[TUNE_K_MIN, TUNE_K_MAX]` and scores each on a bounded subsample with scikit-learn's
`silhouette_score` (the metric is O(n²), so it is capped at a few thousand pixels),
picking the k with the best coefficient and formatting its report the same way.
Every one of these switches defaults to **off**, so an existing run is byte-for-byte
unchanged unless the user opts in; a QGIS-backed test asserts the default-off
contract, and `tests/test_ml_shared.py` covers `tuning_grid`/`resolve_tune_folds`
against a fake QGIS.

`ClassifyOverYearsAlgorithm` (`classify_years.py`, id `alphaearth:classifyyears`)
is the classification analogue of *Change over years*, and the only scikit-learn
algorithm that also fetches. The user ticks two or more years and sets one area of
interest, target CRS and resolution; the algorithm sizes the snapped grid once and
fetches every ticked year onto that same grid through the shared
`_fetch.fetch_year_cube` (so the years are inherently pixel-aligned), trains **one**
classifier on the most recent ticked year — or reuses a saved model — and applies it
to every year, so a class code means the same land cover across the whole series. It
reuses *Classify*'s `_build_classifier`, `_cross_validate` and `_tune` helpers (one
source of truth for the estimator and its tuning) and writes a **folder** of per-year
`classification_<year>.tif` rasters. Two optional extra outputs summarise the
series: a first-to-last **transition** raster (`ml.transition_codes` encodes each
`(from, to)` class pair as one code, auto-styled from `ml.transition_labels` with
`"<from> -> <to>"` legend entries) and a **changed/unchanged mask**
(`ml.changed_mask`); both `int64` core arrays are cast to `int32` before
`write_geotiff` for portable GeoTIFF output with `-1` no-data. Because it shares the
fetch path with the loaders and the classifier plumbing with *Classify*, it cannot
drift from either.

## 6. Reading pixels: the GDAL layer

`aecore/_raster.py` is the only compute-core module that imports GDAL, and it is
imported only at runtime inside `processAlgorithm`. It offers four operations:
`read_cube` (read an entire raster as a `(bands, rows, cols)` cube plus
its geotransform, WKT and no-data), `warp_cube_to_grid` (warp a single source onto
a target grid, reading only the needed window), `warp_many_to_grid` (mosaic *many*
tiles onto one target grid), and `write_geotiff` (write a cube back out, DEFLATE-
compressed). Two header-only helpers support tile discovery: `dataset_bounds_in_crs`
reads only a source's header and reprojects its four corners into the target CRS to
return a footprint — cheap enough to scan many tiles before committing to a full
warp — and `dataset_footprint_and_pixels` does the same but also returns the source's
pixel dimensions, which the fetch-by-year scene-culling step (see the Methodology)
uses to over-estimate a whole scene's extent from a single anchor tile. Both return `None` on *any* failure (a
transient network fault, an untransformable edge tile, or a GDAL error raised as an
exception) rather than propagating: because a year fetch scans many tiles in a zone
concurrently, one bad tile must be skipped, never allowed to abort the whole run.
This matters under GDAL 3.9+ (shipped in QGIS 4), where the Python bindings raise
exceptions by default instead of returning a quiet failure.

`read_cube` returns a **float32** cube by default. AlphaEarth pixels are int8
values de-quantised by `÷127.5`, and every such value is exactly representable in
float32, so float32 is lossless here while halving the resident cube versus
float64 (`dtype=np.float64` forces double precision and `dtype=None` keeps the
source's native dtype). The pure compute cores (`similarity`, `change`, `rgb`,
`ml`, `sampling`) preserve that floating dtype through their maths via the shared
`aecore/_num.as_float` helper — a float32 cube stays float32 end-to-end instead of
being silently promoted back to float64 — so the memory saving is real and not
undone by the first arithmetic op. The warp helpers take an `out_dtype` (default
float64, `None` = native) and an optional `progress` sink that reports the warp's
completion fraction and, by returning `False`, cancels the in-flight `gdal.Warp`
(raising `WarpCanceledError`).

Remote reads are made resilient. `_is_remote_source` recognises `http(s)://` and
the `/vsi*` prefixes. For those, a context manager (`_gdal_config`) temporarily
applies a tuned configuration — bounded HTTP retries and timeouts,
`GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR`, a VSI cache, and `GDAL_HTTP_MULTIRANGE`
+ HTTP/2 so a windowed read (or a many-tile mosaic) needs far fewer round-trips —
and restores the previous values afterwards, so QGIS's own GDAL configuration is
never permanently changed. On top of GDAL's HTTP-layer retries, `_warp_with_retry`
adds a bounded outer loop with linear back-off for errors GDAL surfaces without
retrying. Local sources skip all of this and fail fast.

`warp_many_to_grid` is the key to the fetch-by-year workflow: `gdal.Warp` accepts
a *list* of sources and reprojects each independently, so tiles that live in
different UTM-zone CRSs (as AlphaEarth's per-zone COGs do) are stitched straight
onto one output grid in the target CRS in a single pass.

## 7. Getting a raster layer into the core: source resolution

The compute core opens plain file paths (or GDAL descriptors), but a QGIS raster
layer's "source" is not always a file — it can be a provider descriptor, a
subdataset, a `/vsicurl` URL or a geodatabase raster. `_qgis_io.raster_source_path`
turns any such layer into something the core can open, and it delegates the
*decision* to the dependency-free `_raster_source.resolve_raster_source` so the
branch logic is unit-testable without QGIS or GDAL.

The policy has three tiers, tried in order. A candidate that is a plain file on
disk is used unchanged (zero-copy). Otherwise the first candidate GDAL can open as
a *dataset descriptor* is wrapped in a tiny VRT — a real file that references the
descriptor and reads it lazily, so a large scene is never copied in full; the VRT
XML is hand-assembled by `build_vrt_xml` (guaranteeing a well-formed document with
metadata baked in). If neither works, the layer is materialised to a temporary
GeoTIFF via `QgsRasterFileWriter`. The QGIS/GDAL-specific probes are injected into
`resolve_raster_source` as callables, so the pure policy module has no geospatial
imports at all.

## 8. Progress, messages and cancellation

`_progress.py` bridges QGIS Processing feedback to the compute core through plain
callables and performs no `qgis` import — it relies only on the duck-typed
protocol (`pushInfo`, `setProgress`, `setProgressText`, `isCanceled`). `make_message`
returns a `message(text)` status callable, `make_scene_progress` returns a
`(done, total)` callback that advances the progress bar linearly between two
percentages, and `make_cancel` returns an `is_cancelled()` predicate. Passing
`None`, or an object missing some of those methods, is tolerated (the adapters
no-op), which keeps the core unit-testable with a trivial fake.

Because a long run executes on a background `QgsProcessingAlgRunnerTask` (§9), an
unhandled exception inside the algorithm reaches the user only as the runner's
opaque `Task failed: Executing '<algorithm>'`, with the real cause buried. To keep
failures diagnosable, *Load embedding* wraps its body in a catch-all: expected,
already-explained failures (`QgsProcessingException` — empty AOI, no source, no
intersecting tiles, user cancel) re-raise unchanged, but any *other* exception is
first written in full — `traceback.format_exc()` — to the Processing log via
`feedback.reportError(..., fatalError=True)` and then re-raised as a
`QgsProcessingException` with a concise one-line summary. So the dialog still fails,
but the log it points at now shows the underlying error rather than the runner's
generic message.

## 9. The GUI: the tabbed wizard

`gui/dialog.py` is a guided tabbed wizard. `AlphaEarthDialog` takes the `iface` and
builds four tabs — **Load / Visualise**, **Classify**, **Similarity** and
**Change** — over the common workflows for non-scripters. Shared widgets come from
QGIS's own GUI toolkit: a `QgsExtentGroupBox` bound to the map canvas (with a
draw-on-canvas rectangle) for the area of interest, a `QgsProjectionSelectionWidget`
for the target CRS (kept in sync with the extent's output CRS), `QgsMapLayerComboBox`
raster/vector pickers, a `QgsFieldComboBox` that follows its layer, and a year
`QComboBox` populated from `intake.AVAILABLE_YEARS`. Because QGIS 3.34 (Qt5) and
QGIS 4 (Qt6) moved several enums, version-robust helpers resolve the raster/vector
filter flags across both.

Several tabs also surface the preset library from `aecore/presets.py` as a
drop-down of named starting points — for RGB, similarity, clustering and change —
whose entries store *semantic* values (`method="pca"`, `metric="cosine"`) rather
than dropdown indices, so a preset means the same thing regardless of widget order.
The *Change* tab additionally offers the one-step *Change over years* flow: tick the
years, set one AOI/CRS/resolution and a change mode, and it runs
`alphaearth:changeyears` to fetch and compare in a single pass. The *Classify* tab
exposes the model save/reuse and accuracy-report fields backed by
`algorithms/_ml_shared.py`, an opt-in *Tune hyper-parameters* switch with an
iterations spin-box (which simply sets the `TUNE`/`TUNE_ITERS` parameters on
whichever classify run the tab launches, so both the single-year classify and the
fetch-years flow can tune), and — mirroring the *Change* tab — a one-step
*Classify over years* flow: tick the years on the Load tab, set the training layer
and field once, and a **Fetch ticked years and classify** button runs
`alphaearth:classifyyears`, with two extra tick-boxes to also request the
first-to-last transition raster and the changed/unchanged mask. The wizard adds no
new compute — each control just marshals parameters for an existing Processing
algorithm, run off-thread through `_execute` — so the automatic-k option for
*Cluster* remains the one tuning-style control surfaced only on the Processing
algorithm rather than in the wizard. Finally, an opt-in advanced tick-box, *Remember the tile
index across sessions*, switches the per-`(year, zone)` listing/anchor/footprint
cache from the volatile OS temp directory to a durable location under the QGIS
profile directory; the profile-path resolution is the only QGIS-dependent piece,
while the directory logic (`intake.persistent_cache_dir`) is pure and unit-tested,
and the temp-dir cache remains the default.

Crucially, every run executes **off the GUI thread**. `_execute` builds a
`QgsProcessingContext` and `QgsProcessingFeedback`, wraps the chosen algorithm in a
`QgsProcessingAlgRunnerTask`, connects its `progressChanged` and `executed` signals
to update a progress bar and load results, and hands it to the task manager — so a
long cloud fetch or a model fit never freezes the interface, and a **Cancel** button
cancels both the feedback and the task. Successful file outputs are added to the
project as raster/vector layers when the task finishes. (If the runner task is
unavailable, it falls back to `processing.execAlgorithmDialog`.)

The Classify tab also hosts the scikit-learn **detect-and-install** UX. On open it
calls `_sklearn.sklearn_status` and shows whether the library is available; the
**Install scikit-learn** button runs `python -m pip install "scikit-learn>=1.0"`
against the running interpreter on a background `QgsTask` (so the UI stays live),
then re-checks status when it finishes. The install plumbing lives in the pure
`algorithms/_sklearn.py` (a stdlib-only `find_spec` probe plus a `subprocess`
runner injected as a callable, so it is unit-tested without actually installing
anything), and the algorithms call the same module's `ensure_sklearn` guard — one
source of truth for "is scikit-learn here?" shared by the GUI and the headless
algorithms.

## 10. Testing and CI

Tests live under `tests/` and cover the pure tiers with no network access — the
`intake` tile-discovery helpers are exercised with an *injected* `fetch_json` so
the GCS listing logic is verified deterministically offline (including scene
grouping, the conservative scene-extent over-estimate — asserted to contain every
tile of a synthetic scene under *either* pixel-axis orientation — and the footprint
and listing/anchor caches, plus the multi-year helpers — `years_from_indices`
index→year mapping with sort/dedup and out-of-range rejection, and
`year_raster_name`), and the NumPy similarity, change, sampling, RGB and ML
cores (including the cross-validation, confusion-matrix and class-balance maths),
the pure `aecore.model_io` `.pkl` round-trip and its version/band-count validation,
the `aecore.presets` registries, the pure `intake.persistent_cache_dir` directory
logic, the scikit-learn detection plumbing (with an injected subprocess runner), and
the pure source-resolution policy are tested directly. The change-report core is
tested the same way — `tests/test_report.py` exercises the summary statistics,
histogram, monotone area-fractions and hectares, over-time series, scatter
sub-sampling, the `ReportModel` builder and the standalone HTML/CSV rendering with
no QGIS, GDAL or matplotlib present (the PNG/PDF test `importorskip`s matplotlib) —
and `tests/test_change_years.py` reaches the qgis-facing dispatcher through a fake
`qgis.core` that now also stands in for the `_report`/`_ml_shared` imports
`change_detection` pulls in. The v0.9.0 tuning maths — `ml.randomized_search` (with
an in-memory scorer, so no scikit-learn is needed), its `ml.sample_parameter_grid`,
`ml.select_best_params` and `ml.format_search_report` primitives, and the
transition/changed-mask/transition-label helpers — are unit-tested directly, and
`tests/test_ml_shared.py` exercises `tuning_grid` and `resolve_tune_folds` through
the same fake-`qgis.core` shim. (The cluster automatic-k helper `_select_k` builds
on `silhouette_score` from scikit-learn, so it is covered by the smoke test rather
than the pure suite, but it reuses the same unit-tested `select_best_params` and
`format_search_report`.) As of v0.9.0 the pure suite is 266 passing tests.
`tests/test_raster_gdal.py` adds GDAL round-trip and rasterio-parity tests for
`aecore._raster`; they `pytest.importorskip` GDAL (and rasterio) so they simply
skip in the pure environment and run only where the geo stack is present.
`tests/test_algorithms_qgis.py` goes one step further: it registers the provider in
a headless QGIS and runs each offline algorithm (RGB, similarity, change, extract,
and classify when scikit-learn is present) end-to-end through `processing.run` on
synthetic 64-band inputs, asserts the provider exposes all twelve algorithm ids, and
checks that every optional tuning switch defaults to off. It
skips automatically wherever `qgis.core` is absent, so it is a no-op in the pure
"checks" job and the sandbox and runs for real in the QGIS/GDAL container and in
live QGIS.

`.github/workflows/ci.yml` runs two jobs. The `checks` job runs on a plain Ubuntu
Python (matrix 3.10 and 3.12) and executes the four gates: `ruff check`,
`ruff format --check`, `mypy` (strict) and `pytest`. Because that runner has no
GDAL, only the pure modules are importable there — which is exactly why the
import-light boundary matters. A second `qgis` job runs inside a QGIS container
under `xvfb` over a matrix of two images: `release-3_34` (gating) and the moving
`latest` tag (Qt6/QGIS 4, `continue-on-error` so it is informational), where the
GDAL-backed parity tests and the QGIS-backed algorithm regression tests
(`tests/test_algorithms_qgis.py`) actually execute.

`pyproject.toml` pins the toolchain: ruff selects `E,F,I,N,UP,B,C4,SIM,RUF` at a
100-character line length, mypy is `strict` with the QGIS/GDAL-facing glue modules
relaxed (they subclass QGIS's untyped base classes) and third-party geo modules
marked as missing stubs, and mypy is pinned to `1.11.2`.

## 11. Packaging

`scripts/build_plugin_zip.py` packages exactly the `alphaearth_toolbox/` folder —
skipping byte-code caches — into `alphaearth_toolbox.zip` with fixed entry
timestamps, so repeated builds of an unchanged tree are byte-identical. It builds
into a temporary file and then copies the bytes over the destination, because on a
cloud-synced folder (OneDrive) unlinking or replacing a file can be blocked while
truncate-and-write in place (what `copyfile` does) is allowed — a lesson carried
over from the sibling plugins. The resulting ZIP installs through QGIS's *Install
from ZIP*.

## 12. Extending the toolbox

Adding an algorithm follows a fixed recipe. Put the reusable maths in `aecore/`
as a pure module (NumPy or standard library only) with its own unit tests; write a
`QgsProcessingAlgorithm` subclass under `algorithms/` that reads parameters,
converts layers with the `_qgis_io` bridge, calls the pure core, reports progress
through `_progress`, and writes outputs via `export_raster` (styling through
`_styling` if wanted); register it by adding one lazy import and one list entry in
`algorithms/__init__._algorithms()` (and, if you want it exposed as a class
attribute, in `__getattr__` and `__all__`). Keep every new import of
`qgis`/`osgeo`/`sklearn` inside a function so the import-light boundary — and the
CI "checks" job — stays intact. If the algorithm needs scikit-learn, call
`_sklearn.ensure_sklearn` at the top of `processAlgorithm` so a missing library is
a clean `QgsProcessingException` with a guided-install message, and never a hard
dependency — the pattern `classify`, `cluster` and `regress` already follow. The
wizard picks up new work through its tabs rather than a fixed algorithm list.
