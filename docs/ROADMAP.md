# Roadmap & Future Possibilities

*AlphaEarth Toolbox for QGIS — v0.10.0*

This document lays out where the toolbox is heading. The scoped feature set from
`SCOPE.md` was completed at v0.4.0 — the guided wizard and the machine-learning
tier are in place — the v0.5.x line adds performance and memory hardening for large
areas (v0.5.0 caching and float32; v0.5.1 scene-grouped culling of the first-fetch
tile scan), v0.6.0 adds a ninth algorithm, *Load embeddings (multiple years)*,
that fetches several years over one area of interest as aligned per-year rasters,
v0.7.0 adds a tenth, *Change over years* (a one-step fetch-and-compare),
together with model save/reuse and an optional cross-validation accuracy report for
the machine-learning tier, a preset library, a durable cross-session tile cache and
QGIS-backed regression tests, v0.8.0 adds an eleventh, *Change report*, that
turns any single-band change raster into a distribution, threshold, over-time and
scatter report written as HTML, PNG, CSV and/or PDF (with the same report offered
as an opt-in extra on both change algorithms and in the wizard), and v0.9.0 adds a
twelfth, *Classify over years* (a one-step fetch-and-classify across a tick-box set
of years, the classification analogue of *Change over years*), together with an
opt-in hyper-parameter-tuning switch for the supervised tier and an automatic-k
option for clustering. The roadmap now centres on the remaining path to a public
listing and the longer-horizon possibilities beyond it. It is a plan of intent, not
a commitment to dates.

## 1. Where things stand

The no-dependency tier is complete: *Load embedding* (including no-sign-in
fetch-by-year from the public GCS bucket), *Load embeddings (multiple years)* (the
same fetch across a tick-box set of years onto one shared grid), *Embedding → RGB*,
*Similarity search*, *Change / trajectory*, *Change over years* (the one-step
fetch-and-compare added in v0.7.0), *Change report* (the distribution/threshold/
over-time/scatter report added in v0.8.0, written as HTML, PNG, CSV and/or PDF)
and *Extract training samples*. The
scikit-learn tier is complete: *Classify* (with a per-pixel confidence output),
*Classify over years* (the one-step fetch-and-classify added in v0.9.0, writing a
folder of per-year class rasters plus an optional transition raster and
changed/unchanged mask), *Cluster* and *Regress*, each detecting scikit-learn at
runtime and offering a one-click guided install rather than being a hard dependency;
*Classify* and *Regress* also save and reuse a fitted model and can produce an
optional cross-validation accuracy report, *Classify*, *Regress* and *Classify over
years* can opt in to a cross-validated hyper-parameter search, and *Cluster* can
choose its k automatically by silhouette. The dialog is the guided tabbed wizard with
a draw-AOI extent, year and CRS pickers, a multi-year tick-list, preset drop-downs, an
RGB preview, an install button and off-GUI-thread execution with progress and
Cancel. Four quality gates are green, the pure suite is 266 tests, GDAL
round-trip/parity tests and QGIS-backed algorithm regression tests run in the
QGIS/GDAL CI container, and the ZIP builder is byte-reproducible. What remains is
real-QGIS verification and the mechanics of publishing.

## 2. Milestones M1–M3 — delivered

The no-dependency completion (M1), the scikit-learn tier (M2) and the guided dialog
(M3) have all shipped. *Embedding → RGB* (`alphaearth:torgb`) closed out M1 by
reducing the 64 bands to a three-band view (PCA-to-3 or a band triplet, with a
percentile stretch). M2 added *Classify*, *Cluster* and *Regress* on top of a
pure-NumPy ML core (`aecore.ml`) and a stdlib detect/guided-install helper
(`algorithms._sklearn`), reusing analytics proven in the sibling
`alphaearth-classify`, `alphaearth-lulc` and `alphaearth-urban-growth` packages. M3
replaced the launcher stub with the tabbed wizard (Load / Visualise, Classify,
Similarity, Change), a draw-AOI map tool and off-thread execution via
`QgsProcessingAlgRunnerTask`. Each followed the house pattern — pure maths in
`aecore/` with unit tests, thin `QgsProcessingAlgorithm` wrappers, import-light
boundary preserved.

## 3. Milestone M4 — hardening and public release (in progress)

The in-sandbox hardening is done: GDAL round-trip and rasterio-parity tests
(gated to skip where GDAL is absent), a CI QGIS/GDAL job that runs over both the
3.34 LTR image (gating) and a moving Qt6/QGIS 4 image (informational), the version
bump and changelog, and the release notes and smoke-test checklist. The v0.5.0
release extended this with performance and memory work for large areas — a
persistent tile-footprint cache that skips the network scan on repeat runs, a
float32-throughout cube that halves working memory, a large-AOI warning with a
single-pass GeoTIFF write, and warp progress/cancel with dtype-control tests — and
v0.5.1 attacked the one cost the cache could not, the *first* fetch: tiles are now
grouped by scene, one anchor tile per scene is read, and scenes that cannot touch
the AOI are culled before their tiles are scanned (roughly one read per scene
instead of one per tile), with the zone listing and scene anchors also cached and
unreadable tiles reported as a warning. v0.6.0 then added the *Load embeddings
(multiple years)* algorithm and its wizard tick-list, reusing that same fetch path
per year over a grid sized once so the per-year outputs are pixel-aligned, and
v0.7.0 added the tenth algorithm, *Change over years*, the model save/reuse and
cross-validation accuracy report for the ML tier, the preset library, the durable
cross-session tile cache, and a QGIS-backed regression suite
(`tests/test_algorithms_qgis.py`) that runs each offline algorithm end-to-end in the
container, and v0.8.0 added the eleventh, *Change report*, with a pure-NumPy report
core (`aecore.report`), dependency-free HTML/CSV rendering and matplotlib-gated
PNG/PDF, offered both standalone and as an opt-in extra on the two change
algorithms. v0.9.0 added the twelfth, *Classify over years* — a one-step
fetch-and-classify that trains one model on the most recent ticked year and maps
every year onto a shared grid, with an optional transition raster and
changed/unchanged mask — plus an opt-in cross-validated hyper-parameter search for
*Classify*, *Regress* and *Classify over years* and an automatic-k option for
*Cluster*, all default-off and backed by a pure-NumPy search driver. The remaining
steps need a real QGIS and a maintainer's machine: work
through `docs/SMOKE_TEST.md` in QGIS 3.34 LTR (Qt5) and a Qt6/QGIS 4.x build — an
actual fetch-by-year, a multi-year tick-box fetch, RGB, similarity, change, the
one-step change-over-years, the change report, extract and an ML run with the guided
install — then push the repository, cut a GitHub Release with the ZIP as an asset,
submit to plugins.qgis.org, and clear the `experimental` flag once that coverage
justifies it.

## 4. Longer-horizon possibilities

Beyond the scoped milestones, several directions would deepen the toolbox.
Pluggable embedding backends would let the same algorithms run over Clay, Prithvi
or SatlasPretrain embeddings alongside AlphaEarth, turning the plugin into a
general embedding workbench rather than an AlphaEarth-only tool. Model persistence
and reuse — delivered in v0.7.0 — now lets a classifier or regressor trained on one
area be saved and applied to another without retraining. An active-learning
labelling loop would let users iteratively refine seeds or training data against the
model's current predictions. Cloud-native COG outputs and time-series dashboards
would suit operational monitoring. And the preset library, also delivered in v0.7.0,
gives domain users a running start with named starting points for RGB, similarity,
clustering and change; extending it to targeted profiles — fuel and land cover for
the wildland-urban interface, built-up fraction, canopy — is the natural next step.

## 5. Known follow-ups carried into future releases

Several items once queued here have since shipped. In v0.7.0: built-in accuracy
assessment via a stratified cross-validation report moved the ML tier from
"first-pass model" toward "defensible result"; *Change over years* added the
one-step fetch-and-align-and-compare convenience so *Change* no longer always needs
its years pre-loaded onto a shared grid; and container-run regression tests
(`tests/test_algorithms_qgis.py`) now exercise the QGIS-backed `processAlgorithm`
paths end-to-end in CI rather than waiting on the manual smoke test. The two items
that remained open in this vein both shipped in v0.9.0: the matching one-step "fetch
years and classify" flow arrived as *Classify over years* (`alphaearth:classifyyears`),
and the ML tier gained opt-in hyper-parameter tuning (a cross-validated randomized
search for *Classify*, *Regress* and *Classify over years*, and an automatic-k
choice for *Cluster*) beyond the fixed defaults. Both are default-off and fall back
to the defaults with a warning when the training data is too small to search safely.
With those delivered, the near-term work is now the real-QGIS smoke test and the
publishing mechanics rather than new algorithms.
