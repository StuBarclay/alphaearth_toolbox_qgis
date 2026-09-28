# AlphaEarth Toolbox — Documentation

*Version 0.10.0 (experimental)*

This folder is the detailed documentation set for the AlphaEarth Toolbox QGIS
plugin, which brings Google DeepMind's **AlphaEarth Satellite Embedding V1** into
QGIS — loading, visualising and analysing the 64-band annual embeddings with no
Earth Engine account, API key or GPU, and no extra Python packages for the core
workflow.

## Documents

The set is organised by audience and question:

- **[Architecture](ARCHITECTURE.md)** — for developers and reviewers. How the
  plugin is layered, the import-light boundary that keeps the core testable
  without QGIS, how an algorithm is discovered and run, how the optional
  scikit-learn tier is detected and installed, how remote tiles are read, and how
  the code is tested and packaged.
- **[Methodology](METHODOLOGY.md)** — for technical users. What an AlphaEarth
  embedding is, why its unit-vector format makes similarity an exact dot product,
  exactly how a year of data is discovered and mosaicked from the public cloud
  bucket, how the grid is sized, why nearest-neighbour resampling is used, and the
  maths behind RGB reduction, change, sample extraction and the ML tier.
- **[User Guide](USER_GUIDE.md)** — for end users. Installing the plugin, using
  the tabbed wizard, and running all twelve algorithms step by step, including the
  no-sign-in fetch-by-year workflow (single year, the multi-year tick-box fetch
  and the one-step change-over-years compare), the optional scikit-learn install
  with model save/reuse and accuracy reporting, batch and Modeler use, and
  required attribution.
- **[Limitations](LIMITATIONS.md)** — what the toolbox does not do yet, the
  constraints inherited from the annual 10 m data, and how mature the testing is.
- **[Roadmap & Future Possibilities](ROADMAP.md)** — from the current
  twelve-algorithm release to a public listing, and longer-horizon directions.
- **[Release notes](RELEASE_NOTES.md)** and the **[smoke-test
  checklist](SMOKE_TEST.md)** — what changed in this release and the manual
  real-QGIS pass to run before publishing.

For the top-level overview and install instructions see the repository
[`README.md`](../README.md); for the full design rationale and positioning against
the EMBED-CD plugin see [`SCOPE.md`](../SCOPE.md).

## At a glance

The toolbox registers a Processing provider, **AlphaEarth Toolbox**
(id `alphaearth`), with twelve algorithms.

Eight are in the no-dependency (NumPy + GDAL) tier. *Load embedding*
(`alphaearth:loadembedding`) clips and warps a 64-band embedding to an area of
interest from a layer, a COG URL, **or just a year (2017–2025)** fetched straight
from the public `gs://alphaearth_foundations` bucket. *Load embeddings (multiple
years)* (`alphaearth:loadyears`) does the same for a **set** of years chosen with
tick-boxes over the *same* area of interest, writing one pixel-aligned 64-band
GeoTIFF per year (`alphaearth_<year>.tif`) into an output folder — ready to feed
straight into *Change / trajectory* or *Similarity* with no re-projection.
*Embedding → RGB*
(`alphaearth:torgb`) paints the 64 bands to a 3-band preview, by PCA-to-3 or a
band triplet with a percentile stretch. *Similarity search*
(`alphaearth:similarity`) scores every pixel by cosine similarity to seed
features — an exact dot product, because AlphaEarth pixels are unit vectors —
producing a 0–1 raster and an optional "more like this" mask. *Change /
trajectory* (`alphaearth:change`) compares two or more already-loaded years by
cosine or Euclidean distance, in pairwise, trajectory-magnitude and
anomaly-vs-baseline modes — the capability that takes the toolbox past the
bi-temporal EMBED-CD plugin. *Change over years* (`alphaearth:changeyears`) folds
the multi-year fetch and *Change / trajectory* into one step — tick the years,
set one area of interest, get the change raster directly (every year fetched onto
the same grid, so always pixel-aligned). *Extract training samples*
(`alphaearth:extract`) samples the 64-D vector at point/polygon features into a
point layer (one row per pixel, `A00…A63` plus source attributes) for external ML
or QA. *Change report* (`alphaearth:changereport`) turns any single-band change
raster into a small report — a change-magnitude distribution with summary
statistics, the percentage of area changed at each threshold, an optional
over-time series and an optional change-vs-embedding-band scatter — written as any
combination of HTML, PNG, CSV and PDF (HTML and CSV need nothing extra; PNG and
PDF use matplotlib when present and are skipped with a message when it is not). The
same report is offered as an opt-in extra on both change algorithms and in the
wizard's Change tab.

Four more form the optional scikit-learn tier. *Classify embedding*
(`alphaearth:classify`) trains a random-forest or gradient-boosting classifier on
labelled features and maps land cover across every valid pixel, optionally with a
per-pixel confidence raster. *Classify over years* (`alphaearth:classifyyears`) is
the classification analogue of *Change over years*: tick two or more years, set one
area of interest, and it fetches every year onto the same grid and maps land cover
for each in one step — one model trained on the most recent year (or reused from a
saved file) applied across the series, writing a folder of per-year class rasters
plus an optional first-to-last transition raster and a changed/unchanged mask.
*Cluster embedding* (`alphaearth:cluster`) runs unsupervised K-means. *Regress
embedding* (`alphaearth:regress`) predicts a continuous target from a numeric
field. Classify and Regress can save the fitted model and reuse it on another scene
(version- and band-count-stamped, so a mismatch fails loudly) and optionally write
a k-fold accuracy report with a class-balance summary. Classify, Regress and
Classify over years can optionally **tune their hyper-parameters** (a
cross-validated randomized search), and Cluster can **choose k automatically** by
silhouette — all opt-in and default-off, falling back to the defaults with a
warning rather than failing when the data is too small. scikit-learn is never a
hard dependency — it is detected at runtime, and the wizard offers a one-click
install into the QGIS Python environment when it is missing.

All twelve are standard `QgsProcessingAlgorithm`s, so they run headless, in batch
and in the Graphical Modeler; the plugin's own dialog is a guided tabbed wizard
with curated presets and off-GUI-thread execution.

## Licensing & attribution

The plugin code is licensed **GPL-2.0-or-later**. The AlphaEarth data is **CC BY
4.0** and requires attribution — *"Contains modified Google DeepMind AlphaEarth
Satellite Embedding V1 data (CC BY 4.0)."* — which is shown in each algorithm's
help and should accompany any published output. AlphaEarth is commercially usable
with attribution.
