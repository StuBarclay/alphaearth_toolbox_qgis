# Methodology

*AlphaEarth Toolbox for QGIS — v0.10.0*

This document explains the data the toolbox consumes and the maths it performs:
what an AlphaEarth embedding is, why its unit-vector format makes similarity an
exact dot product and multi-year change a direct distance, exactly how a year of
data is discovered and mosaicked from the public cloud bucket, how the output grid
is sized, why nearest-neighbour resampling is the right choice, and how the RGB
reduction, similarity, change (both the raster-based *Change / trajectory* and the
one-step *Change over years*), the change report / charts, sample-extraction and
scikit-learn (classify, classify over years, cluster, regress) algorithms work,
including the optional cross-validation accuracy metrics and hyper-parameter
tuning. It is aimed at technical users and reviewers who
need to trust — and reproduce — what the algorithms compute.

## 1. The dataset: AlphaEarth Satellite Embedding V1

AlphaEarth Satellite Embedding V1, from Google DeepMind, is a global, annual
(2017–2025), 10-metre, 64-band learned embedding of the Earth's land surface. It
is not imagery: each pixel is a 64-dimensional feature vector that summarises a
year of multi-sensor observations at that location. The dataset is licensed
**CC BY 4.0** and published as Cloud-Optimised GeoTIFFs (COGs) in the public
Google Cloud Storage bucket `gs://alphaearth_foundations`, readable anonymously.

Two properties of the format matter for everything that follows. First, each
pixel vector is (very close to) **unit length** in 64-D — it lies on the unit
hypersphere. Second, the published tiles are quantised to **int8** to save space:
the stored integers are the unit-length float components multiplied by a scale of
127.5, and a sentinel of **−128** marks fill/no-data (ocean, scene edges, gaps).
The toolbox constants that encode this live in `aecore/intake.py`:
`EMBEDDING_BANDS = 64`, `AVAILABLE_YEARS = 2017…2025`, `NATIVE_RESOLUTION_M = 10.0`,
`QUANT_SCALE = 127.5` and `FILL_VALUE = -128`.

The consequence of the unit-length property is the toolbox's headline capability.
For two unit vectors **a** and **b**, cosine similarity is
`cos θ = (a · b) / (‖a‖‖b‖)`, and because `‖a‖ = ‖b‖ = 1` this collapses to the
plain dot product `a · b`. So "how similar are these two pixels" is answered by a
single multiply-and-add per band — no scikit-learn, no distance library, no
training. The same property makes *change* between two years a direct distance:
cosine distance `1 − a · b`, or Euclidean distance `‖a − b‖`, and for unit vectors
the two are tied by `‖a − b‖ = √(2·(1 − a · b))`. That is why both the Similarity
and Change algorithms sit in the no-dependency tier.

## 2. The storage layout

Objects in the bucket are named:

```
satellite_embedding/v1/annual/{YEAR}/{UTM_ZONE}/{hash}-{row}-{col}.tiff
```

`{YEAR}` is 2017–2025. `{UTM_ZONE}` is a UTM zone number (1–60) with a hemisphere
suffix `N` or `S` — for example `56S` for Sydney. Every tile in a zone folder is a
64-band int8 COG at 10 m in that zone's own UTM CRS. This per-zone-CRS layout is
important: an area of interest that straddles two UTM zones will draw tiles that
live in *different* projected coordinate systems, and the mosaic step has to
reconcile them (§4).

The endpoint is not guessed. The bucket name, prefix, host and quantisation
constants are grounded in the proven intake code of the sibling
`alphaearth-bushfire` package, and are recorded as named constants in
`intake.py` (`GCS_BUCKET`, `GCS_PREFIX`, `GCS_HOST`, `GCS_LIST_URL`) so there is a
single source of truth.

## 3. Discovering the tiles for an area of interest

When the *Load embedding* algorithm is given a year (rather than an explicit layer
or URL), it discovers the covering tiles in five steps, all of which are honoured
against the user's Cancel button.

The area of interest arrives in the user's chosen target CRS (default EPSG:3577).
Its bounds are first snapped to a whole number of output pixels (§5), then
reprojected to WGS84 with a `QgsCoordinateTransform` so the toolbox can decide
which UTM-zone folders the area touches. `intake.utm_zones_for_bbox` does this by
*interval overlap*: each zone `z` owns the longitude window `[6z − 186, 6z − 180)`
degrees, and a zone is included only if that window genuinely overlaps the area's
longitude span. This deliberately avoids a subtle bug — an eastern edge that lands
exactly on a zone boundary must not pull in the next, empty zone — and the
hemisphere suffix follows the latitude sign, emitting both `N` and `S` when the
area straddles the equator.

For each touched zone, `intake.list_zone_tiles` enumerates the objects under that
`year/zone/` folder using the public GCS JSON list API, following `nextPageToken`
pagination in pages of 1000 and keeping only `.tif`/`.tiff` objects (the bucket's
folder-placeholder objects are skipped). The network call is a small injectable
`fetch_json` helper, which is what allows the whole discovery path to be unit-
tested offline. The listing itself, together with the per-scene anchors described
below, is persisted per `(year, zone)` via `intake.save_zone_index` /
`load_zone_index`, so a later run over a *different* area of interest in a zone
already visited skips the list API entirely and reuses the cached index.

Before any tile is opened, each zone's cached footprints are loaded with
`intake.load_footprint_cache`. AlphaEarth's annual embeddings are immutable, so a
tile's footprint never changes; the toolbox therefore persists the footprints it
scans to a small JSON file per `(year, zone)` (under the system temp directory,
keyed by a schema version plus the bucket/prefix so a stale or foreign cache is
ignored). Footprints are stored in **WGS84**, not the run's target CRS, so the
cache is reused whatever CRS a later run requests. `intake.partition_by_cache`
then splits the zone's tiles into those already known and those still to scan.

The uncached tiles are not all scanned blindly. An AlphaEarth object name ends
`{hash}-{offset_a}-{offset_b}.tiff`, where the two integers are that scene's
per-tile *pixel* offsets — each `{hash}` is an independent export with its own
local origin, so the numbers are not positions on a global grid and a tile's
location cannot be read from its name alone. But tiles *can* be grouped by their
scene hash (`intake.group_by_scene`, keyed by `intake.scene_key`; an unparseable
name becomes its own singleton scene and is therefore always scanned, never culled).
For each scene with uncached tiles, one **anchor** tile is opened with the new
`_raster.dataset_footprint_and_pixels` helper — a header-only read returning the
anchor's WGS84 footprint *and* its pixel dimensions. From that footprint, the
anchor's pixel size and the (known) pixel offsets of the scene's other tiles,
`intake.conservative_scene_bounds` computes a deliberately generous over-estimate of
the whole scene's extent: it takes the larger per-pixel degree step of the two axes,
multiplies by the maximum pixel reach across the scene plus one tile, inflates by a
50 % margin, and pads the anchor box symmetrically on all four sides. Because the
estimate can only ever *over*-include (it is proven by unit test to contain every
tile's true footprint under *either* pixel-axis→lon/lat orientation), a scene whose
over-estimate does not intersect the area of interest can be dropped without reading
any of its tiles, and no true intersection is ever lost.

Only the tiles of scenes that survive that cull are then header-scanned
concurrently. A `ThreadPoolExecutor` with 16 workers calls
`_raster.dataset_bounds_in_crs` on each — a *header-only* open that reads the tile's
geotransform and size, computes its four corner coordinates, and reprojects them
into WGS84 to produce a footprint. Both the anchor reads and this second scan share
that bounded worker pool, so a fetch never opens an unbounded number of remote
connections at once. The freshly scanned footprints (and the scene anchors) are
written back to the caches, so a repeat area of interest over the same zones does no
network I/O in this phase at all, and turning a first fetch from roughly one read
per tile into roughly one read per *scene* — commonly several-fold fewer reads —
without changing which tiles are ultimately selected: the exact intersection test
below always uses real scanned footprints, never the over-estimate. A tile (or
anchor) that cannot be opened or transformed — a transient `/vsicurl` fault, or an
odd edge tile — returns `None` and is simply skipped rather than aborting the run
(a failed *anchor* just means its scene cannot be culled, so all its tiles are
scanned as a safe fallback). `dataset_bounds_in_crs` deliberately catches *any*
error to enforce this, which matters under GDAL 3.9+ (shipped in QGIS 4), where the
Python bindings raise exceptions by default instead of returning a quiet failure:
without the broad guard, a single bad tile among the thousands scanned concurrently
would propagate out of the worker pool and collapse the whole run into an opaque
background-task failure. The number of tiles skipped this way is counted and
surfaced to the user as a single warning ("the mosaic may have small gaps there")
rather than passing silently. This whole discovery phase is the 0–90 % span of the
progress bar (and completes instantly on a full cache hit).

Only tiles whose (WGS84) footprint actually intersects the area of interest —
itself reprojected to WGS84 — are kept, tested with `intake.tiles_intersecting`
over `intake.bboxes_intersect` (touching edges do not count).

Finally the surviving tiles — possibly spanning several UTM-zone CRSs — are handed
to `_raster.warp_many_to_grid` and mosaicked in one pass (§4). If no tiles were
published for the year/zones, or none intersect the area, the algorithm raises a
clear, specific error rather than emitting an empty raster.

## 4. Mosaicking across UTM zones

`gdal.Warp` accepts a list of source datasets and reprojects each one
independently onto the requested output grid. The toolbox exploits this in
`warp_many_to_grid`: it passes the full list of intersecting `/vsicurl` tiles with
a single set of output options — target width and height in pixels, output bounds,
target CRS as WKT, and the resampling algorithm — and lets GDAL stitch tiles from
different UTM zones straight onto one grid in the target CRS. Source and
destination no-data are both set to the int8 fill value (−128) so genuinely empty
pixels stay empty through the warp. The whole set is treated as a remote read and
retried as a unit with the tuned HTTP/2 + multi-range configuration described in
the Architecture document.

## 5. Sizing the output grid

`intake.grid_dimensions` turns an extent and a resolution into an integer pixel
grid. Width and height are computed by rounding the extent *out* (ceiling) to a
whole number of pixels — never shrinking it — so the requested area is always
fully covered, with a floor of at least one pixel in each direction. The returned
"snapped" bounds are the extent re-expanded to that whole-pixel grid, anchored at
the top-left corner (the maximum-Y edge is held fixed and the grid grows downward).
Using these snapped bounds for both tile selection and the warp guarantees the
footprint test and the mosaic agree on exactly the same grid. The function guards
against a non-positive resolution and an empty extent with explicit errors.

Once the grid is sized, the algorithm estimates the output's memory footprint
(width × height × 64 bands × 4 bytes for the float32 cube) and, above roughly
2 GiB, emits an up-front warning that the read may be slow or exhaust memory and
that a smaller extent or coarser resolution would help. It is only a warning — a
deliberate large export still proceeds. When the chosen output is itself a
`.tif`/`.tiff` file, the de-quantised cube is written straight to that destination
in a single pass; other formats are written to a scratch GeoTIFF and translated,
so the common case avoids an extra full-size copy on disk.

At the default EPSG:3577 (Australian Albers) the units are metres, so the native
10 m resolution maps directly to a 10-unit pixel. EPSG:3577 is chosen as the
default because it is equal-area, which keeps any downstream area statistics
honest; the CRS is a per-run parameter, so the toolbox is equally usable outside
Australia.

This snapped grid is also what makes the multi-year loader (*Load embeddings
(multiple years)*, `alphaearth:loadyears`) produce analysis-ready output. It sizes
the grid **once** from the shared area of interest, resolution and CRS, then fetches
each ticked year (2017–2025) through the identical fetch-and-mosaic path onto that
same grid — so the per-year rasters it writes (`alphaearth_<year>.tif`) are exactly
pixel-aligned. Because *Change / trajectory* (§10) requires its inputs to share one
`(bands, rows, cols)` grid, and *Similarity* compares pixel-for-pixel, this
alignment is precisely what lets the multi-year outputs feed those algorithms with
no intermediate re-projection or resampling — which is the point of fetching the
years together rather than one at a time. The maths of each year's fetch is
unchanged from the single-year loader (§3–§6); only the orchestration differs (the
grid and large-area check are computed once, and each year takes an equal share of
the progress bar).

## 6. Dequantising to unit vectors

After the mosaic, the int8 cube is converted back to float embedding vectors. The
fill pixels (equal to −128) are recorded, every value is divided by
`QUANT_SCALE = 127.5` and cast to float32, and the fill pixels are then written as
the float no-data sentinel `-9999.0`. That sentinel is chosen to sit far outside
the legitimate unit-vector range of roughly [−1, 1], so downstream algorithms can
mask no-data without ever colliding with a real embedding value. The output
GeoTIFF records `-9999.0` as its no-data value. When *Load embedding* is instead
given an explicit layer or URL, the source is warped verbatim with no
dequantisation (it is assumed already to be in whatever units the user supplied).

## 7. Why nearest-neighbour resampling

Both the fetch-by-year mosaic and the single-source warp resample with
nearest-neighbour. This is a considered choice, not a default. The embedding
vectors are *learned* features whose meaning depends on the exact 64 components
staying together; averaging neighbouring pixels (bilinear/cubic) would produce a
blended vector that is no longer a genuine embedding and, critically, would break
the near-unit-length property the similarity maths relies on. Nearest-neighbour
carries each source pixel's vector through unchanged, preserving both its meaning
and its magnitude.

## 8. Embedding → RGB: the maths

The RGB algorithm reduces the 64-band embedding to a three-band image so it can be
seen at a glance, and is implemented in the pure-NumPy `aecore/rgb.py`. It offers
two methods, both ending in the same percentile contrast stretch.

The default is **PCA-to-3**. The valid pixels (finite across all bands, and not
equal to the no-data value where one is set) are collected into an `(n, 64)`
matrix; if there are more than a cap, a deterministic random subsample is drawn so
the fit is fast on large scenes. The subsample is mean-centred and decomposed with
a singular value decomposition; the top three right-singular vectors are the
principal axes, and every valid pixel is projected onto them to give three
components. PCA is used because it packs the most variance into the fewest bands,
so the three channels carry as much of the embedding's structure as three numbers
can — but note the axes are data-dependent, so PCA-RGB colours are not comparable
between two separate runs. The alternative method, **band triplet**, simply takes
three user-chosen bands (e.g. `10,20,30`) as R, G and B, which *is* stable across
scenes at the cost of showing only three of the 64 dimensions.

Either way the three channels are then contrast-stretched independently: each is
clipped to its 2nd–98th percentile (over valid pixels) and linearly rescaled to
0–255 `uint8`, which spreads the visible range without letting a few outliers
flatten the image. No-data pixels are written as the fill colour (0,0,0). The
result is a standard 3-band RGB raster; it is a visualisation aid only and must not
be fed back into the analytical algorithms, which expect the 64-band embedding.

## 9. Similarity search: the maths

The Similarity algorithm answers "find more like this" and is implemented in the
pure-NumPy `aecore/similarity.py`. It proceeds in four stages.

First, **seed extraction**. The user's seed points or polygons are rasterised onto
the embedding grid (points mark the pixel they fall in; polygons mark every pixel
whose centre they contain, with a centroid fallback for features smaller than a
pixel), reprojecting the seed geometries into the embedding's CRS first.
`extract_vectors` then pulls the `(n, bands)` matrix of embedding vectors at those
seed pixels.

Second, **seed aggregation** (`aggregate_seeds`) reduces those `n` vectors to a
single reference vector, either as their mean (then re-normalised to unit length)
or as their *medoid* — the seed most similar on average to the others, found from
the pairwise cosine-similarity matrix. Both paths return a unit-length reference.

Third, the **similarity map** (`similarity_map`). The embedding cube and the
reference are each L2-normalised, then the reference is dotted against every pixel:
mechanically, a `(k, bands)` reference matrix is multiplied by the
`(bands, rows·cols)` reshaped cube to give `(k, rows·cols)` scores, and each pixel
takes its best (maximum) score over the `k` seeds. Explicit normalisation makes
the result exact even if the input vectors are only approximately unit length
(as int8 dequantisation leaves them). By default the cosine result is rescaled
from `[−1, 1]` to `[0, 1]` via `(x + 1) / 2` for convenient styling; raw cosine can
be requested instead. No-data pixels — identified as any pixel that is not finite
across all bands, or (where a no-data value is set) equal to it in every band —
are written as `NaN` so they are never scored as a spurious match.

Fourth, an optional **threshold mask** (`threshold_mask`) produces a `uint8`
"more like this" layer that is 1 where similarity ≥ the threshold. `NaN` (no-data)
pixels are treated as below any threshold, so they are never included. The
similarity raster is written with a `-9999.0` no-data sentinel and given a
best-effort blue-to-red pseudocolour style once loaded.

## 10. Change / trajectory: the maths

The Change algorithm answers "how much did each pixel change between years?" and is
implemented in the pure-NumPy `aecore/change.py`. It takes two or more embedding
cubes — one per year, on a shared `(bands, rows, cols)` grid — in temporal order,
and reduces them to a single-band per-pixel change raster under one of three modes.

The per-pixel distance between two cubes is the primitive everything builds on.
For **cosine** distance the two cubes are L2-normalised along the band axis and
dotted band-wise, giving `1 − Σ(â·b̂)` per pixel, bounded in `[0, 2]` (0 = no
change, 1 = orthogonal, 2 = opposite). For **Euclidean** distance it is
`√(Σ(a − b)²)`. Explicit normalisation for the cosine path makes the result exact
even though int8 dequantisation leaves the vectors only approximately unit length.

*Pairwise* mode is simply this distance between the first and last supplied year —
the classic bi-temporal change map, and the case EMBED-CD covers. *Trajectory
magnitude* sums the distance across every consecutive pair of years (`itertools.
pairwise` over the ordered sequence), so a pixel that changes and then reverts
scores its full round-trip travel rather than the ~zero a first-vs-last comparison
would show; it needs three or more years to differ from pairwise. *Anomaly vs
baseline* forms the element-wise mean ("baseline") cube across all supplied years
and returns the distance of the most recent year from that baseline — flagging
where the latest year departs from the multi-year norm. A pixel is valid only where
every input year is valid; the per-year no-data masks are combined with logical AND
and invalid pixels are written as `NaN`, then the float sentinel `-9999.0` on
output. The result is auto-styled blue (little change) to red (most change).

*Change over years* (`alphaearth:changeyears`) computes exactly this same change
surface but removes the manual step of loading each year onto a matching grid. The
user ticks two or more years and gives one area of interest, target CRS and
resolution; the algorithm sizes the snapped output grid once (§5) and then fetches
each ticked year through the identical fetch-and-mosaic-and-dequantise path used by
*Load embedding* and *Load embeddings (multiple years)* (§3–§6, via the shared
`_fetch.fetch_year_cube`) directly onto that one grid. Because every year is warped
onto the same `(bands, rows, cols)` grid in a single pass, the per-year cubes are
pixel-aligned by construction — there is no separate alignment test to satisfy,
unlike the raster-based *Change / trajectory*, which must verify its inputs share a
grid. The stacked cubes, in temporal order, are then handed to the very same
`aecore.change` `compute_change` dispatch, so the choice of mode (pairwise,
trajectory magnitude, anomaly vs baseline) and metric (cosine or Euclidean) and the
resulting distance maths are identical to those above; only the provenance of the
inputs differs. The fetched per-year cubes may optionally also be written to a
folder as `alphaearth_<year>.tif`. Sharing both `fetch_year_cube` and
`compute_change` with the loaders and with *Change / trajectory* guarantees the
one-step flow cannot drift from the multi-step one.

## 11. Change report: the maths

The Change report turns a change raster into a small, readable summary and is
implemented in the pure-NumPy/stdlib `aecore/report.py`. It reads one band of a
single-band magnitude raster (no-data mapped to `NaN`), and every statistic is
computed over the **finite** values only (`finite_values` drops the `NaN`s), so
ocean and missing tiles never distort the numbers. Four content types are computed
independently and any combination can be requested.

The **change-magnitude distribution** is a `Stats` summary plus a histogram.
`summarise` returns the count of valid pixels, the minimum and maximum, the mean,
the median and the standard deviation, together with a fixed set of percentiles
(the 5th, 25th, 50th, 75th and 95th) computed with `numpy.percentile`; it raises
rather than inventing a number when there are no finite values. `histogram` bins the
finite values into a chosen number of equal-width bins between the observed minimum
and maximum, returning the bin edges and counts (which sum to the valid-pixel count)
for the bar chart.

The **percentage of area changed** answers "how much of the scene changed by at
least *t*?" for each of several thresholds. `area_fractions` counts the finite
pixels at or above each threshold and divides by the valid-pixel count, so the
fractions are monotone non-increasing in the threshold and the fraction at the
lowest threshold that every pixel clears is 1.0. When a per-pixel ground area is
known — that is, when the raster's CRS is projected, so a pixel covers a constant
number of square metres — each row also carries the corresponding **hectares**
(pixels × pixel-area-m² ÷ 10 000); over a geographic (degrees) CRS, where a pixel
has no constant ground area, the hectares are omitted and only the fraction is
given. The pixel area is derived from the raster's geotransform by
`_report.pixel_area_m2`, which returns `None` for a geographic CRS.

The **change-over-time series** is built only when several change rasters are
supplied in order (the optional *Change rasters over time* input, or, when the
report is run from a change algorithm, the consecutive-year pairs). `series_points`
reduces each period's raster to one point — its label (the layer name), its mean and
median change and its valid-pixel count — and silently skips a period with no finite
pixels, giving a line/point series of how the typical change magnitude moves from
period to period.

The **scatter** relates change to one embedding band. `scatter_sample` pairs the
per-pixel change value with the chosen band's value at the same pixel, keeps only
the pairs finite in *both*, and — because a full 10 m scene is far too many points to
plot — draws a deterministic random subsample (a fixed seed, so the same inputs give
the same picture) capped at a maximum number of points. It requires the two arrays
to be the same size and raises otherwise; `_report.scatter_from_band` returns `None`
(and the algorithm warns) when the embedding raster's band or grid does not match the
change raster, so a mismatched scatter is omitted rather than misleading.

`build_model` assembles just the requested contents into a `ReportModel`, recording a
note for any content that was asked for but could not be produced (no series rasters,
or a mismatched scatter). Rendering is then format-specific but reads only that
model. `render_html` emits a **standalone** HTML page — every chart is inline SVG and
every label is HTML-escaped, with no external asset or fetch — and `render_csv`
writes the same numbers as CSV sections (summary, histogram, area-by-threshold,
over-time, scatter), quoting any field that contains a comma. `render_png` and
`render_pdf` draw the same charts with matplotlib, but only when
`matplotlib_available()` reports it importable; when it is not, those two formats are
skipped with a message rather than failing the run — the same optional-dependency
discipline the scikit-learn tier uses. HTML and CSV therefore always work with
nothing beyond NumPy and the standard library.

## 12. Extract training samples: the maths

The Extract algorithm turns marked features into a table of embedding vectors, for
external ML or QA, and is implemented with the pure helpers in
`aecore/sampling.py`. It samples **one row per covered pixel**: a point samples the
pixel its coordinate falls in (`rowcol_for_xy` maps a map coordinate to a
`(row, col)` via the inverse geotransform), while a polygon samples every pixel
whose centre lies inside it (a bounding-box scan testing `geom.contains` at each
`pixel_centre_xy`, with a centroid fallback for sub-pixel features and a safety cap
on the pixels one feature may cover). Seed geometries are reprojected into the
embedding's CRS first.

For each feature the covered pixel indices are gathered in order and
`gather_vectors` reads the `(n, bands)` matrix of embedding vectors at exactly those
pixels — order-preserving, so each output row lines up with its source feature.
When *Skip no-data pixels* is on, `valid_rows` drops any sampled row that is not
finite across all bands, or (where a no-data value is set) equals it in every band.
Every surviving pixel becomes an output point at the pixel centre, in the
embedding's CRS, carrying the source feature's full attribute record (so class
labels and ids follow the data) plus four pixel-metadata columns and the band
values `A00 … A{n−1}` — the AlphaEarth band naming. The result is a standard vector
layer, so exporting to CSV for scikit-learn, R or a notebook is a right-click away.

## 13. The scikit-learn tier: classify, classify over years, cluster, regress

Four algorithms add supervised and unsupervised learning over the embedding. They
need scikit-learn, which QGIS does not bundle, so each first calls a runtime guard
(`_sklearn.ensure_sklearn`) and stops with a clear, actionable error when the
library is absent — the analytics never become a silent hard dependency. The
per-pixel plumbing they share lives in the pure-NumPy `aecore/ml.py`, which is unit
tested with an injected `predict` function so the flow is verifiable without
scikit-learn installed.

The common shape is: turn the embedding cube into a design matrix, fit an
estimator, predict across every valid pixel, and scatter the predictions back onto
the grid. `cube_to_matrix` reshapes the `(64, rows, cols)` cube to an
`(rows·cols, 64)` matrix in row-major order, and `flat_valid_mask` marks which of
those rows are worth predicting on (finite across all bands, and — where a no-data
value is set — not equal to it in every band). Prediction runs through
`predict_in_chunks`, which splits the design matrix into row-blocks and applies the
estimator block by block, so peak memory stays bounded on large scenes while the
estimator still sees ordinary 2-D arrays. `scatter_to_grid` then writes each
prediction back to its pixel and fills the masked-out cells, returning a
`(rows, cols)` raster of the requested dtype (an integer type for class/cluster
codes, float for regression).

*Classify* is supervised. Labelled point/polygon features are sampled against the
embedding (reusing the `aecore.sampling` maths via `_features`) to build a training
matrix whose rows are 64-D vectors and whose targets are the class labels;
`encode_labels` maps arbitrary label values to contiguous integer codes and back. A
`RandomForestClassifier` or `GradientBoostingClassifier` is fitted and applied to
every valid pixel. When a confidence output is requested, the class probabilities
(`predict_proba`) are computed in the same chunked way and the winning-class
probability (the row maximum, 0–1) is written as a second raster — a direct read of
how sure the model is at each pixel. *Regress* is the continuous analogue: the
target is a numeric field (a non-numeric field is rejected up front), a
`RandomForestRegressor` or `GradientBoostingRegressor` is fitted, and the
prediction is a pseudocolour surface. *Cluster* is unsupervised: it needs no
training features, fits `KMeans` (with `k` chosen by the user) on a subsample of
the valid pixels for speed, and then assigns every valid pixel to its nearest
cluster in the same chunked prediction pass. Cluster codes are arbitrary labels,
not ranks, so the output is styled with a categorical palette. All three honour the
no-data mask throughout, so ocean and missing tiles are never fed to the estimator
and never receive a spurious prediction.

*Classify* and *Regress* can optionally cross-validate the model before it is
applied. Given a fold count *k* and a report path, the training matrix is split by
**stratified** k-fold — each fold is drawn so that every class keeps (as nearly as
integer counts allow) its overall proportion, which keeps rare classes present in
each fold — and the estimator is refitted *k* times on the other folds and scored on
the held-out one. For classification the report gives overall accuracy (the fraction
of held-out samples predicted correctly) together with, per class, precision
`TP / (TP + FP)`, recall `TP / (TP + FN)` and their harmonic mean
`F1 = 2·precision·recall / (precision + recall)`, plus the full confusion matrix
(rows = true class, columns = predicted). For regression it gives per-fold and
pooled coefficient of determination `R² = 1 − SS_res / SS_tot` and root-mean-square
error `RMSE = √(mean((y − ŷ)²))`. Both reports append a class-balance summary — the
count of training samples per class — and raise a warning when the set is heavily
imbalanced (a large majority-to-minority ratio), because an accuracy figure from a
skewed set flatters the model. All of this — the stratified fold assignment, the
confusion matrix, precision/recall/F1, R²/RMSE and the balance check — is pure NumPy
in `aecore/ml.py`, unit-tested without scikit-learn, QGIS or GDAL; scikit-learn is
used only to fit and predict within each fold.

A fitted *Classify* or *Regress* estimator can be saved and reused. The pure
`aecore/model_io.py` pickles the estimator to a `.pkl` together with a stamp
recording the scikit-learn version it was trained under and the 64-band embedding
count it expects. Reusing that file on another scene skips training entirely (so no
training layer or label field is needed), but `model_io` first checks the stamp: if
the saved band count or scikit-learn version does not match the current environment
it raises a clear error rather than predicting from an incompatible model, since a
silent mismatch would produce a confidently wrong map.

*Classify over years* (`alphaearth:classifyyears`) is the supervised analogue of
*Change over years*: it applies one classifier to a whole run of years at once.
The user ticks two or more years and gives one area of interest, target CRS and
resolution; the algorithm sizes the snapped grid once (§5) and fetches each ticked
year onto it through the same `_fetch.fetch_year_cube` path as the loaders (§3–§6),
so the years are pixel-aligned by construction. Crucially, **one** model is trained
— on the most recent ticked year, using the same feature-sampling, encoding and
`_build_classifier` fit as *Classify* — or a saved model is reused, and that single
model is then applied to *every* year. Training once and predicting many keeps the
class codes comparable across the series: a pixel labelled code 3 in 2018 and code 3
in 2024 means the same land cover, which a per-year retrain could not promise. Each
year is written to `classification_<year>.tif` in an output folder. Two optional
extra outputs summarise the change between the first and last year: a **transition**
raster encodes each ordered `(from-class, to-class)` pair as a single integer code
`first * n_classes + last` (so every combination is distinct, and the "no change"
transitions fall on the diagonal), auto-styled with a legend reading
`"<from> -> <to>"`; and a **changed/unchanged mask** is simply `1` where the
first- and last-year class differ and `0` where they agree. Both are computed in the
pure `aecore.ml` (`transition_codes`, `transition_labels`, `changed_mask`) as `int64`
and cast to `int32` with `-1` no-data before writing, because GDAL GeoTIFF support
for 64-bit integers is not universal. Reusing the loaders' fetch and *Classify*'s
estimator means this algorithm cannot drift from either.

Any of *Classify*, *Regress* and *Classify over years* can optionally **tune the
estimator's hyper-parameters** before the final fit, and *Cluster* can **choose k
automatically** — both opt-in and off by default, so an untouched run is unchanged.
When tuning is requested, a small preset search space for the chosen estimator (for
a random forest: the number of trees, maximum depth, feature-subsample rule and
minimum leaf size; for gradient boosting: the number of stages, learning rate and
depth) is searched by a **randomized** cross-validated search — a fixed budget of
random combinations is scored by stratified k-fold accuracy (classification) or R²
(regression), reusing the same fold machinery as the accuracy report, and the best
combination is used for the final fit and recorded in the report. Sampling random
combinations rather than an exhaustive grid keeps the cost fixed and predictable
regardless of how large the space is. The search *driver* is pure NumPy
(`ml.randomized_search`) and takes a scoring closure that owns scikit-learn, so the
driver is testable without it. If the training set has fewer samples than folds the
tuning is **skipped with a warning** and the defaults are used, never a failure.
*Cluster*'s automatic k fits K-means for each k over a user range and keeps the k
with the best mean **silhouette** coefficient (how well each pixel sits in its own
cluster versus the nearest other, in `[-1, 1]`), scored on a bounded pixel subsample
because the silhouette is quadratic in sample count.

Several algorithms also offer a **preset** — a named starting point drawn from the
pure-stdlib `aecore/presets.py` for RGB, similarity, clustering and change. A preset
stores *semantic* values (for example `method="pca"` or `metric="cosine"`) rather
than a dropdown index, so it selects the same behaviour regardless of widget
ordering or future additions. Presets are guidance, not guarantees: because the
embedding has no fixed per-band semantics, a preset is a sensible default to adjust,
not a calibrated recipe.

## 14. Reproducibility and honesty about no-data

Two design choices support trustworthy results. Tile discovery is deterministic
and offline-testable because the only network call is an injectable function, and
the ZIP builder is byte-reproducible. And no-data is surfaced explicitly at every
stage rather than being silently treated as a real value: the int8 −128 fill is
carried through the warp, mapped to a distinct float sentinel on load, propagated
as `NaN` through the similarity and change maths (a change pixel is no-data unless
*every* input year is valid there), excluded from threshold masks, and skipped when
extracting samples. A missing tile or an ocean pixel therefore shows up as no-data,
never as a misleading "zero similarity" or "no change".
