# Limitations

*AlphaEarth Toolbox for QGIS — v0.10.0*

This document is candid about what the toolbox does *not* do yet and where its
results should be treated with care. It covers the current release's scope, the
constraints inherited from the data, and the maturity of testing. Read it
alongside the Methodology document, which explains the maths these limitations
qualify.

## 1. Release scope

All twelve algorithms are now implemented: *Load embedding*, *Load embeddings
(multiple years)*, *Embedding → RGB*, *Similarity search*, *Change / trajectory*,
*Change over years*, *Change report* and *Extract training samples* in the
no-dependency (NumPy + GDAL) tier, plus *Classify*, *Classify over years*, *Cluster*
and *Regress* in the optional scikit-learn tier. The dialog is the guided tabbed wizard (draw-AOI
extent, year picker, target-CRS chooser, off-GUI-thread execution with progress and
Cancel). The plugin is nonetheless still marked `experimental=True`: real-QGIS
coverage is only beginning to broaden (see §10), so treat this as a capable but
not-yet-battle-tested release.

The scikit-learn tier is an optional add-on, not part of a fresh install.
scikit-learn is never bundled and never a hard dependency; until you install it
(one click in the wizard, or `pip` by hand), *Classify*, *Classify over years*,
*Cluster* and *Regress* will decline to run. Installing it changes the QGIS Python environment for every
plugin, and the install needs network access and appropriate permissions — on a
locked-down or offline QGIS it may fail, in which case the no-dependency tier still
works fully.

## 2. Data constraints

The limitations of AlphaEarth V1 itself flow straight through the toolbox. The
data is **annual**, so it captures a year's worth of surface behaviour and cannot
resolve within-year or single-date events — there is one embedding per calendar
year. It covers **2017–2025** only; requests outside that range are rejected. The
native resolution is **10 m**, so features smaller than that are not resolvable,
and asking for a finer output resolution only resamples, it does not add detail.

The embedding is a *learned* representation, not a physical measurement. Its 64
bands have no direct physical units, and their meaning is defined only relative to
other embeddings. That is exactly why similarity and change work well, but it also
means an embedding value on its own is not interpretable the way a reflectance or
a temperature is.

Coverage gaps and ocean are represented as fill (int8 −128), which the toolbox
surfaces as explicit no-data rather than as a real value. This is a feature, but it
does mean areas over water or with missing tiles will simply have no output there.

## 3. Embedding → RGB caveats

The RGB preview is a *visualisation aid, not data*. It compresses 64 dimensions to
three, so it necessarily discards most of the embedding — never feed an RGB output
back into the analytical algorithms, which expect the full 64-band embedding. The
default PCA method chooses its colour axes from the pixels in each run, so the same
land cover can appear in different colours in two different rasters (or two AOIs of
one year); when you need colours that mean the same thing across scenes, use the
band-triplet method instead, at the cost of seeing only three of the 64 bands. The
percentile stretch is cosmetic and per-channel, so absolute colours carry no
physical meaning either way.

## 4. Similarity-search caveats

Similarity search is only as good as its seeds. A handful of unrepresentative seed
pixels will produce a misleading map, and mixing distinct targets into one seed set
blends them into an average vector that may match neither well. The *mean*
aggregation is sensitive to outlier seeds; *medoid* is more robust but reduces the
whole seed set to a single most-representative example. The result is a
*relative* similarity surface — high values mean "like your seeds", not any
absolute class membership — so the threshold that separates "similar enough" is a
judgement call the user makes per task, not a calibrated constant.

Seed geometries must overlap the embedding raster; polygon seeds are rasterised by
pixel-centre containment (with a centroid fallback for sub-pixel features), and a
very large polygon seed is capped to protect the run. Similarity is computed in the
embedding's own grid and CRS, so results inherit whatever resolution and projection
the loaded embedding has.

## 5. Change / trajectory caveats

Change analysis compares pixel-for-pixel, so every input year must already sit on
exactly the same grid — the same extent, CRS, resolution and pixel alignment. The
algorithm enforces this and refuses mismatched inputs rather than silently
misregistering them, which means the burden of loading each year identically (with
*Load embedding* using one area of interest, CRS and resolution) falls on the user.
If you would rather skip that manual alignment, *Change over years*
(`alphaearth:changeyears`) fetches every ticked year onto one snapped grid and
compares them in a single step, so its inputs are aligned by construction and there
is no grid check to fail; the raster-based *Change / trajectory* remains the tool
for years you have already loaded or prepared yourself.

The result is a *relative* distance surface, not a labelled "what changed" map. A
high value says the embedding moved a lot between years, but not whether that is
regrowth, clearing, flooding, a crop rotation or a sensor artefact — attributing a
cause needs domain knowledge or reference data. Because the data is annual, change
is only ever resolved year-to-year: a disturbance and full recovery inside one
calendar year can leave little annual signal, and the exact within-year timing of a
change is invisible. A pixel is treated as no-data unless *every* input year is
valid there, so the change layer is only defined where all years have coverage. The
two metrics (cosine, Euclidean) are monotonically related for unit vectors, so they
rank pixels almost identically; cosine distance is the bounded (0–2) default.

## 6. Change-report caveats

The change report summarises a change raster's *magnitude and extent*; it does not
add information the raster does not already contain, so every *Change / trajectory*
caveat above flows straight through it. A distribution, a threshold table or a
scatter shows how much and how widely the embedding moved, never *why* — attributing
a cause still needs domain knowledge or reference data. The report reads whatever
single-band raster you give it (it does not have to be an AlphaEarth change layer),
so its numbers are only as meaningful as that input.

The "percentage of area changed" reports **hectares only when the raster's CRS is
projected (in metres)**; over a geographic (degrees) CRS a pixel has no constant
ground area, so the report gives the fraction of valid pixels but omits hectares.
The over-time series is built from the rasters you supply *in order* and labels each
period with that layer's name, so a mislabelled or mis-ordered input produces a
mislabelled or mis-ordered series; it makes no attempt to parse years from filenames
or to check that the periods are contiguous. The scatter needs an embedding raster on
the *same grid* as the change raster and is omitted (with a warning) if the band or
grid does not match. HTML and CSV always work with nothing beyond NumPy, but the
**PNG and PDF outputs require matplotlib** in the QGIS Python environment: when it is
absent those two formats are skipped with a message rather than failing the run, so a
report requesting only PNG/PDF on a matplotlib-less install produces no file.

## 7. Extract-samples caveats

Sample extraction writes one output point per covered pixel, so a large polygon
over a 10 m grid can produce a very large table; a per-feature safety cap protects
the run, and hitting it means splitting the feature or coarsening the resolution.
Pixel assignment is by centre containment (with a centroid fallback for sub-pixel
features), so a polygon edge that clips a pixel without covering its centre does not
sample that pixel — expect slightly fewer rows than a naïve area estimate. Samples
are emitted in the embedding's own grid and CRS at pixel centres, not at the
original feature vertices. The band columns carry whatever the loaded embedding
holds; if you extract from a raster that was not dequantised (an explicit-URL or
raster-layer *Load* run), the values are the raw stored units, not unit-length
float vectors. With *Skip no-data pixels* on, pixels that are no-data in the
embedding are dropped, so features lying entirely over ocean or missing tiles
yield no rows.

## 8. scikit-learn tier caveats

Classification, clustering and regression are only as good as their inputs and are
easy to over-trust. A classifier or regressor is trained purely on the labelled
features you supply, so too few examples, an unbalanced set (most points in one
class), or labels that do not actually match what is on the ground will produce a
confident-looking but wrong map. The per-pixel **confidence** raster is the model's
own probability, not a validated accuracy — a high value means the model is sure,
not that it is right — so it should never be read as ground-truth reliability.
Clustering is unsupervised: cluster numbers are arbitrary labels with no inherent
meaning or order, and the "right" number of clusters is a judgement call, not a
result — the optional *Choose k automatically* switch picks a k by the best mean
silhouette over a range, but that is a heuristic to inform the judgement, not a
substitute for it. For speed the cluster fit (and any large training set) uses a
subsample, so results can shift slightly between runs. *Classify* and *Regress* now
offer an optional accuracy report — a stratified k-fold cross-validation giving
overall accuracy plus per-class precision, recall and F1 and a confusion matrix for
classification, and per-fold and overall R² and RMSE for regression, each with a
class-balance summary that warns on a heavily imbalanced training set — but it is
off unless you set a fold count and a report path, and it reflects only the labelled
data you supply (so an unrepresentative training set gives an optimistic score).
*Classify*, *Regress* and *Classify over years* also offer an opt-in *Tune
hyper-parameters* switch (a small cross-validated randomized search), but it is
off by default, searches only a modest preset grid rather than exhaustively, and
falls back to the defaults when the training set is too small; these remain
first-pass models best checked against independent reference data. A fitted
*Classify* or *Regress* model can be saved to a `.pkl` and reused on another scene
without retraining, but a saved model is tied to the 64-band embedding and the
scikit-learn version it was trained under: reusing it against a mismatched band
count or a different scikit-learn version fails loudly with a clear message rather
than producing a quietly wrong map. Finally, these four algorithms (*Classify*,
*Classify over years*, *Cluster* and *Regress*) will not run at all until
scikit-learn is installed (§1).

## 9. Reproducibility and the cloud dependency

Fetch-by-year depends on a live, anonymous connection to the public
`gs://alphaearth_foundations` bucket. If the bucket is unreachable, throttled, or
its object naming changes, fetch-by-year will fail (with retries and a timeout
rather than a hang). The tile-selection maths and listing are pure and unit-tested
offline, but the *pixels* still come over the network, so a fetch-by-year run is
only as reproducible as the remote data and your connection. For fully repeatable
work, load once and save the clipped embedding GeoTIFF, then operate on that.

A multi-year fetch (*Load embeddings (multiple years)*) inherits every one of
these remote-read caveats, once per ticked year: the years are fetched
sequentially, not in parallel, so wall-clock time scales with the number of years
(each shares the caches, so years or zones already fetched are much faster), and a
single unreadable tile degrades that one year's mosaic exactly as it would a
single-year run. It also does no cross-year gap-filling or harmonisation — each
year is fetched and de-quantised independently — so a pixel that is no-data in one
year is simply no-data in that year's raster.

There is also an inherent first-run cost, though v0.5.1 cuts it substantially.
AlphaEarth tile filenames carry no *global* grid position — the two integers in a
name are only that scene's per-tile pixel offsets, and there is no published grid
origin — so the toolbox still cannot read a tile's location from its name alone.
What it *can* do is group the zone's tiles by their scene hash, read one "anchor"
tile per scene, and from that anchor's footprint plus the scene's known pixel
offsets compute a deliberately generous over-estimate of the whole scene's extent;
scenes whose over-estimate cannot touch the area of interest are dropped without
reading their tiles, and only the surviving scenes are header-scanned tile-by-tile
for exact footprints. That turns the slow 0–90 % phase of a first fetch from
roughly one remote read per *tile* into roughly one per *scene* — commonly a few
times fewer reads — without changing which tiles are ultimately selected (the
over-estimate can only over-include, and the final choice always uses exact scanned
footprints). The cost still scales with the *zone* rather than the (possibly tiny)
area requested, and a first fetch of a new year-and-region combination still pays
for the scene anchors; eliminating it entirely would need a published tile index or
grid origin that AlphaEarth V1 does not provide. On top of this, the per-`(year,
zone)` footprint cache removes already-seen tiles on repeat runs, and a listing +
anchor cache lets a later run over a *different* AOI in the same zone skip
re-listing and re-cull its scenes with no new reads. By default these caches live
under the OS temp directory, so they are typically discarded between sessions and a
fresh QGIS session pays the discovery cost again; an opt-in advanced option
("Remember the tile index across sessions") instead persists them under the QGIS
profile directory so a later session over the same zone reuses that work. A single tile whose header
cannot be read is skipped rather than failing the run (see Methodology §3), and the
count of such skipped tiles is now reported as a warning, so a flaky tile degrades
completeness slightly (and visibly) rather than aborting the fetch.

## 10. Testing maturity

The pure tiers are well covered — as of v0.9.0, 266 unit tests exercise the NumPy
similarity, change, sampling, RGB and ML cores (including the stratified
cross-validation, confusion-matrix and class-balance maths, the post-classification
transition/changed-mask/label helpers, and the randomized-search and
parameter-grid primitives that back the opt-in tuning), the change-report core
(summary statistics, histogram, area-fractions with hectares, over-time series,
scatter sub-sampling and the standalone HTML/CSV rendering — all with no matplotlib
present), the pure model-save/
load round-trip and its version/band-count validation, the preset registries, the
scikit-learn detection plumbing (with an injected subprocess runner, so nothing is
actually installed in tests), the standard-library tile-discovery helpers (with an
injected fetch, so no network) — including scene grouping, the conservative extent
over-estimate, the listing/anchor and footprint caches added in v0.5.1, the
persistent-cache directory logic, and the v0.6.0 multi-year helpers (the
tick-box-index → year mapping with sort/dedup and out-of-range rejection, and the
per-year output naming) — and the source-resolution policy. The GDAL layer now has round-trip and rasterio-parity
tests (`tests/test_raster_gdal.py`) for `aecore._raster`; these skip automatically
where GDAL is absent, so they run in the QGIS/GDAL container job (which now covers
both the 3.34 LTR image and a moving Qt6/QGIS 4 image) but not in the pure "checks"
job.

Coverage of the QGIS-backed paths has broadened. `tests/test_algorithms_qgis.py`
now registers the provider in a headless QGIS and runs each offline algorithm (RGB,
similarity, change, extract, and classify where scikit-learn is present) end-to-end
through `processing.run` on synthetic inputs, asserting the provider exposes all twelve
algorithm ids; it skips where `qgis.core` is absent and so runs for real in the
QGIS/GDAL container. Even so, the cloud fetch-by-year path (which needs live network
access, and so is not exercised for *Classify over years* either), the wizard GUI and
the guided scikit-learn install are still only compile-checked and covered by design,
not by an automated run, and none of the twelve
`processAlgorithm` bodies has yet been exercised inside a full, interactive QGIS. A
real-QGIS smoke test (installing the ZIP into QGIS 3.34 LTR and a Qt6/QGIS 4.x build
and working through fetch-by-year, RGB, similarity, change, extract and the ML tier
— see `SMOKE_TEST.md`) is the outstanding pre-release step. Until it is signed off, treat
the QGIS-backed behaviour as "expected to work" rather than "verified", which is why
the plugin stays flagged experimental.

## 11. Not a substitute for field validation

Finally, the toolbox produces analytical layers, not ground truth. Similarity maps,
change surfaces and classifications are hypotheses to be checked against reference
data and local knowledge, especially for decisions with real consequences. The
plugin's own metadata says as much: "not a substitute for field validation."
