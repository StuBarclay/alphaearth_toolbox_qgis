# User Guide

*AlphaEarth Toolbox for QGIS — v0.10.0*

This guide walks through installing the plugin and running its twelve algorithms.
No Earth Engine account, API key or GPU is required, and the core workflow needs no
extra Python package — NumPy and GDAL ship with QGIS, and the embedding data is
read anonymously from a public cloud bucket. The optional machine-learning tier
adds scikit-learn, which the plugin can install for you with one click.

## 1. What the toolbox does

AlphaEarth Satellite Embedding V1 is a global, 10-metre, annual (2017–2025),
64-band learned "fingerprint" of the Earth's surface, published free under CC BY
4.0. This toolbox brings it into QGIS as a Processing provider called **AlphaEarth
Toolbox** with twelve algorithms.

Eight need only NumPy and GDAL:

*Load embedding* (`alphaearth:loadembedding`) fetches the 64-band embedding for an
area of interest — you can give it just a year and it finds the data for you.

*Load embeddings (multiple years)* (`alphaearth:loadyears`) does the same for
several years at once, chosen with tick-boxes, over the same area — one aligned
raster per year, ready for Change / trajectory or Similarity.

*Embedding → RGB* (`alphaearth:torgb`) paints the 64 bands to a 3-band image so you
can actually see the embedding, by PCA or a chosen band triplet.

*Similarity search* (`alphaearth:similarity`) takes example locations you mark and
scores every pixel by how similar it is — a "find more like this" map.

*Change / trajectory* (`alphaearth:change`) compares two or more years of
embeddings and maps how much each pixel changed — the standout multi-year
capability.

*Change over years* (`alphaearth:changeyears`) folds the two steps above into one:
tick the years, set a single area, and it fetches each year onto one aligned grid
and produces the change map directly — no separate Load step needed.

*Extract training samples* (`alphaearth:extract`) reads the embedding vector at
your marked features into a point table, ready for external machine learning or QA.

*Change report* (`alphaearth:changereport`) turns a change raster into a small
report with charts — a change-magnitude distribution and summary statistics, the
percentage of area changed at each threshold, an optional over-time series and an
optional change-vs-band scatter — as any mix of HTML, PNG, CSV and PDF.

Four more form an optional scikit-learn tier — *Classify embedding*
(`alphaearth:classify`), *Classify over years* (`alphaearth:classifyyears`),
*Cluster embedding* (`alphaearth:cluster`) and *Regress embedding*
(`alphaearth:regress`) — covered in §10.

All run from the Processing Toolbox, from the plugin's guided wizard, in batch, and
in the Graphical Modeler.

## 2. Installing

Build the plugin ZIP from the repository root:

```bash
python scripts/build_plugin_zip.py
```

This writes a clean, reproducible `alphaearth_toolbox.zip`. Then in QGIS go to
**Plugins → Manage and Install Plugins → Install from ZIP**, choose that file, and
enable *AlphaEarth Toolbox*. QGIS 3.34 LTR or newer is required (both Qt5/PyQt5 and
Qt6/PyQt6 builds are supported). After installing you will find an **AlphaEarth
Toolbox** group in the Processing Toolbox and an entry under the **Plugins** menu.

Because it is marked experimental, you may need to tick *Show also experimental
plugins* in the Plugin Manager settings.

## 3. Load embedding

Open it from **Processing Toolbox → AlphaEarth Toolbox → Load embedding**, or from
the plugin dialog. It clips and warps a 64-band AlphaEarth source to your area of
interest, CRS and resolution, and there are three ways to tell it where the data
comes from.

The parameters are: an optional **Embedding raster** (a layer already in your
project), an optional **Embedding COG URL** (an `http(s)://` or `/vsicurl/…` path),
a **Embedding year to fetch from public GCS** (2017–2025; leave at 0 if using a
layer or URL), the **Area of interest** extent, the **Target CRS** (default
EPSG:3577), the **Output resolution** in target-CRS units (default 10, matching the
native resolution), and the **Embedding (clipped)** output.

When more than one source is provided, precedence is: a chosen raster layer wins,
then an explicit URL, then the year.

### The simplest workflow: fetch by year

Set the **Area of interest** (draw it on the map, use a layer's extent, or type
bounds), choose a **year**, and leave the raster and URL empty. The toolbox then
does everything for you: it works out which UTM-zone folders your area touches,
lists the tiles in the public bucket, keeps only the ones that actually overlap
your area, mosaics them — even across UTM zones — and dequantises the result back
to proper unit-length embedding vectors. Progress runs to about 90 % during the
tile scan and completes on the mosaic; you can Cancel at any point.

Because GDAL reads only the window it needs, a small area is pulled from large
remote tiles without downloading whole scenes, and remote reads retry with
back-off and time out on a stall rather than hanging.

The first fetch over a given year and area spends most of its time in that 0–90 %
tile scan, working out which tiles cover you — but it is smart about it. Rather than
reading every candidate tile, it groups the tiles by their source scene, reads just
one tile per scene, and from that quickly rules out whole scenes that cannot touch
your area before reading only the tiles of the scenes that survive. That commonly
cuts the number of remote reads on a first fetch several-fold, without changing
which tiles end up in the mosaic. The toolbox also remembers each tile's footprint
(and each zone's scene index) on disk, so a **second run over the same year and
zones is much faster** — the scan is skipped for tiles it has already seen and jumps
almost straight to the mosaic, and even a run over a *different* area in a zone it
has already visited reuses the cached index instead of re-listing. That tile index
normally lives in a temporary folder that QGIS may clear between sessions; if you
work the same region day to day, tick **Remember the tile index across sessions**
(an advanced option, also on the wizard) to keep it under your QGIS profile instead,
so a fresh session over the same zone still skips the scan. If a handful of tiles
cannot be read they are skipped and reported in a warning (the mosaic may have small
gaps there) rather than failing the run. If you ask for a very large area, it
warns you up front that the read may be slow or run short of memory and suggests a
smaller extent or coarser resolution; the run still proceeds if you want it to. When
your output file ends in `.tif` or `.tiff` the result is written straight there in
one pass.

### Using an explicit URL or an existing layer

If you already have a specific tile URL, paste it into **Embedding COG URL** (a
plain `http(s)://` link is fine — it is wrapped as `/vsicurl` automatically). If
you have already loaded an embedding raster into your project, pick it under
**Embedding raster**. In both cases the source is warped to your grid verbatim,
without the year-mode tile discovery or dequantisation, so use these when you want
one specific tile or a scene you have already prepared.

### Tips

Keep the resolution at 10 (the native ground sample distance) unless you have a
reason to coarsen it. Nearest-neighbour resampling is always used so the learned
vectors are preserved unchanged. If you get "No AlphaEarth tiles intersect the area
of interest", double-check that your extent's CRS is set correctly — an extent in
the wrong CRS is the usual cause.

## 4. Load embeddings (multiple years)

Open it from **Processing Toolbox → AlphaEarth Toolbox → Load embeddings
(multiple years)**, or use the year tick-list and **Load selected years** button on
the wizard's *Load / Visualise* tab. It fetches several AlphaEarth years for the
*same* area of interest in one run and writes one 64-band GeoTIFF per year.

The parameters are: **Years to fetch from public GCS** (tick one or more of
2017–2025; the most recent year is ticked by default, so a stray run does not
pull all nine), the **Area of interest** extent, the **Target CRS** (default
EPSG:3577), the **Output resolution** in target-CRS units (default 10), and an
**Output folder** to hold the per-year rasters.

Every year is clipped and warped onto the *same* snapped grid, so the outputs are
pixel-aligned — which means they drop straight into *Change / trajectory* (§7)
and *Similarity* (§6) with no re-projection or resampling step in between. Each
year is fetched exactly the way single-year *Load embedding* does it (the same
public-bucket tile discovery, scene-grouped culling and de-quantisation, and the
same per-year on-disk caches), so a run over years or zones you have fetched before
is fast. The grid is sized once and the large-area heads-up, if any, is given once;
the progress bar is split evenly across the ticked years, and you can Cancel between
them.

Files are named `alphaearth_<year>.tif` inside the output folder (for example
`alphaearth_2019.tif`, `alphaearth_2024.tif`). Run from the wizard, each fetched
raster is added to your project automatically; run headless, every path is returned
under a `YEAR_<year>` result key alongside the `OUTPUT` folder.

This is the quickest way to assemble the multi-year stack that *Change / trajectory*
needs: tick the years, pick a folder, run once, then point Change at the resulting
rasters.

## 5. Embedding → RGB

Open it from **Processing Toolbox → AlphaEarth Toolbox → Embedding to RGB**, or use
the **Preview RGB** button on the wizard's *Load / Visualise* tab. It turns the
64-band embedding into an ordinary 3-band colour image so you can see structure at
a glance — the embedding itself is not a picture, so this is how you look at it.

The parameters are: the **Embedding raster** (64-band), the **Method** (*PCA* or
*Band triplet*), the **Bands** for the triplet method (three comma-separated,
0-based indices, e.g. `10,20,30`; ignored for PCA), and the **RGB preview** output.

*PCA* (the default) finds the three directions of greatest variation across the
embedding and maps them to red, green and blue, so it shows as much of the
embedding's structure as three channels can. Its colours are decided by the data in
*this* run, so they are vivid and informative but not comparable between separate
rasters. *Band triplet* just shows three bands you name; the colours mean the same
thing from one raster to the next, but only three of the 64 dimensions are visible.
Either way each channel is contrast-stretched (2nd–98th percentile) so the image is
not washed out. Use the result for display only — feed the original 64-band
embedding, not the RGB, into the analytical algorithms.

## 6. Similarity search

Open it from **Processing Toolbox → AlphaEarth Toolbox → Similarity search**. It
answers "where else looks like here?" by scoring every pixel's cosine similarity to
one or more example ("seed") locations.

First produce an embedding raster with *Load embedding*. Then digitise your
examples: create or pick a **Seed features** layer of points or polygons over
places that represent what you are looking for (a few burnt-vegetation points, some
built-up polygons, a stand of a particular land cover). Point seeds mark the pixel
they land in; polygon seeds mark every pixel whose centre falls inside them.

The parameters are: the **Embedding raster** (64-band), the **Seed features**,
**Combine seed pixels by** (*Mean of seeds*, the default, or *Medoid* — the single
most representative seed), **Rescale similarity to 0–1** (on by default; turn it off
for raw cosine in −1…1), a **Mask threshold** (0–1; leave at 0 for no mask), the
**Similarity** output raster, and an optional **More-like-this mask** output.

Run it and you get a similarity raster, automatically styled blue (low) to red
(high). If you set a threshold above 0 *and* choose a mask output, you also get a
binary layer flagging every pixel at or above that threshold — your "more like
this" selection. A good way to work is to run once with threshold 0, inspect the
similarity map to see where a natural break falls, then re-run with a threshold at
that break to get a clean mask.

Seeds are reprojected into the embedding's CRS automatically, so they do not need
to be in the same CRS as the raster — but they must actually overlap it. A very
large polygon seed is capped for safety; if you hit that limit, use a smaller seed
or a coarser resolution.

## 7. Change / trajectory

Open it from **Processing Toolbox → AlphaEarth Toolbox → Change / trajectory**. It
answers "how much, and where, did the surface change between years?" by measuring
the distance between embedding years at every pixel.

First produce one embedding raster **per year** with *Load embedding*, using the
**same area of interest, CRS and resolution each time** so the grids line up
exactly — the algorithm compares pixel-for-pixel and will refuse mismatched grids.
Then select those rasters as the input, **ordered oldest to newest**.

The parameters are: the **Embedding rasters** (two or more, in temporal order), the
**Distance metric** (*Cosine distance*, the default, = 1 − cosine similarity; or
*Euclidean distance*), the **Change mode**, and the **Change** output raster.

There are three modes. *Pairwise (first vs last)* is the classic bi-temporal change
map — the distance between your first and last year. *Trajectory magnitude* adds up
the distance across every consecutive pair, so a pixel that keeps changing
year-on-year scores higher than one that changed once and settled (supply three or
more years to see the difference). *Anomaly vs baseline* measures how far the most
recent year sits from the average of all the years you supplied — useful for
flagging where the latest year departs from the multi-year norm.

The output is a single-band raster, automatically styled blue (little change) to
red (most change); higher values always mean more change. Any pixel that is
no-data in even one input year is no-data in the result. Because AlphaEarth pixels
are unit vectors the two metrics are closely related, but cosine distance is the
natural default and is bounded in 0–2.

### Change over years — fetch and compare in one step

If you have not already loaded the years, *Change over years*
(`alphaearth:changeyears`) does the whole job at once. Open it from **Processing
Toolbox → AlphaEarth Toolbox → Change over years**, or use the one-step control on
the wizard's *Change* tab. It fetches each year you tick onto a single shared grid
and produces the change raster directly, so you never have to load and align the
years by hand — and because every year is fetched onto the same grid there is no
alignment step that can fail.

The parameters are: **Years to compare** (tick two or more of 2017–2025, in
increasing order), the **Area of interest** extent, the **Target CRS** (default
EPSG:3577), the **Output resolution** in target-CRS units (default 10), the
**Distance metric** (*Cosine distance* or *Euclidean distance*), the **Change mode**
(*Pairwise (first vs last)*, *Trajectory magnitude*, or *Anomaly vs baseline* — the
same three modes as *Change / trajectory*), the **Change** output raster, and an
optional **Output folder** in which to also save the fetched per-year rasters
(`alphaearth_<year>.tif`) if you want to keep them.

Each year is fetched exactly the way *Load embedding* does it (same public-bucket
tile discovery, scene-grouped culling, caches and de-quantisation), so a run over
years and zones you have fetched before is fast, and the change maths is identical
to *Change / trajectory*. Use *Change over years* when you just want the change map
for a set of years; use *Change / trajectory* when you have already prepared the
per-year rasters yourself and want to compare those.

## 8. Change report

Open it from **Processing Toolbox → AlphaEarth Toolbox → Change report**. It turns a
change raster into a small report you can read at a glance — with charts — instead of
just a coloured surface. Give it the change raster from *Change / trajectory* or
*Change over years* (or any single-band magnitude raster) and choose which contents
you want and which file formats to write.

The parameters are: the **Change raster to summarise** and the **Band to analyse**
(1-based, default 1); an optional **Change rasters over time** input (several change
rasters, oldest first, for the over-time content); an optional **Embedding raster**
and a **Scatter band** for the scatter; the four content tick-boxes; and the four
output paths — **HTML report**, **PNG chart**, **CSV data** and **PDF report**. Set
one or more output paths (at least one is required) and tick the contents you want.

The four contents are independent. The **change-magnitude distribution** gives
summary statistics — count, minimum and maximum, mean, median, standard deviation and
percentiles — and a histogram. The **percentage of area changed** reports, for each
of several thresholds, the share of the scene that changed by at least that much
(with hectares too when the raster is in a metric CRS). The **change over time**
series plots one point per period from the *Change rasters over time* you supply, in
order. The **scatter of change vs an embedding band** plots the change magnitude
against a chosen band of the optional embedding raster (it is omitted, with a warning,
if that raster's grid or band does not match).

Any combination of the four formats can be written at once. **HTML** (a standalone
page with inline charts) and **CSV** (the underlying numbers) need nothing beyond what
QGIS already has. **PNG** and **PDF** are drawn with **matplotlib**: if it is not
installed in the QGIS Python environment those two are skipped with a message (the run
still succeeds and still writes the HTML/CSV), so request PNG or PDF only once
matplotlib is present — you can add it the same way as scikit-learn, with
`pip install matplotlib` into the QGIS Python environment.

The same report is also available without a separate step: on both *Change /
trajectory* and *Change over years* the report outputs appear as **advanced**
parameters, so setting one or more report paths there writes the report alongside the
change raster (its over-time series taken from the consecutive-year pairs, its scatter
against the most recent year). The wizard's *Change* tab offers the same content
tick-boxes and format pickers, applied to whichever change run you launch.

## 9. Extract training samples

Open it from **Processing Toolbox → AlphaEarth Toolbox → Extract training
samples**. It turns your marked features into a table of embedding vectors — the
raw material for training a classifier in scikit-learn, R or elsewhere, or for
checking what a class looks like in embedding space.

Provide an **Embedding raster** (from *Load embedding*) and a **Sample features**
layer of points or polygons — typically labelled with a class field. Each pixel
covered by a feature becomes one output row: a point at the pixel centre carrying
**all of the source feature's attributes** (so your class labels and ids come
along) plus the 64 embedding band values in columns `A00, A01, … A63`. Points
sample the pixel they fall in; polygons sample every pixel whose centre lies inside
them (with a centroid fallback for features smaller than a pixel).

The **Skip no-data pixels** option (on by default) drops pixels that are no-data in
the embedding, so you do not get empty rows. The output CRS matches the embedding
raster; to use the samples outside QGIS, right-click the result layer and **Export
→ Save Features As… → CSV** (tick *Add geometry columns* if you want the
coordinates).

A very large sample polygon is capped for safety; if you hit that limit, split the
feature or coarsen the resolution.

## 10. The scikit-learn tier: classify, classify over years, cluster and regress

Four algorithms add machine learning over the embedding. They need **scikit-learn**,
which QGIS does not ship. You do not have to install it by hand: open the wizard's
**Classify** tab and click **Install scikit-learn** — it installs into the QGIS
Python environment on a background task and the status updates to "available" when
done (it may take a minute, and only needs doing once). If scikit-learn is missing
when you run one of these algorithms, it stops with a clear message rather than a
crash.

*Classify embedding* trains a classifier from your labelled examples and maps land
cover everywhere. Give it the **Embedding raster**, a **Training features** layer of
points/polygons, the **Class field** holding the label, and a **Model**
(*Random forest* or *Gradient boosting*). Tick **Also output a confidence raster**
to get a second layer holding the model's certainty (0–1) at each pixel. The class
raster is styled with a categorical palette; confidence is styled red (unsure) to
green (sure).

*Classify over years* is the classification analogue of *Change over years*: it
folds the multi-year fetch and *Classify embedding* into one step, so a classify run
no longer needs its years pre-loaded and aligned by hand. Tick **two or more years**
on the Load tab (or supply them to the algorithm), set one **area of interest**,
**CRS** and **resolution**, and give it labelled **Training features** with a
**Class field**; it fetches every year onto the same grid and maps land cover for
each. To keep the class codes comparable from year to year it trains **one model on
the most recent year** (or reuses one you load from a saved file) and applies it
across the whole series, so a "2" means the same class in every year. The result is
an **output folder** holding one `classification_<year>.tif` per year. Two optional
extras summarise the change: a **transition raster** encoding the first-to-last
move as a single "<from> -> <to>" class per pixel, and a **changed / unchanged
mask** flagging where the class differs between the first and last year. Both are
auto-styled, and the transition legend spells out each pair as "<from> -> <to>".

*Cluster embedding* needs no training data: give it the **Embedding raster** and a
**Number of clusters**, and it groups pixels into that many unsupervised classes
(K-means) — a quick way to see natural groupings. The output is a paletted raster;
cluster numbers are labels, not an order.

*Regress embedding* predicts a continuous value. Give it the **Embedding raster**, a
**Training features** layer with a **numeric** target field, and a **Model**; it
returns a pseudocolour prediction surface. A non-numeric field is rejected with a
clear message.

*Classify* and *Regress* can save the trained model and reuse it later. Set a
**Save model to** path and the fitted estimator is written to a `.pkl` alongside the
raster. On another scene, set **Load model from** to that file instead of supplying
training features: the saved model is applied directly, with no retraining, so you
do not need the training layer or the label/target field at all. The saved file
records the scikit-learn version and the 64-band embedding count it was trained on,
and reusing it against a mismatched band count or a different scikit-learn version
stops with a clear error rather than producing a quietly wrong map — so retrain (or
match the environment) if you see that message.

Both also offer an optional accuracy check. Set **Cross-validation folds** to a
number (say 5) and an **Accuracy report** path, and the algorithm runs a stratified
k-fold cross-validation before mapping. For *Classify* the report gives overall
accuracy, per-class precision, recall and F1, and a confusion matrix; for *Regress*
it gives per-fold and overall R² and RMSE. Each report also summarises the class
balance and warns if your training set is heavily skewed toward one class. Leave the
folds at 0 to skip the check. Remember the score reflects only the labelled data you
supplied, so treat it as a sanity check, not a substitute for independent
validation.

The estimators run with sensible fixed defaults, but you can let them **tune their
hyper-parameters** instead. Tick **Tune hyper-parameters** on *Classify*, *Regress*
or *Classify over years* and the algorithm runs a cross-validated randomized search
over a small preset space for the chosen model before mapping, and (when an accuracy
report is requested) records the winning settings in the report. *Cluster embedding*
has the matching option **Choose k automatically**: give it a small range and it
picks the number of clusters with the best silhouette score instead of a fixed k,
logging the value it chose. Tuning reuses the same fold machinery as the accuracy
check, and both options are opt-in and default-off — if the training set is too
small to search safely the run falls back to the defaults with a warning rather than
failing.

All four run over every valid pixel in bounded chunks, so they cope with large
scenes, and they skip no-data pixels (ocean, missing tiles) rather than inventing a
prediction there.

## 11. Running headless, in batch, or in the Modeler

All twelve algorithms are standard Processing algorithms, so anything Processing can
do applies. From the Python console:

```python
import processing

processing.run(
    "alphaearth:loadembedding",
    {
        "YEAR": 2024,
        "EXTENT": "151.0,151.1,-33.9,-33.8 [EPSG:4326]",
        "TARGET_CRS": "EPSG:3577",
        "RESOLUTION": 10.0,
        "OUTPUT": "/path/to/embedding_2024.tif",
    },
)
```

To fetch several years over one area in a single call, use `alphaearth:loadyears`
— `YEARS` is a list of tick-box indices into 2017–2025 (0 = 2017 … 8 = 2025) and
`OUTPUT` is a folder; each year is written as `alphaearth_<year>.tif` inside it:

```python
processing.run(
    "alphaearth:loadyears",
    {
        "YEARS": [1, 7],  # 2018 and 2024
        "EXTENT": "151.0,151.1,-33.9,-33.8 [EPSG:4326]",
        "TARGET_CRS": "EPSG:3577",
        "RESOLUTION": 10.0,
        "OUTPUT": "/path/to/years_folder",
    },
)
```

To report on a change raster, use `alphaearth:changereport`. `REPORT_CONTENT` is a
list of tick-box indices (0 = distribution, 1 = area changed, 2 = over time,
3 = scatter), and each output format is its own optional path — set the ones you
want:

```python
processing.run(
    "alphaearth:changereport",
    {
        "INPUT": "/path/to/change_2018_2024.tif",
        "BAND": 1,
        "REPORT_CONTENT": [0, 1],  # distribution + area changed
        "REPORT_HTML": "/path/to/change_report.html",
        "REPORT_CSV": "/path/to/change_report.csv",
    },
)
```

Use the batch interface (right-click an algorithm → *Execute as Batch Process*) to
run many areas or years at once, and chain the algorithms in the Graphical Modeler
— for example *Load embeddings (multiple years)* → Change into a repeatable
multi-year change model, or Load → Extract to build a training table for a study
area.

## 12. The tabbed wizard

**Plugins → AlphaEarth Toolbox** (or the toolbar button) opens a guided wizard with
four tabs — **Load / Visualise**, **Classify**, **Similarity** and **Change** — for
the common workflows without touching the Processing dialogs. Draw an area of
interest straight on the map, pick a year, choose a target CRS, and pick input
layers and fields from drop-downs; the *Load / Visualise* tab also has a **Preview
RGB** button and, below the single-year Load, a **tick-list of years** with a
**Load selected years** button that fetches each ticked year over the same area and
adds every resulting raster to the project. On the tabs where they apply — RGB,
Similarity, clustering and Change — a **Preset** drop-down offers named starting
points (for example a PCA RGB or a cosine-distance change) that fill in sensible
choices for you to adjust; a preset is a good default, not a fixed recipe, because
the embedding has no fixed band meanings. The *Change* tab additionally lets you
**tick the years and compare in one step**: choose the years, the mode and the
metric, and it fetches and compares them for you (the *Change over years* algorithm
of §7) without a separate Load. The same tab carries the **change-report controls** —
content tick-boxes and HTML / PNG / CSV / PDF output pickers (see §8) — which are
applied to whichever change run you launch, so you can produce the change raster and
its report in one go. The *Classify* tab has the **Install scikit-learn**
button and the fields to **save or load a model** and to request a
**cross-validation accuracy report**, an opt-in **Tune hyper-parameters** switch
(with an iterations spin-box) that runs the same cross-validated randomized search
as the Processing algorithm (§10) and applies to both runs on the tab, and — like
the *Change* tab — a one-step **Fetch ticked years and classify** mode: tick two or
more years on the *Load / Visualise* tab, set the training layer and label field
once, and it fetches every year onto one shared grid and maps land cover for each
(the *Classify over years* algorithm of §10), optionally also writing a
first-to-last transition raster and a changed/unchanged mask from its two extra
tick-boxes. An advanced **Remember the tile index across
sessions** tick-box keeps the fetch cache under your QGIS profile so a later session
over the same zone stays fast.

Every run happens on a background task with a progress bar and a **Cancel** button,
so a long cloud fetch or a model fit never freezes QGIS, and results are added to
your project automatically when the run finishes. Anything the wizard does you can
also do (with every parameter exposed) from the Processing Toolbox.

## 13. Attribution (required)

The embedding data is licensed CC BY 4.0, which requires attribution. Each
algorithm's help text ends with the required credit line:

> Contains modified Google DeepMind AlphaEarth Satellite Embedding V1 data
> (CC BY 4.0).

Include that wherever you publish maps or figures derived from the toolbox's
outputs. AlphaEarth is commercially usable with attribution.

## 14. Troubleshooting

If the toolbox does not appear, confirm the plugin is enabled and that
experimental plugins are shown, and that QGIS is 3.34 or newer (the plugin refuses
to load on Python older than 3.10 with an explanatory message). "No AlphaEarth
tiles intersect the area of interest" almost always means the extent's CRS is
wrong or the area is over ocean — check the extent. A *fetch-by-year* run that sits
at a low percentage for a while on its **first** use of a given year and region is
normal, not a hang: to find the tiles covering your area the toolbox header-scans
tiles in each UTM zone it touches, and that is the 0–90 % phase. It keeps this as
short as it can — tiles are grouped by scene, one tile per scene is read, and whole
scenes that cannot reach your area are ruled out before their tiles are read — but a
first fetch still has to anchor every scene in the zone (AlphaEarth tile names carry
no global position, so there is no way to skip them all). A second run over the same
year and zones reuses a cached tile index and jumps almost straight to the mosaic,
so the wait is a once-per-region cost. If some tiles cannot be read, the run
finishes and warns you how many were skipped (the mosaic may have small gaps there)
rather than failing. Any fetch retries and times out rather than hanging outright,
and can be cancelled at any point. If a raster layer
you pass to Similarity cannot be read, load the embedding with *Load embedding*
first (which normalises the source) and feed that output in.

If a run stops with **"Task failed: Executing '&lt;algorithm&gt;'"** and no obvious
reason, open **View ▸ Panels ▸ Log Messages** and select the **Processing** tab:
*Load embedding* now writes the full underlying error (with its traceback) there
before failing, so the real cause — for example a network or GDAL problem — is
recorded even though the dialog only shows the background task's generic message.
Include that log text if you report a bug.

If *Change* reports that the inputs are not aligned or have different shapes, the
years were not loaded onto the same grid — re-run *Load embedding* for each year
with an identical area of interest, target CRS and resolution, then compare. If
*Extract* produces no rows, the features either fall outside the embedding extent
or (with *Skip no-data pixels* on) land only on no-data pixels — check the overlap
and CRS.

If *Classify*, *Cluster* or *Regress* reports that scikit-learn is missing, install
it with the **Install scikit-learn** button on the wizard's *Classify* tab (or
`pip install scikit-learn` into the QGIS Python environment) and re-run. If *Regress*
refuses a field, it needs a **numeric** target — pick a number column. If *Classify*
produces a single class everywhere, your training features are probably all one
label or too few; add more, well-spread examples per class.
