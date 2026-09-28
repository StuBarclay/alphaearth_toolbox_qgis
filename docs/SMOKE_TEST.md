# Smoke-test checklist (real QGIS)

A manual pass to run before publishing a release. It cannot be automated in CI
(no interactive QGIS there), so run it by hand in **QGIS 3.34 LTR** and, if
convenient, a **QGIS 4 / Qt6** build. Tick each box; note anything unexpected.

The offline algorithms are also exercised end-to-end by
`tests/test_algorithms_qgis.py` in the QGIS CI container, but this checklist
covers the interactive wizard, styling and the network-backed fetches, which CI
cannot.

Keep the area of interest tiny (a few km across) so fetches and fits are quick.

## 1. Install and load

- [ ] Install the ZIP: *Plugins ▸ Manage and Install Plugins ▸ Install from ZIP*,
      choose `alphaearth_toolbox.zip`, enable it. (Tick *Show also experimental
      plugins* if it is hidden.)
- [ ] No errors on load; an **AlphaEarth Toolbox** entry appears in the *Plugins*
      menu and a toolbar icon is present.
- [ ] The Processing Toolbox shows an **AlphaEarth Toolbox** provider with twelve
      algorithms: Load embedding, Load embeddings (multiple years), Embedding →
      RGB, Similarity search, Change / trajectory, Change over years (fetch +
      compare), Change report (chart / summary), Extract training samples, Classify
      embedding, Classify over years (fetch + classify), Cluster embedding, Regress
      embedding.
- [ ] *Help ▸ About* / plugin metadata shows version **0.10.0**.

## 2. Load and visualise

- [ ] Open the toolbar dialog. It is a **tabbed wizard** (Load / Visualise,
      Classify, Similarity, Change), not a launcher list.
- [ ] On **Load / Visualise**: pick a recent year, draw an extent on the canvas,
      leave the target CRS at EPSG:3577, click **Load embedding**. A progress bar
      and **Cancel** appear; the GUI stays responsive.
- [ ] A 64-band embedding raster is added to the project when it finishes.
- [ ] Click **Cancel** on a fresh load partway through — the run stops and the
      dialog returns to idle without crashing.
- [ ] Tick **two or more years** and click **Load selected years** — one
      `alphaearth_<year>.tif` per year is written to the chosen folder and added
      to the project, all on the same grid.
- [ ] With the embedding selected as *Embedding to preview*, pick an **RGB preset**
      (or leave the default), click **Preview RGB** (PCA). A 3-band RGB raster is
      added and looks sensible.
- [ ] Switch the method to **Band triplet**, set bands e.g. `10,20,30`, preview
      again — the band field is only editable for the triplet method.

## 3. Similarity and change

- [ ] Digitise 2–3 seed points over one land-cover type, pick a **similarity
      preset** if desired, run the **Similarity** tab. A 0–1 similarity raster is
      added; high values sit where expected.
- [ ] Tick *Also output a threshold mask* with a threshold (e.g. 0.7) — a mask
      raster is also produced.
- [ ] Load a second year for the same extent, then on the **Change** tab select
      both rasters (oldest→newest) and run. A change raster is added and
      auto-styled.
- [ ] On the **Change** tab, tick two or more years on the Load tab and click
      **Fetch ticked years and compare** — a single run fetches every year and
      produces the auto-styled change raster directly (no manual re-selection).
- [ ] With the change-report content boxes ticked and an **HTML** and a **CSV**
      output path set on the Change tab, re-run a change — the change raster is
      produced *and* an HTML report (with charts) and a CSV are written; open the
      HTML and confirm the distribution, threshold table and any over-time/scatter
      panels render.
- [ ] Run **Change report** on its own from the Processing Toolbox against an
      existing change raster, requesting **PNG** and/or **PDF**. If matplotlib is
      present those files are written; if it is absent the run still succeeds and
      logs that PNG/PDF were skipped (HTML/CSV, if requested, are still written).

## 4. scikit-learn tier

- [ ] On the **Classify** tab, if scikit-learn is absent the status reads "not
      installed" and **Run classification** refuses with a clear message.
- [ ] Click **Install scikit-learn**. It installs on a background task; the
      status updates to "available" when done (may take a minute).
- [ ] Digitise a few labelled polygons (a text/integer class field), run
      **Classify** with *Also output a confidence raster* ticked. A paletted
      class raster and a red→green confidence raster are added.
- [ ] Set a **Save model** path and a small **accuracy folds** value (e.g. 3) with
      a **report** path, and re-run — a `.pkl` model and a text report (overall
      accuracy, per-class precision/recall/F1, confusion matrix, class balance)
      are written.
- [ ] Set only a **Load model** path (leave training empty) on a *different*
      scene and run — it classifies from the saved model without retraining. Point
      it at a model saved from a raster with a different band count or a different
      scikit-learn version and confirm it **fails with a clear mismatch message**.
- [ ] Run **Classify** (or **Regress**) from the Processing Toolbox with **Tune
      hyper-parameters** ticked and an accuracy-folds value and report path set — the
      report includes the chosen hyper-parameters and the run still produces the
      class/prediction raster. With a very small training set, the run still succeeds
      and logs that tuning was **skipped** (fell back to the defaults) rather than
      failing.
- [ ] On the wizard's **Classify** tab, tick **Tune hyper-parameters**, set the
      tuning-iterations spinbox (it enables only when the box is ticked) and an
      accuracy-folds value + report path, then **Run classification** — the same tuned
      report is written, confirming the wizard drives `TUNE`/`TUNE_ITERS` (parity with
      the Processing algorithm).
- [ ] Run **Classify over years (fetch + classify)** from the Processing Toolbox:
      tick two or more years, set one AOI/CRS/resolution, supply labelled training and
      an output **folder** — one `classification_<year>.tif` per year is written to
      the folder. Set the optional **transition** and **changed/unchanged mask**
      outputs — both are produced and auto-styled (the transition legend reads
      "<from> -> <to>").
- [ ] On the wizard's **Classify** tab, tick two or more years on the Load /
      Visualise tab, supply training features and a label field, optionally tick
      **Also output a first-to-last transition raster** / **changed/unchanged mask**,
      then click **Fetch ticked years and classify** — a background run fetches every
      year and adds one per-year class raster (plus the ticked transition/mask) to the
      project, auto-styled. With training left empty and only a **Reuse model** path
      set, it classifies every year from the saved model without retraining.
- [ ] Run **Cluster embedding** from the Processing Toolbox (e.g. 6 clusters, or a
      cluster preset) — a paletted cluster raster is produced. Tick **Choose k
      automatically** over a small range and re-run — it picks a k by silhouette and
      logs the chosen value.
- [ ] Run **Regress embedding** with a numeric field — a pseudocolour prediction
      raster is produced; a non-numeric field fails with a clear message. With an
      accuracy-folds value and report path set, per-fold and overall R²/RMSE are
      written.

## 5. Robustness

- [ ] Run an algorithm with no input selected — it fails gracefully with a
      readable message, not a Python traceback dialog.
- [ ] Nothing writes into the plugin folder; outputs are temporary layers or the
      paths you chose. With **Remember the tile index across sessions** ticked
      (advanced), the tile cache is written under your QGIS profile and a repeat
      fetch over the same zone in a *new* QGIS session is noticeably faster.
- [ ] Repeat the key steps (load, RGB, classify) in a **QGIS 4 / Qt6** build if
      available — no enum/import errors, widgets render, runs complete.

## Sign-off

- [ ] QGIS 3.34 LTR: pass
- [ ] QGIS 4 / Qt6: pass (or noted issues)
- [ ] Version, changelog and docs match the release being cut.

## Recorded run log

Because this pass cannot run in CI, its evidence is this log rather than a green
build. Copy the block below into the release notes (or a `SMOKE_LOG.md`) and fill
it in for each environment you test, so the "verified in a real QGIS" claim is
backed by a dated, named record rather than being taken on trust. Leaving a
section blank is itself a signal — it says that path was *not* exercised for this
release.

```
Release:        0.10.0
Tester:         <name>
Date:           <YYYY-MM-DD>
QGIS build:     <e.g. 3.34.15 LTR, Qt5>   |   <e.g. 3.99 nightly, Qt6>
OS:             <Windows / macOS / Linux + version>
scikit-learn:   <version, or "installed via wizard button">
matplotlib:     <version, or "absent (PNG/PDF skip expected)">

 1. Install and load .............. PASS / FAIL / SKIP   notes:
 2. Load and visualise ............ PASS / FAIL / SKIP   notes:
 3. Similarity and change ......... PASS / FAIL / SKIP   notes:
 4. scikit-learn tier ............. PASS / FAIL / SKIP   notes:
    - wizard Tune hyper-parameters . PASS / FAIL / SKIP  notes:
    - wizard Fetch years + classify. PASS / FAIL / SKIP  notes:
 5. Robustness .................... PASS / FAIL / SKIP   notes:

AOI used (bbox + CRS):
Years fetched:
Anything unexpected / follow-ups:
```

Keep at least the most recent completed log with the release so a reviewer can see
which QGIS builds the release was actually exercised against.
