"""The AlphaEarth Toolbox guided wizard (milestone M3).

A tabbed :class:`~qgis.PyQt.QtWidgets.QDialog` that walks the user through the
toolbox without their having to open each Processing algorithm by hand:

* **Load / Visualise** -- fetch a 64-band embedding for a drawn/canvas extent and
  a chosen year (``alphaearth:loadembedding``), then paint any embedding to a
  3-band RGB preview via PCA or a band triplet (``alphaearth:torgb``).
* **Classify** -- supervised land-cover classification from labelled features
  (``alphaearth:classify``), with a one-click *Install scikit-learn* action for
  the optional ML tier, optional save/reuse of the trained model, an optional
  k-fold accuracy report and an opt-in hyper-parameter tuning switch; the same tab
  can also fetch the ticked years straight from AlphaEarth and classify them in
  one step (``alphaearth:classifyyears``), optionally writing a first-to-last
  transition raster and a changed/unchanged mask.
* **Similarity** -- "find more like these" from seed features
  (``alphaearth:similarity``).
* **Change** -- multi-year change from two or more embeddings already in the
  project (``alphaearth:change``), or fetch the ticked years straight from
  AlphaEarth and compare in one step (``alphaearth:changeyears``); either run can
  optionally write a report / charts (HTML, PNG, CSV, PDF) alongside the raster.

Every run is handed to :class:`qgis.core.QgsProcessingAlgRunnerTask` so the GUI
never blocks; a shared progress bar and Cancel button drive the running task, and
file outputs are added to the project when it finishes. The dialog only collects
parameters and delegates -- all numerical work lives in the algorithms -- so it
is deliberately import-light and holds no analysis logic of its own.

This module imports Qt/QGIS and is therefore only ever imported lazily by
:mod:`alphaearth_toolbox.plugin` when the user opens the dialog.
"""

from __future__ import annotations

import tempfile
from typing import Any

from qgis.core import QgsCoordinateReferenceSystem, QgsProcessing
from qgis.gui import (
    QgsExtentGroupBox,
    QgsFieldComboBox,
    QgsMapLayerComboBox,
    QgsProjectionSelectionWidget,
)
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from alphaearth_toolbox.aecore import change, intake, presets, report, rgb, similarity
from alphaearth_toolbox.algorithms import _report, _sklearn

#: Algorithm ids driven by the wizard tabs.
_ALG_LOAD = "alphaearth:loadembedding"
_ALG_LOAD_YEARS = "alphaearth:loadyears"
_ALG_RGB = "alphaearth:torgb"
_ALG_CLASSIFY = "alphaearth:classify"
_ALG_CLASSIFY_YEARS = "alphaearth:classifyyears"
_ALG_SIMILARITY = "alphaearth:similarity"
_ALG_CHANGE = "alphaearth:change"
_ALG_CHANGE_YEARS = "alphaearth:changeyears"

#: Sentinel string every Processing algorithm accepts for "write a temp file".
_TEMP = QgsProcessing.TEMPORARY_OUTPUT


def _option_index(order: tuple[str, ...], value: str) -> int:
    """Return the dropdown index for a preset's semantic value (0 if unknown).

    ``order`` is the canonical option order the core exposes (e.g.
    :data:`rgb.METHODS`); the wizard dropdowns are built in the same order, so a
    preset value always maps to the matching dropdown row. The fallback keeps the
    UI usable even if a future preset carried a value the dropdown lacks.
    """
    try:
        return order.index(value)
    except ValueError:  # pragma: no cover - guarded by the preset tests
        return 0


def _raster_filter() -> Any:
    """Return the "raster layers only" filter flag, across QGIS 3/4 enum moves."""
    try:
        from qgis.core import QgsMapLayerProxyModel

        return QgsMapLayerProxyModel.RasterLayer
    except (ImportError, AttributeError):  # pragma: no cover - QGIS 4 fallback
        from qgis.core import Qgis

        return Qgis.LayerFilter.RasterLayer


def _vector_filter() -> Any:
    """Return the "vector layers only" filter flag, across QGIS 3/4 enum moves."""
    try:
        from qgis.core import QgsMapLayerProxyModel

        return QgsMapLayerProxyModel.VectorLayer
    except (ImportError, AttributeError):  # pragma: no cover - QGIS 4 fallback
        from qgis.core import Qgis

        return Qgis.LayerFilter.VectorLayer


class AlphaEarthDialog(QDialog):
    """Tabbed wizard for loading, visualising and analysing AlphaEarth embeddings."""

    def __init__(self, iface: Any = None, parent: Any = None) -> None:
        super().__init__(parent)
        self.iface = iface
        self._canvas = iface.mapCanvas() if iface is not None else None

        # Live task state (kept as attributes so nothing is garbage-collected
        # while a background task or feedback object is still in use).
        self._task: Any = None
        self._context: Any = None
        self._feedback: Any = None
        self._install_task: Any = None
        self._last_label = ""

        self.setWindowTitle("AlphaEarth Toolbox")
        self.setMinimumWidth(540)
        self._build_ui()
        self._refresh_sklearn_state()
        self._refresh_change_layers()

    # ------------------------------------------------------------------ UI ---
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._tabs = QTabWidget(self)
        self._tabs.addTab(self._build_load_tab(), "Load / Visualise")
        self._tabs.addTab(self._build_classify_tab(), "Classify")
        self._tabs.addTab(self._build_similarity_tab(), "Similarity")
        self._tabs.addTab(self._build_change_tab(), "Change")
        self._tabs.currentChanged.connect(lambda _index: self._refresh_change_layers())
        layout.addWidget(self._tabs)

        self._progress = QProgressBar(self)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.hide()
        self._cancel_button = QPushButton("Cancel", self)
        self._cancel_button.hide()
        self._cancel_button.clicked.connect(self._cancel)
        progress_row = QHBoxLayout()
        progress_row.addWidget(self._progress, 1)
        progress_row.addWidget(self._cancel_button)
        layout.addLayout(progress_row)

        self._status = QLabel("", self)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        close_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        close_box.rejected.connect(self.reject)
        layout.addWidget(close_box)

    def _build_load_tab(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)

        intro = QLabel(
            "Fetch a 64-band AlphaEarth embedding for an extent and year (no Earth "
            "Engine account needed), or re-project an embedding you already have. "
            "Then paint any embedding to an RGB preview."
        )
        intro.setWordWrap(True)
        form.addRow(intro)

        self.load_layer = QgsMapLayerComboBox()
        self.load_layer.setFilters(_raster_filter())
        self.load_layer.setAllowEmptyLayer(True)
        form.addRow("Re-project existing layer (optional)", self.load_layer)

        self.load_url = QLineEdit()
        self.load_url.setPlaceholderText("http(s):// or /vsicurl/... COG (optional)")
        form.addRow("Source URL (optional)", self.load_url)

        self.load_year = QComboBox()
        self.load_year.addItem("(none)", 0)
        for year in intake.AVAILABLE_YEARS:
            self.load_year.addItem(str(year), int(year))
        self.load_year.setCurrentIndex(self.load_year.count() - 1)
        form.addRow("AlphaEarth year", self.load_year)

        self.load_extent = QgsExtentGroupBox()
        if self._canvas is not None:
            self.load_extent.setMapCanvas(self._canvas)
        form.addRow(self.load_extent)

        self.load_crs = QgsProjectionSelectionWidget()
        self.load_crs.setCrs(QgsCoordinateReferenceSystem(intake.DEFAULT_CRS))
        self.load_extent.setOutputCrs(self.load_crs.crs())
        self.load_crs.crsChanged.connect(self.load_extent.setOutputCrs)
        form.addRow("Target CRS", self.load_crs)

        self.load_res = QDoubleSpinBox()
        self.load_res.setDecimals(3)
        self.load_res.setRange(0.001, 1_000_000.0)
        self.load_res.setValue(intake.NATIVE_RESOLUTION_M)
        self.load_res.setSuffix(" units/px")
        form.addRow("Resolution", self.load_res)

        load_button = QPushButton("Load embedding")
        load_button.clicked.connect(self._run_load)
        form.addRow(load_button)

        years_divider = QLabel("...or fetch several years for the same area:")
        years_divider.setWordWrap(True)
        form.addRow(years_divider)

        self.load_years = QListWidget()
        self.load_years.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.load_years.setMaximumHeight(150)
        for year in intake.AVAILABLE_YEARS:
            item = QListWidgetItem(str(year))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.load_years.addItem(item)
        form.addRow("Years (tick to fetch)", self.load_years)

        load_years_button = QPushButton("Load selected years")
        load_years_button.clicked.connect(self._run_load_years)
        form.addRow(load_years_button)

        divider = QLabel("Visualise (embedding -> RGB)")
        form.addRow(divider)

        self.viz_layer = QgsMapLayerComboBox()
        self.viz_layer.setFilters(_raster_filter())
        form.addRow("Embedding to preview", self.viz_layer)

        self.viz_preset = QComboBox()
        self.viz_preset.addItems(presets.preset_labels(presets.RGB_PRESETS))
        self.viz_preset.setToolTip(
            "Pick a starting point; it fills the fields below, which you can then tweak."
        )
        form.addRow("Preset", self.viz_preset)

        self.viz_method = QComboBox()
        self.viz_method.addItems(["PCA (top 3 components)", "Band triplet"])
        self.viz_method.currentIndexChanged.connect(self._sync_viz_bands_enabled)
        form.addRow("Projection method", self.viz_method)

        self.viz_bands = QLineEdit("1,2,3")
        self.viz_bands.setEnabled(False)
        form.addRow("Bands (for triplet)", self.viz_bands)

        self.viz_low = QDoubleSpinBox()
        self.viz_low.setRange(0.0, 50.0)
        self.viz_low.setValue(2.0)
        self.viz_low.setSuffix(" %")
        form.addRow("Stretch low percentile", self.viz_low)

        self.viz_high = QDoubleSpinBox()
        self.viz_high.setRange(50.0, 100.0)
        self.viz_high.setValue(98.0)
        self.viz_high.setSuffix(" %")
        form.addRow("Stretch high percentile", self.viz_high)

        viz_button = QPushButton("Preview RGB")
        viz_button.clicked.connect(self._run_visualise)
        form.addRow(viz_button)

        # Connect after every dependent widget exists (addItems above fired no
        # handler), then seed the fields from the default preset.
        self.viz_preset.currentIndexChanged.connect(self._apply_rgb_preset)
        self._apply_rgb_preset(self.viz_preset.currentIndex())

        return widget

    def _build_classify_tab(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)

        intro = QLabel(
            "Train a classifier on labelled features and map land cover across an "
            "embedding. Needs the optional scikit-learn package."
        )
        intro.setWordWrap(True)
        form.addRow(intro)

        self.sklearn_status = QLabel("")
        self.sklearn_status.setWordWrap(True)
        form.addRow(self.sklearn_status)
        self.sklearn_button = QPushButton("Install scikit-learn")
        self.sklearn_button.clicked.connect(self._install_sklearn)
        form.addRow(self.sklearn_button)

        self.cls_layer = QgsMapLayerComboBox()
        self.cls_layer.setFilters(_raster_filter())
        form.addRow("Embedding raster", self.cls_layer)

        self.cls_train = QgsMapLayerComboBox()
        self.cls_train.setFilters(_vector_filter())
        form.addRow("Labelled training features", self.cls_train)

        self.cls_field = QgsFieldComboBox()
        self.cls_field.setLayer(self.cls_train.currentLayer())
        self.cls_train.layerChanged.connect(self.cls_field.setLayer)
        form.addRow("Class label field", self.cls_field)

        self.cls_estimator = QComboBox()
        self.cls_estimator.addItems(["Random forest", "Gradient boosting"])
        form.addRow("Estimator", self.cls_estimator)

        self.cls_trees = QSpinBox()
        self.cls_trees.setRange(1, 5000)
        self.cls_trees.setValue(200)
        form.addRow("Trees / boosting stages", self.cls_trees)

        self.cls_seed = QSpinBox()
        self.cls_seed.setRange(0, 1_000_000)
        self.cls_seed.setValue(42)
        form.addRow("Random seed", self.cls_seed)

        self.cls_confidence = QCheckBox("Also output a per-pixel confidence raster")
        form.addRow(self.cls_confidence)

        model_divider = QLabel("Reuse or assess the model (optional)")
        model_divider.setWordWrap(True)
        form.addRow(model_divider)

        self.cls_load_model, load_row = self._path_row(
            save=False,
            caption="Choose a saved model to reuse",
            file_filter="Model files (*.pkl);;All files (*)",
            placeholder="Reuse a saved .pkl model instead of training (optional)",
        )
        self.cls_load_model.setToolTip(
            "When set, the saved model is applied directly and the training features "
            "and label field are not needed."
        )
        form.addRow("Reuse model", load_row)

        self.cls_save_model, save_row = self._path_row(
            save=True,
            caption="Save the trained model as",
            file_filter="Model files (*.pkl);;All files (*)",
            placeholder="Save the trained model to a .pkl file (optional)",
        )
        form.addRow("Save model", save_row)

        self.cls_folds = QSpinBox()
        self.cls_folds.setRange(0, 20)
        self.cls_folds.setValue(0)
        self.cls_folds.setToolTip(
            "0 skips the accuracy assessment. 2+ runs stratified k-fold cross-validation "
            "and reports overall accuracy plus per-class precision/recall/F1."
        )
        form.addRow("Accuracy folds (0 = skip)", self.cls_folds)

        self.cls_report, report_row = self._path_row(
            save=True,
            caption="Write the accuracy report to",
            file_filter="Text files (*.txt);;All files (*)",
            placeholder="Write the class-balance / accuracy report to a .txt file (optional)",
        )
        form.addRow("Report file", report_row)

        tune_divider = QLabel("Hyper-parameter tuning (optional)")
        tune_divider.setWordWrap(True)
        form.addRow(tune_divider)

        self.cls_tune = QCheckBox("Tune hyper-parameters (cross-validated randomized search)")
        self.cls_tune.setToolTip(
            "Off by default. When on, a small cross-validated randomized search picks the "
            "estimator's hyper-parameters before the final fit, falling back to the defaults "
            "when the training set is too small to search. Applies to both runs below."
        )
        form.addRow(self.cls_tune)

        self.cls_tune_iters = QSpinBox()
        self.cls_tune_iters.setRange(1, 100)
        self.cls_tune_iters.setValue(10)
        self.cls_tune_iters.setEnabled(False)
        self.cls_tune_iters.setToolTip(
            "How many random hyper-parameter combinations to try when tuning is on."
        )
        self.cls_tune.toggled.connect(self.cls_tune_iters.setEnabled)
        form.addRow("Tuning iterations", self.cls_tune_iters)

        run_button = QPushButton("Run classification")
        run_button.clicked.connect(self._run_classify)
        form.addRow(run_button)

        years_divider = QLabel(
            "...or fetch straight from AlphaEarth: tick the years and set the extent on "
            "the Load / Visualise tab, then classify them in one step. One model is "
            "trained on the most recent ticked year (or reused from a saved model) and "
            "applied to every year, so a class code means the same thing across years."
        )
        years_divider.setWordWrap(True)
        form.addRow(years_divider)

        self.cls_years_transition = QCheckBox("Also output a first-to-last transition raster")
        self.cls_years_transition.setToolTip(
            "Encodes the class change from the first to the last ticked year, one colour "
            "per from/to pair. Used only by 'Fetch ticked years and classify'."
        )
        form.addRow(self.cls_years_transition)

        self.cls_years_mask = QCheckBox("Also output a changed/unchanged mask")
        self.cls_years_mask.setToolTip(
            "1 where the class differs between the first and last ticked year, else 0. "
            "Used only by 'Fetch ticked years and classify'."
        )
        form.addRow(self.cls_years_mask)

        classify_years_button = QPushButton("Fetch ticked years and classify")
        classify_years_button.clicked.connect(self._run_classify_years)
        form.addRow(classify_years_button)

        return widget

    def _build_similarity_tab(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)

        intro = QLabel(
            "Score every pixel by cosine similarity to one or more seed features "
            "-- 'find more places like these'."
        )
        intro.setWordWrap(True)
        form.addRow(intro)

        self.sim_layer = QgsMapLayerComboBox()
        self.sim_layer.setFilters(_raster_filter())
        form.addRow("Embedding raster", self.sim_layer)

        self.sim_seeds = QgsMapLayerComboBox()
        self.sim_seeds.setFilters(_vector_filter())
        form.addRow("Seed features", self.sim_seeds)

        self.sim_preset = QComboBox()
        self.sim_preset.addItems(presets.preset_labels(presets.SIMILARITY_PRESETS))
        self.sim_preset.setToolTip(
            "Pick a starting point; it fills the fields below, which you can then tweak."
        )
        form.addRow("Preset", self.sim_preset)

        self.sim_aggregation = QComboBox()
        self.sim_aggregation.addItems(["Mean of seeds", "Medoid (most representative seed)"])
        form.addRow("Combine seed pixels by", self.sim_aggregation)

        self.sim_rescale = QCheckBox("Rescale similarity to 0-1")
        self.sim_rescale.setChecked(True)
        form.addRow(self.sim_rescale)

        self.sim_threshold = QDoubleSpinBox()
        self.sim_threshold.setRange(0.0, 1.0)
        self.sim_threshold.setSingleStep(0.05)
        self.sim_threshold.setValue(0.0)
        form.addRow("Mask threshold (0 = none)", self.sim_threshold)

        self.sim_mask = QCheckBox("Also output a threshold mask")
        form.addRow(self.sim_mask)

        run_button = QPushButton("Run similarity")
        run_button.clicked.connect(self._run_similarity)
        form.addRow(run_button)

        self.sim_preset.currentIndexChanged.connect(self._apply_similarity_preset)
        self._apply_similarity_preset(self.sim_preset.currentIndex())

        return widget

    def _build_change_tab(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)

        intro = QLabel(
            "Compare two or more embeddings (e.g. successive years) to map how much "
            "each pixel changed. Tick the rasters to compare; they are read in the "
            "listed (oldest-to-newest) order."
        )
        intro.setWordWrap(True)
        form.addRow(intro)

        self.chg_list = QListWidget()
        self.chg_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.chg_list.setMinimumHeight(120)
        form.addRow("Embedding rasters (tick to compare)", self.chg_list)

        refresh_button = QPushButton("Refresh layer list")
        refresh_button.clicked.connect(self._refresh_change_layers)
        form.addRow(refresh_button)

        self.chg_preset = QComboBox()
        self.chg_preset.addItems(presets.preset_labels(presets.CHANGE_PRESETS))
        self.chg_preset.setToolTip(
            "Pick a starting point; it fills the fields below, which you can then tweak."
        )
        form.addRow("Preset", self.chg_preset)

        self.chg_metric = QComboBox()
        self.chg_metric.addItems(["Cosine distance (1 - similarity)", "Euclidean distance"])
        form.addRow("Distance metric", self.chg_metric)

        self.chg_mode = QComboBox()
        self.chg_mode.addItems(
            [
                "Pairwise (first vs last)",
                "Trajectory magnitude (cumulative)",
                "Anomaly vs baseline (latest vs mean)",
            ]
        )
        form.addRow("Change mode", self.chg_mode)

        run_button = QPushButton("Run change detection")
        run_button.clicked.connect(self._run_change)
        form.addRow(run_button)

        years_divider = QLabel(
            "...or fetch straight from AlphaEarth: tick the years and set the extent on "
            "the Load / Visualise tab, then compare them in one step (no need to load "
            "each year first)."
        )
        years_divider.setWordWrap(True)
        form.addRow(years_divider)

        change_years_button = QPushButton("Fetch ticked years and compare")
        change_years_button.clicked.connect(self._run_change_years)
        form.addRow(change_years_button)

        self._add_change_report_controls(form)

        self.chg_preset.currentIndexChanged.connect(self._apply_change_preset)
        self._apply_change_preset(self.chg_preset.currentIndex())

        return widget

    def _add_change_report_controls(self, form: QFormLayout) -> None:
        """Add the optional report / chart controls shared by both change runs.

        The report is opt-in: leave every output path blank and the run behaves
        exactly as before, writing only the change raster. Set one or more of the
        HTML / PNG / CSV / PDF paths and a report is written alongside it, covering
        the ticked contents. The same controls drive both the raster-based *Run
        change detection* and the *Fetch ticked years and compare* buttons.
        """
        report_divider = QLabel(
            "Optional report / charts -- written alongside the change raster. Leave "
            "the paths blank to skip. Tick the contents to include; PNG and PDF need "
            "matplotlib in the QGIS Python environment (HTML and CSV never do)."
        )
        report_divider.setWordWrap(True)
        form.addRow(report_divider)

        content_box = QWidget()
        content_layout = QVBoxLayout(content_box)
        content_layout.setContentsMargins(0, 0, 0, 0)
        self.chg_report_content: dict[str, QCheckBox] = {}
        for key in report.CONTENTS:
            checkbox = QCheckBox(report.CONTENT_LABELS[key])
            checkbox.setChecked(True)
            self.chg_report_content[key] = checkbox
            content_layout.addWidget(checkbox)
        form.addRow("Report contents", content_box)

        self.chg_report_html, html_row = self._path_row(
            save=True,
            caption="Write the HTML report to",
            file_filter="HTML files (*.html);;All files (*)",
            placeholder="Standalone HTML report with inline charts (optional)",
        )
        form.addRow("Report HTML", html_row)

        self.chg_report_png, png_row = self._path_row(
            save=True,
            caption="Write the PNG charts to",
            file_filter="PNG files (*.png);;All files (*)",
            placeholder="Charts as a PNG image -- needs matplotlib (optional)",
        )
        form.addRow("Report PNG", png_row)

        self.chg_report_csv, csv_row = self._path_row(
            save=True,
            caption="Write the CSV data to",
            file_filter="CSV files (*.csv);;All files (*)",
            placeholder="Report data as CSV (optional)",
        )
        form.addRow("Report CSV", csv_row)

        self.chg_report_pdf, pdf_row = self._path_row(
            save=True,
            caption="Write the PDF report to",
            file_filter="PDF files (*.pdf);;All files (*)",
            placeholder="Report as a PDF -- needs matplotlib (optional)",
        )
        form.addRow("Report PDF", pdf_row)

        self.chg_report_scatter_band = QSpinBox()
        self.chg_report_scatter_band.setRange(1, 64)
        self.chg_report_scatter_band.setValue(1)
        self.chg_report_scatter_band.setToolTip(
            "1-based embedding band plotted on the scatter's x-axis (against the most "
            "recent year), used only for the 'scatter' content."
        )
        form.addRow("Scatter embedding band", self.chg_report_scatter_band)

    def _change_report_params(self) -> dict[str, Any]:
        """Collect the optional report parameters shared by both change runs.

        Returns an empty dict when no output path is set (so the run is unchanged),
        otherwise the ``REPORT_*`` keys for the chosen formats, ticked contents and
        scatter band. Bins and thresholds keep their defaults here; tune them from
        the algorithm's own dialog if needed.
        """
        paths = {
            _report.REPORT_HTML: self.chg_report_html.text().strip(),
            _report.REPORT_PNG: self.chg_report_png.text().strip(),
            _report.REPORT_CSV: self.chg_report_csv.text().strip(),
            _report.REPORT_PDF: self.chg_report_pdf.text().strip(),
        }
        params: dict[str, Any] = {name: path for name, path in paths.items() if path}
        if not params:
            return {}
        indices = [
            index
            for index, key in enumerate(report.CONTENTS)
            if self.chg_report_content[key].isChecked()
        ]
        if indices:
            params[_report.REPORT_CONTENT] = indices
        params[_report.REPORT_SCATTER_BAND] = self.chg_report_scatter_band.value()
        return params

    # -------------------------------------------------------------- helpers ---
    def _apply_rgb_preset(self, index: int) -> None:
        """Fill the Visualise fields from the chosen RGB preset."""
        try:
            preset = presets.preset_by_index(presets.RGB_PRESETS, index)
        except IndexError:  # pragma: no cover - guarded by preset tests
            return
        self.viz_method.setCurrentIndex(_option_index(rgb.METHODS, preset.method))
        if preset.band_indices is not None:
            self.viz_bands.setText(",".join(str(band + 1) for band in preset.band_indices))
        self.viz_low.setValue(preset.low_percent)
        self.viz_high.setValue(preset.high_percent)
        self._sync_viz_bands_enabled()

    def _apply_similarity_preset(self, index: int) -> None:
        """Fill the Similarity fields from the chosen preset."""
        try:
            preset = presets.preset_by_index(presets.SIMILARITY_PRESETS, index)
        except IndexError:  # pragma: no cover - guarded by preset tests
            return
        self.sim_aggregation.setCurrentIndex(
            _option_index(similarity.AGGREGATIONS, preset.aggregation)
        )
        self.sim_rescale.setChecked(preset.rescale)
        self.sim_threshold.setValue(preset.threshold)

    def _apply_change_preset(self, index: int) -> None:
        """Fill the Change fields from the chosen preset."""
        try:
            preset = presets.preset_by_index(presets.CHANGE_PRESETS, index)
        except IndexError:  # pragma: no cover - guarded by preset tests
            return
        self.chg_metric.setCurrentIndex(_option_index(change.METRICS, preset.metric))
        self.chg_mode.setCurrentIndex(_option_index(change.MODES, preset.mode))

    def _sync_viz_bands_enabled(self) -> None:
        """Enable the band-triplet field only when the triplet method is chosen."""
        self.viz_bands.setEnabled(self.viz_method.currentIndex() == 1)

    def _path_row(
        self, *, save: bool, caption: str, file_filter: str, placeholder: str = ""
    ) -> tuple[QLineEdit, QWidget]:
        """Build a "line edit + Browse" row for an optional file path.

        Returns the editable :class:`QLineEdit` (so the caller can read/tooltip it)
        and the composed row widget to drop into a form. ``save`` picks a save-file
        dialog (for outputs the run will create) versus an open-file dialog (for an
        existing file to read).
        """
        edit = QLineEdit()
        if placeholder:
            edit.setPlaceholderText(placeholder)
        button = QPushButton("Browse...")

        def _browse() -> None:
            if save:
                path, _ = QFileDialog.getSaveFileName(self, caption, edit.text(), file_filter)
            else:
                path, _ = QFileDialog.getOpenFileName(self, caption, edit.text(), file_filter)
            if path:
                edit.setText(path)

        button.clicked.connect(_browse)
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(edit, 1)
        row_layout.addWidget(button)
        return edit, row

    def _refresh_change_layers(self) -> None:
        """Repopulate the change-tab raster list from the current project."""
        if not hasattr(self, "chg_list"):
            return
        from qgis.core import QgsProject, QgsRasterLayer

        checked = self._checked_change_layer_ids()
        self.chg_list.clear()
        for layer in QgsProject.instance().mapLayers().values():
            if isinstance(layer, QgsRasterLayer):
                item = QListWidgetItem(layer.name())
                item.setData(Qt.ItemDataRole.UserRole, layer.id())
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Checked if layer.id() in checked else Qt.CheckState.Unchecked
                )
                self.chg_list.addItem(item)

    def _checked_change_layer_ids(self) -> set[str]:
        """Layer ids whose change-tab tick-box is currently checked."""
        if not hasattr(self, "chg_list"):
            return set()
        ids: set[str] = set()
        for row in range(self.chg_list.count()):
            item = self.chg_list.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                ids.add(item.data(Qt.ItemDataRole.UserRole))
        return ids

    def _refresh_sklearn_state(self) -> None:
        """Reflect whether scikit-learn is importable in the ML-tier controls."""
        if _sklearn.is_available():
            self.sklearn_status.setText("scikit-learn is available -- the ML tools are ready.")
            self.sklearn_button.setEnabled(False)
        else:
            self.sklearn_status.setText(
                "scikit-learn is not installed. The Classify tool needs it; install it "
                "once into the QGIS Python environment with the button below."
            )
            self.sklearn_button.setEnabled(True)

    def _install_sklearn(self) -> None:
        """Install scikit-learn on a background task, then refresh the state."""
        from qgis.core import QgsApplication, QgsTask

        self.sklearn_button.setEnabled(False)
        self.sklearn_status.setText("Installing scikit-learn... this can take a minute.")

        def _job(_task: Any) -> Any:
            return _sklearn.install()

        def _done(exception: Any, result: Any = None) -> None:
            self._install_task = None
            if exception is not None:
                self.sklearn_status.setText(f"Install failed: {exception}")
            elif result is not None and result.ok:
                self.sklearn_status.setText(
                    "scikit-learn installed. Re-open the algorithm to use it."
                )
            elif result is not None:
                self.sklearn_status.setText(f"Install did not complete: {result.output[:300]}")
            else:  # pragma: no cover - defensive
                self.sklearn_status.setText("Install did not complete.")
            self._refresh_sklearn_state()

        try:
            task = QgsTask.fromFunction("Install scikit-learn", _job, on_finished=_done)
        except Exception as error:  # pragma: no cover - GUI-only
            self.sklearn_status.setText(f"Could not start the install: {error}")
            self.sklearn_button.setEnabled(True)
            return
        self._install_task = task
        QgsApplication.taskManager().addTask(task)

    def _set_status(self, message: str) -> None:
        self._status.setText(message)

    def _extent_string(self) -> str | None:
        """Build a Processing ``xmin,xmax,ymin,ymax [CRS]`` extent string, or None."""
        rectangle = self.load_extent.outputExtent()
        crs = self.load_extent.outputCrs()
        if rectangle is None or rectangle.isEmpty():
            return None
        authid = crs.authid() if crs is not None and crs.isValid() else ""
        suffix = f" [{authid}]" if authid else ""
        return (
            f"{rectangle.xMinimum()},{rectangle.xMaximum()},"
            f"{rectangle.yMinimum()},{rectangle.yMaximum()}{suffix}"
        )

    # ----------------------------------------------------------- run actions ---
    def _run_load(self) -> None:
        layer = self.load_layer.currentLayer()
        url = self.load_url.text().strip()
        year = int(self.load_year.currentData())
        extent = self._extent_string()

        if layer is None and not url and year == 0:
            self._set_status(
                "Choose a source: an existing layer to re-project, a source URL, or "
                "an AlphaEarth year to fetch."
            )
            return
        if layer is None and not extent:
            self._set_status(
                "Set an extent (draw on the canvas or use the layer/canvas extent) so "
                "the fetch has an area of interest."
            )
            return

        params: dict[str, Any] = {
            "INPUT": layer,
            "SOURCE_URL": url,
            "YEAR": year,
            "TARGET_CRS": self.load_crs.crs(),
            "RESOLUTION": self.load_res.value(),
            "OUTPUT": _TEMP,
        }
        if extent:
            params["EXTENT"] = extent
        self._execute(_ALG_LOAD, params, "Load embedding")

    def _checked_year_indices(self) -> list[int]:
        """Return the AVAILABLE_YEARS indices whose tick-box is checked."""
        indices: list[int] = []
        for row in range(self.load_years.count()):
            item = self.load_years.item(row)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                indices.append(row)
        return indices

    def _run_load_years(self) -> None:
        indices = self._checked_year_indices()
        if not indices:
            self._set_status("Tick at least one year to fetch.")
            return
        extent = self._extent_string()
        if not extent:
            self._set_status(
                "Set an extent (draw on the canvas or use the layer/canvas extent) so "
                "the multi-year fetch has an area of interest."
            )
            return
        # Each year is written to a fresh temp folder as alphaearth_<year>.tif;
        # the finished run returns one path per year, which the runner adds to
        # the project.
        folder = tempfile.mkdtemp(prefix="alphaearth_years_")
        params: dict[str, Any] = {
            "YEARS": indices,
            "EXTENT": extent,
            "TARGET_CRS": self.load_crs.crs(),
            "RESOLUTION": self.load_res.value(),
            "OUTPUT": folder,
        }
        self._execute(_ALG_LOAD_YEARS, params, "Load years")

    def _run_visualise(self) -> None:
        layer = self.viz_layer.currentLayer()
        if layer is None:
            self._set_status("Choose an embedding raster to preview.")
            return
        params = {
            "INPUT": layer,
            "METHOD": self.viz_method.currentIndex(),
            "BANDS": self.viz_bands.text().strip() or "1,2,3",
            "STRETCH_LOW": self.viz_low.value(),
            "STRETCH_HIGH": self.viz_high.value(),
            "OUTPUT": _TEMP,
        }
        self._execute(_ALG_RGB, params, "RGB preview")

    def _run_classify(self) -> None:
        if not _sklearn.is_available():
            self._set_status(
                "scikit-learn is not installed. Use the Install scikit-learn button first."
            )
            self._tabs.setCurrentIndex(1)
            return
        raster = self.cls_layer.currentLayer()
        training = self.cls_train.currentLayer()
        field = self.cls_field.currentField()
        load_model = self.cls_load_model.text().strip()
        if raster is None:
            self._set_status("Classification needs an embedding raster.")
            return
        # Training features and a label field are only required when training a
        # fresh model; reusing a saved model applies it straight to the raster.
        if not load_model and (training is None or not field):
            self._set_status(
                "Classification needs training features and a label field, or a saved "
                "model to reuse."
            )
            return
        params: dict[str, Any] = {
            "INPUT": raster,
            "ALGORITHM": self.cls_estimator.currentIndex(),
            "N_ESTIMATORS": self.cls_trees.value(),
            "SEED": self.cls_seed.value(),
            "OUTPUT": _TEMP,
        }
        if training is not None and field:
            params["TRAINING"] = training
            params["LABEL_FIELD"] = field
        if self.cls_confidence.isChecked():
            params["CONFIDENCE"] = _TEMP
        if load_model:
            params["LOAD_MODEL"] = load_model
        save_model = self.cls_save_model.text().strip()
        if save_model:
            params["SAVE_MODEL"] = save_model
        if self.cls_folds.value() > 0:
            params["ACCURACY_FOLDS"] = self.cls_folds.value()
        report = self.cls_report.text().strip()
        if report:
            params["REPORT"] = report
        if self.cls_tune.isChecked():
            params["TUNE"] = True
            params["TUNE_ITERS"] = self.cls_tune_iters.value()
        self._execute(_ALG_CLASSIFY, params, "Classification")

    def _run_classify_years(self) -> None:
        """Fetch the ticked years for the Load-tab extent and classify each in one step.

        Reuses the Load / Visualise tab's ticked years and extent (exactly like
        *Fetch ticked years and compare* on the Change tab) together with this tab's
        estimator, training, model-reuse, report and tuning settings. One model is
        trained on the most recent ticked year -- or reused from a saved .pkl -- and
        applied to every year, writing a folder of per-year class rasters plus the
        optionally ticked transition raster and changed/unchanged mask.
        """
        if not _sklearn.is_available():
            self._set_status(
                "scikit-learn is not installed. Use the Install scikit-learn button first."
            )
            self._tabs.setCurrentIndex(1)
            return
        indices = self._checked_year_indices()
        if len(indices) < 2:
            self._set_status(
                "Tick at least two years on the Load / Visualise tab to fetch and classify."
            )
            self._tabs.setCurrentIndex(0)
            return
        extent = self._extent_string()
        if not extent:
            self._set_status(
                "Set an extent on the Load / Visualise tab (draw on the canvas or use the "
                "layer/canvas extent) so the fetch has an area of interest."
            )
            self._tabs.setCurrentIndex(0)
            return
        training = self.cls_train.currentLayer()
        field = self.cls_field.currentField()
        load_model = self.cls_load_model.text().strip()
        # Training features and a label field are only required when training a fresh
        # model; reusing a saved model applies it straight to every fetched year.
        if not load_model and (training is None or not field):
            self._set_status(
                "Classification needs training features and a label field, or a saved "
                "model to reuse."
            )
            self._tabs.setCurrentIndex(1)
            return
        # Per-year class rasters land in a fresh temp folder as classification_<year>.tif;
        # the finished run returns one path per year for the runner to add.
        folder = tempfile.mkdtemp(prefix="alphaearth_classyears_")
        params: dict[str, Any] = {
            "YEARS": indices,
            "EXTENT": extent,
            "TARGET_CRS": self.load_crs.crs(),
            "RESOLUTION": self.load_res.value(),
            "ALGORITHM": self.cls_estimator.currentIndex(),
            "N_ESTIMATORS": self.cls_trees.value(),
            "SEED": self.cls_seed.value(),
            "OUTPUT": folder,
        }
        if training is not None and field:
            params["TRAINING"] = training
            params["LABEL_FIELD"] = field
        if load_model:
            params["LOAD_MODEL"] = load_model
        save_model = self.cls_save_model.text().strip()
        if save_model:
            params["SAVE_MODEL"] = save_model
        if self.cls_folds.value() > 0:
            params["ACCURACY_FOLDS"] = self.cls_folds.value()
        report = self.cls_report.text().strip()
        if report:
            params["REPORT"] = report
        if self.cls_tune.isChecked():
            params["TUNE"] = True
            params["TUNE_ITERS"] = self.cls_tune_iters.value()
        if self.cls_years_transition.isChecked():
            params["TRANSITION"] = _TEMP
        if self.cls_years_mask.isChecked():
            params["CHANGED_MASK"] = _TEMP
        self._execute(_ALG_CLASSIFY_YEARS, params, "Classify over years")

    def _run_similarity(self) -> None:
        raster = self.sim_layer.currentLayer()
        seeds = self.sim_seeds.currentLayer()
        if raster is None or seeds is None:
            self._set_status("Similarity needs an embedding raster and seed features.")
            return
        params = {
            "INPUT": raster,
            "SEEDS": seeds,
            "AGGREGATION": self.sim_aggregation.currentIndex(),
            "RESCALE": self.sim_rescale.isChecked(),
            "THRESHOLD": self.sim_threshold.value(),
            "OUTPUT": _TEMP,
        }
        if self.sim_mask.isChecked():
            params["OUTPUT_MASK"] = _TEMP
        self._execute(_ALG_SIMILARITY, params, "Similarity")

    def _run_change(self) -> None:
        from qgis.core import QgsProject

        layers = []
        for row in range(self.chg_list.count()):
            item = self.chg_list.item(row)
            if item.checkState() != Qt.CheckState.Checked:
                continue
            layer = QgsProject.instance().mapLayer(item.data(Qt.ItemDataRole.UserRole))
            if layer is not None:
                layers.append(layer)
        if len(layers) < 2:
            self._set_status("Tick at least two embedding rasters (oldest to newest).")
            return
        params: dict[str, Any] = {
            "INPUT": layers,
            "METRIC": self.chg_metric.currentIndex(),
            "MODE": self.chg_mode.currentIndex(),
            "OUTPUT": _TEMP,
        }
        params.update(self._change_report_params())
        self._execute(_ALG_CHANGE, params, "Change detection")

    def _run_change_years(self) -> None:
        """Fetch the ticked years for the Load-tab extent and compare in one step."""
        indices = self._checked_year_indices()
        if len(indices) < 2:
            self._set_status(
                "Tick at least two years on the Load / Visualise tab to fetch and compare."
            )
            self._tabs.setCurrentIndex(0)
            return
        extent = self._extent_string()
        if not extent:
            self._set_status(
                "Set an extent on the Load / Visualise tab (draw on the canvas or use the "
                "layer/canvas extent) so the fetch has an area of interest."
            )
            self._tabs.setCurrentIndex(0)
            return
        params: dict[str, Any] = {
            "YEARS": indices,
            "EXTENT": extent,
            "TARGET_CRS": self.load_crs.crs(),
            "RESOLUTION": self.load_res.value(),
            "METRIC": self.chg_metric.currentIndex(),
            "MODE": self.chg_mode.currentIndex(),
            "OUTPUT": _TEMP,
        }
        params.update(self._change_report_params())
        self._execute(_ALG_CHANGE_YEARS, params, "Change over years")

    # --------------------------------------------------------------- runner ---
    def _execute(self, algorithm_id: str, parameters: dict[str, Any], label: str) -> None:
        """Run an algorithm off the GUI thread with progress and Cancel."""
        if self._task is not None:
            self._set_status("A task is already running; wait for it or cancel it first.")
            return

        from qgis.core import (
            QgsApplication,
            QgsProcessingAlgRunnerTask,
            QgsProcessingContext,
            QgsProcessingFeedback,
            QgsProject,
        )

        registry = QgsApplication.processingRegistry()
        algorithm = registry.algorithmById(algorithm_id)
        if algorithm is None:
            self._set_status(
                f"Algorithm '{algorithm_id}' is not available. Is the AlphaEarth "
                "provider loaded in the Processing Toolbox?"
            )
            return

        self._last_label = label
        context = QgsProcessingContext()
        context.setProject(QgsProject.instance())
        feedback = QgsProcessingFeedback()

        try:
            task = QgsProcessingAlgRunnerTask(algorithm, parameters, context, feedback)
        except Exception as error:  # pragma: no cover - GUI-only fallback
            self._fallback_exec(algorithm_id, parameters, error)
            return

        self._context = context
        self._feedback = feedback
        self._task = task
        task.progressChanged.connect(self._on_progress)
        task.executed.connect(self._on_executed)
        self._set_running(True, f"Running {label}...")
        QgsApplication.taskManager().addTask(task)

    def _fallback_exec(
        self, algorithm_id: str, parameters: dict[str, Any], error: Exception
    ) -> None:  # pragma: no cover - GUI-only
        """Fall back to the modal Processing dialog if a background task can't start."""
        try:
            import processing

            processing.execAlgorithmDialog(algorithm_id, parameters)
            self._set_status(
                f"Opened '{algorithm_id}' in the Processing dialog "
                f"(background task unavailable: {error})."
            )
        except Exception as second_error:
            self._set_status(f"Could not run '{algorithm_id}': {error}; {second_error}")

    def _on_progress(self, value: float) -> None:
        self._progress.setValue(int(value))

    def _on_executed(self, successful: bool, results: Any) -> None:  # pragma: no cover - GUI-only
        self._task = None
        self._set_running(False)
        if successful:
            added = self._load_results(results if isinstance(results, dict) else {})
            noun = "output" if added == 1 else "outputs"
            self._set_status(f"{self._last_label} finished. Added {added} {noun} to the project.")
        else:
            self._set_status(
                f"{self._last_label} did not finish. See Log Messages "
                "(View > Panels > Log Messages) for details."
            )
        self._context = None
        self._feedback = None
        self._refresh_change_layers()

    def _load_results(self, results: dict[str, Any]) -> int:  # pragma: no cover - GUI-only
        """Add file-path outputs from a finished run to the project; return the count."""
        from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer

        added = 0
        for key, value in results.items():
            if not isinstance(value, str) or not value:
                continue
            name = f"{self._last_label} - {key.lower()}"
            raster = QgsRasterLayer(value, name)
            if raster.isValid():
                QgsProject.instance().addMapLayer(raster)
                added += 1
                continue
            vector = QgsVectorLayer(value, name, "ogr")
            if vector.isValid():
                QgsProject.instance().addMapLayer(vector)
                added += 1
        return added

    def _cancel(self) -> None:  # pragma: no cover - GUI-only
        if self._feedback is not None:
            self._feedback.cancel()
        if self._task is not None:
            self._task.cancel()
        self._set_status("Cancelling...")

    def _set_running(self, running: bool, message: str = "") -> None:
        self._tabs.setEnabled(not running)
        self._progress.setVisible(running)
        self._cancel_button.setVisible(running)
        if running:
            self._progress.setValue(0)
        if message:
            self._set_status(message)
