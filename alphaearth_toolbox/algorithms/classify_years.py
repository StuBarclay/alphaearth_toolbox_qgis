"""``alphaearth:classifyyears`` -- fetch several years then classify them in one step.

This is the classification analogue of *Change over years (fetch + compare)*: it
folds the two-step workflow -- *Load embeddings (multiple years)* followed by
*Classify embedding* -- into a single algorithm. Tick the years, set one area of
interest, provide labelled training features (or reuse a saved model), and it
fetches every year onto the same snapped grid, trains one classifier, and maps
land cover for each year -- without the intermediate per-year rasters ever having
to be written out and re-selected by hand.

**Training follows the "one model, most recent year" rule:** the training samples
are gathered from the most recent ticked year's embedding, one classifier is
fitted, and that single model is applied to every ticked year, so the class codes
are directly comparable across years.

Three kinds of output are produced:

* a folder of **per-year class rasters** (``classification_<year>.tif``), one per
  ticked year;
* an optional **transition raster** encoding the ``first -> last`` class change
  per pixel (auto-styled, one colour per from/to pair); and
* an optional **changed/unchanged mask** (1 where the class differs between the
  first and last ticked year, else 0).

It belongs to the **optional scikit-learn tier**: scikit-learn is imported lazily
and, when missing, the algorithm fails with a clear message plus a one-click
install. It reuses the shared fetch path
(:func:`alphaearth_toolbox.algorithms._fetch.fetch_year_cube`) and the classifier
building / tuning / cross-validation helpers of
:mod:`alphaearth_toolbox.algorithms.classify`, so it cannot drift from either.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterCrs,
    QgsProcessingParameterEnum,
    QgsProcessingParameterExtent,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingUtils,
)

from alphaearth_toolbox.aecore import intake, model_io
from alphaearth_toolbox.algorithms import _features, _fetch, _ml_shared, _sklearn
from alphaearth_toolbox.algorithms._progress import make_message
from alphaearth_toolbox.algorithms._qgis_io import export_raster
from alphaearth_toolbox.algorithms._styling import apply_paletted_style
from alphaearth_toolbox.algorithms.classify import (
    _ALGORITHMS,
    _NODATA_CODE,
    _build_classifier,
    _class_legend,
    _cross_validate,
    _tune,
)

#: Fraction of the progress bar given to fetching (the rest is train + classify).
_FETCH_PROGRESS_END = 85.0

#: Labels for the two values of the changed/unchanged mask (index == code).
_MASK_LABELS = ["unchanged", "changed"]


class ClassifyOverYearsAlgorithm(QgsProcessingAlgorithm):
    """Fetch several AlphaEarth years over one AOI and classify each in one step."""

    YEARS = "YEARS"
    EXTENT = "EXTENT"
    TARGET_CRS = "TARGET_CRS"
    RESOLUTION = "RESOLUTION"
    TRAINING = "TRAINING"
    LABEL_FIELD = "LABEL_FIELD"
    ALGORITHM = "ALGORITHM"
    N_ESTIMATORS = "N_ESTIMATORS"
    SEED = "SEED"
    OUTPUT = "OUTPUT"
    TRANSITION = "TRANSITION"
    CHANGED_MASK = "CHANGED_MASK"

    def __init__(self) -> None:
        super().__init__()
        self._results: dict[str, Any] = {}
        self._classes: list[Any] = []

    def name(self) -> str:
        return "classifyyears"

    def displayName(self) -> str:
        return "Classify over years (fetch + classify)"

    def group(self) -> str:
        return "AlphaEarth"

    def groupId(self) -> str:
        return "alphaearth"

    def createInstance(self) -> ClassifyOverYearsAlgorithm:
        return ClassifyOverYearsAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fetch several AlphaEarth Satellite Embedding V1 years for one area of "
            "interest and classify each of them -- in a single step.\n\n"
            "This does 'Load embeddings (multiple years)' and 'Classify embedding' "
            "together: tick two or more years, set one area of interest, CRS and "
            "resolution, and provide labelled training features (or reuse a saved "
            "model). Every year is fetched onto the same grid, so the class maps are "
            "pixel-aligned and directly comparable.\n\n"
            "One classifier is trained on the most recent ticked year and applied to "
            "every year (one model, most recent year), so the class codes mean the "
            "same thing in each map.\n\n"
            "Outputs:\n"
            "- a folder of per-year class rasters (classification_<year>.tif);\n"
            "- optionally, a transition raster encoding the first->last class change "
            "per pixel (one colour per from/to pair);\n"
            "- optionally, a changed/unchanged mask (1 where the class differs between "
            "the first and last ticked year).\n\n"
            "Requires the optional scikit-learn package. If it is not installed, the "
            "algorithm stops with instructions to install it once into the QGIS "
            "Python environment.\n\n"
            f"{intake.ATTRIBUTION}"
        )

    def initAlgorithm(self, config: dict[str, Any] | None = None) -> None:
        year_options = [str(y) for y in intake.AVAILABLE_YEARS]
        self.addParameter(
            QgsProcessingParameterEnum(
                self.YEARS,
                "Years to fetch and classify (tick two or more)",
                options=year_options,
                allowMultiple=True,
                # Default to the two most recent years: the minimum for a comparison.
                defaultValue=[len(year_options) - 2, len(year_options) - 1],
            )
        )
        self.addParameter(QgsProcessingParameterExtent(self.EXTENT, "Area of interest"))
        self.addParameter(
            QgsProcessingParameterCrs(
                self.TARGET_CRS, "Target CRS", defaultValue=intake.DEFAULT_CRS
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RESOLUTION,
                "Output resolution (target CRS units)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=intake.NATIVE_RESOLUTION_M,
                minValue=0.0001,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.TRAINING, "Labelled training features", optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.LABEL_FIELD,
                "Class label field",
                parentLayerParameterName=self.TRAINING,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.ALGORITHM,
                "Estimator",
                options=["Random forest", "Gradient boosting"],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.N_ESTIMATORS,
                "Number of trees / boosting stages",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=200,
                minValue=1,
                maxValue=5000,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SEED,
                "Random seed",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=42,
                minValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterFolderDestination(self.OUTPUT, "Per-year class rasters (folder)")
        )
        transition = QgsProcessingParameterRasterDestination(
            self.TRANSITION,
            "First->last transition raster (optional)",
            optional=True,
            createByDefault=False,
        )
        self.addParameter(transition)
        changed = QgsProcessingParameterRasterDestination(
            self.CHANGED_MASK,
            "Changed/unchanged mask (optional)",
            optional=True,
            createByDefault=False,
        )
        self.addParameter(changed)
        _fetch.add_cache_parameter(self)
        _ml_shared.add_reuse_and_report_parameters(self)
        _ml_shared.add_tuning_parameters(self)

    def processAlgorithm(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict[str, Any]:
        if not _sklearn.is_available():
            raise QgsProcessingException(_sklearn.missing_message("Classify over years"))

        import numpy as np

        from alphaearth_toolbox.aecore import _raster, ml

        message = make_message(feedback)

        try:
            years = intake.years_from_indices(
                self.parameterAsEnums(parameters, self.YEARS, context)
            )
        except ValueError as error:
            raise QgsProcessingException(str(error)) from error
        if len(years) < 2:
            raise QgsProcessingException(
                "Classify over years needs at least two years; tick two or more (2017-2025)."
            )

        target_crs = self.parameterAsCrs(parameters, self.TARGET_CRS, context)
        if target_crs is None or not target_crs.isValid():
            raise QgsProcessingException("A valid target CRS is required.")
        dst_wkt = target_crs.toWkt()
        resolution = float(self.parameterAsDouble(parameters, self.RESOLUTION, context))

        extent = self.parameterAsExtent(parameters, self.EXTENT, context, target_crs)
        if extent.isEmpty():
            raise QgsProcessingException("The area of interest is empty.")
        bounds = (extent.xMinimum(), extent.yMinimum(), extent.xMaximum(), extent.yMaximum())
        # Size the grid once; every year shares it, so the fetched cubes are
        # inherently pixel-aligned and the class maps line up pixel-for-pixel.
        width, height, snapped = intake.grid_dimensions(bounds, resolution)
        _fetch.warn_if_large(width, height, feedback)

        source = self.parameterAsSource(parameters, self.TRAINING, context)
        label_field = self.parameterAsString(parameters, self.LABEL_FIELD, context)
        algorithm = _ALGORITHMS[self.parameterAsEnum(parameters, self.ALGORITHM, context)]
        n_estimators = self.parameterAsInt(parameters, self.N_ESTIMATORS, context)
        seed = self.parameterAsInt(parameters, self.SEED, context)
        load_model_path = self.parameterAsString(parameters, _ml_shared.LOAD_MODEL, context)
        save_model_path = self.parameterAsFileOutput(parameters, _ml_shared.SAVE_MODEL, context)
        report_path = self.parameterAsFileOutput(parameters, _ml_shared.REPORT, context)
        requested_folds = self.parameterAsInt(parameters, _ml_shared.ACCURACY_FOLDS, context)
        tune = self.parameterAsBool(parameters, _ml_shared.TUNE, context)
        tune_iters = self.parameterAsInt(parameters, _ml_shared.TUNE_ITERS, context)

        if not load_model_path and (source is None or not label_field):
            raise QgsProcessingException(
                "Provide labelled training features and a class label field, or a saved "
                "model to reuse."
            )

        out_folder = self.parameterAsString(parameters, self.OUTPUT, context)
        if not out_folder:
            raise QgsProcessingException(
                "An output folder for the per-year class rasters is required."
            )
        os.makedirs(out_folder, exist_ok=True)

        transform_context = context.transformContext()
        persist = self.parameterAsBool(parameters, _fetch.PERSIST_CACHE, context)
        cache_dir = _fetch.resolve_cache_dir(persist)
        n_years = len(years)
        message(
            f"Fetching {n_years} year(s) for the same area and classifying each: "
            f"{', '.join(map(str, years))}."
        )

        cubes: list[np.ndarray] = []
        geotransform: tuple[float, ...] | None = None
        report_sections: list[str] = []
        results: dict[str, Any] = {}

        try:
            for i, year in enumerate(years):
                if feedback.isCanceled():
                    raise QgsProcessingException("Canceled.")
                message(f"--- Year {year} ({i + 1} of {n_years}) ---")
                span_start = _FETCH_PROGRESS_END * i / n_years
                span_end = _FETCH_PROGRESS_END * (i + 1) / n_years
                cube, gt = _fetch.fetch_year_cube(
                    year,
                    snapped=snapped,
                    width=width,
                    height=height,
                    target_crs=target_crs,
                    transform_context=transform_context,
                    feedback=feedback,
                    message=message,
                    progress_start=span_start,
                    progress_end=span_end,
                    cache_dir=cache_dir,
                )
                cube, _out_nodata = _fetch.dequantise(cube)
                if geotransform is None:
                    geotransform = gt
                cubes.append(cube)

            assert geotransform is not None
            feedback.setProgress(_FETCH_PROGRESS_END)

            # --- Train once (on the most recent year) or reuse a saved model. ---
            estimator, classes = self._fit_or_load(
                cubes[-1],
                geotransform=geotransform,
                target_crs=target_crs,
                source=source,
                label_field=label_field,
                algorithm=algorithm,
                n_estimators=n_estimators,
                seed=seed,
                load_model_path=load_model_path,
                save_model_path=save_model_path,
                requested_folds=requested_folds,
                tune=tune,
                tune_iters=tune_iters,
                context=context,
                feedback=feedback,
                message=message,
                report_sections=report_sections,
            )
            self._classes = classes
            n_classes = len(classes)

            # --- Classify each year with the one model. ---
            class_grids: list[np.ndarray] = []
            for i, (year, cube) in enumerate(zip(years, cubes, strict=True)):
                if feedback.isCanceled():
                    raise QgsProcessingException("Canceled.")
                rows, cols = cube.shape[1], cube.shape[2]
                full = ml.cube_to_matrix(cube)
                valid = ml.flat_valid_mask(cube, _fetch.OUTPUT_NODATA)
                valid_matrix = full[valid]
                message(f"Classifying {valid_matrix.shape[0]} valid pixel(s) for {year}...")
                predicted = ml.predict_in_chunks(valid_matrix, estimator.predict)
                grid = ml.scatter_to_grid(
                    predicted, rows, cols, valid=valid, fill=_NODATA_CODE, dtype=np.int32
                )
                class_grids.append(grid)
                out_path = os.path.join(out_folder, f"classification_{year}.tif")
                _raster.write_geotiff(
                    out_path, grid, geotransform=geotransform, wkt=dst_wkt, nodata=_NODATA_CODE
                )
                results[f"CLASS_{year}"] = out_path
                message(f"Wrote {out_path}.")
                progress = _FETCH_PROGRESS_END + (95.0 - _FETCH_PROGRESS_END) * (i + 1) / n_years
                feedback.setProgress(progress)

            del cubes

            # --- First-vs-last transition and changed/unchanged mask. ---
            self._write_change_outputs(
                parameters,
                context,
                feedback,
                first=class_grids[0],
                last=class_grids[-1],
                n_classes=n_classes,
                geotransform=geotransform,
                wkt=dst_wkt,
                results=results,
                first_year=years[0],
                last_year=years[-1],
                message=message,
            )
        except QgsProcessingException:
            raise
        except Exception as exc:
            raise _fetch.report_and_raise(feedback, exc) from exc

        if report_path and report_sections:
            _ml_shared.write_text_report("\n\n".join(report_sections), report_path, feedback)

        results[self.OUTPUT] = out_folder
        message(f"Class codes: {_class_legend(classes)}.")
        feedback.setProgress(100.0)
        self._results = results
        return results

    def _fit_or_load(
        self,
        train_cube: Any,
        *,
        geotransform: tuple[float, ...],
        target_crs: Any,
        source: Any,
        label_field: str,
        algorithm: str,
        n_estimators: int,
        seed: int,
        load_model_path: str,
        save_model_path: str,
        requested_folds: int,
        tune: bool,
        tune_iters: int,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
        message: Any,
        report_sections: list[str],
    ) -> tuple[Any, list[Any]]:
        """Return a fitted (or reused) classifier and its class list.

        Trains on ``train_cube`` (the most recent year) unless a saved model is
        supplied, in which case the model is loaded and its recorded classes used.
        """
        from alphaearth_toolbox.aecore import ml

        bands = train_cube.shape[0]

        if load_model_path:
            bundle = _ml_shared.load_reused_model(
                load_model_path, n_bands=bands, kind=model_io.KIND_CLASSIFIER, feedback=feedback
            )
            classes = list(bundle.classes or [])
            if len(classes) < 2:
                raise QgsProcessingException(
                    "The reused model records fewer than two classes and cannot classify."
                )
            label = bundle.algorithm.replace("_", " ") or "saved model"
            message(f"Reusing {label} with {len(classes)} classes; training features ignored.")
            if requested_folds > 0 or tune:
                _ml_shared.push_warning(
                    feedback,
                    "Accuracy assessment and tuning need training data; both are skipped "
                    "when reusing a saved model.",
                )
            return bundle.estimator, classes

        cube_crs = _features.resolve_cube_crs(target_crs.toWkt(), target_crs)
        message("Collecting training samples from the most recent year...")
        matrix, labels, used = _features.collect_training(
            source,
            cube=train_cube,
            geotransform=geotransform,
            cube_crs=cube_crs,
            context=context,
            label_field=label_field,
            nodata=_fetch.OUTPUT_NODATA,
            skip_nodata=True,
            feedback=feedback,
        )
        if matrix.shape[0] == 0:
            raise QgsProcessingException(
                "No training samples were gathered. Check the training features overlap the "
                "area of interest and fall on valid pixels."
            )
        codes, classes = ml.encode_labels(labels)
        if len(classes) < 2:
            raise QgsProcessingException(
                f"Classification needs at least two classes; found only {len(classes)} "
                f"({classes!r}). Add training features for more classes."
            )

        balance = ml.class_balance_report(codes, classes)
        balance_text = ml.format_class_balance(balance)
        message(balance_text)
        report_sections.append(balance_text)
        if balance["imbalanced"]:
            _ml_shared.push_warning(
                feedback,
                f"Imbalanced training set (rarest '{balance['min_label']}' vs commonest "
                f"'{balance['max_label']}'); rare classes may map poorly.",
            )

        message(
            f"Fitting {algorithm.replace('_', ' ')} on {matrix.shape[0]} samples "
            f"from {used} feature(s) across {len(classes)} classes."
        )

        tuned_params: dict[str, Any] | None = None
        if tune:
            tune_folds = _ml_shared.resolve_tune_folds(requested_folds, matrix.shape[0], feedback)
            if tune_folds:
                tuned = _tune(
                    matrix,
                    codes,
                    classes,
                    algorithm,
                    n_estimators,
                    seed,
                    tune_iters,
                    tune_folds,
                    feedback,
                    message,
                )
                if tuned is not None:
                    tuned_params, tune_report = tuned
                    report_sections.append(tune_report)

        folds = _ml_shared.resolve_folds(requested_folds, matrix.shape[0], feedback)
        if folds:
            accuracy_text = _cross_validate(
                matrix,
                codes,
                classes,
                folds,
                algorithm,
                n_estimators,
                seed,
                feedback,
                message,
                params=tuned_params,
            )
            if accuracy_text:
                report_sections.append(accuracy_text)

        estimator = _build_classifier(algorithm, n_estimators, seed, params=tuned_params)
        estimator.fit(matrix, codes)

        if save_model_path:
            _ml_shared.save_fitted_model(
                save_model_path,
                estimator=estimator,
                kind=model_io.KIND_CLASSIFIER,
                n_bands=bands,
                algorithm=algorithm,
                classes=classes,
                feedback=feedback,
            )
        return estimator, classes

    def _write_change_outputs(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
        *,
        first: Any,
        last: Any,
        n_classes: int,
        geotransform: tuple[float, ...],
        wkt: str,
        results: dict[str, Any],
        first_year: int,
        last_year: int,
        message: Any,
    ) -> None:
        """Write the optional first->last transition and changed/unchanged mask rasters."""
        import numpy as np

        from alphaearth_toolbox.aecore import _raster, ml

        transition_dest = self.parameterAsOutputLayer(parameters, self.TRANSITION, context)
        mask_dest = self.parameterAsOutputLayer(parameters, self.CHANGED_MASK, context)
        if not transition_dest and not mask_dest:
            return

        scratch = Path(tempfile.mkdtemp(prefix="alphaearth_classifyyears_"))
        try:
            if transition_dest:
                message(f"Building {first_year}->{last_year} class-transition raster...")
                # int64 from the pure core -> int32 for a portable GeoTIFF (nodata -1).
                transition = ml.transition_codes(first, last, n_classes, nodata=_NODATA_CODE)
                tmp = str(scratch / "transition.tif")
                _raster.write_geotiff(
                    tmp,
                    transition.astype(np.int32),
                    geotransform=geotransform,
                    wkt=wkt,
                    nodata=_NODATA_CODE,
                )
                results[self.TRANSITION] = export_raster(tmp, transition_dest, feedback)
            if mask_dest:
                message(f"Building {first_year}->{last_year} changed/unchanged mask...")
                mask = ml.changed_mask(first, last, nodata=_NODATA_CODE)
                tmp = str(scratch / "changed.tif")
                _raster.write_geotiff(
                    tmp,
                    mask.astype(np.int32),
                    geotransform=geotransform,
                    wkt=wkt,
                    nodata=_NODATA_CODE,
                )
                results[self.CHANGED_MASK] = export_raster(tmp, mask_dest, feedback)
        finally:
            _rmtree_quiet(scratch)

    def postProcessAlgorithm(
        self, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ) -> dict[str, Any]:
        from alphaearth_toolbox.aecore import ml

        transition = self._results.get(self.TRANSITION)
        if transition:
            layer = QgsProcessingUtils.mapLayerFromString(transition, context)
            if layer is not None and self._classes:
                apply_paletted_style(layer, ml.transition_labels(self._classes), feedback)
        mask = self._results.get(self.CHANGED_MASK)
        if mask:
            layer = QgsProcessingUtils.mapLayerFromString(mask, context)
            if layer is not None:
                apply_paletted_style(layer, _MASK_LABELS, feedback)
        return self._results


def _rmtree_quiet(path: Path) -> None:
    """Best-effort recursive delete that never raises."""
    import shutil

    shutil.rmtree(path, ignore_errors=True)
