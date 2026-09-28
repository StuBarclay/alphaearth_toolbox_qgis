"""QGIS-backed regression tests for the AlphaEarth Processing algorithms.

These exercise the real :class:`QgsProcessingAlgorithm` wrappers end to end
through ``processing.run``: the provider is registered, tiny synthetic 64-band
embedding rasters and seed layers are fed in, and each algorithm is asserted to
produce a valid output. They complement the pure-core unit tests (which need no
QGIS at all) by proving the parameter wiring, the GDAL/QGIS I/O helpers and
provider registration actually work *inside* QGIS -- the part that cannot be
covered without ``qgis.core``.

The whole module is skipped where ``qgis.core`` (or GDAL) is unavailable, so it
is a no-op in the lightweight ``checks`` CI job and the local sandbox, and runs
for real in the ``qgis`` CI container (``qgis/qgis:release-3_34``) and in a live
QGIS install. The network-backed, public-GCS year-fetch algorithms
(``loadembedding`` in year mode, ``loadyears``, ``changeyears``, ``classifyyears``)
are deliberately *not* exercised here -- they need internet access and are covered
by the shared ``_fetch``/``intake`` unit tests plus the manual smoke test.
Everything tested here runs offline on synthetic pixels.
"""

from __future__ import annotations

from typing import Any

import pytest

qgis_core = pytest.importorskip("qgis.core", reason="QGIS (qgis.core) not available")
pytest.importorskip("osgeo.gdal", reason="GDAL (osgeo) not available")

import numpy as np  # noqa: E402
from qgis.core import (  # noqa: E402
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsPointXY,
    QgsRasterLayer,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QVariant  # noqa: E402

from alphaearth_toolbox.aecore import _raster  # noqa: E402
from alphaearth_toolbox.provider import AlphaEarthProvider  # noqa: E402

# --------------------------------------------------------------------------- #
# Synthetic-fixture geometry                                                  #
# --------------------------------------------------------------------------- #

#: A plain projected CRS in metres; the actual location is irrelevant because
#: the algorithms operate on band values, and seeds share this CRS with the
#: raster so pixel look-ups line up.
_CRS = "EPSG:3857"
#: Small square grid: 8x8 px at 10 m -> a 0..80 m extent on both axes.
_SIZE = 8
_PIX = 10.0
_BANDS = 64
#: Every algorithm the provider must expose (full ids are ``alphaearth:<name>``).
_EXPECTED_NAMES = (
    "loadembedding",
    "loadyears",
    "torgb",
    "similarity",
    "change",
    "changeyears",
    "extract",
    "classify",
    "classifyyears",
    "cluster",
    "regress",
)


def _wkt() -> str:
    return str(QgsCoordinateReferenceSystem(_CRS).toWkt())


def _write_embedding(path: str, *, seed: int, size: int = _SIZE) -> str:
    """Write a synthetic 64-band unit-vector embedding GeoTIFF to ``path``.

    Each pixel is a random 64-D vector normalised to unit length, mimicking a
    real AlphaEarth embedding (so cosine similarity behaves sensibly). The grid
    is north-up with square pixels anchored at the origin.
    """
    rng = np.random.default_rng(seed)
    cube = rng.standard_normal((_BANDS, size, size)).astype(np.float32)
    norms = np.sqrt((cube.astype(np.float64) ** 2).sum(axis=0, keepdims=True))
    norms[norms == 0.0] = 1.0
    cube = (cube / norms).astype(np.float32)
    geotransform = (0.0, _PIX, 0.0, float(size * _PIX), 0.0, -_PIX)
    _raster.write_geotiff(path, cube, geotransform=geotransform, wkt=_wkt(), nodata=None)
    return path


def _seed_points() -> QgsVectorLayer:
    """A tiny in-memory point layer of labelled seeds inside the raster extent."""
    layer = QgsVectorLayer(f"Point?crs={_CRS}", "seeds", "memory")
    provider = layer.dataProvider()
    provider.addAttributes([QgsField("class", QVariant.String)])
    layer.updateFields()
    # Two points per class, each landing on a distinct pixel of the 0..80 grid.
    coords = [(15.0, 65.0, "a"), (25.0, 55.0, "a"), (55.0, 25.0, "b"), (65.0, 15.0, "b")]
    feats = []
    for x, y, label in coords:
        feat = QgsFeature(layer.fields())
        feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
        feat.setAttributes([label])
        feats.append(feat)
    provider.addFeatures(feats)
    layer.updateExtents()
    return layer


# --------------------------------------------------------------------------- #
# QGIS / Processing environment                                               #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def alphaearth_env() -> Any:
    """Boot a headless QGIS, initialise Processing and register the provider.

    Yields the provider id. The ``QgsApplication`` and provider are kept
    referenced for the module's lifetime so the C++ objects are not collected
    mid-test, and QGIS is shut down cleanly afterwards.
    """
    app = QgsApplication([], False)
    app.initQgis()
    from processing.core.Processing import Processing

    Processing.initialize()
    provider = AlphaEarthProvider()
    QgsApplication.processingRegistry().addProvider(provider)
    try:
        yield provider.id()
    finally:
        QgsApplication.processingRegistry().removeProvider(provider.id())
        app.exitQgis()


def _run(alg_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Run one algorithm headless with a fresh context/feedback and return results."""
    import processing
    from qgis.core import QgsProcessingContext, QgsProcessingFeedback

    context = QgsProcessingContext()
    feedback = QgsProcessingFeedback()
    result: dict[str, Any] = processing.run(alg_id, params, context=context, feedback=feedback)
    return result


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #


def test_provider_registers_every_algorithm(alphaearth_env: str) -> None:
    provider_id = alphaearth_env
    registry = QgsApplication.processingRegistry()
    provider = registry.providerById(provider_id)
    assert provider is not None
    ids = {alg.id() for alg in provider.algorithms()}
    assert ids == {f"{provider_id}:{name}" for name in _EXPECTED_NAMES}


def test_torgb_produces_three_band_image(alphaearth_env: str, tmp_path: Any) -> None:
    embedding = _write_embedding(str(tmp_path / "embed.tif"), seed=1)
    out = str(tmp_path / "rgb.tif")
    result = _run(
        f"{alphaearth_env}:torgb",
        {"INPUT": embedding, "METHOD": 0, "STRETCH_LOW": 2.0, "STRETCH_HIGH": 98.0, "OUTPUT": out},
    )
    layer = QgsRasterLayer(result["OUTPUT"], "rgb")
    assert layer.isValid()
    assert layer.bandCount() == 3


def test_change_produces_single_band(alphaearth_env: str, tmp_path: Any) -> None:
    year1 = _write_embedding(str(tmp_path / "y1.tif"), seed=1)
    year2 = _write_embedding(str(tmp_path / "y2.tif"), seed=2)
    out = str(tmp_path / "change.tif")
    result = _run(
        f"{alphaearth_env}:change",
        {"INPUT": [year1, year2], "METRIC": 0, "MODE": 0, "OUTPUT": out},
    )
    layer = QgsRasterLayer(result["OUTPUT"], "change")
    assert layer.isValid()
    assert layer.bandCount() == 1
    assert layer.width() == _SIZE and layer.height() == _SIZE


def test_similarity_produces_single_band(alphaearth_env: str, tmp_path: Any) -> None:
    embedding = _write_embedding(str(tmp_path / "embed.tif"), seed=3)
    seeds = _seed_points()  # kept referenced for the duration of the run
    out = str(tmp_path / "similarity.tif")
    result = _run(
        f"{alphaearth_env}:similarity",
        {
            "INPUT": embedding,
            "SEEDS": seeds,
            "AGGREGATION": 0,
            "RESCALE": True,
            "THRESHOLD": 0.0,
            "OUTPUT": out,
        },
    )
    layer = QgsRasterLayer(result["OUTPUT"], "similarity")
    assert layer.isValid()
    assert layer.bandCount() == 1


def test_extract_writes_embedding_columns(alphaearth_env: str, tmp_path: Any) -> None:
    embedding = _write_embedding(str(tmp_path / "embed.tif"), seed=4)
    seeds = _seed_points()
    out = str(tmp_path / "samples.gpkg")
    result = _run(
        f"{alphaearth_env}:extract",
        {"INPUT": embedding, "SEEDS": seeds, "SKIP_NODATA": True, "OUTPUT": out},
    )
    layer = QgsVectorLayer(result["OUTPUT"], "samples", "ogr")
    assert layer.isValid()
    # One row per sampled pixel (the four seed points hit four distinct pixels).
    assert layer.featureCount() == 4
    field_names = {field.name() for field in layer.fields()}
    # The 64 embedding bands come through as A00 ... A63 alongside the label.
    assert {"A00", "A63", "class"} <= field_names


def test_tuning_is_opt_in_by_default(alphaearth_env: str) -> None:
    """The optional tuning switches must all default to off across the ML tier.

    House rule: every ML enhancement is opt-in and default-OFF, so an existing run
    is unchanged unless the user ticks it. This asserts the contract on the real
    QGIS parameter definitions (Classify, Regress and Classify-over-years expose
    ``TUNE``; Cluster exposes ``TUNE_K``).
    """
    from alphaearth_toolbox.algorithms.classify import ClassifyEmbeddingAlgorithm
    from alphaearth_toolbox.algorithms.classify_years import ClassifyOverYearsAlgorithm
    from alphaearth_toolbox.algorithms.cluster import ClusterEmbeddingAlgorithm
    from alphaearth_toolbox.algorithms.regress import RegressEmbeddingAlgorithm

    for cls in (
        ClassifyEmbeddingAlgorithm,
        RegressEmbeddingAlgorithm,
        ClassifyOverYearsAlgorithm,
    ):
        alg = cls()
        alg.initAlgorithm({})
        assert not alg.parameterDefinition("TUNE").defaultValue()

    cluster = ClusterEmbeddingAlgorithm()
    cluster.initAlgorithm({})
    assert not cluster.parameterDefinition("TUNE_K").defaultValue()


def test_classify_produces_class_raster(alphaearth_env: str, tmp_path: Any) -> None:
    pytest.importorskip("sklearn", reason="scikit-learn not available")
    embedding = _write_embedding(str(tmp_path / "embed.tif"), seed=5)
    seeds = _seed_points()
    out = str(tmp_path / "classes.tif")
    result = _run(
        f"{alphaearth_env}:classify",
        {
            "INPUT": embedding,
            "TRAINING": seeds,
            "LABEL_FIELD": "class",
            "ALGORITHM": 0,
            "N_ESTIMATORS": 50,
            "SEED": 42,
            "OUTPUT": out,
        },
    )
    layer = QgsRasterLayer(result["OUTPUT"], "classes")
    assert layer.isValid()
    assert layer.bandCount() == 1
    assert layer.width() == _SIZE and layer.height() == _SIZE
