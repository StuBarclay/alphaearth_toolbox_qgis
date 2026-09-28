"""Processing algorithms for the AlphaEarth Toolbox.

The provider discovers algorithms through :func:`_algorithms`, which imports the
algorithm classes lazily and returns fresh instances. Keeping the import inside
the function means importing this package (or the top-level plugin package) does
not pull in ``qgis`` / GDAL, so the pure compute core stays testable outside
QGIS.

The no-dependency tier ships eight algorithms: Load embedding, Load embeddings
(multiple years), Embedding to RGB, Similarity search, Change / trajectory,
Change over years (fetch + compare), Change report (chart / summary) and Extract
training samples. The scikit-learn tier adds four more -- Classify, Classify over
years (fetch + classify), Cluster and Regress -- which import scikit-learn lazily
and fail gracefully (with a one-click guided install) when it is not present,
rather than making it a hard dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qgis.core import QgsProcessingAlgorithm

__all__ = [
    "ChangeDetectionAlgorithm",
    "ChangeOverYearsAlgorithm",
    "ChangeReportAlgorithm",
    "ClassifyEmbeddingAlgorithm",
    "ClassifyOverYearsAlgorithm",
    "ClusterEmbeddingAlgorithm",
    "EmbeddingToRgbAlgorithm",
    "ExtractSamplesAlgorithm",
    "LoadEmbeddingAlgorithm",
    "LoadYearsAlgorithm",
    "RegressEmbeddingAlgorithm",
    "SimilaritySearchAlgorithm",
    "_algorithms",
]


def _algorithms() -> list[QgsProcessingAlgorithm]:
    """Return one fresh instance of every algorithm the toolbox provides."""
    from alphaearth_toolbox.algorithms.change_detection import ChangeDetectionAlgorithm
    from alphaearth_toolbox.algorithms.change_report import ChangeReportAlgorithm
    from alphaearth_toolbox.algorithms.change_years import ChangeOverYearsAlgorithm
    from alphaearth_toolbox.algorithms.classify import ClassifyEmbeddingAlgorithm
    from alphaearth_toolbox.algorithms.classify_years import ClassifyOverYearsAlgorithm
    from alphaearth_toolbox.algorithms.cluster import ClusterEmbeddingAlgorithm
    from alphaearth_toolbox.algorithms.embedding_rgb import EmbeddingToRgbAlgorithm
    from alphaearth_toolbox.algorithms.extract_samples import ExtractSamplesAlgorithm
    from alphaearth_toolbox.algorithms.load_embedding import LoadEmbeddingAlgorithm
    from alphaearth_toolbox.algorithms.load_years import LoadYearsAlgorithm
    from alphaearth_toolbox.algorithms.regress import RegressEmbeddingAlgorithm
    from alphaearth_toolbox.algorithms.similarity_search import SimilaritySearchAlgorithm

    return [
        LoadEmbeddingAlgorithm(),
        LoadYearsAlgorithm(),
        EmbeddingToRgbAlgorithm(),
        SimilaritySearchAlgorithm(),
        ChangeDetectionAlgorithm(),
        ChangeOverYearsAlgorithm(),
        ChangeReportAlgorithm(),
        ExtractSamplesAlgorithm(),
        ClassifyEmbeddingAlgorithm(),
        ClassifyOverYearsAlgorithm(),
        ClusterEmbeddingAlgorithm(),
        RegressEmbeddingAlgorithm(),
    ]


def __getattr__(name: str) -> object:
    """Lazily expose the algorithm classes as attributes without eager imports."""
    if name == "LoadEmbeddingAlgorithm":
        from alphaearth_toolbox.algorithms.load_embedding import LoadEmbeddingAlgorithm

        return LoadEmbeddingAlgorithm
    if name == "LoadYearsAlgorithm":
        from alphaearth_toolbox.algorithms.load_years import LoadYearsAlgorithm

        return LoadYearsAlgorithm
    if name == "EmbeddingToRgbAlgorithm":
        from alphaearth_toolbox.algorithms.embedding_rgb import EmbeddingToRgbAlgorithm

        return EmbeddingToRgbAlgorithm
    if name == "SimilaritySearchAlgorithm":
        from alphaearth_toolbox.algorithms.similarity_search import SimilaritySearchAlgorithm

        return SimilaritySearchAlgorithm
    if name == "ChangeDetectionAlgorithm":
        from alphaearth_toolbox.algorithms.change_detection import ChangeDetectionAlgorithm

        return ChangeDetectionAlgorithm
    if name == "ChangeOverYearsAlgorithm":
        from alphaearth_toolbox.algorithms.change_years import ChangeOverYearsAlgorithm

        return ChangeOverYearsAlgorithm
    if name == "ChangeReportAlgorithm":
        from alphaearth_toolbox.algorithms.change_report import ChangeReportAlgorithm

        return ChangeReportAlgorithm
    if name == "ExtractSamplesAlgorithm":
        from alphaearth_toolbox.algorithms.extract_samples import ExtractSamplesAlgorithm

        return ExtractSamplesAlgorithm
    if name == "ClassifyEmbeddingAlgorithm":
        from alphaearth_toolbox.algorithms.classify import ClassifyEmbeddingAlgorithm

        return ClassifyEmbeddingAlgorithm
    if name == "ClassifyOverYearsAlgorithm":
        from alphaearth_toolbox.algorithms.classify_years import ClassifyOverYearsAlgorithm

        return ClassifyOverYearsAlgorithm
    if name == "ClusterEmbeddingAlgorithm":
        from alphaearth_toolbox.algorithms.cluster import ClusterEmbeddingAlgorithm

        return ClusterEmbeddingAlgorithm
    if name == "RegressEmbeddingAlgorithm":
        from alphaearth_toolbox.algorithms.regress import RegressEmbeddingAlgorithm

        return RegressEmbeddingAlgorithm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
