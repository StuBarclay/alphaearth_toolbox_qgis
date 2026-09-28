"""Named starting-point presets for the AlphaEarth toolbox algorithms.

AlphaEarth Satellite Embedding V1 is a *learned* 64-D representation: no band
carries a fixed physical meaning (there is no "red" or "NIR" band), so there is
no universally correct set of parameters for a given task. What there *is* is a
set of sensible starting points for the questions users most often ask -- "show
me this scene", "find more pixels like these", "split this area into land-cover
groups", "how much changed" -- and, for the grouping/similarity tasks, defaults
that tend to work for a few recurring themes (wildland-urban-interface fuel,
built-up land, tree canopy).

This module holds those presets as plain frozen dataclasses in small ordered
registries, with lookup helpers by key and by position. It is pure standard
library -- no NumPy, GDAL, scikit-learn or QGIS -- so the algorithm wrappers and
the wizard can both read it, and it is trivially unit-testable. Each preset
stores *semantic* values (``method="pca"``, ``metric="cosine"``) rather than
enum indices, so it never has to track a dropdown's option order.

The presets are guidance, not guarantees: because the embedding has no fixed
semantics, a "canopy" cluster preset only makes the grouping *finer* in a way
that usually helps separate vegetation structure -- it does not label anything
as canopy. Users should treat every preset as a first guess to refine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

# --------------------------------------------------------------------------- #
# Preset dataclasses                                                          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RgbPreset:
    """A recipe for rendering a 64-band embedding to a 3-band RGB preview.

    Band-triplet presets are *visualisation aids*: different bands emphasise
    different structure in the learned representation, but the numbers are not
    physical channels. ``pca`` (the default) is the most informative general
    choice because it packs the most variance into three channels.
    """

    key: str
    label: str
    description: str
    method: str  # "pca" or "bands"
    band_indices: tuple[int, int, int] | None
    low_percent: float
    high_percent: float


@dataclass(frozen=True)
class SimilarityPreset:
    """Defaults for the *Similarity search* ("find more like this") algorithm."""

    key: str
    label: str
    description: str
    aggregation: str  # "mean" or "medoid"
    rescale: bool
    threshold: float


@dataclass(frozen=True)
class ClusterPreset:
    """Defaults for the *Cluster* (unsupervised K-means) algorithm."""

    key: str
    label: str
    description: str
    n_clusters: int
    subsample: int
    seed: int


@dataclass(frozen=True)
class ChangePreset:
    """Defaults for the *Change / trajectory* algorithm."""

    key: str
    label: str
    description: str
    mode: str  # "pairwise", "trajectory" or "anomaly"
    metric: str  # "cosine" or "euclidean"


# --------------------------------------------------------------------------- #
# Registries (ordered; first entry is the sensible default)                   #
# --------------------------------------------------------------------------- #

RGB_PRESETS: tuple[RgbPreset, ...] = (
    RgbPreset(
        key="pca",
        label="PCA (recommended)",
        description=(
            "Project onto the top three principal components, 2-98% stretch. The most "
            "informative general view: visually distinct surfaces get distinct colours."
        ),
        method="pca",
        band_indices=None,
        low_percent=2.0,
        high_percent=98.0,
    ),
    RgbPreset(
        key="pca_tight",
        label="PCA, high contrast",
        description=(
            "Top three principal components with a tighter 5-95% stretch, for scenes "
            "whose interesting variation sits in a narrow value range."
        ),
        method="pca",
        band_indices=None,
        low_percent=5.0,
        high_percent=95.0,
    ),
    RgbPreset(
        key="bands_first",
        label="First three bands",
        description=(
            "Map embedding bands 1, 2 and 3 straight to R, G, B. A stable, model-free "
            "mapping that always renders the same regardless of the pixels in view."
        ),
        method="bands",
        band_indices=(0, 1, 2),
        low_percent=2.0,
        high_percent=98.0,
    ),
    RgbPreset(
        key="bands_spread",
        label="Spread bands (1, 22, 43)",
        description=(
            "Three bands spread across the 64 dimensions, a model-free alternative view "
            "that can separate surfaces the first three bands render similarly."
        ),
        method="bands",
        band_indices=(0, 21, 42),
        low_percent=2.0,
        high_percent=98.0,
    ),
)

SIMILARITY_PRESETS: tuple[SimilarityPreset, ...] = (
    SimilarityPreset(
        key="find_similar",
        label="Find similar (general)",
        description=(
            "Mean of the seed pixels, similarity rescaled to 0-1, no mask. A good "
            "general 'more like this' score to inspect before choosing a threshold."
        ),
        aggregation="mean",
        rescale=True,
        threshold=0.0,
    ),
    SimilarityPreset(
        key="strict_match",
        label="Strict match (tight)",
        description=(
            "Medoid of the seeds (most representative single seed) with a high 0.85 "
            "mask threshold -- picks out only pixels very close to the target surface."
        ),
        aggregation="medoid",
        rescale=True,
        threshold=0.85,
    ),
    SimilarityPreset(
        key="broad_match",
        label="Broad match (inclusive)",
        description=(
            "Mean of the seeds with a lower 0.6 mask threshold -- a more inclusive "
            "'more like this' layer that accepts looser matches."
        ),
        aggregation="mean",
        rescale=True,
        threshold=0.6,
    ),
)

CLUSTER_PRESETS: tuple[ClusterPreset, ...] = (
    ClusterPreset(
        key="landcover",
        label="Land cover (6 groups)",
        description=(
            "Six clusters -- a broad land-cover split (e.g. water, built-up, bare, and "
            "a few vegetation groups). A good first pass over a mixed scene."
        ),
        n_clusters=6,
        subsample=100_000,
        seed=42,
    ),
    ClusterPreset(
        key="wui_fuel",
        label="WUI fuel detail (10 groups)",
        description=(
            "Ten clusters to resolve finer vegetation-structure differences that map to "
            "fuel types at the wildland-urban interface. More groups, more to interpret."
        ),
        n_clusters=10,
        subsample=150_000,
        seed=42,
    ),
    ClusterPreset(
        key="builtup",
        label="Built-up split (8 groups)",
        description=(
            "Eight clusters, aimed at separating built-up and hard surfaces from open "
            "and vegetated land in and around settlements."
        ),
        n_clusters=8,
        subsample=120_000,
        seed=42,
    ),
    ClusterPreset(
        key="canopy",
        label="Canopy / vegetation (12 groups)",
        description=(
            "Twelve clusters to draw out gradations within vegetation (canopy density, "
            "structure). Finer still; expect several vegetation groups to interpret."
        ),
        n_clusters=12,
        subsample=150_000,
        seed=42,
    ),
)

CHANGE_PRESETS: tuple[ChangePreset, ...] = (
    ChangePreset(
        key="bitemporal",
        label="Bi-temporal (first vs last)",
        description=(
            "Cosine distance between the first and last year -- the classic two-date "
            "change map. Higher means more change."
        ),
        mode="pairwise",
        metric="cosine",
    ),
    ChangePreset(
        key="trajectory",
        label="Trajectory (cumulative)",
        description=(
            "Cumulative cosine distance across every consecutive year, so year-on-year "
            "churn scores higher than a single step change. Needs three or more years."
        ),
        mode="trajectory",
        metric="cosine",
    ),
    ChangePreset(
        key="anomaly",
        label="Anomaly vs baseline",
        description=(
            "Cosine distance of the most recent year from the mean of all supplied "
            "years -- how far the latest state departs from the period average."
        ),
        mode="anomaly",
        metric="cosine",
    ),
)


# --------------------------------------------------------------------------- #
# Lookup helpers                                                              #
# --------------------------------------------------------------------------- #

_Preset = TypeVar("_Preset", RgbPreset, SimilarityPreset, ClusterPreset, ChangePreset)


def preset_labels(presets: Sequence[_Preset]) -> list[str]:
    """Return the presets' labels in registry order (for a dropdown's options)."""
    return [preset.label for preset in presets]


def preset_by_index(presets: Sequence[_Preset], index: int) -> _Preset:
    """Return the preset at ``index`` in the registry (as a dropdown selection).

    Raises:
        IndexError: If ``index`` is out of range for ``presets``.
    """
    if not 0 <= index < len(presets):
        raise IndexError(f"preset index {index} out of range for {len(presets)} presets.")
    return presets[index]


def preset_by_key(presets: Sequence[_Preset], key: str) -> _Preset:
    """Return the preset whose ``key`` matches (case-insensitively).

    Raises:
        KeyError: If no preset in ``presets`` has that key.
    """
    wanted = key.strip().lower()
    for preset in presets:
        if preset.key == wanted:
            return preset
    available = ", ".join(preset.key for preset in presets)
    raise KeyError(f"no preset {key!r}; available keys: {available}.")
