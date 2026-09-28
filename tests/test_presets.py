"""Unit tests for the pure-stdlib preset registries (no NumPy / QGIS / GDAL)."""

from __future__ import annotations

import pytest

from alphaearth_toolbox.aecore import presets
from alphaearth_toolbox.aecore.change import METRICS as CHANGE_METRICS
from alphaearth_toolbox.aecore.change import MODES as CHANGE_MODES
from alphaearth_toolbox.aecore.rgb import METHODS as RGB_METHODS
from alphaearth_toolbox.aecore.similarity import AGGREGATIONS as SIM_AGGREGATIONS

_ALL_REGISTRIES = (
    presets.RGB_PRESETS,
    presets.SIMILARITY_PRESETS,
    presets.CLUSTER_PRESETS,
    presets.CHANGE_PRESETS,
)


def test_every_registry_is_non_empty() -> None:
    for registry in _ALL_REGISTRIES:
        assert len(registry) >= 1


def test_keys_unique_within_each_registry() -> None:
    for registry in _ALL_REGISTRIES:
        keys = [preset.key for preset in registry]
        assert len(keys) == len(set(keys))


def test_labels_and_descriptions_present() -> None:
    for registry in _ALL_REGISTRIES:
        for preset in registry:
            assert preset.label.strip()
            assert preset.description.strip()
            # Keys are lower-case tokens the wizard/algorithm match on.
            assert preset.key == preset.key.strip().lower()


def test_preset_labels_matches_registry_order() -> None:
    labels = presets.preset_labels(presets.CLUSTER_PRESETS)
    assert labels == [p.label for p in presets.CLUSTER_PRESETS]
    assert labels[0] == "Land cover (6 groups)"


def test_preset_by_index_round_trip() -> None:
    for i, preset in enumerate(presets.SIMILARITY_PRESETS):
        assert presets.preset_by_index(presets.SIMILARITY_PRESETS, i) is preset


def test_preset_by_index_out_of_range_raises() -> None:
    with pytest.raises(IndexError, match="out of range"):
        presets.preset_by_index(presets.RGB_PRESETS, len(presets.RGB_PRESETS))


def test_preset_by_key_case_insensitive() -> None:
    got = presets.preset_by_key(presets.CLUSTER_PRESETS, "WUI_FUEL")
    assert got.key == "wui_fuel"
    assert got.n_clusters == 10


def test_preset_by_key_unknown_raises() -> None:
    with pytest.raises(KeyError, match="no preset"):
        presets.preset_by_key(presets.CHANGE_PRESETS, "nope")


# --------------------------------------------------------------------------- #
# Presets must only carry values the algorithms actually accept                #
# --------------------------------------------------------------------------- #


def test_rgb_presets_valid_method_and_bands() -> None:
    for preset in presets.RGB_PRESETS:
        assert preset.method in RGB_METHODS
        if preset.method == "bands":
            assert preset.band_indices is not None
            assert len(preset.band_indices) == 3
            assert all(0 <= b < 64 for b in preset.band_indices)
        else:
            assert preset.band_indices is None
        assert 0.0 <= preset.low_percent < preset.high_percent <= 100.0


def test_similarity_presets_valid_fields() -> None:
    for preset in presets.SIMILARITY_PRESETS:
        assert preset.aggregation in ("mean", "medoid")
        assert isinstance(preset.rescale, bool)
        assert 0.0 <= preset.threshold <= 1.0


def test_cluster_presets_sane_counts() -> None:
    for preset in presets.CLUSTER_PRESETS:
        assert preset.n_clusters >= 2
        assert preset.subsample > 0
        assert preset.seed >= 0


def test_change_presets_valid_mode_and_metric() -> None:
    for preset in presets.CHANGE_PRESETS:
        assert preset.mode in CHANGE_MODES
        assert preset.metric in CHANGE_METRICS


def test_every_preset_value_maps_to_a_dropdown_index() -> None:
    # The wizard turns each preset's semantic value into a dropdown row via
    # ``order.index(value)`` on these canonical tuples; if any preset carried a
    # value outside its tuple the wizard would silently fall back to row 0. This
    # guards that contract without needing QGIS to import the dialog.
    for rgb_preset in presets.RGB_PRESETS:
        assert rgb_preset.method in RGB_METHODS
    for sim_preset in presets.SIMILARITY_PRESETS:
        assert sim_preset.aggregation in SIM_AGGREGATIONS
    for change_preset in presets.CHANGE_PRESETS:
        assert change_preset.metric in CHANGE_METRICS
        assert change_preset.mode in CHANGE_MODES


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
