"""Unit tests for the pure intake configuration/helpers (no network)."""

from __future__ import annotations

from typing import Any

import pytest

from alphaearth_toolbox.aecore import intake


def test_dataset_constants() -> None:
    assert intake.EMBEDDING_BANDS == 64
    assert intake.AVAILABLE_YEARS[0] == 2017
    assert intake.AVAILABLE_YEARS[-1] == 2025
    assert len(intake.AVAILABLE_YEARS) == 9
    assert intake.DEFAULT_CRS == "EPSG:3577"
    assert intake.GCS_BUCKET == "alphaearth_foundations"
    assert intake.GCS_PREFIX == "satellite_embedding/v1/annual"
    assert intake.QUANT_SCALE == 127.5
    assert intake.FILL_VALUE == -128


def test_vsicurl_wraps_http_and_is_idempotent() -> None:
    assert intake.vsicurl("https://x/y.tif") == "/vsicurl/https://x/y.tif"
    assert intake.vsicurl("http://x/y.tif") == "/vsicurl/http://x/y.tif"
    assert intake.vsicurl("/vsicurl/https://x/y.tif") == "/vsicurl/https://x/y.tif"
    # A local path is left untouched.
    assert intake.vsicurl("/data/embed.tif") == "/data/embed.tif"


def test_is_supported_year() -> None:
    assert intake.is_supported_year(2017)
    assert intake.is_supported_year(2025)
    assert not intake.is_supported_year(2016)
    assert not intake.is_supported_year(2026)


def test_years_from_indices_maps_sorts_and_dedupes() -> None:
    # Enum indices are positions into AVAILABLE_YEARS (2017 == 0 ... 2025 == 8).
    assert intake.years_from_indices([0]) == [2017]
    assert intake.years_from_indices([8, 0, 4]) == [2017, 2021, 2025]
    # Out-of-order and duplicate indices collapse to sorted unique years.
    assert intake.years_from_indices([4, 4, 1]) == [2018, 2021]
    assert intake.years_from_indices([]) == []


def test_years_from_indices_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        intake.years_from_indices([len(intake.AVAILABLE_YEARS)])
    with pytest.raises(ValueError, match="out of range"):
        intake.years_from_indices([-1])


def test_year_raster_name() -> None:
    assert intake.year_raster_name(2018) == "alphaearth_2018.tif"
    assert intake.year_raster_name(2025) == "alphaearth_2025.tif"


def test_utm_zone_number() -> None:
    # Zone 1 begins at -180; each zone spans 6 degrees.
    assert intake.utm_zone_number(-180.0) == 1
    assert intake.utm_zone_number(0.0) == 31
    # Sydney (~151 E) is in zone 56.
    assert intake.utm_zone_number(151.0) == 56
    # Clamped to the valid 1-60 range at the antimeridian.
    assert intake.utm_zone_number(180.0) == 60


def test_utm_zones_for_bbox_south() -> None:
    # An Australian AOI spanning ~150-152 E, all south of the equator.
    zones = intake.utm_zones_for_bbox(150.5, -34.0, 151.5, -33.0)
    assert zones == ["56S"]


def test_utm_zones_for_bbox_spans_two_zones() -> None:
    zones = intake.utm_zones_for_bbox(149.0, -35.0, 151.0, -34.0)
    assert zones == ["55S", "56S"]


def test_utm_zones_for_bbox_edge_does_not_leak_next_zone() -> None:
    # 150-156 E is exactly zone 56's window; an eastern edge landing on the
    # 56/57 boundary (156 E) must not pull in the (empty) next zone.
    zones = intake.utm_zones_for_bbox(150.0, -34.0, 156.0, -33.0)
    assert zones == ["56S"]


def test_utm_zones_for_bbox_straddles_equator() -> None:
    zones = intake.utm_zones_for_bbox(150.5, -1.0, 151.5, 1.0)
    assert set(zones) == {"56S", "56N"}


def test_zone_prefix_and_object_url() -> None:
    prefix = intake.zone_prefix(2021, "56S")
    assert prefix == "satellite_embedding/v1/annual/2021/56S/"
    url = intake.gcs_object_url(f"{prefix}abc-0-0.tif")
    assert url == (
        "/vsicurl/https://storage.googleapis.com/alphaearth_foundations/"
        "satellite_embedding/v1/annual/2021/56S/abc-0-0.tif"
    )


def test_normalise_bucket_strips_scheme() -> None:
    assert intake.normalise_bucket("gs://alphaearth_foundations/") == "alphaearth_foundations"
    assert intake.normalise_bucket("alphaearth_foundations") == "alphaearth_foundations"


def test_bboxes_intersect() -> None:
    assert intake.bboxes_intersect((0, 0, 10, 10), (5, 5, 15, 15))
    # Only touching along an edge does not count as intersecting.
    assert not intake.bboxes_intersect((0, 0, 10, 10), (10, 0, 20, 10))
    assert not intake.bboxes_intersect((0, 0, 10, 10), (20, 20, 30, 30))


def test_parse_listing_keeps_only_cogs_and_returns_token() -> None:
    doc = {
        "items": [
            {"name": "satellite_embedding/v1/annual/2021/56S/a-0-0.tif"},
            {"name": "satellite_embedding/v1/annual/2021/56S/b-0-1.tiff"},
            {"name": "satellite_embedding/v1/annual/2021/56S/"},  # folder placeholder
            {"name": "satellite_embedding/v1/annual/2021/56S/notes.json"},
        ],
        "nextPageToken": "PAGE2",
    }
    names, token = intake.parse_listing(doc)
    assert names == [
        "satellite_embedding/v1/annual/2021/56S/a-0-0.tif",
        "satellite_embedding/v1/annual/2021/56S/b-0-1.tiff",
    ]
    assert token == "PAGE2"


def test_parse_listing_no_token() -> None:
    names, token = intake.parse_listing({"items": []})
    assert names == []
    assert token is None


def test_list_zone_tiles_paginates_and_sorts() -> None:
    pages: list[dict[str, Any]] = [
        {
            "items": [{"name": "satellite_embedding/v1/annual/2021/56S/b.tif"}],
            "nextPageToken": "T2",
        },
        {"items": [{"name": "satellite_embedding/v1/annual/2021/56S/a.tif"}]},
    ]
    calls: list[str] = []

    def fake_fetch(url: str) -> dict[str, Any]:
        calls.append(url)
        return pages[len(calls) - 1]

    names = intake.list_zone_tiles(2021, "56S", fetch_json=fake_fetch)
    assert names == [
        "satellite_embedding/v1/annual/2021/56S/a.tif",
        "satellite_embedding/v1/annual/2021/56S/b.tif",
    ]
    # Two calls (one per page); the second carried the page token.
    assert len(calls) == 2
    assert "pageToken=T2" in calls[1]
    assert "prefix=satellite_embedding%2Fv1%2Fannual%2F2021%2F56S%2F" in calls[0]


def test_list_zone_tiles_rejects_unsupported_year() -> None:
    with pytest.raises(ValueError, match="no data for year"):
        intake.list_zone_tiles(1999, "56S", fetch_json=lambda _url: {"items": []})


def test_grid_dimensions_exact_multiple() -> None:
    width, height, snapped = intake.grid_dimensions((0.0, 0.0, 100.0, 100.0), 10.0)
    assert (width, height) == (10, 10)
    assert snapped == (0.0, 0.0, 100.0, 100.0)


def test_grid_dimensions_rounds_out() -> None:
    width, height, snapped = intake.grid_dimensions((0.0, 0.0, 95.0, 95.0), 10.0)
    assert (width, height) == (10, 10)
    # Anchored at the top-left corner, expanded to cover the AOI.
    assert snapped == (0.0, -5.0, 100.0, 95.0)


def test_grid_dimensions_guards() -> None:
    with pytest.raises(ValueError, match="resolution must be positive"):
        intake.grid_dimensions((0.0, 0.0, 10.0, 10.0), 0.0)
    with pytest.raises(ValueError, match="positive width and height"):
        intake.grid_dimensions((10.0, 0.0, 0.0, 10.0), 10.0)


# --------------------------------------------------------------------------- #
# Scene grouping and conservative culling                                     #
# --------------------------------------------------------------------------- #

_P = "satellite_embedding/v1/annual/2021/56S/"


def test_parse_tile_offsets() -> None:
    assert intake.parse_tile_offsets(f"{_P}abc-0000008192-0000000000.tiff") == (8192, 0)
    assert intake.parse_tile_offsets(f"{_P}abc-0-16384.tif") == (0, 16384)
    # A name without the two trailing integer groups is unparseable.
    assert intake.parse_tile_offsets(f"{_P}abc.tif") is None
    assert intake.parse_tile_offsets(f"{_P}abc-0-0.json") is None


def test_scene_key_groups_tiles_of_one_scene() -> None:
    a = f"{_P}sceneA-0000000000-0000000000.tiff"
    b = f"{_P}sceneA-0000008192-0000000000.tiff"
    c = f"{_P}sceneB-0000000000-0000000000.tiff"
    assert intake.scene_key(a) == intake.scene_key(b) == f"{_P}sceneA"
    assert intake.scene_key(c) == f"{_P}sceneB"
    # An unparseable name is its own scene (so it is always scanned, never culled).
    assert intake.scene_key(f"{_P}weird.tif") == f"{_P}weird.tif"


def test_group_by_scene_preserves_order() -> None:
    names = [
        f"{_P}sceneA-0-0.tiff",
        f"{_P}sceneB-0-0.tiff",
        f"{_P}sceneA-8192-0.tiff",
    ]
    groups = intake.group_by_scene(names)
    assert set(groups) == {f"{_P}sceneA", f"{_P}sceneB"}
    assert groups[f"{_P}sceneA"] == [f"{_P}sceneA-0-0.tiff", f"{_P}sceneA-8192-0.tiff"]


def test_conservative_scene_bounds_contains_every_tile_either_axis_mapping() -> None:
    # Anchor tile at offset (0, 0): a 0.1x0.1 deg, 8192x8192 px footprint.
    anchor_bounds = (150.0, -34.0, 150.1, -33.9)
    px = (8192, 8192)
    tile = 8192
    offsets = [(0, 0), (tile, 0), (0, tile), (tile, tile), (2 * tile, 0)]
    box = intake.conservative_scene_bounds(anchor_bounds, px, (0, 0), offsets)

    # The true footprint of each tile, for BOTH possible pixel-axis -> lon/lat
    # mappings, must lie inside the (deliberately generous) conservative box.
    dlon = anchor_bounds[2] - anchor_bounds[0]
    dlat = anchor_bounds[3] - anchor_bounds[1]
    for off_a, off_b in offsets:
        for maps_a_to_lon in (True, False):
            steps_lon = (off_a if maps_a_to_lon else off_b) / tile
            steps_lat = (off_b if maps_a_to_lon else off_a) / tile
            true_fp = (
                anchor_bounds[0] + steps_lon * dlon,
                anchor_bounds[1] + steps_lat * dlat,
                anchor_bounds[2] + steps_lon * dlon,
                anchor_bounds[3] + steps_lat * dlat,
            )
            assert box[0] <= true_fp[0] and box[1] <= true_fp[1]
            assert box[2] >= true_fp[2] and box[3] >= true_fp[3]


def test_conservative_scene_bounds_culls_far_and_keeps_near() -> None:
    anchor_bounds = (150.0, -34.0, 150.1, -33.9)
    offsets = [(0, 0), (8192, 0), (0, 8192)]
    box = intake.conservative_scene_bounds(anchor_bounds, (8192, 8192), (0, 0), offsets)
    # A far-away AOI is safely culled ...
    assert not intake.bboxes_intersect(box, (10.0, 10.0, 10.1, 10.1))
    # ... while an AOI over the anchor keeps the scene.
    assert intake.bboxes_intersect(box, (150.05, -33.95, 150.06, -33.94))


def test_conservative_scene_bounds_single_tile_is_anchor_bounds() -> None:
    # No sibling offsets -> nothing to extrapolate -> just the anchor's own box.
    assert intake.conservative_scene_bounds((1.0, 2.0, 3.0, 4.0), (8192, 8192), (0, 0), []) == (
        1.0,
        2.0,
        3.0,
        4.0,
    )


# --------------------------------------------------------------------------- #
# Persistent caches (footprint + zone index)                                  #
# --------------------------------------------------------------------------- #


def test_partition_by_cache_splits_preserving_order() -> None:
    have, missing = intake.partition_by_cache(["a", "b", "c"], {"b": (0.0, 0.0, 1.0, 1.0)})
    assert have == ["b"]
    assert missing == ["a", "c"]


def test_tiles_intersecting_filters_and_sorts() -> None:
    footprints = {
        "b-0-0.tif": (10.0, 10.0, 11.0, 11.0),
        "a-0-0.tif": (0.0, 0.0, 1.0, 1.0),
    }
    assert intake.tiles_intersecting(footprints, (0.5, 0.5, 0.6, 0.6)) == ["a-0-0.tif"]


def test_footprint_cache_roundtrip(tmp_path: Any) -> None:
    footprints = {"abc-0-0.tif": (150.0, -34.0, 150.1, -33.9)}
    intake.save_footprint_cache(2021, "56S", footprints, cache_dir=tmp_path)
    assert intake.load_footprint_cache(2021, "56S", cache_dir=tmp_path) == footprints


def test_footprint_cache_bucket_mismatch_is_ignored(tmp_path: Any) -> None:
    intake.save_footprint_cache(
        2021, "56S", {"a-0-0.tif": (1.0, 2.0, 3.0, 4.0)}, cache_dir=tmp_path, bucket="other_bucket"
    )
    # Written for a different bucket -> a default-bucket load must not reuse it.
    assert intake.load_footprint_cache(2021, "56S", cache_dir=tmp_path) == {}


def test_footprint_cache_absent_returns_empty(tmp_path: Any) -> None:
    assert intake.load_footprint_cache(2021, "56S", cache_dir=tmp_path) == {}


def test_zone_index_roundtrip(tmp_path: Any) -> None:
    index = intake.ZoneIndex(
        names=[f"{_P}sceneA-0-0.tiff", f"{_P}sceneA-8192-0.tiff"],
        anchors={
            f"{_P}sceneA": intake.SceneAnchor(
                name=f"{_P}sceneA-0-0.tiff",
                bounds=(150.0, -34.0, 150.1, -33.9),
                pixels=(8192, 8192),
                offset=(0, 0),
            )
        },
    )
    intake.save_zone_index(2021, "56S", index, cache_dir=tmp_path)
    loaded = intake.load_zone_index(2021, "56S", cache_dir=tmp_path)
    assert loaded is not None
    assert loaded.names == index.names
    assert loaded.anchors == index.anchors


def test_zone_index_missing_returns_none(tmp_path: Any) -> None:
    assert intake.load_zone_index(2021, "56S", cache_dir=tmp_path) is None


def test_zone_index_bucket_mismatch_returns_none(tmp_path: Any) -> None:
    intake.save_zone_index(
        2021, "56S", intake.ZoneIndex(names=["x"], anchors={}), cache_dir=tmp_path, bucket="other"
    )
    assert intake.load_zone_index(2021, "56S", cache_dir=tmp_path) is None


def test_persistent_cache_dir_is_cache_name_under_base(tmp_path: Any) -> None:
    # The durable, opt-in cache directory is just the cache sub-folder under a
    # stable base (the QGIS profile at runtime); it accepts a str or a Path and
    # does not touch the filesystem.
    from pathlib import Path

    assert intake.persistent_cache_dir(tmp_path) == tmp_path / intake.CACHE_DIR_NAME
    assert intake.persistent_cache_dir(str(tmp_path)) == Path(tmp_path) / intake.CACHE_DIR_NAME
    # A durable base is never the volatile OS-temp default.
    assert intake.persistent_cache_dir(tmp_path) != intake.default_cache_dir()


def test_cache_roundtrips_through_persistent_cache_dir(tmp_path: Any) -> None:
    # End-to-end contract the year fetch relies on: footprints and the zone
    # index written under a persistent cache dir are read back from that same
    # dir (so a later session reuses the discovery work instead of rescanning).
    cache = intake.persistent_cache_dir(tmp_path)
    footprints = {"abc-0-0.tif": (150.0, -34.0, 150.1, -33.9)}
    index = intake.ZoneIndex(names=["abc-0-0.tif"], anchors={})

    intake.save_footprint_cache(2021, "56S", footprints, cache_dir=cache)
    intake.save_zone_index(2021, "56S", index, cache_dir=cache)

    assert intake.load_footprint_cache(2021, "56S", cache_dir=cache) == footprints
    loaded = intake.load_zone_index(2021, "56S", cache_dir=cache)
    assert loaded is not None and loaded.names == index.names
