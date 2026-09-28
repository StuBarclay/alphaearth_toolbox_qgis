"""Unit tests for the QGIS-free raster-source resolution policy."""

from __future__ import annotations

from alphaearth_toolbox.algorithms._raster_source import (
    build_vrt_xml,
    resolve_raster_source,
    source_candidates,
)


def test_source_candidates_dedupes_and_strips_pipe() -> None:
    candidates = source_candidates("/data/x.tif|layername=a", "/data/x.tif")
    assert candidates == ["/data/x.tif", "/data/x.tif|layername=a"]


def test_source_candidates_orders_specific_first() -> None:
    candidates = source_candidates("raw", "uri")
    assert candidates == ["raw", "uri"]


def test_resolve_prefers_existing_file() -> None:
    chosen = resolve_raster_source(
        ["/data/a.tif", "/data/b.tif"],
        exists=lambda c: c == "/data/b.tif",
        wrap_descriptor=lambda c: None,
        materialise=lambda: "materialised",
    )
    assert chosen == "/data/b.tif"


def test_resolve_wraps_descriptor_when_no_file() -> None:
    chosen = resolve_raster_source(
        ["OpenFileGDB:x.gdb:layer"],
        exists=lambda c: False,
        wrap_descriptor=lambda c: "/tmp/wrap.vrt",
        materialise=lambda: "materialised",
    )
    assert chosen == "/tmp/wrap.vrt"


def test_resolve_materialises_as_last_resort() -> None:
    chosen = resolve_raster_source(
        ["weird://source"],
        exists=lambda c: False,
        wrap_descriptor=lambda c: None,
        materialise=lambda: "materialised",
    )
    assert chosen == "materialised"


def test_build_vrt_xml_is_wellformed() -> None:
    xml = build_vrt_xml(
        width=2,
        height=3,
        source="/vsicurl/https://x/y.tif",
        bands=[("Float32", None), ("Float32", -9999.0)],
        srs_wkt="PROJCS[...]",
        geotransform=[0.0, 10.0, 0.0, 30.0, 0.0, -10.0],
    )
    assert xml.startswith('<VRTDataset rasterXSize="2" rasterYSize="3">')
    assert xml.rstrip().endswith("</VRTDataset>")
    assert xml.count("<VRTRasterBand") == 2
    assert 'relativeToVRT="0"' in xml
    assert "<NoDataValue>-9999.0</NoDataValue>" in xml
