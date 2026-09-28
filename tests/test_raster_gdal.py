"""GDAL-backed I/O tests for :mod:`alphaearth_toolbox.aecore._raster`.

Unlike the rest of the suite, these exercise the real GDAL read/write/warp paths,
so the whole module is skipped where GDAL (``osgeo``) cannot be imported -- i.e.
the pure "checks" CI job and a plain developer virtualenv. They run in the
QGIS/GDAL container job (and inside QGIS), where GDAL is always present.

Two things are checked: that ``write_geotiff`` / ``read_cube`` round-trip a cube
faithfully (values, dtype promotion, geotransform, projection, no-data), and that
the results are byte-for-value compatible with an independent reader/writer
(rasterio) when it is available -- the "parity" the toolbox promises by using GDAL
directly instead of rasterio. The warp and footprint helpers are covered against
small local GeoTIFFs so no network is touched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

# Skip the entire module unless GDAL is importable.
gdal = pytest.importorskip("osgeo.gdal", reason="GDAL (osgeo) not available")
gdalconst = pytest.importorskip("osgeo.gdalconst", reason="GDAL (osgeo) not available")

from alphaearth_toolbox.aecore import _raster  # noqa: E402  (after importorskip)

# A projected geotransform in metres: origin (100, 200), 10 m pixels, north-up.
_GEOTRANSFORM = (100.0, 10.0, 0.0, 200.0, 0.0, -10.0)


def _wkt(epsg: int = 3577) -> str:
    """Return the WKT for an EPSG code via GDAL's own SRS machinery."""
    from osgeo import osr

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    return str(srs.ExportToWkt())


def _same_crs(wkt_a: str, wkt_b: str) -> bool:
    """True if two WKT strings describe the same CRS (avoids string-equality churn)."""
    from osgeo import osr

    a = osr.SpatialReference()
    a.ImportFromWkt(wkt_a)
    b = osr.SpatialReference()
    b.ImportFromWkt(wkt_b)
    return bool(a.IsSame(b))


def test_write_then_read_roundtrips_a_cube(tmp_path: Path) -> None:
    cube = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    path = str(tmp_path / "cube.tif")

    _raster.write_geotiff(path, cube, geotransform=_GEOTRANSFORM, wkt=_wkt(), nodata=-1.0)
    read, geotransform, wkt, nodata = _raster.read_cube(path)

    assert read.shape == (2, 3, 4)
    assert read.dtype == np.float32  # read_cube now defaults to float32
    assert np.allclose(read, cube)
    assert geotransform == _GEOTRANSFORM
    assert nodata == -1.0
    assert _same_crs(wkt, _wkt())


def test_single_band_2d_is_promoted_and_nodata_optional(tmp_path: Path) -> None:
    array = np.arange(6, dtype=np.int32).reshape(2, 3)
    path = str(tmp_path / "single.tif")

    _raster.write_geotiff(path, array, geotransform=_GEOTRANSFORM, wkt=_wkt(), nodata=None)
    read, _, _, nodata = _raster.read_cube(path)

    assert read.shape == (1, 2, 3)
    assert np.allclose(read[0], array)
    assert nodata is None


def test_write_geotiff_preserves_integer_band_type(tmp_path: Path) -> None:
    array = np.arange(6, dtype=np.int32).reshape(2, 3)
    path = str(tmp_path / "int.tif")

    _raster.write_geotiff(path, array, geotransform=_GEOTRANSFORM, wkt=_wkt())
    dataset = gdal.Open(path)
    try:
        assert dataset.RasterCount == 1
        assert dataset.GetRasterBand(1).DataType == gdal.GDT_Int32
    finally:
        dataset = None


def test_read_cube_matches_raw_gdal_read(tmp_path: Path) -> None:
    cube = (np.random.default_rng(0).standard_normal((3, 5, 6))).astype(np.float32)
    path = str(tmp_path / "raw.tif")
    _raster.write_geotiff(path, cube, geotransform=_GEOTRANSFORM, wkt=_wkt(), nodata=None)

    read, geotransform, _, _ = _raster.read_cube(path)

    dataset = gdal.Open(path)
    try:
        raw = np.asarray(dataset.ReadAsArray(), dtype=np.float64)
        raw_gt = tuple(float(v) for v in dataset.GetGeoTransform())
    finally:
        dataset = None
    assert np.allclose(read, raw)
    assert geotransform == raw_gt


def test_read_cube_dtype_is_selectable(tmp_path: Path) -> None:
    # int16 source so a "native" read is visibly distinct from float32/float64.
    array = np.arange(6, dtype=np.int16).reshape(2, 3)
    path = str(tmp_path / "dtype.tif")
    _raster.write_geotiff(path, array, geotransform=_GEOTRANSFORM, wkt=_wkt())

    default_read, _, _, _ = _raster.read_cube(path)
    assert default_read.dtype == np.float32  # the memory-halving default

    float64_read, _, _, _ = _raster.read_cube(path, dtype=np.float64)
    assert float64_read.dtype == np.float64

    native_read, _, _, _ = _raster.read_cube(path, dtype=None)
    assert native_read.dtype == np.int16  # None preserves the source dtype

    # Whatever the dtype, the values (and the promotion of a 2-D band to 3-D)
    # are identical.
    for read in (default_read, float64_read, native_read):
        assert read.shape == (1, 2, 3)
        assert np.array_equal(read[0], array)


def test_warp_cube_to_grid_downsamples_local_source(tmp_path: Path) -> None:
    cube = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
    # Bounds: x 0..40, y 0..40 with 10 m pixels.
    geotransform = (0.0, 10.0, 0.0, 40.0, 0.0, -10.0)
    path = str(tmp_path / "warp_src.tif")
    _raster.write_geotiff(path, cube, geotransform=geotransform, wkt=_wkt())

    warped, out_gt = _raster.warp_cube_to_grid(
        path,
        width=2,
        height=2,
        output_bounds=(0.0, 0.0, 40.0, 40.0),
        dst_wkt=_wkt(),
        resample_alg=gdalconst.GRA_NearestNeighbour,
    )

    assert warped.shape == (1, 2, 2)
    assert out_gt[1] == pytest.approx(20.0)  # pixel width doubled
    assert out_gt[5] == pytest.approx(-20.0)


def _write_warp_source(tmp_path: Path, *, dtype: npt.DTypeLike = np.int16) -> str:
    """Write a small local GeoTIFF suitable for driving warp tests."""
    cube: npt.NDArray[Any] = np.arange(1 * 4 * 4, dtype=dtype).reshape(1, 4, 4)
    geotransform = (0.0, 10.0, 0.0, 40.0, 0.0, -10.0)
    path = str(tmp_path / "warp_progress_src.tif")
    _raster.write_geotiff(path, cube, geotransform=geotransform, wkt=_wkt())
    return path


def test_warp_cube_to_grid_reports_progress(tmp_path: Path) -> None:
    path = _write_warp_source(tmp_path)
    seen: list[float] = []

    def _record(fraction: float) -> bool:
        seen.append(fraction)
        return True

    _raster.warp_cube_to_grid(
        path,
        width=2,
        height=2,
        output_bounds=(0.0, 0.0, 40.0, 40.0),
        dst_wkt=_wkt(),
        resample_alg=gdalconst.GRA_NearestNeighbour,
        progress=_record,
    )

    assert seen, "the progress sink was never called"
    assert all(0.0 <= f <= 1.0 for f in seen)
    assert seen[-1] == pytest.approx(1.0)  # the warp finishes at 100%


def test_warp_cube_to_grid_cancels_when_progress_returns_false(tmp_path: Path) -> None:
    path = _write_warp_source(tmp_path)

    with pytest.raises(_raster.WarpCanceledError):
        _raster.warp_cube_to_grid(
            path,
            width=2,
            height=2,
            output_bounds=(0.0, 0.0, 40.0, 40.0),
            dst_wkt=_wkt(),
            resample_alg=gdalconst.GRA_NearestNeighbour,
            progress=lambda _fraction: False,  # ask to cancel immediately
        )


def test_warp_cube_out_dtype_defaults_to_float64_but_can_preserve_native(tmp_path: Path) -> None:
    path = _write_warp_source(tmp_path, dtype=np.int16)
    kwargs: dict[str, Any] = {
        "width": 2,
        "height": 2,
        "output_bounds": (0.0, 0.0, 40.0, 40.0),
        "dst_wkt": _wkt(),
        "resample_alg": gdalconst.GRA_NearestNeighbour,
    }

    default_cube, _ = _raster.warp_cube_to_grid(path, **kwargs)
    assert default_cube.dtype == np.float64  # default upcasts to float64

    native_cube, _ = _raster.warp_cube_to_grid(path, out_dtype=None, **kwargs)
    assert native_cube.dtype == np.int16  # None keeps the warp's native dtype


def test_dataset_bounds_in_crs_for_local_source(tmp_path: Path) -> None:
    cube = np.zeros((1, 4, 4), dtype=np.float32)
    geotransform = (0.0, 10.0, 0.0, 40.0, 0.0, -10.0)
    path = str(tmp_path / "bounds.tif")
    _raster.write_geotiff(path, cube, geotransform=geotransform, wkt=_wkt())

    bounds = _raster.dataset_bounds_in_crs(path, _wkt())

    assert bounds is not None
    minx, miny, maxx, maxy = bounds
    assert (minx, miny, maxx, maxy) == pytest.approx((0.0, 0.0, 40.0, 40.0))


def test_dataset_bounds_in_crs_returns_none_for_unopenable() -> None:
    assert _raster.dataset_bounds_in_crs("/no/such/raster.tif", _wkt()) is None


# --------------------------------------------------------------------------- #
# Parity with an independent GeoTIFF reader/writer (rasterio), when installed.  #
# --------------------------------------------------------------------------- #


def test_written_geotiff_reads_identically_in_rasterio(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")

    cube = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    path = str(tmp_path / "for_rasterio.tif")
    _raster.write_geotiff(path, cube, geotransform=_GEOTRANSFORM, wkt=_wkt(), nodata=-1.0)

    with rasterio.open(path) as src:
        assert src.count == 2
        assert np.allclose(src.read(), cube)
        assert src.nodata == -1.0
        transform = src.transform
        # rasterio Affine (a,b,c,d,e,f) maps to GDAL gt (c,a,b,f,d,e).
        assert (transform.c, transform.a, transform.b) == pytest.approx(_GEOTRANSFORM[:3])
        assert (transform.f, transform.d, transform.e) == pytest.approx(_GEOTRANSFORM[3:])


def test_read_cube_matches_rasterio_written_geotiff(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")
    from rasterio.transform import Affine

    cube = (np.random.default_rng(1).standard_normal((3, 5, 6))).astype(np.float32)
    path = str(tmp_path / "rasterio_src.tif")
    transform = Affine(10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=cube.shape[1],
        width=cube.shape[2],
        count=cube.shape[0],
        dtype="float32",
        crs="EPSG:3577",
        transform=transform,
        nodata=-1.0,
    ) as dst:
        dst.write(cube)

    read, geotransform, _, nodata = _raster.read_cube(path)

    assert read.shape == (3, 5, 6)
    assert np.allclose(read, cube)
    assert geotransform == pytest.approx(_GEOTRANSFORM)
    assert nodata == -1.0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
