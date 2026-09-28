"""Access configuration and tile discovery for AlphaEarth Satellite Embedding V1.

AlphaEarth V1 is a 64-band, 10 m, annual (2017-2025) global embedding, licensed
CC BY 4.0. It is published as Cloud-Optimised GeoTIFFs in the **public** Google
Cloud Storage bucket ``gs://alphaearth_foundations`` under::

    satellite_embedding/v1/annual/{YEAR}/{UTM_ZONE}/{hash}-{offsetA}-{offsetB}.tiff

where ``UTM_ZONE`` is a UTM zone number (1-60) with a hemisphere suffix
(``N``/``S``) -- e.g. ``55S`` -- and the two trailing integers are per-*scene*
pixel offsets (each ``{hash}`` is an independent scene with its own local
origin), **not** a global grid row/col, so a tile's world position is unknown
until a header is read. Each tile is a 64-band int8 COG at 10 m
native resolution in that zone's UTM CRS. Anonymous HTTPS reads through GDAL's
``/vsicurl`` work with no authentication and no billing project, and the objects
in a year/zone folder are enumerated with the public GCS JSON list API.

Everything here is pure standard library (no NumPy, GDAL or QGIS): the
constants, the tile-selection maths (which UTM zones an AOI touches, GCS object
prefixes/URLs, listing parsing) and a small ``urllib``-based listing helper whose
network call is injectable, so the logic is unit-testable in isolation. The
actual pixel reads live in :mod:`alphaearth_toolbox.aecore._raster`.

The endpoint here is grounded in the proven intake code of the sibling
``alphaearth-bushfire`` package, not guessed.
"""

from __future__ import annotations

import json
import math
import re
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

Bounds = tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)

#: Number of bands in an AlphaEarth Satellite Embedding V1 raster.
EMBEDDING_BANDS = 64

#: Years for which AlphaEarth V1 is published (2017-2025 inclusive).
AVAILABLE_YEARS: tuple[int, ...] = tuple(range(2017, 2026))

#: Native ground sample distance, in metres.
NATIVE_RESOLUTION_M = 10.0

#: Default working CRS: Australian Albers (equal-area, good for area stats).
#: Exposed as a per-algorithm parameter so the toolbox is usable outside
#: Australia.
DEFAULT_CRS = "EPSG:3577"

#: Attribution required by the CC BY 4.0 licence; bake into outputs/metadata.
ATTRIBUTION = (
    "Contains modified Google DeepMind AlphaEarth Satellite Embedding V1 data (CC BY 4.0)."
)

#: Public GCS bucket holding the AlphaEarth Foundations embeddings.
GCS_BUCKET = "alphaearth_foundations"

#: Object-name prefix of the annual V1 release inside :data:`GCS_BUCKET`.
GCS_PREFIX = "satellite_embedding/v1/annual"

#: HTTPS host for anonymous object reads of a public GCS bucket.
GCS_HOST = "https://storage.googleapis.com"

#: GCS JSON API endpoint for listing the objects in a bucket.
GCS_LIST_URL = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"

#: The tiles are int8, de-quantised to (near) unit-length float vectors by
#: dividing by this scale.
QUANT_SCALE = 127.5

#: Fill/no-data sentinel in the int8 tiles (ocean and scene edges).
FILL_VALUE = -128

#: Objects are listed in pages of up to this many items.
_LIST_PAGE_SIZE = 1000

#: Default timeout (seconds) for a single GCS list request.
_LIST_TIMEOUT_S = 60

#: Bump when the on-disk tile-footprint cache layout changes so stale caches are
#: silently ignored rather than misread.
CACHE_SCHEMA_VERSION = 1

#: Folder name (under the OS temp dir) holding the per-(year, zone) footprint
#: caches. AlphaEarth V1 is static/immutable annual data, so a cached footprint
#: never expires -- the cache only ever grows as new zones are scanned.
CACHE_DIR_NAME = "alphaearth_toolbox_tile_cache"


def vsicurl(url: str) -> str:
    """Wrap an ``http(s)`` URL as a GDAL ``/vsicurl`` path (idempotent).

    A path that is already a ``/vsicurl`` path, or that is not an ``http(s)``
    URL at all (e.g. a local file), is returned unchanged.

    Args:
        url: A URL or path.

    Returns:
        A ``/vsicurl/``-prefixed URL for remote http(s) inputs, else ``url``.
    """
    cleaned = url.strip()
    if cleaned.startswith("/vsicurl/"):
        return cleaned
    if cleaned.startswith(("http://", "https://")):
        return "/vsicurl/" + cleaned
    return cleaned


def normalise_bucket(bucket: str) -> str:
    """Strip a leading ``gs://`` and any surrounding slashes from a bucket ref."""
    return bucket.replace("gs://", "").strip("/")


def gcs_object_url(obj: str, bucket: str = GCS_BUCKET) -> str:
    """GDAL ``/vsicurl`` path for anonymous HTTPS reads of a public GCS object."""
    return f"/vsicurl/{GCS_HOST}/{normalise_bucket(bucket)}/{obj.lstrip('/')}"


def is_supported_year(year: int) -> bool:
    """Return whether AlphaEarth V1 publishes data for ``year``."""
    return int(year) in AVAILABLE_YEARS


def years_from_indices(indices: Iterable[int]) -> list[int]:
    """Resolve multi-select enum indices into :data:`AVAILABLE_YEARS`.

    The multi-year loader offers the years as a QGIS "enum" parameter whose
    checked entries arrive as positional indices into :data:`AVAILABLE_YEARS`.
    This turns them back into actual years, sorted ascending (oldest first, so a
    per-year output set is naturally chronological) and de-duplicated.

    Args:
        indices: Positions into :data:`AVAILABLE_YEARS` (e.g. ``[0, 4, 8]``).

    Returns:
        The corresponding years, sorted and unique.

    Raises:
        ValueError: If any index is out of range for :data:`AVAILABLE_YEARS`.
    """
    years: set[int] = set()
    for i in indices:
        index = int(i)
        if not 0 <= index < len(AVAILABLE_YEARS):
            raise ValueError(f"year index {index} is out of range 0-{len(AVAILABLE_YEARS) - 1}.")
        years.add(AVAILABLE_YEARS[index])
    return sorted(years)


def year_raster_name(year: int) -> str:
    """Filename for a single year's embedding in a multi-year output folder.

    Kept here (rather than inline in the algorithm) so the naming is one obvious,
    unit-testable place: ``alphaearth_2018.tif``.
    """
    return f"alphaearth_{int(year)}.tif"


def utm_zone_number(lon: float) -> int:
    """UTM zone number (1-60) containing longitude ``lon`` (degrees)."""
    zone = int((lon + 180.0) // 6.0) + 1
    return min(60, max(1, zone))


def utm_zones_for_bbox(lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> list[str]:
    """Return the UTM zone folder names (e.g. ``["54S", "55S"]``) an AOI touches.

    Longitude spans are matched by *interval overlap* against each zone's
    ``[6z-186, 6z-180)`` degree window, so an AOI whose eastern edge lands
    exactly on a zone boundary does not spuriously pull in the next (empty)
    zone. The hemisphere suffix follows the AOI latitude sign, including both
    ``N`` and ``S`` when the box straddles the equator.
    """
    if lon_min > lon_max:
        lon_min, lon_max = lon_max, lon_min
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min

    hemis: list[str] = []
    if lat_min < 0:
        hemis.append("S")
    if lat_max >= 0:
        hemis.append("N")
    if not hemis:  # pragma: no cover - defensive; lat range always hits a branch
        hemis = ["S"]

    z_lo = utm_zone_number(lon_min)
    z_hi = utm_zone_number(lon_max)
    zones: list[str] = []
    for z in range(z_lo, z_hi + 1):
        west = 6 * z - 186
        east = 6 * z - 180
        if west < lon_max and east > lon_min:
            zones.extend(f"{z}{h}" for h in hemis)
    return zones


def bboxes_intersect(a: Bounds, b: Bounds) -> bool:
    """True if two axis-aligned bounding boxes overlap (touching edges excluded)."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def zone_prefix(year: int, zone: str, prefix: str = GCS_PREFIX) -> str:
    """GCS object-name prefix for one year/zone folder (trailing slash)."""
    return f"{prefix.strip('/')}/{int(year)}/{zone}/"


def parse_listing(doc: dict[str, Any]) -> tuple[list[str], str | None]:
    """Extract COG object names + ``nextPageToken`` from a GCS list response.

    Only ``.tif``/``.tiff`` objects are returned (the folder placeholder objects
    the bucket stores for each prefix are skipped).
    """
    names = [
        str(item["name"])
        for item in doc.get("items", [])
        if str(item.get("name", "")).lower().endswith((".tif", ".tiff"))
    ]
    token = doc.get("nextPageToken")
    return names, (str(token) if token is not None else None)


def _default_fetch_json(url: str) -> dict[str, Any]:
    """Fetch and JSON-decode ``url`` with the standard library (anonymous GET)."""
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=_LIST_TIMEOUT_S) as response:
        payload = response.read()
    decoded: dict[str, Any] = json.loads(payload.decode("utf-8"))
    return decoded


def list_zone_tiles(
    year: int,
    zone: str,
    *,
    bucket: str = GCS_BUCKET,
    prefix: str = GCS_PREFIX,
    fetch_json: Callable[[str], dict[str, Any]] | None = None,
) -> list[str]:
    """List every COG object name under one ``year/zone`` folder on public GCS.

    Args:
        year: Embedding year (must be in :data:`AVAILABLE_YEARS`).
        zone: UTM zone folder name, e.g. ``"55S"``.
        bucket: GCS bucket (``gs://`` prefix optional).
        prefix: Object-name prefix of the annual release.
        fetch_json: Callable mapping a list URL to the decoded JSON response.
            Defaults to an anonymous ``urllib`` GET; injectable for testing.

    Returns:
        Sorted object names (``.tif``/``.tiff`` only) under the folder.

    Raises:
        ValueError: If ``year`` is outside the published range.
    """
    if not is_supported_year(year):
        raise ValueError(
            f"AlphaEarth V1 has no data for year {year}; available years are "
            f"{AVAILABLE_YEARS[0]}-{AVAILABLE_YEARS[-1]}."
        )
    fetch = fetch_json if fetch_json is not None else _default_fetch_json
    base = GCS_LIST_URL.format(bucket=normalise_bucket(bucket))
    folder = zone_prefix(year, zone, prefix)

    names: list[str] = []
    token: str | None = None
    while True:
        params: dict[str, Any] = {"prefix": folder, "maxResults": _LIST_PAGE_SIZE}
        if token:
            params["pageToken"] = token
        doc = fetch(f"{base}?{urlencode(params)}")
        page_names, token = parse_listing(doc)
        names.extend(page_names)
        if not token:
            break
    return sorted(names)


# --------------------------------------------------------------------------- #
# Persistent tile-footprint cache                                             #
# --------------------------------------------------------------------------- #
#
# Discovering which tiles cover an AOI means header-scanning tiles in each touched
# UTM zone (one small remote read each) -- the slow "0..90%" phase of a year fetch.
# The ``{hash}-{offset_a}-{offset_b}`` object names carry only *per-scene* pixel
# offsets, not a global grid position, so an intersecting tile's footprint cannot
# be computed from its name alone (see the scene-culling section above, which uses
# one anchor read per scene to cut the scan). Whatever footprints are read are then
# cached here, keyed by (year, zone), so subsequent AOI runs over the same zone
# skip the network for tiles already seen. Footprints are stored in **WGS84**
# (lon/lat) so the cache is independent of any run's target CRS.
#
# The store is pure stdlib JSON and every failure is swallowed: the cache is only
# ever an optimisation, so a missing/corrupt/unwritable cache must degrade to a
# normal (uncached) scan, never break a run.


def default_cache_dir() -> Path:
    """Return the default footprint-cache directory (under the OS temp dir).

    The OS temp dir is fine within a session but is periodically cleaned by the
    system, so a cache written there is not guaranteed to survive to the next
    QGIS session. For a durable, opt-in cache point at a stable base directory
    (e.g. the QGIS profile) via :func:`persistent_cache_dir`.
    """
    return Path(tempfile.gettempdir()) / CACHE_DIR_NAME


def persistent_cache_dir(base: Path | str) -> Path:
    """Return a durable footprint-cache directory under a stable ``base``.

    Given a directory that survives across sessions (typically the running
    QGIS profile directory), this returns the cache sub-folder within it, so a
    year fetch can reuse the ``(year, zone)`` listing and scene anchors it built
    in an earlier session instead of re-listing and re-reading tile headers.

    This is a pure path join (it does not touch the filesystem); the caller
    passes it to the cache read/write functions as their ``cache_dir``. The
    QGIS-aware resolution of ``base`` lives in the algorithm layer so this core
    stays importable without QGIS.
    """
    return Path(base) / CACHE_DIR_NAME


def _cache_file(year: int, zone: str, *, cache_dir: Path) -> Path:
    """Path of the JSON cache file for one (year, zone) folder."""
    return Path(cache_dir) / f"v{CACHE_SCHEMA_VERSION}_{int(year)}_{zone}.json"


def load_footprint_cache(
    year: int,
    zone: str,
    *,
    cache_dir: Path | None = None,
    bucket: str = GCS_BUCKET,
    prefix: str = GCS_PREFIX,
) -> dict[str, Bounds]:
    """Load cached ``{object_name: WGS84 bounds}`` for one year/zone folder.

    Returns an empty mapping when the cache is absent, unreadable, or was written
    for a different schema/bucket/prefix (so a mismatch is refreshed, not misused).

    Args:
        year: Embedding year.
        zone: UTM zone folder name, e.g. ``"55S"``.
        cache_dir: Cache root (defaults to :func:`default_cache_dir`).
        bucket: GCS bucket the cache must have been written for.
        prefix: Object-name prefix the cache must have been written for.

    Returns:
        A mapping of object name to its ``(min_lon, min_lat, max_lon, max_lat)``
        WGS84 footprint. Empty on any miss or error.
    """
    root = cache_dir if cache_dir is not None else default_cache_dir()
    path = _cache_file(year, zone, cache_dir=root)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if (
        not isinstance(doc, dict)
        or doc.get("schema_version") != CACHE_SCHEMA_VERSION
        or int(doc.get("year", -1)) != int(year)
        or str(doc.get("zone", "")) != str(zone)
        or str(doc.get("bucket", "")) != normalise_bucket(bucket)
        or str(doc.get("prefix", "")) != prefix.strip("/")
    ):
        return {}
    tiles = doc.get("tiles", {})
    out: dict[str, Bounds] = {}
    if isinstance(tiles, dict):
        for name, bounds in tiles.items():
            try:
                values = tuple(float(v) for v in bounds)
            except (TypeError, ValueError):
                continue
            if len(values) == 4:
                out[str(name)] = (values[0], values[1], values[2], values[3])
    return out


def save_footprint_cache(
    year: int,
    zone: str,
    footprints: dict[str, Bounds],
    *,
    cache_dir: Path | None = None,
    bucket: str = GCS_BUCKET,
    prefix: str = GCS_PREFIX,
) -> None:
    """Write ``{object_name: WGS84 bounds}`` for one year/zone folder (best effort).

    The write is atomic (temp file + replace) and never raises: a failure to
    persist the cache just means the next run rescans, which is correct if slow.
    """
    root = cache_dir if cache_dir is not None else default_cache_dir()
    path = _cache_file(year, zone, cache_dir=root)
    doc = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "year": int(year),
        "zone": str(zone),
        "bucket": normalise_bucket(bucket),
        "prefix": prefix.strip("/"),
        "tiles": {str(name): [float(v) for v in b] for name, b in footprints.items()},
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(doc), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        return  # the cache is an optimisation; never let a write failure abort a run


@dataclass(frozen=True)
class SceneAnchor:
    """One header-scanned representative tile of a scene, cached for reuse.

    Its WGS84 ``bounds`` and ``pixels`` (x, y raster size) plus its parsed
    ``offset`` are exactly what :func:`conservative_scene_bounds` needs, so a
    cached anchor lets a scene be culled on a later run with no network read.
    """

    name: str
    bounds: Bounds
    pixels: tuple[int, int]
    offset: tuple[int, int]


@dataclass
class ZoneIndex:
    """Cached discovery state for one (year, zone) folder: listing + scene anchors.

    ``names`` is the full object listing (immutable data, so it never has to be
    re-listed once cached). ``anchors`` maps a :func:`scene_key` to a
    :class:`SceneAnchor`, letting previously-seen scenes be culled arithmetically
    on later runs over a different AOI in the same zone.
    """

    names: list[str]
    anchors: dict[str, SceneAnchor]


def _zone_index_file(year: int, zone: str, *, cache_dir: Path) -> Path:
    """Path of the JSON zone-index (listing + anchors) file for one folder."""
    return Path(cache_dir) / f"idx_v{CACHE_SCHEMA_VERSION}_{int(year)}_{zone}.json"


def load_zone_index(
    year: int,
    zone: str,
    *,
    cache_dir: Path | None = None,
    bucket: str = GCS_BUCKET,
    prefix: str = GCS_PREFIX,
) -> ZoneIndex | None:
    """Load the cached listing + scene anchors for one year/zone, or ``None``.

    Returns ``None`` when the cache is absent, unreadable, or was written for a
    different schema/bucket/prefix, so a mismatch triggers a fresh listing rather
    than being misused. Anchors that fail to parse are skipped individually; a
    scene without a valid cached anchor simply gets an anchor read next run.
    """
    root = cache_dir if cache_dir is not None else default_cache_dir()
    path = _zone_index_file(year, zone, cache_dir=root)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        not isinstance(doc, dict)
        or doc.get("schema_version") != CACHE_SCHEMA_VERSION
        or int(doc.get("year", -1)) != int(year)
        or str(doc.get("zone", "")) != str(zone)
        or str(doc.get("bucket", "")) != normalise_bucket(bucket)
        or str(doc.get("prefix", "")) != prefix.strip("/")
    ):
        return None
    raw_names = doc.get("names", [])
    if not isinstance(raw_names, list):
        return None
    names = [str(n) for n in raw_names]

    anchors: dict[str, SceneAnchor] = {}
    raw_anchors = doc.get("anchors", {})
    if isinstance(raw_anchors, dict):
        for key, rec in raw_anchors.items():
            try:
                bounds = tuple(float(v) for v in rec["bounds"])
                pixels = tuple(int(v) for v in rec["pixels"])
                offset = tuple(int(v) for v in rec["offset"])
                name = str(rec["name"])
            except (TypeError, ValueError, KeyError):
                continue
            if len(bounds) == 4 and len(pixels) == 2 and len(offset) == 2:
                anchors[str(key)] = SceneAnchor(
                    name=name,
                    bounds=(bounds[0], bounds[1], bounds[2], bounds[3]),
                    pixels=(pixels[0], pixels[1]),
                    offset=(offset[0], offset[1]),
                )
    return ZoneIndex(names=names, anchors=anchors)


def save_zone_index(
    year: int,
    zone: str,
    index: ZoneIndex,
    *,
    cache_dir: Path | None = None,
    bucket: str = GCS_BUCKET,
    prefix: str = GCS_PREFIX,
) -> None:
    """Write the listing + scene anchors for one year/zone (atomic, best effort).

    Never raises: a failed write just means the next run re-lists and re-anchors.
    """
    root = cache_dir if cache_dir is not None else default_cache_dir()
    path = _zone_index_file(year, zone, cache_dir=root)
    doc = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "year": int(year),
        "zone": str(zone),
        "bucket": normalise_bucket(bucket),
        "prefix": prefix.strip("/"),
        "names": [str(n) for n in index.names],
        "anchors": {
            str(key): {
                "name": anchor.name,
                "bounds": [float(v) for v in anchor.bounds],
                "pixels": [int(v) for v in anchor.pixels],
                "offset": [int(v) for v in anchor.offset],
            }
            for key, anchor in index.anchors.items()
        },
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(doc), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        return  # the cache is an optimisation; never let a write failure abort a run


def partition_by_cache(names: list[str], cached: dict[str, Bounds]) -> tuple[list[str], list[str]]:
    """Split object ``names`` into ``(already_cached, needs_scan)`` preserving order."""
    have = [name for name in names if name in cached]
    missing = [name for name in names if name not in cached]
    return have, missing


def tiles_intersecting(footprints: dict[str, Bounds], aoi_wgs84: Bounds) -> list[str]:
    """Return the object names whose cached WGS84 footprint intersects the AOI bbox.

    Args:
        footprints: A ``{object_name: WGS84 bounds}`` mapping (as cached).
        aoi_wgs84: The AOI's ``(min_lon, min_lat, max_lon, max_lat)`` bounding box.

    Returns:
        The intersecting object names, sorted for a deterministic mosaic order.
    """
    return sorted(
        name for name, bounds in footprints.items() if bboxes_intersect(bounds, aoi_wgs84)
    )


# --------------------------------------------------------------------------- #
# Scene grouping and conservative culling                                     #
# --------------------------------------------------------------------------- #
#
# An AlphaEarth object name ends ``{hash}-{offset_a}-{offset_b}.tiff``. The two
# trailing integers are that *scene's* per-tile pixel offsets (multiples of the
# tile size), NOT positions on a global grid: each ``{hash}`` is an independent
# export with its own local (0, 0) origin, so many distinct scenes each own a
# ``-0000000000-0000000000`` tile. A tile's world position therefore cannot be
# derived from its name alone. But tiles *can* be grouped by their scene hash,
# and once one tile of a scene has been header-read its footprint plus pixel size
# turn the (known) pixel offsets of the scene's other tiles into a geographic
# extent. That lets a whole scene be dismissed from a single read when its
# (deliberately over-estimated) extent cannot touch the AOI, instead of scanning
# every tile -- roughly one read per scene rather than one per tile.

#: Matches the trailing ``-{offset_a}-{offset_b}.tif[f]`` of a tile object name.
_TILE_OFFSET_RE = re.compile(r"-(\d+)-(\d+)\.tiff?$", re.IGNORECASE)


def parse_tile_offsets(name: str) -> tuple[int, int] | None:
    """Return a tile's ``(offset_a, offset_b)`` pixel offsets, or ``None``.

    The offsets are per-*scene* (each scene hash has its own origin), so they only
    give relative position *within* one scene, never a global grid position. A
    name that does not match the expected pattern yields ``None``.
    """
    match = _TILE_OFFSET_RE.search(name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def scene_key(name: str) -> str:
    """Return the scene identifier of a tile (its name minus the offset suffix).

    Tiles of one AlphaEarth scene share this key. A name that cannot be parsed is
    returned unchanged, so it forms its own singleton scene and is always scanned
    (the safe fallback -- an unrecognised name is never culled arithmetically).
    """
    match = _TILE_OFFSET_RE.search(name)
    if match is None:
        return name
    return name[: match.start()]


def group_by_scene(names: Iterable[str]) -> dict[str, list[str]]:
    """Group object names by their :func:`scene_key`, preserving input order."""
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(scene_key(name), []).append(name)
    return groups


def conservative_scene_bounds(
    anchor_bounds: Bounds,
    anchor_pixels: tuple[int, int],
    anchor_offset: tuple[int, int],
    scene_offsets: Iterable[tuple[int, int]],
    *,
    margin: float = 0.5,
) -> Bounds:
    """Over-estimate the WGS84 extent of a whole scene from one anchor tile.

    Given a single header-scanned tile of a scene (its WGS84 ``anchor_bounds``,
    its pixel size ``anchor_pixels`` and its parsed ``anchor_offset``) and the
    pixel offsets of every tile in the scene, return a WGS84 bounding box
    *guaranteed to contain* the scene. The estimate is deliberately generous --
    it converts pixel offsets to degrees using the anchor's own (envelope,
    already an over-estimate) degrees-per-pixel, takes the larger axis so the
    unknown pixel-axis-to-lon/lat mapping cannot cause an under-estimate, adds a
    whole extra tile of reach, and inflates by ``margin`` on top. Its only use is
    culling: if this box misses the AOI the scene certainly does, so its tiles can
    be skipped; a surviving scene is still scanned tile-by-tile for exact
    footprints, so accuracy of this estimate never affects correctness.

    Falls back to ``anchor_bounds`` when there is nothing to extrapolate from
    (no offsets, or degenerate pixel dimensions).
    """
    min_lon, min_lat, max_lon, max_lat = anchor_bounds
    xpix, ypix = anchor_pixels
    a0, b0 = anchor_offset
    offsets = list(scene_offsets)
    if not offsets or xpix <= 0 or ypix <= 0:
        return anchor_bounds
    # Degrees-per-pixel from the anchor's footprint; use the larger axis so we
    # never under-scale whichever offset axis maps to the wider geographic span.
    deg_per_pixel = max((max_lon - min_lon) / xpix, (max_lat - min_lat) / ypix)
    max_pixel_reach = 0
    for off_a, off_b in offsets:
        max_pixel_reach = max(max_pixel_reach, abs(off_a - a0), abs(off_b - b0))
    # +one full tile covers the far tile's own extent; (1 + margin) is slack for
    # projection rotation and latitude variation across the scene.
    pad = deg_per_pixel * (max_pixel_reach + max(xpix, ypix)) * (1.0 + margin)
    return (min_lon - pad, min_lat - pad, max_lon + pad, max_lat + pad)


def grid_dimensions(
    bounds: tuple[float, float, float, float], resolution: float
) -> tuple[int, int, tuple[float, float, float, float]]:
    """Size a pixel grid for an extent at a given resolution.

    The extent is expanded (never shrunk) to a whole number of pixels so the
    requested area is fully covered, and the snapped bounds are returned
    alongside the pixel dimensions.

    Args:
        bounds: ``(minx, miny, maxx, maxy)`` of the area of interest, in the
            target CRS's units (metres for EPSG:3577).
        resolution: Pixel size in the same units (e.g. ``10.0`` for AlphaEarth's
            native 10 m).

    Returns:
        ``(width, height, snapped_bounds)`` -- integer pixel counts (at least 1
        each) and the bounds rounded out to a whole number of pixels.

    Raises:
        ValueError: If ``resolution`` is not positive or the extent is empty.
    """
    if resolution <= 0:
        raise ValueError(f"resolution must be positive; got {resolution}.")
    minx, miny, maxx, maxy = (float(v) for v in bounds)
    if maxx <= minx or maxy <= miny:
        raise ValueError(f"bounds must have positive width and height; got {bounds}.")

    width = max(1, math.ceil((maxx - minx) / resolution))
    height = max(1, math.ceil((maxy - miny) / resolution))
    snapped = (minx, maxy - height * resolution, minx + width * resolution, maxy)
    return width, height, snapped
