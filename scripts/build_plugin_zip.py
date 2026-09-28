#!/usr/bin/env python3
"""Build the installable QGIS plugin ZIP, cleanly and reproducibly.

The QGIS Plugin Manager installs a ZIP whose single top-level entry is the
plugin package folder (``alphaearth_toolbox/``). This script packages exactly
that folder and nothing else: byte-code caches (``__pycache__``, ``*.pyc``), the
test suite, tooling caches and the dev/packaging scaffolding all live outside
the package or are skipped here, so they never reach a published ZIP.

Run it from anywhere::

    python scripts/build_plugin_zip.py

It writes ``alphaearth_toolbox.zip`` next to the package and prints a summary.
Entries are added in sorted order with a fixed timestamp so repeated builds of
an unchanged tree are byte-identical.
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

#: The plugin package folder that must sit at the root of the ZIP.
PACKAGE = "alphaearth_toolbox"

#: Directory names skipped anywhere in the tree.
SKIP_DIRS = frozenset({"__pycache__"})

#: File suffixes skipped anywhere in the tree.
SKIP_SUFFIXES = frozenset({".pyc", ".pyo"})

#: Fixed ZIP entry timestamp (Y, M, D, h, m, s) for reproducible builds.
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def _included_files(package_dir: Path) -> list[Path]:
    """Return the files to pack, sorted, excluding caches and byte-code."""
    files: list[Path] = []
    for path in package_dir.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(package_dir).parts):
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        files.append(path)
    return sorted(files)


def build_zip(repo_root: Path) -> Path:
    """Build ``<repo_root>/alphaearth_toolbox.zip`` and return its path."""
    package_dir = repo_root / PACKAGE
    if not package_dir.is_dir():
        raise SystemExit(f"Plugin package not found: {package_dir}")

    output = repo_root / f"{PACKAGE}.zip"

    # Build into a scratch file first, then copy the bytes over the destination.
    # On a cloud-synced folder (e.g. OneDrive) unlinking/replacing a file can be
    # blocked, but truncate-and-write in place (what copyfile does) is allowed --
    # so this stays robust without changing the reproducible output.
    files = _included_files(package_dir)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        scratch = Path(tmp.name)
    try:
        with zipfile.ZipFile(scratch, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in files:
                # Arc names are rooted at the package folder, e.g.
                # ``alphaearth_toolbox/metadata.txt``.
                arcname = file.relative_to(repo_root).as_posix()
                info = zipfile.ZipInfo(arcname, date_time=_FIXED_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, file.read_bytes())
        shutil.copyfile(scratch, output)
    finally:
        scratch.unlink(missing_ok=True)

    print(f"Built {output.name} ({len(files)} files, {output.stat().st_size} bytes)")
    return output


def main() -> None:
    """Build the ZIP relative to this script's repository root."""
    repo_root = Path(__file__).resolve().parent.parent
    build_zip(repo_root)


if __name__ == "__main__":
    main()
