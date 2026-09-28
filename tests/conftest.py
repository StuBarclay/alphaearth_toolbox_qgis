"""Pytest configuration: make the plugin package importable.

The plugin package ``alphaearth_toolbox`` lives directly under the project root
(one level above this ``tests`` directory). ``pyproject.toml`` already sets
``pythonpath = ["."]`` for the common case; this belt-and-braces insert keeps
the tests importable when pytest is launched from another directory (for example
from within a QGIS Python console).
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
