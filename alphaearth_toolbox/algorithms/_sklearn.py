"""Runtime detection and guided install of the optional scikit-learn tier.

scikit-learn is **not** bundled with QGIS and is **not** a hard dependency of
this plugin. The machine-learning algorithms (classify / cluster / regress)
import it lazily and, when it is missing, fail gracefully with a clear message
and offer a one-click install into the running QGIS Python environment.

Everything here is pure standard library -- no ``numpy``, ``qgis`` or
``sklearn`` import at module load -- so the detection and command-building logic
is importable and unit-testable in a plain Python environment. The actual
``import sklearn`` only ever happens inside the algorithm modules at run time.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

#: Import name of the optional dependency the ML tier needs.
SKLEARN_MODULE = "sklearn"

#: PyPI distribution name (differs from the import name).
SKLEARN_DISTRIBUTION = "scikit-learn"

#: Conservative version floor known to provide the estimators used here.
_MIN_SKLEARN = "1.0"


class InstallResult(NamedTuple):
    """Outcome of an attempted scikit-learn install."""

    ok: bool
    args: list[str]
    returncode: int | None
    output: str


def is_available(module: str = SKLEARN_MODULE) -> bool:
    """Return whether ``module`` can be imported in this interpreter.

    Uses :func:`importlib.util.find_spec` so it does not actually import (or pay
    the import cost of) scikit-learn just to check for it.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def pip_install_args(
    executable: str | None = None,
    *,
    distribution: str = SKLEARN_DISTRIBUTION,
    minimum: str = _MIN_SKLEARN,
) -> list[str]:
    """Return the argv that pip-installs scikit-learn into ``executable``.

    Args:
        executable: The Python interpreter to install into; defaults to the one
            running QGIS (``sys.executable``).
        distribution: PyPI distribution to install.
        minimum: Minimum acceptable version.

    Returns:
        An argv list suitable for :func:`subprocess.run`.
    """
    python = executable or sys.executable or "python"
    return [python, "-m", "pip", "install", f"{distribution}>={minimum}"]


def missing_message(context: str = "") -> str:
    """Return a clear, actionable message for when scikit-learn is absent.

    Args:
        context: Optional phrase naming what needs it (e.g. "Classify").
    """
    where = f" ({context})" if context else ""
    command = " ".join(pip_install_args())
    return (
        f"This algorithm{where} needs the optional scikit-learn package, which is not "
        "installed in this QGIS Python environment. Install it once -- from the AlphaEarth "
        "Toolbox dialog's 'Install scikit-learn' button, or by running:\n\n"
        f"    {command}\n\n"
        "in the QGIS Python console (Plugins > Python Console) -- then run the algorithm again."
    )


def _default_runner(args: Sequence[str]) -> Any:
    """Run ``args`` capturing output, without raising on a non-zero exit."""
    return subprocess.run(args, capture_output=True, text=True, check=False)


def install(
    executable: str | None = None,
    *,
    runner: Callable[[Sequence[str]], Any] | None = None,
) -> InstallResult:
    """Attempt to pip-install scikit-learn, returning a structured result.

    Args:
        executable: Interpreter to install into (defaults to ``sys.executable``).
        runner: Callable that runs an argv and returns an object exposing
            ``returncode`` / ``stdout`` / ``stderr`` (defaults to
            :func:`subprocess.run`). Injectable so the flow is unit-testable
            without actually shelling out.

    Returns:
        An :class:`InstallResult` with the command run and its combined output.
    """
    args = pip_install_args(executable)
    run = runner if runner is not None else _default_runner
    try:
        completed = run(args)
    except OSError as error:
        return InstallResult(False, args, None, str(error))

    returncode = int(getattr(completed, "returncode", 1) or 0)
    stdout = getattr(completed, "stdout", "") or ""
    stderr = getattr(completed, "stderr", "") or ""
    return InstallResult(returncode == 0, args, returncode, f"{stdout}{stderr}".strip())
