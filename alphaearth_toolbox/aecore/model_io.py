"""Save and reload fitted Classify/Regress models, with compatibility checks.

Training a random forest over a large scene is the slow part of the ML tier, so
being able to fit once and re-apply the same model to another year or a
neighbouring area is a real time-saver. This module persists a fitted estimator
together with the metadata needed to reuse it safely -- the band count it was
trained on, whether it is a classifier or regressor, the class list, and the
scikit-learn version it was fitted under -- and checks that metadata before a
saved model is applied to fresh data.

The on-disk format is a single :mod:`pickle` file holding a :class:`ModelBundle`.
This module is pure standard library (``pickle`` + ``dataclasses`` +
``importlib.metadata``): it never imports scikit-learn, NumPy or QGIS, so it is
importable and unit-testable on its own -- a fitted estimator is just an opaque
picklable payload here. scikit-learn is only actually needed by the interpreter
that *unpickles* a real estimator, which is exactly where the ML algorithms run.

Security note: a pickle can execute arbitrary code on load, so only load model
files you produced or trust. The bundle stores a format version and a magic
marker so an unrelated file fails fast with a clear message rather than
executing.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Bumped if the :class:`ModelBundle` layout changes incompatibly.
FORMAT_VERSION = 1

#: Marker stored in every bundle so a foreign pickle is rejected with a clear
#: message instead of being trusted as one of ours.
_MAGIC = "alphaearth_toolbox.model"

#: The two estimator roles the toolbox persists.
KIND_CLASSIFIER = "classifier"
KIND_REGRESSOR = "regressor"
KINDS = (KIND_CLASSIFIER, KIND_REGRESSOR)


class ModelError(Exception):
    """A saved model could not be read, or is not an AlphaEarth model bundle."""


@dataclass
class ModelBundle:
    """A fitted estimator plus the metadata needed to reuse it safely.

    Attributes:
        estimator: The fitted estimator (an opaque picklable payload here; a
            scikit-learn model in practice).
        kind: :data:`KIND_CLASSIFIER` or :data:`KIND_REGRESSOR`.
        n_bands: The number of input bands (features) the model was trained on;
            fresh data must match.
        algorithm: The estimator family used (e.g. ``"random_forest"``), for
            reporting only.
        classes: For a classifier, the ordered class labels (``classes[code]``);
            ``None`` for a regressor.
        sklearn_version: The scikit-learn version string the model was fitted
            under, or ``None`` if it could not be determined at save time.
        created: ISO-8601 UTC timestamp of when the bundle was written.
        format_version: The on-disk format version (:data:`FORMAT_VERSION`).
        magic: Internal marker identifying an AlphaEarth model file.
    """

    estimator: Any
    kind: str
    n_bands: int
    algorithm: str = ""
    classes: list[Any] | None = None
    sklearn_version: str | None = None
    created: str = ""
    format_version: int = FORMAT_VERSION
    magic: str = _MAGIC


def installed_sklearn_version() -> str | None:
    """Return the installed scikit-learn version string, or ``None`` if absent.

    Uses :mod:`importlib.metadata`, so it reads the version without importing
    scikit-learn (keeping this module dependency-free).
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("scikit-learn")
    except PackageNotFoundError:
        return None
    except Exception:  # pragma: no cover - defensive; metadata backend quirks
        return None


def save_model(
    path: str | Path,
    *,
    estimator: Any,
    kind: str,
    n_bands: int,
    algorithm: str = "",
    classes: list[Any] | None = None,
    sklearn_version: str | None = None,
) -> ModelBundle:
    """Pickle a fitted estimator and its metadata to ``path``.

    Args:
        path: Destination file (``.pkl`` by convention; parent dirs are created).
        estimator: The fitted estimator to store.
        kind: :data:`KIND_CLASSIFIER` or :data:`KIND_REGRESSOR`.
        n_bands: Number of input bands the estimator was trained on.
        algorithm: Estimator family label, for reporting.
        classes: Ordered class labels for a classifier; ``None`` for a regressor.
        sklearn_version: Override the recorded scikit-learn version; when
            ``None`` the installed version is detected and stored.

    Returns:
        The :class:`ModelBundle` that was written.

    Raises:
        ValueError: If ``kind`` is not a known kind or ``n_bands`` is not
            positive.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}; got {kind!r}.")
    if n_bands <= 0:
        raise ValueError(f"n_bands must be positive; got {n_bands}.")

    bundle = ModelBundle(
        estimator=estimator,
        kind=kind,
        n_bands=int(n_bands),
        algorithm=algorithm,
        classes=list(classes) if classes is not None else None,
        sklearn_version=(
            sklearn_version if sklearn_version is not None else installed_sklearn_version()
        ),
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return bundle


def load_model(path: str | Path) -> ModelBundle:
    """Unpickle a :class:`ModelBundle` from ``path``, validating the format.

    Args:
        path: A model file previously written by :func:`save_model`.

    Returns:
        The loaded :class:`ModelBundle`.

    Raises:
        ModelError: If the file is missing, unreadable, not an AlphaEarth model
            bundle, or written in an unsupported format version. (Unpickling a
            real estimator additionally needs a compatible scikit-learn present;
            a failure there surfaces here as a ``ModelError``.)
    """
    src = Path(path)
    if not src.is_file():
        raise ModelError(f"model file not found: {src}")
    try:
        with src.open("rb") as handle:
            obj = pickle.load(handle)
    except ModelError:
        raise
    except Exception as error:
        raise ModelError(
            f"could not read model file {src.name}: {error}. If it was trained with a "
            "different scikit-learn version, retrain or match that version."
        ) from error

    if not isinstance(obj, ModelBundle) or obj.magic != _MAGIC:
        raise ModelError(f"{src.name} is not an AlphaEarth model file.")
    if obj.format_version != FORMAT_VERSION:
        raise ModelError(
            f"{src.name} uses model format v{obj.format_version}, but this plugin reads "
            f"v{FORMAT_VERSION}. Retrain and save the model again."
        )
    return obj


def _major_minor(version_string: str) -> tuple[int, int] | None:
    """Parse ``"1.3.2"`` -> ``(1, 3)``; ``None`` if it cannot be parsed."""
    parts = version_string.split(".")
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return None


@dataclass
class CompatibilityReport:
    """The outcome of checking a loaded model against fresh input data."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when there are no blocking errors (warnings are allowed)."""
        return not self.errors


def check_compatibility(
    bundle: ModelBundle,
    *,
    n_bands: int,
    kind: str,
    current_sklearn: str | None = None,
) -> CompatibilityReport:
    """Check a loaded model can be applied to the data at hand.

    Band-count and estimator-kind mismatches are **errors** (applying the model
    would be wrong or impossible). A scikit-learn *minor* version mismatch is
    also an error, because a pickled estimator is not guaranteed to behave
    correctly across minor versions; a mere patch-level difference (or an unknown
    version on either side) is a **warning** only.

    Args:
        bundle: The loaded model bundle.
        n_bands: Band count of the embedding the model is about to be applied to.
        kind: The kind the calling algorithm needs
            (:data:`KIND_CLASSIFIER` / :data:`KIND_REGRESSOR`).
        current_sklearn: The installed scikit-learn version; when ``None`` it is
            detected via :func:`installed_sklearn_version`.

    Returns:
        A :class:`CompatibilityReport`; ``report.ok`` is ``False`` when the model
        must not be used.
    """
    report = CompatibilityReport()

    if bundle.kind != kind:
        report.errors.append(
            f"this model is a {bundle.kind}, but the {kind} algorithm needs a {kind}."
        )
    if bundle.n_bands != n_bands:
        report.errors.append(
            f"this model was trained on {bundle.n_bands}-band input, but the embedding "
            f"has {n_bands} bands."
        )

    saved = bundle.sklearn_version
    current = current_sklearn if current_sklearn is not None else installed_sklearn_version()
    if saved and current and saved != current:
        saved_mm, current_mm = _major_minor(saved), _major_minor(current)
        if saved_mm is not None and current_mm is not None and saved_mm != current_mm:
            report.errors.append(
                f"this model was trained with scikit-learn {saved}, but {current} is "
                "installed; pickled models are not guaranteed to load correctly across "
                f"minor versions. Retrain the model, or install scikit-learn {saved}."
            )
        else:
            report.warnings.append(
                f"this model was trained with scikit-learn {saved}; {current} is "
                "installed (patch-level difference, usually fine)."
            )
    elif saved and not current:
        report.warnings.append(
            f"this model was trained with scikit-learn {saved}; the installed version "
            "could not be determined."
        )
    elif not saved:
        report.warnings.append(
            "this model does not record the scikit-learn version it was trained with."
        )

    return report
