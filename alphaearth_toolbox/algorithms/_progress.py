"""Feedback adapters bridging QGIS Processing feedback to the compute core.

The compute core reports progress through plain callables -- a status
``message(str)``, a ``scene_progress(done, total)`` and an
``is_cancelled() -> bool`` predicate -- rather than depending on QGIS. This
module builds those callables from a ``QgsProcessingFeedback``.

It deliberately performs no ``qgis`` import: it needs only the duck-typed
feedback protocol (``pushInfo`` / ``setProgress`` / ``setProgressText`` /
``isCanceled``), so it imports and unit-tests without a QGIS runtime, just like
the compute core.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def make_message(feedback: Any) -> Callable[[str], None]:
    """Return a ``message(text)`` callable that pushes info lines to feedback.

    Args:
        feedback: A ``QgsProcessingFeedback`` (or any object with ``pushInfo``);
            ``None`` yields a no-op.

    Returns:
        A callable taking a single status string.
    """

    def message(text: str) -> None:
        push = getattr(feedback, "pushInfo", None)
        if callable(push):
            push(text)

    return message


def make_scene_progress(
    feedback: Any,
    *,
    start: float = 10.0,
    end: float = 90.0,
) -> Callable[[int, int], None]:
    """Return a ``(done, total)`` progress callback that drives feedback.

    Each call advances ``feedback`` linearly from ``start`` to ``end`` percent
    and sets a matching progress-text line.

    Args:
        feedback: A ``QgsProcessingFeedback`` (or any object exposing
            ``setProgress`` / optionally ``setProgressText``). ``None`` and
            partial objects are tolerated.
        start: Progress percent already reached when the step begins.
        end: Progress percent reached once the step is complete.

    Returns:
        A ``(done, total)`` callable.
    """
    span = end - start

    def scene_progress(done: int, total: int) -> None:
        setter = getattr(feedback, "setProgress", None)
        if callable(setter) and total > 0:
            setter(max(0.0, min(100.0, start + span * done / total)))
        text_setter = getattr(feedback, "setProgressText", None)
        if callable(text_setter):
            text_setter(f"Processed {done} of {total}")

    return scene_progress


def make_cancel(feedback: Any) -> Callable[[], bool]:
    """Return an ``is_cancelled()`` predicate reading the feedback's state.

    Args:
        feedback: A ``QgsProcessingFeedback`` (or any object with
            ``isCanceled``); ``None`` yields a predicate that is always false.

    Returns:
        A zero-argument callable returning whether the run is cancelled.
    """

    def is_cancelled() -> bool:
        checker = getattr(feedback, "isCanceled", None)
        if callable(checker):
            return bool(checker())
        return False

    return is_cancelled
