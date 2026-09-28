"""Unit tests for the QGIS-free progress/feedback adapters."""

from __future__ import annotations

from alphaearth_toolbox.algorithms._progress import (
    make_cancel,
    make_message,
    make_scene_progress,
)


class _FakeFeedback:
    """A minimal stand-in for QgsProcessingFeedback."""

    def __init__(self, cancelled: bool = False) -> None:
        self.messages: list[str] = []
        self.progress: list[float] = []
        self.texts: list[str] = []
        self._cancelled = cancelled

    def pushInfo(self, text: str) -> None:  # noqa: N802 - QGIS API shape
        self.messages.append(text)

    def setProgress(self, value: float) -> None:  # noqa: N802 - QGIS API shape
        self.progress.append(value)

    def setProgressText(self, text: str) -> None:  # noqa: N802 - QGIS API shape
        self.texts.append(text)

    def isCanceled(self) -> bool:  # noqa: N802 - QGIS API shape
        return self._cancelled


def test_make_message_pushes_info() -> None:
    feedback = _FakeFeedback()
    make_message(feedback)("hello")
    assert feedback.messages == ["hello"]


def test_make_message_none_is_noop() -> None:
    make_message(None)("ignored")  # must not raise


def test_make_scene_progress_interpolates() -> None:
    feedback = _FakeFeedback()
    progress = make_scene_progress(feedback, start=10.0, end=90.0)
    progress(0, 4)
    progress(2, 4)
    progress(4, 4)
    assert feedback.progress == [10.0, 50.0, 90.0]
    assert feedback.texts[-1] == "Processed 4 of 4"


def test_make_scene_progress_zero_total_is_safe() -> None:
    feedback = _FakeFeedback()
    make_scene_progress(feedback)(0, 0)  # must not divide by zero
    assert feedback.progress == []


def test_make_cancel_reads_state() -> None:
    assert make_cancel(_FakeFeedback(cancelled=True))() is True
    assert make_cancel(_FakeFeedback(cancelled=False))() is False
    assert make_cancel(None)() is False
