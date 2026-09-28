"""Unit tests for the scikit-learn detection / guided-install helper.

These are pure standard-library tests: scikit-learn need not be installed, and
the install flow is exercised with an injected runner so nothing is shelled out.
"""

from __future__ import annotations

import sys

import pytest

from alphaearth_toolbox.algorithms import _sklearn


def test_is_available_false_for_missing_module() -> None:
    assert _sklearn.is_available("definitely_not_a_real_module_zzz") is False


def test_is_available_true_for_stdlib() -> None:
    # A module guaranteed to exist stands in for scikit-learn here.
    assert _sklearn.is_available("json") is True


def test_pip_install_args_uses_given_executable() -> None:
    args = _sklearn.pip_install_args("/opt/py/bin/python")
    assert args == ["/opt/py/bin/python", "-m", "pip", "install", "scikit-learn>=1.0"]


def test_pip_install_args_defaults_to_sys_executable() -> None:
    args = _sklearn.pip_install_args()
    assert args[0] == (sys.executable or "python")
    assert args[1:4] == ["-m", "pip", "install"]


def test_missing_message_mentions_package_and_context() -> None:
    message = _sklearn.missing_message("Classify embedding")
    assert "scikit-learn" in message
    assert "Classify embedding" in message
    assert "pip install" in message


def test_missing_message_without_context() -> None:
    assert "scikit-learn" in _sklearn.missing_message()


class _FakeCompleted:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_install_success() -> None:
    result = _sklearn.install("/py", runner=lambda args: _FakeCompleted(0, "Installed"))
    assert result.ok is True
    assert result.returncode == 0
    assert "Installed" in result.output
    assert result.args[0] == "/py"


def test_install_failure_captures_output() -> None:
    result = _sklearn.install(runner=lambda args: _FakeCompleted(1, "", "No matching dist"))
    assert result.ok is False
    assert result.returncode == 1
    assert "No matching dist" in result.output


def test_install_oserror_is_reported_not_raised() -> None:
    def raiser(args: object) -> object:
        raise OSError("pip not found")

    result = _sklearn.install(runner=raiser)
    assert result.ok is False
    assert result.returncode is None
    assert "pip not found" in result.output


def test_install_result_is_a_namedtuple() -> None:
    result = _sklearn.install(runner=lambda args: _FakeCompleted(0))
    ok, args, returncode, output = result
    assert ok is True
    assert isinstance(args, list)
    assert returncode == 0
    assert isinstance(output, str)


def test_constants() -> None:
    assert _sklearn.SKLEARN_MODULE == "sklearn"
    assert _sklearn.SKLEARN_DISTRIBUTION == "scikit-learn"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
