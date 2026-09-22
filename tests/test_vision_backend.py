"""Tests for vision backend factory and platform backends."""

from __future__ import annotations

import sys

import pytest

from wechat_agent.access_layer.vision.backends.factory import get_backend
from wechat_agent.access_layer.vision.backends.macos import MacOSVisionBackend
from wechat_agent.access_layer.vision.backends.windows import WindowsVisionBackend


def _deps_available() -> bool:
    """Check whether the optional native vision dependencies are installed."""
    try:
        import PIL  # noqa: F401
        import Quartz  # noqa: F401

        return True
    except ImportError:
        return False


def test_factory_raises_without_deps() -> None:
    """get_backend should raise if platform deps are missing."""
    if _deps_available():
        pytest.skip("vision dependencies are installed")
    with pytest.raises(RuntimeError):
        get_backend()


def test_factory_selects_correct_backend_class(monkeypatch: pytest.MonkeyPatch) -> None:
    """Factory imports the correct backend class for the mocked platform."""
    monkeypatch.setattr(sys, "platform", "darwin")
    if _deps_available():
        backend = get_backend()
        assert isinstance(backend, MacOSVisionBackend)
    else:
        with pytest.raises(RuntimeError, match="macOS vision backend is not available"):
            get_backend()

    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="Windows vision backend is not available"):
        get_backend()


def test_macos_backend_can_be_instantiated() -> None:
    backend = MacOSVisionBackend()
    assert backend.platform == "macos"


def test_windows_backend_can_be_instantiated() -> None:
    backend = WindowsVisionBackend()
    assert backend.platform == "windows"


def test_windows_backend_methods_raise_not_implemented() -> None:
    backend = WindowsVisionBackend()
    with pytest.raises(NotImplementedError):
        backend.find_window(".*")
    with pytest.raises(NotImplementedError):
        backend.capture(None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        backend.activate(None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        backend.send_text(None, "hi")  # type: ignore[arg-type]
