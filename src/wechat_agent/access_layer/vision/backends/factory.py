"""Factory for selecting the correct vision backend."""

import sys

import structlog

from wechat_agent.access_layer.vision.backends.base import BaseVisionBackend

logger = structlog.get_logger(__name__)


def get_backend() -> BaseVisionBackend:
    """Return the platform-appropriate vision backend."""
    if sys.platform == "darwin":
        from wechat_agent.access_layer.vision.backends.macos import MacOSVisionBackend

        backend = MacOSVisionBackend()
        if not backend.is_available():
            raise RuntimeError("macOS vision backend is not available")
        return backend

    if sys.platform == "win32":  # type: ignore[unreachable]
        from wechat_agent.access_layer.vision.backends.windows import WindowsVisionBackend

        backend = WindowsVisionBackend()
        if not backend.is_available():
            raise RuntimeError("Windows vision backend is not available")
        return backend

    raise RuntimeError(f"Vision gateway is not supported on platform: {sys.platform}")
