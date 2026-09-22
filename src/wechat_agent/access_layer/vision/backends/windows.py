"""Windows implementation of the vision backend (stub)."""

import structlog

from wechat_agent.access_layer.vision.backends.base import BaseVisionBackend
from wechat_agent.access_layer.vision.models import WindowInfo

logger = structlog.get_logger(__name__)


class WindowsVisionBackend(BaseVisionBackend):
    """Placeholder for Windows screenshot and keyboard automation."""

    platform = "windows"

    def is_available(self) -> bool:
        """Return True if the Windows automation libraries are installed."""
        # The Windows backend is a placeholder; avoid importing platform-only
        # libraries that may fail to initialize on non-Windows systems.
        return False

    def find_window(self, title_pattern: str) -> WindowInfo | None:
        """Find a window whose title matches the regex pattern."""
        raise NotImplementedError("Windows find_window not yet implemented")

    def capture(self, window: WindowInfo) -> bytes:
        """Capture the window and return PNG bytes."""
        raise NotImplementedError("Windows capture not yet implemented")

    def activate(self, window: WindowInfo) -> bool:
        """Bring the window to the foreground."""
        raise NotImplementedError("Windows activate not yet implemented")

    def send_text(self, window: WindowInfo, text: str) -> bool:
        """Type/paste the given text into the active window and submit it."""
        raise NotImplementedError("Windows send_text not yet implemented")
