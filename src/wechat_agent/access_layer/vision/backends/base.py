"""Abstract backend for vision-based window capture and input."""

from abc import ABC, abstractmethod

from wechat_agent.access_layer.vision.models import WindowInfo


class BaseVisionBackend(ABC):
    """Platform-specific screenshot and keyboard automation backend."""

    platform: str = ""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend can run on this platform."""

    @abstractmethod
    def find_window(self, title_pattern: str) -> WindowInfo | None:
        """Find a window whose title matches the regex pattern."""

    @abstractmethod
    def capture(self, window: WindowInfo) -> bytes:
        """Capture the window and return PNG bytes."""

    @abstractmethod
    def activate(self, window: WindowInfo) -> bool:
        """Bring the window to the foreground."""

    @abstractmethod
    def send_text(self, window: WindowInfo, text: str) -> bool:
        """Type/paste the given text into the active window and submit it."""
