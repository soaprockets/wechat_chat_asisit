"""macOS implementation of the vision backend using Quartz and AppleScript."""

from __future__ import annotations

import io
import re
import subprocess
import time
from typing import Any

import structlog

from wechat_agent.access_layer.vision.backends.base import BaseVisionBackend
from wechat_agent.access_layer.vision.models import WindowInfo

logger = structlog.get_logger(__name__)


class MacOSVisionBackend(BaseVisionBackend):
    """Screenshot and keyboard automation on macOS via native APIs."""

    platform = "macos"

    def __init__(self) -> None:
        self._Quartz: Any = None

    def _import_deps(self) -> None:
        """Lazy-import optional dependencies."""
        if self._Quartz is None:
            import Quartz

            self._Quartz = Quartz

    def is_available(self) -> bool:
        """Return True if all macOS dependencies are importable."""
        try:
            self._import_deps()
            return True
        except ImportError as exc:
            logger.warning("macos_vision_deps_missing", error=str(exc))
            return False

    def find_window(self, title_pattern: str) -> WindowInfo | None:
        """Find the first visible WeChat window matching the title regex."""
        self._import_deps()
        quartz = self._Quartz
        pattern = re.compile(title_pattern)

        window_list = quartz.CGWindowListCopyWindowInfo(
            quartz.kCGWindowListOptionOnScreenOnly
            | quartz.kCGWindowListExcludeDesktopElements,
            quartz.kCGNullWindowID,
        )
        for info in window_list:
            title = info.get(quartz.kCGWindowName, "") or ""
            if not title or not pattern.search(title):
                continue

            bounds = info.get(quartz.kCGWindowBounds)
            if not bounds:
                continue

            width = int(bounds.get("Width", 0))
            height = int(bounds.get("Height", 0))
            if width <= 0 or height <= 0:
                continue

            window_id = info.get(quartz.kCGWindowNumber)
            if not window_id:
                continue

            return WindowInfo(
                handle=window_id,
                title=title,
                x=int(bounds.get("X", 0)),
                y=int(bounds.get("Y", 0)),
                width=width,
                height=height,
                platform=self.platform,
            )
        return None

    def capture(self, window: WindowInfo) -> bytes:
        """Capture the window and return PNG bytes.

        Uses Quartz ``CGWindowListCreateImage`` so the window can be captured
        even if partially occluded, as long as Screen Recording permission is
        granted. Falls back to a screen-region capture when Quartz returns no
        image (e.g. permission not yet allowed).
        """
        self._import_deps()
        quartz = self._Quartz

        try:
            cg_image = quartz.CGWindowListCreateImage(
                quartz.CGRectNull,
                quartz.kCGWindowListOptionIncludingWindow,
                window.handle,
                quartz.kCGWindowImageBoundsIgnoreFraming,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Screen capture failed. Ensure Screen Recording permission is granted."
            ) from exc

        if cg_image is None:
            logger.debug("quartz_window_capture_none", window_id=window.handle)
            return self._capture_region_fallback(window)

        return self._cgimage_to_png(cg_image)

    def _cgimage_to_png(self, cg_image: Any) -> bytes:
        """Serialize a CGImage to PNG bytes via ImageIO."""
        quartz = self._Quartz
        data = quartz.CFDataCreateMutable(quartz.kCFAllocatorDefault, 0)
        destination = quartz.CGImageDestinationCreateWithData(
            data, "public.png", 1, None
        )
        if destination is None:
            raise RuntimeError("Failed to create PNG image destination")

        quartz.CGImageDestinationAddImage(destination, cg_image, None)
        if not quartz.CGImageDestinationFinalize(destination):
            raise RuntimeError("Failed to finalize PNG image destination")

        return bytes(data)

    def _capture_region_fallback(self, window: WindowInfo) -> bytes:
        """Capture the on-screen region occupied by the window."""
        try:
            from PIL import ImageGrab
        except ImportError as exc:
            raise RuntimeError(
                "Screen capture failed and Pillow ImageGrab is not available"
            ) from exc

        bbox = (window.x, window.y, window.x + window.width, window.y + window.height)
        img = ImageGrab.grab(bbox=bbox)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def activate(self, window: WindowInfo) -> bool:
        """Bring the WeChat window to the foreground."""
        script = 'tell application "WeChat" to activate'
        return self._run_applescript(script)

    def send_text(self, window: WindowInfo, text: str) -> bool:
        """Set the clipboard and send the text via WeChat's input box.

        Uses AppleScript UI scripting (``click at`` inside the WeChat process)
        to focus the input box, paste, and press Return. This tends to work
        even when WeChat's hardened runtime ignores global synthetic events.
        """
        if not self.activate(window):
            return False

        # Click target: roughly the center of the input box at the bottom of
        # the chat pane. Empirically these ratios land inside WeChat's input box.
        input_x = int(window.x + window.width * 0.55)
        input_y = int(window.y + window.height * 0.96)

        escaped_text = text.replace("\\", "\\\\").replace('"', '\\"')
        script = f'''
            set the clipboard to "{escaped_text}"
            tell application "System Events"
                tell process "WeChat"
                    set frontmost to true
                    delay 0.2
                    click at {{{input_x}, {input_y}}}
                    delay 0.2
                    keystroke "v" using command down
                    delay 0.2
                    key code 36
                end tell
            end tell
        '''
        if not self._run_applescript(script):
            return False

        # Give WeChat a moment to send before the next poll tick.
        time.sleep(0.2)
        return True

    @staticmethod
    def _run_applescript(script: str) -> bool:
        """Execute an AppleScript and log failures."""
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("applescript_failed", error=str(exc))
            return False

        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "不允许发送按键" in stderr or "not allowed to send keystrokes" in stderr:
                logger.error(
                    "macos_accessibility_permission_denied",
                    stderr=stderr,
                    help=(
                        "Grant Accessibility permission to your terminal and Python "
                        "in System Settings > Privacy & Security > Accessibility."
                    ),
                )
            else:
                logger.warning(
                    "applescript_error",
                    stderr=stderr,
                    stdout=result.stdout.strip(),
                )
            return False
        return True
