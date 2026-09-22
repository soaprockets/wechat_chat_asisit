"""Vision-based WeChat gateway implementation."""

from __future__ import annotations

import base64
import io
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import structlog

from wechat_agent.access_layer.base_gateway import BaseGateway
from wechat_agent.access_layer.vision.backends.base import BaseVisionBackend
from wechat_agent.access_layer.vision.backends.factory import get_backend
from wechat_agent.access_layer.vision.extraction import VisionExtractor
from wechat_agent.access_layer.vision.models import (
    ExtractedChat,
    ExtractedMessage,
    VisionTarget,
    WindowInfo,
)
from wechat_agent.config import Settings, settings
from wechat_agent.models.message import MessageType, WeChatMessage

logger = structlog.get_logger(__name__)


def _hamming_distance(a: str, b: str) -> int:
    """Compute Hamming distance between two hex strings."""
    if len(a) != len(b):
        return 999
    return sum(c1 != c2 for c1, c2 in zip(a, b, strict=True))


class VisionGateway(BaseGateway):
    """Monitor a WeChat chat window and drive replies through the agent graph."""

    name = "vision"

    def __init__(
        self,
        backend: BaseVisionBackend | None = None,
        extractor: VisionExtractor | None = None,
        settings_obj: Settings | None = None,
    ) -> None:
        self._settings = settings_obj or settings
        self._backend = backend or get_backend()
        self._extractor = extractor or VisionExtractor()

        self._handler: Callable[[WeChatMessage], Any] | None = None
        self._window: WindowInfo | None = None
        self._targets: list[VisionTarget] = [
            VisionTarget(friend_name=t["friend_name"], chat_id=t["chat_id"])
            for t in self._settings.vision_target_list
        ]

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._last_image_hash: str | None = None
        self._last_seen_digest: str = ""

    def start(self, on_message: Callable[[WeChatMessage], Any]) -> None:
        """Find the WeChat window and start the polling thread."""
        self._handler = on_message
        self._window = self._backend.find_window(self._settings.vision_window_title_regex)
        if self._window is None:
            raise RuntimeError(
                f"No WeChat window matching '{self._settings.vision_window_title_regex}' found"
            )
        logger.info(
            "vision_gateway_started",
            title=self._window.title,
            rect=self._window.rect,
        )
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        """Polling loop."""
        while not self._stop_event.is_set():
            start_time = time.monotonic()
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001
                logger.error("vision_tick_failed", error=str(exc))
            elapsed = time.monotonic() - start_time
            sleep_time = max(0.0, self._settings.vision_poll_interval - elapsed)
            self._stop_event.wait(sleep_time)

    def _tick(self) -> None:
        """Capture, diff, extract, and possibly emit a new message."""
        with self._lock:
            if self._window is None:
                return
            image_bytes = self._backend.capture(self._window)
            image_bytes = self._resize_for_vision(image_bytes)
            image_hash = self._compute_hash(image_bytes)

            if (
                self._last_image_hash is not None
                and _hamming_distance(image_hash, self._last_image_hash)
                <= self._settings.vision_change_threshold
            ):
                return

            self._last_image_hash = image_hash
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            extracted = self._extractor.extract(image_b64, self._targets)

        if extracted is None:
            return

        newest = self._newest_friend_message(extracted)
        if newest is None:
            return

        digest = f"{extracted.chat_title}:{newest.sender_id}:{newest.content}"
        with self._lock:
            if digest == self._last_seen_digest:
                return
            self._last_seen_digest = digest
            chat_id = self._resolve_chat_id(extracted.chat_title)

        message = WeChatMessage(
            message_id=f"vision_{datetime.now(timezone.utc).isoformat()}",
            chat_id=chat_id,
            sender_id=newest.sender_id,
            sender_name=newest.sender_name,
            content=newest.content,
            message_type=MessageType.TEXT,
            timestamp=datetime.now(timezone.utc),
            is_group=False,
            extra={"chat_title": extracted.chat_title},
        )

        if self._handler is not None:
            self._handler(message)

    def send_message(self, chat_id: str, content: str) -> bool:
        """Send a reply via the backend keyboard automation."""
        if not self._settings.vision_send_enabled or self._settings.vision_dry_run:
            logger.info(
                "vision_send_skipped",
                chat_id=chat_id,
                content=content,
                dry_run=self._settings.vision_dry_run,
            )
            with self._lock:
                self._last_seen_digest = f"{chat_id}:me:{content}"
            return True

        with self._lock:
            if self._window is None:
                return False
            result = self._backend.send_text(self._window, content)
            if result:
                self._last_seen_digest = f"{chat_id}:me:{content}"
            return result

    def stop(self) -> None:
        """Stop the polling thread."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("vision_gateway_stopped")

    @staticmethod
    def _newest_friend_message(extracted: ExtractedChat) -> ExtractedMessage | None:
        """Return the latest message not sent by me, if any."""
        for msg in reversed(extracted.messages):
            if msg.sender_id != "me":
                return msg
        return None

    def _resolve_chat_id(self, chat_title: str) -> str:
        """Map an extracted chat title to the configured stable chat_id."""
        title_lower = chat_title.lower()
        for target in self._targets:
            if target.friend_name.lower() in title_lower:
                return target.chat_id
        return chat_title

    def _resize_for_vision(self, image_bytes: bytes) -> bytes:
        """Downscale the screenshot if it exceeds the configured max width."""
        max_width = self._settings.vision_image_max_width
        if max_width <= 0:
            return image_bytes
        try:
            from PIL import Image
        except ImportError:
            return image_bytes
        img = Image.open(io.BytesIO(image_bytes))
        if img.width <= max_width:
            return image_bytes
        ratio = max_width / img.width
        new_size = (max_width, int(img.height * ratio))
        resized = img.resize(new_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _compute_hash(image_bytes: bytes) -> str:
        """Compute a simple difference hash of the image."""
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for vision image hashing") from exc
        img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((9, 8))
        pixels = img.tobytes()
        diff_bits = []
        for row in range(8):
            row_start = row * 9
            for col in range(8):
                diff_bits.append(pixels[row_start + col] > pixels[row_start + col + 1])
        value = 0
        for bit in diff_bits:
            value = (value << 1) | int(bit)
        return f"{value:0{len(diff_bits) // 4}x}"
