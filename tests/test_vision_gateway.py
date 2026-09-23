"""Tests for VisionGateway polling and send logic."""

from __future__ import annotations

import io
from typing import Any

import pytest

from wechat_agent.access_layer.vision.gateway import VisionGateway
from wechat_agent.access_layer.vision.models import (
    ExtractedChat,
    ExtractedMessage,
    VisionTarget,
    WindowInfo,
)
from wechat_agent.config import Settings, settings

pytest.importorskip("PIL")


def _make_image(seed: int = 0) -> bytes:
    """Return a noisy PNG for hashing; different seeds produce different hashes."""
    import random

    from PIL import Image

    rng = random.Random(seed)
    data = bytes(rng.randint(0, 255) for _ in range(100 * 100 * 3))
    img = Image.frombytes("RGB", (100, 100), data)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class FakeBackend:
    """Vision backend that returns canned data and records send calls."""

    platform = "fake"

    def __init__(self, image_bytes: bytes) -> None:
        self.image_bytes = image_bytes
        self.window = WindowInfo(
            handle="fake-handle",
            title="微信",
            x=0,
            y=0,
            width=100,
            height=100,
            platform="fake",
        )
        self.sent: list[str] = []

    def is_available(self) -> bool:
        return True

    def find_window(self, title_pattern: str) -> WindowInfo | None:
        return self.window

    def capture(self, window: WindowInfo) -> bytes:
        return self.image_bytes

    def activate(self, window: WindowInfo) -> bool:
        return True

    def send_text(self, window: WindowInfo, text: str) -> bool:
        self.sent.append(text)
        return True


class FakeExtractor:
    """Extractor that returns a canned ExtractedChat."""

    def __init__(self, chat: ExtractedChat | None) -> None:
        self.chat = chat
        self.calls = 0

    def extract(self, image_b64: str, targets: list[VisionTarget]) -> ExtractedChat | None:
        self.calls += 1
        return self.chat


def _build_settings(targets: list[VisionTarget], dry_run: bool = False) -> Settings:
    import json

    return settings.model_copy(
        update={
            "vision_targets": json.dumps(
                [{"friend_name": t.friend_name, "chat_id": t.chat_id} for t in targets],
                ensure_ascii=False,
            ),
            "vision_dry_run": dry_run,
            "vision_poll_interval": 0.1,
        }
    )


def test_tick_emits_new_friend_message() -> None:
    image = _make_image()
    backend = FakeBackend(image)
    chat = ExtractedChat(
        chat_title="Alice",
        messages=[ExtractedMessage(sender_name="Alice", sender_id="Alice", content="hello")],
    )
    extractor = FakeExtractor(chat)
    gateway = VisionGateway(
        backend=backend,
        extractor=extractor,
        settings_obj=_build_settings([VisionTarget(friend_name="Alice", chat_id="alice")]),
    )

    received: list[Any] = []
    gateway._handler = lambda msg: received.append(msg)
    gateway._window = backend.window

    gateway._tick()

    assert len(received) == 1
    assert received[0].chat_id == "alice"
    assert received[0].content == "hello"
    assert received[0].sender_id == "Alice"


def test_tick_deduplicates_same_message() -> None:
    backend = FakeBackend(_make_image())
    chat = ExtractedChat(
        chat_title="Alice",
        messages=[ExtractedMessage(sender_name="Alice", sender_id="Alice", content="hello")],
    )
    extractor = FakeExtractor(chat)
    gateway = VisionGateway(
        backend=backend,
        extractor=extractor,
        settings_obj=_build_settings([VisionTarget(friend_name="Alice", chat_id="alice")]),
    )

    received: list[Any] = []
    gateway._handler = lambda msg: received.append(msg)
    gateway._window = backend.window

    gateway._tick()
    # Change the image so the hash diff does not short-circuit the second tick.
    backend.image_bytes = _make_image(seed=1)
    gateway._tick()

    assert len(received) == 1
    assert extractor.calls == 2


def test_tick_ignores_self_message() -> None:
    image = _make_image()
    backend = FakeBackend(image)
    chat = ExtractedChat(
        chat_title="Alice",
        messages=[ExtractedMessage(sender_name="我", sender_id="me", content="ok")],
    )
    extractor = FakeExtractor(chat)
    gateway = VisionGateway(
        backend=backend,
        extractor=extractor,
        settings_obj=_build_settings([VisionTarget(friend_name="Alice", chat_id="alice")]),
    )

    received: list[Any] = []
    gateway._handler = lambda msg: received.append(msg)
    gateway._window = backend.window

    gateway._tick()

    assert len(received) == 0


def test_tick_ignores_empty_friend_message() -> None:
    image = _make_image()
    backend = FakeBackend(image)
    chat = ExtractedChat(
        chat_title="Alice",
        messages=[
            ExtractedMessage(sender_name="Alice", sender_id="Alice", content=""),
            ExtractedMessage(sender_name="我", sender_id="me", content="ok"),
        ],
    )
    extractor = FakeExtractor(chat)
    gateway = VisionGateway(
        backend=backend,
        extractor=extractor,
        settings_obj=_build_settings([VisionTarget(friend_name="Alice", chat_id="alice")]),
    )

    received: list[Any] = []
    gateway._handler = lambda msg: received.append(msg)
    gateway._window = backend.window

    gateway._tick()

    assert len(received) == 0


def test_send_message_dry_run_does_not_call_backend() -> None:
    image = _make_image()
    backend = FakeBackend(image)
    gateway = VisionGateway(
        backend=backend,
        extractor=FakeExtractor(None),
        settings_obj=_build_settings(
            [VisionTarget(friend_name="Alice", chat_id="alice")],
            dry_run=True,
        ),
    )
    gateway._window = backend.window

    result = gateway.send_message("alice", "reply")

    assert result is True
    assert backend.sent == []
    assert gateway._last_sent_digest == "alice:me:reply"
    # Sending a reply must not overwrite the last seen friend message digest.
    assert gateway._last_seen_digest == ""


def test_send_message_auto_send_calls_backend() -> None:
    image = _make_image()
    backend = FakeBackend(image)
    gateway = VisionGateway(
        backend=backend,
        extractor=FakeExtractor(None),
        settings_obj=_build_settings(
            [VisionTarget(friend_name="Alice", chat_id="alice")],
            dry_run=False,
        ),
    )
    gateway._window = backend.window

    result = gateway.send_message("alice", "reply")

    assert result is True
    assert backend.sent == ["reply"]
    assert gateway._last_sent_digest == "alice:me:reply"
    assert gateway._last_seen_digest == ""


def test_send_message_does_not_cause_duplicate_friend_reply() -> None:
    """After sending a reply, the same friend message must not trigger again."""
    image = _make_image()
    backend = FakeBackend(image)
    chat = ExtractedChat(
        chat_title="Alice",
        messages=[ExtractedMessage(sender_name="Alice", sender_id="Alice", content="hello")],
    )
    extractor = FakeExtractor(chat)
    gateway = VisionGateway(
        backend=backend,
        extractor=extractor,
        settings_obj=_build_settings(
            [VisionTarget(friend_name="Alice", chat_id="alice")],
            dry_run=False,
        ),
    )
    gateway._window = backend.window

    received: list[Any] = []
    gateway._handler = lambda msg: received.append(msg)

    # First tick: process friend message and "send" reply.
    gateway._tick()
    assert len(received) == 1
    gateway.send_message("alice", "reply")

    # Second tick with the same screenshot should not emit again.
    backend.image_bytes = _make_image(seed=1)
    gateway._tick()
    assert len(received) == 1


class SlowBackend(FakeBackend):
    """Backend that returns None for the first N find_window calls."""

    def __init__(self, image_bytes: bytes, fail_count: int = 0) -> None:
        super().__init__(image_bytes)
        self.fail_count = fail_count
        self.find_calls = 0

    def find_window(self, title_pattern: str) -> WindowInfo | None:
        self.find_calls += 1
        if self.find_calls <= self.fail_count:
            return None
        return self.window


def test_start_waits_for_window() -> None:
    backend = SlowBackend(_make_image(), fail_count=2)
    gateway = VisionGateway(
        backend=backend,
        extractor=FakeExtractor(None),
        settings_obj=_build_settings(
            [VisionTarget(friend_name="Alice", chat_id="alice")],
            dry_run=True,
        ),
    )
    gateway._settings = gateway._settings.model_copy(
        update={"vision_window_wait_timeout": 2.0}
    )

    received: list[Any] = []
    gateway.start(lambda msg: received.append(msg))

    assert backend.find_calls >= 2
    assert gateway._window is backend.window
    gateway.stop()


def test_start_raises_when_window_times_out() -> None:
    backend = SlowBackend(_make_image(), fail_count=999)
    gateway = VisionGateway(
        backend=backend,
        extractor=FakeExtractor(None),
        settings_obj=_build_settings(
            [VisionTarget(friend_name="Alice", chat_id="alice")],
            dry_run=True,
        ),
    )
    gateway._settings = gateway._settings.model_copy(
        update={"vision_window_wait_timeout": 0.2}
    )

    with pytest.raises(RuntimeError, match="No WeChat window matching"):
        gateway.start(lambda msg: None)
