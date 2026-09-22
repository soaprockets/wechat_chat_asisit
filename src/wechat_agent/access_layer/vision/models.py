"""Data models for the vision-based WeChat gateway."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WindowInfo:
    """A handle to an on-screen window."""

    handle: Any
    title: str
    x: int
    y: int
    width: int
    height: int
    platform: str

    @property
    def rect(self) -> tuple[int, int, int, int]:
        """Return (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)


@dataclass(frozen=True)
class ExtractedMessage:
    """A single message extracted from a screenshot."""

    sender_name: str
    sender_id: str
    content: str


@dataclass(frozen=True)
class ExtractedChat:
    """The structured content extracted from a chat screenshot."""

    chat_title: str
    messages: list[ExtractedMessage]


@dataclass(frozen=True)
class VisionTarget:
    """Mapping from a WeChat display name to the internal chat_id."""

    friend_name: str
    chat_id: str
