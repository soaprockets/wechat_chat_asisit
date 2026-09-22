"""WeChat message data models."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MessageType(str, Enum):
    """Supported WeChat message types."""

    TEXT = "text"
    IMAGE = "image"
    VOICE = "voice"
    VIDEO = "video"
    FILE = "file"
    EMOTICON = "emoticon"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class WeChatMessage:
    """A single WeChat message."""

    message_id: str
    chat_id: str
    sender_id: str
    sender_name: str
    content: str
    message_type: MessageType = MessageType.TEXT
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_group: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for storage."""
        return {
            "message_id": self.message_id,
            "chat_id": self.chat_id,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "content": self.content,
            "message_type": self.message_type.value,
            "timestamp": self.timestamp.isoformat(),
            "is_group": self.is_group,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WeChatMessage":
        """Deserialize from a dict."""
        return cls(
            message_id=data["message_id"],
            chat_id=data["chat_id"],
            sender_id=data["sender_id"],
            sender_name=data["sender_name"],
            content=data["content"],
            message_type=MessageType(data.get("message_type", "text")),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            is_group=data.get("is_group", False),
            extra=data.get("extra", {}),
        )
