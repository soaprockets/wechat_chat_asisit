"""Profile builder agent: extract structured profiles from historical chats."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import structlog

from wechat_agent.memory.profile_store import ProfileStore, get_profile_store

logger = structlog.get_logger(__name__)

DEFAULT_PROFILE_SCHEMA = {
    "chat_id": "",
    "chat_type": "private",
    "user_name": "我",
    "friend_name": "",
    "my_style": {
        "tone": "casual",
        "emoji_freq": 0.5,
        "avg_sentence_len": 12,
        "catchphrases": [],
        "punctuation_habits": [],
        "response_delay_avg": 180,
    },
    "friend_style": {
        "tone": "casual",
        "topics": [],
        "active_hours": [],
    },
    "relationship": {
        "intimacy": 0.5,
        "last_active": "",
        "conversation_count": 0,
    },
    "metadata": {
        "version": 1,
        "confidence": 0.8,
        "created_at": "",
    },
}


class ProfileBuilderAgent:
    """Build or seed a structured profile from historical message data."""

    def __init__(self, store: ProfileStore | None = None) -> None:
        self._store = store or get_profile_store()

    def build_from_file(self, chat_id: str, history_file: str | Path) -> dict[str, Any]:
        """Load a chat history JSON file and extract a profile."""
        path = Path(history_file)
        with open(path, encoding="utf-8") as f:
            history = json.load(f)
        return self.build_from_history(chat_id, history)

    def build_from_history(
        self,
        chat_id: str,
        history: list[dict[str, Any]],
        friend_name: str | None = None,
    ) -> dict[str, Any]:
        """Extract a profile from a list of raw message dicts."""
        profile = self._empty_profile(chat_id)

        # Basic statistics
        contents = [msg.get("content", "") for msg in history if msg.get("content")]
        timestamps = [
            datetime.fromisoformat(msg["timestamp"]) for msg in history if msg.get("timestamp")
        ]

        profile["relationship"]["conversation_count"] = len(history)
        if timestamps:
            profile["relationship"]["last_active"] = max(timestamps).isoformat()
            profile["friend_style"]["active_hours"] = sorted({t.hour for t in timestamps})

        # Simple heuristic style extraction
        my_messages = [m for m in history if m.get("sender_id") == "me"]
        friend_messages = [m for m in history if m.get("sender_id") != "me"]

        profile["friend_name"] = (
            friend_name
            or self._most_common_sender_name(friend_messages)
            or chat_id
        )

        profile["my_style"]["avg_sentence_len"] = self._avg_length(
            [m.get("content", "") for m in my_messages]
        )
        profile["friend_style"]["tone"] = self._estimate_tone(friend_messages)
        profile["friend_style"]["topics"] = self._extract_topics(contents)

        # Estimate intimacy from message frequency and length
        if len(history) > 100:
            profile["relationship"]["intimacy"] = 0.85
        elif len(history) > 20:
            profile["relationship"]["intimacy"] = 0.6
        else:
            profile["relationship"]["intimacy"] = 0.4

        self._store.save_profile(chat_id, profile)
        logger.info("profile_built", chat_id=chat_id, messages=len(history))
        return profile

    def get_profile(self, chat_id: str) -> dict[str, Any] | None:
        """Load an existing profile."""
        return self._store.get_profile(chat_id)

    def get_or_create_profile(self, chat_id: str) -> dict[str, Any]:
        """Load an existing profile, or create and persist a default one."""
        profile = self.get_profile(chat_id)
        if profile:
            return profile
        profile = self._empty_profile(chat_id)
        self._store.save_profile(chat_id, profile)
        logger.info("default_profile_created", chat_id=chat_id)
        return profile

    def _empty_profile(self, chat_id: str) -> dict[str, Any]:
        """Return a fresh profile template."""
        profile = cast(dict[str, Any], json.loads(json.dumps(DEFAULT_PROFILE_SCHEMA)))
        profile["chat_id"] = chat_id
        profile["metadata"]["created_at"] = datetime.now(timezone.utc).isoformat()
        return profile

    @staticmethod
    def _avg_length(contents: list[str]) -> int:
        """Average sentence length in characters."""
        if not contents:
            return 12
        return round(sum(len(c) for c in contents) / len(contents))

    @staticmethod
    def _estimate_tone(messages: list[dict[str, Any]]) -> str:
        """Estimate friend tone from punctuation and emoji usage."""
        text = "".join(m.get("content", "") for m in messages)
        if "!" in text or "！" in text or any(e in text for e in "😂🤣❤️👍"):
            return "casual"
        if any(word in text for word in ["您好", "请问", "谢谢", "麻烦"]):
            return "formal"
        return "casual"

    @staticmethod
    def _extract_topics(contents: list[str]) -> list[str]:
        """Simple keyword-based topic extraction."""
        topics = []
        text = " ".join(contents)
        topic_keywords = {
            "工作": ["工作", "项目", "开会", "加班", "老板", "同事"],
            "技术": ["代码", "程序", "bug", "开发", "AI", "算法"],
            "生活": ["吃饭", "电影", "旅游", "周末", "健身"],
            "家庭": ["爸妈", "家里", "孩子", "亲戚"],
        }
        for topic, keywords in topic_keywords.items():
            if any(kw in text for kw in keywords):
                topics.append(topic)
        return topics[:5]

    @staticmethod
    def _most_common_sender_name(messages: list[dict[str, Any]]) -> str | None:
        """Return the most common sender_name among the given messages.

        Names that contain only punctuation, whitespace or symbols are ignored,
        because vision models sometimes return garbage like "!!!" for chat
        titles that could not be read clearly.
        """
        import re
        from collections import Counter

        meaningful = re.compile(r"[\w一-鿿]")
        names = [
            name.strip()
            for m in messages
            if (name := m.get("sender_name", ""))
            and meaningful.search(name)
        ]
        if not names:
            return None
        return cast(str, Counter(names).most_common(1)[0][0])
