"""Short-term session memory backed by Redis or an in-memory fallback."""

import json
from abc import ABC, abstractmethod
from typing import Any

import redis
import structlog

from wechat_agent.config import settings

logger = structlog.get_logger(__name__)

DEFAULT_HISTORY_LIMIT = 20
DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days


class SessionStore(ABC):
    """Abstract session store."""

    @abstractmethod
    def get_history(self, chat_id: str, limit: int = DEFAULT_HISTORY_LIMIT) -> list[dict[str, Any]]:
        """Return recent conversation turns for a chat."""

    @abstractmethod
    def add_message(self, chat_id: str, message: dict[str, Any]) -> None:
        """Append a message to the chat history."""

    @abstractmethod
    def clear_history(self, chat_id: str) -> None:
        """Clear history for a chat."""


class RedisSessionStore(SessionStore):
    """Redis-backed session store."""

    def __init__(self, redis_url: str | None = None, ttl: int = DEFAULT_TTL_SECONDS) -> None:
        self._client = redis.from_url(redis_url or settings.redis_url, decode_responses=True)
        self._ttl = ttl
        self._key = lambda chat_id: f"session:{chat_id}:history"

    def get_history(self, chat_id: str, limit: int = DEFAULT_HISTORY_LIMIT) -> list[dict[str, Any]]:
        key = self._key(chat_id)
        try:
            items = self._client.lrange(key, -limit, -1)
            return [json.loads(item) for item in items]
        except redis.RedisError:
            logger.warning("redis_history_failed", chat_id=chat_id)
            return []

    def add_message(self, chat_id: str, message: dict[str, Any]) -> None:
        key = self._key(chat_id)
        try:
            pipe = self._client.pipeline()
            pipe.rpush(key, json.dumps(message, ensure_ascii=False))
            pipe.ltrim(key, -DEFAULT_HISTORY_LIMIT, -1)
            pipe.expire(key, self._ttl)
            pipe.execute()
        except redis.RedisError:
            logger.warning("redis_add_message_failed", chat_id=chat_id)

    def clear_history(self, chat_id: str) -> None:
        key = self._key(chat_id)
        try:
            self._client.delete(key)
        except redis.RedisError:
            logger.warning("redis_clear_history_failed", chat_id=chat_id)


class InMemorySessionStore(SessionStore):
    """In-memory session store for local dev and tests."""

    def __init__(self) -> None:
        self._data: dict[str, list[dict[str, Any]]] = {}

    def get_history(self, chat_id: str, limit: int = DEFAULT_HISTORY_LIMIT) -> list[dict[str, Any]]:
        return self._data.get(chat_id, [])[-limit:]

    def add_message(self, chat_id: str, message: dict[str, Any]) -> None:
        self._data.setdefault(chat_id, [])
        self._data[chat_id].append(message)
        self._data[chat_id] = self._data[chat_id][-DEFAULT_HISTORY_LIMIT:]

    def clear_history(self, chat_id: str) -> None:
        self._data.pop(chat_id, None)


def get_session_store(redis_url: str | None = None) -> SessionStore:
    """Return a Redis store if available, otherwise in-memory."""
    try:
        store = RedisSessionStore(redis_url=redis_url)
        store._client.ping()
        return store
    except Exception:  # noqa: BLE001
        logger.info("using_in_memory_session_store")
        return InMemorySessionStore()
