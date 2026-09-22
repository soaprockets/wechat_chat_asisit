"""Long-term profile storage backed by JSON files or ChromaDB."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

import structlog

from wechat_agent.config import settings

logger = structlog.get_logger(__name__)


class ProfileStore(ABC):
    """Abstract profile store."""

    @abstractmethod
    def get_profile(self, chat_id: str) -> dict[str, Any] | None:
        """Return the profile for a chat, or None if not found."""

    @abstractmethod
    def save_profile(self, chat_id: str, profile: dict[str, Any]) -> None:
        """Persist a profile."""

    @abstractmethod
    def list_profiles(self) -> list[str]:
        """Return all stored chat IDs."""


class JsonProfileStore(ProfileStore):
    """JSON-file profile store for MVP."""

    def __init__(self, base_path: str | None = None) -> None:
        self._base_path = Path(base_path or settings.profile_store_path)
        self._base_path.mkdir(parents=True, exist_ok=True)

    def _profile_path(self, chat_id: str) -> Path:
        # Simple sanitization: replace path separators.
        safe_id = chat_id.replace("/", "_").replace("\\\\", "_")
        return self._base_path / f"{safe_id}.json"

    def get_profile(self, chat_id: str) -> dict[str, Any] | None:
        path = self._profile_path(chat_id)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return cast(dict[str, Any], json.load(f))
        except (json.JSONDecodeError, OSError):
            logger.warning("profile_load_failed", chat_id=chat_id, path=str(path))
            return None

    def save_profile(self, chat_id: str, profile: dict[str, Any]) -> None:
        path = self._profile_path(chat_id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)
        except OSError:
            logger.warning("profile_save_failed", chat_id=chat_id, path=str(path))

    def list_profiles(self) -> list[str]:
        return [p.stem for p in self._base_path.glob("*.json")]


class ChromaProfileStore(ProfileStore):
    """ChromaDB-backed profile store (stub for future RAG enhancement)."""

    def __init__(self, collection_name: str = "profiles") -> None:
        self.collection_name = collection_name
        self._client: Any | None = None
        self._collection: Any | None = None
        self._init_client()

    def _init_client(self) -> None:
        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=settings.chroma_db_path)
            self._collection = self._client.get_or_create_collection(self.collection_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("chromadb_init_failed", error=str(exc))

    def get_profile(self, chat_id: str) -> dict[str, Any] | None:
        if self._collection is None:
            return None
        try:
            result = self._collection.get(ids=[chat_id], include=["metadatas"])
            metadatas = result.get("metadatas")
            if metadatas and metadatas[0]:
                return dict(metadatas[0])
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("chromadb_get_profile_failed", chat_id=chat_id, error=str(exc))
            return None

    def save_profile(self, chat_id: str, profile: dict[str, Any]) -> None:
        if self._collection is None:
            return
        try:
            self._collection.upsert(
                ids=[chat_id],
                metadatas=[profile],
                documents=[json.dumps(profile, ensure_ascii=False)],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("chromadb_save_profile_failed", chat_id=chat_id, error=str(exc))

    def list_profiles(self) -> list[str]:
        if self._collection is None:
            return []
        try:
            result = self._collection.get()
            return cast(list[str], result.get("ids", []))
        except Exception as exc:  # noqa: BLE001
            logger.warning("chromadb_list_profiles_failed", error=str(exc))
            return []


def get_profile_store() -> ProfileStore:
    """Return the configured profile store."""
    if settings.profile_store_type == "chroma":
        return ChromaProfileStore()
    return JsonProfileStore()
