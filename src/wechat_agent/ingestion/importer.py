"""High-level chat history importer.

Takes a user-provided file (text, JSON, CSV, PDF, image) and produces a
persisted profile via ``ProfileBuilderAgent``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wechat_agent.agents.profile_builder import ProfileBuilderAgent
from wechat_agent.ingestion.parsers import get_parser


class ChatHistoryImporter:
    """Import chat history from a file and build/persist a profile."""

    def __init__(self, profile_builder: ProfileBuilderAgent | None = None) -> None:
        self._profile_builder = profile_builder or ProfileBuilderAgent()

    def import_file(self, file_path: str | Path, chat_id: str) -> dict[str, Any]:
        """Parse a chat history file and build a profile for the chat.

        Args:
            file_path: Path to the chat history export file.
            chat_id: Unique ID for the chat (e.g., friend_001).

        Returns:
            The built/persisted profile dict.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        parser = get_parser(path)
        messages = parser.parse(path, chat_id=chat_id)
        if not messages:
            raise ValueError(f"No messages could be extracted from {path}")

        friend_name = getattr(parser, "chat_title", None)
        return self._profile_builder.build_from_history(
            chat_id, messages, friend_name=friend_name
        )
