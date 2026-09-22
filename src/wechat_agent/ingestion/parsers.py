"""Chat history parsers for text, JSON, CSV, PDF and image files.

The parsers convert user-provided chat history exports into a normalized list
of raw message dicts. The dict schema matches ``WeChatMessage.to_dict()``:

    {
        "message_id": str (optional),
        "chat_id": str,
        "sender_id": str,
        "sender_name": str,
        "content": str,
        "timestamp": ISO-8601 string (optional),
        "message_type": "text" (default),
        "is_group": bool (default False),
    }
"""

from __future__ import annotations

import base64
import csv
import json
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wechat_agent.llm.providers import get_provider


class ChatHistoryParser(ABC):
    """Base class for chat history parsers."""

    @abstractmethod
    def parse(self, file_path: Path, chat_id: str | None = None) -> list[dict[str, Any]]:
        """Parse a file into a list of normalized message dicts."""


class TextParser(ChatHistoryParser):
    """Parse plain text files with simple ``Name: message`` lines."""

    def parse(self, file_path: Path, chat_id: str | None = None) -> list[dict[str, Any]]:
        text = file_path.read_text(encoding="utf-8")
        messages: list[dict[str, Any]] = []
        for idx, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            sender_name, content = self._split_line(line)
            messages.append(
                {
                    "message_id": f"txt_{idx}",
                    "chat_id": chat_id or "imported",
                    "sender_id": sender_name,
                    "sender_name": sender_name,
                    "content": content,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "message_type": "text",
                    "is_group": False,
                }
            )
        return messages

    @staticmethod
    def _split_line(line: str) -> tuple[str, str]:
        """Split a line like 'Alice: hello' into sender and content."""
        match = re.match(r"^(?:\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?\s+)?([^:]+):\s*(.*)$", line)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return "对方", line


class JsonParser(ChatHistoryParser):
    """Parse JSON or JSONL files containing message dicts."""

    def parse(self, file_path: Path, chat_id: str | None = None) -> list[dict[str, Any]]:
        text = file_path.read_text(encoding="utf-8")
        if file_path.suffix.lower() == ".jsonl":
            raw_items = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            data = json.loads(text)
            raw_items = data if isinstance(data, list) else data.get("messages", [])

        return [self._normalize(msg, idx, chat_id) for idx, msg in enumerate(raw_items)]

    @staticmethod
    def _normalize(msg: dict[str, Any], idx: int, chat_id: str | None) -> dict[str, Any]:
        sender = msg.get("sender_name") or msg.get("sender_id") or "对方"
        return {
            "message_id": msg.get("message_id", f"json_{idx}"),
            "chat_id": chat_id or msg.get("chat_id", "imported"),
            "sender_id": msg.get("sender_id", sender),
            "sender_name": sender,
            "content": msg.get("content", ""),
            "timestamp": msg.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "message_type": msg.get("message_type", "text"),
            "is_group": msg.get("is_group", False),
        }


class CsvParser(ChatHistoryParser):
    """Parse CSV files with chat history columns."""

    EXPECTED_COLUMNS = {"sender_id", "sender_name", "content"}

    def parse(self, file_path: Path, chat_id: str | None = None) -> list[dict[str, Any]]:
        with open(file_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            columns = set(reader.fieldnames or [])

        if not self.EXPECTED_COLUMNS.issubset(columns):
            missing = self.EXPECTED_COLUMNS - columns
            raise ValueError(f"CSV missing required columns: {missing}")

        messages: list[dict[str, Any]] = []
        with open(file_path, encoding="utf-8", newline="") as f:
            for idx, row in enumerate(csv.DictReader(f)):
                sender_name = row["sender_name"] or row["sender_id"]
                messages.append(
                    {
                        "message_id": f"csv_{idx}",
                        "chat_id": chat_id or "imported",
                        "sender_id": row["sender_id"],
                        "sender_name": sender_name,
                        "content": row["content"],
                        "timestamp": row.get("timestamp", datetime.now(timezone.utc).isoformat())
                        or datetime.now(timezone.utc).isoformat(),
                        "message_type": row.get("message_type", "text") or "text",
                        "is_group": row.get("is_group", "false").lower() == "true",
                    }
                )
        return messages


class PdfParser(ChatHistoryParser):
    """Parse PDF chat exports by extracting text and asking the LLM to structure it."""

    def __init__(self) -> None:
        self._llm = get_provider()

    def parse(self, file_path: Path, chat_id: str | None = None) -> list[dict[str, Any]]:
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise RuntimeError(
                "PDF parsing requires PyMuPDF. Install with: pip install pymupdf"
            ) from exc

        doc = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()

        if not text.strip():
            return []

        return self._llm_structured_parse(text, chat_id)

    def _llm_structured_parse(
        self, text: str, chat_id: str | None
    ) -> list[dict[str, Any]]:
        system = (
            "You are a chat history parser. Convert the following exported chat "
            "text into a JSON array of messages. Each message must have "
            "sender_name and content. Optionally include timestamp if present. "
            "Output ONLY valid JSON, no markdown."
        )
        prompt = f"Parse this chat history into JSON:\n\n{text[:8000]}"
        response = self._llm.generate(prompt, system=system, temperature=0.0)
        return self._extract_messages(response, chat_id)

    @staticmethod
    def _extract_messages(response: str, chat_id: str | None) -> list[dict[str, Any]]:
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if not match:
            return []
        try:
            raw_items = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

        messages: list[dict[str, Any]] = []
        for idx, msg in enumerate(raw_items):
            sender = msg.get("sender_name") or msg.get("sender_id") or "对方"
            messages.append(
                {
                    "message_id": f"pdf_{idx}",
                    "chat_id": chat_id or "imported",
                    "sender_id": msg.get("sender_id", sender),
                    "sender_name": sender,
                    "content": msg.get("content", ""),
                    "timestamp": msg.get(
                        "timestamp", datetime.now(timezone.utc).isoformat()
                    ),
                    "message_type": "text",
                    "is_group": False,
                }
            )
        return messages


class ImageParser(ChatHistoryParser):
    """Parse screenshot exports by sending the image directly to a vision-capable LLM."""

    def __init__(self) -> None:
        self._llm = get_provider()
        self.chat_title: str | None = None

    def parse(self, file_path: Path, chat_id: str | None = None) -> list[dict[str, Any]]:
        image_b64 = self._encode_image(file_path)
        system = (
            "You are a WeChat screenshot parser. Look at the chat screenshot and "
            "output a JSON object with two fields: chat_title (the name shown at "
            "the top of the chat window) and messages (an array of messages). "
            "For each message, output sender_name and content. Green bubbles are "
            "from the user (me), white/gray bubbles are from the other person. "
            "Use sender_name '我' for green bubbles and the contact name for the "
            "other side. Output ONLY valid JSON, no markdown."
        )
        prompt = (
            'Parse this WeChat screenshot into JSON: {"chat_title": "...", '
            '"messages": [{"sender_name": "...", "content": "..."}, ...]}'
        )
        response = self._llm.generate(
            prompt, system=system, temperature=0.0, images=[image_b64]
        )
        return self._extract_messages(response, chat_id)

    @staticmethod
    def _encode_image(file_path: Path) -> str:
        """Return the base64-encoded bytes of an image file."""
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _extract_messages(self, response: str, chat_id: str | None) -> list[dict[str, Any]]:
        """Extract messages from the LLM response and capture chat_title if present."""
        self.chat_title = None
        text = self._strip_markdown_fence(response)

        decoder = json.JSONDecoder()
        try:
            data, _ = decoder.raw_decode(text)
        except json.JSONDecodeError:
            # Fallback: scan for the first object/array if the response has
            # surrounding prose.
            data = self._first_json_value(text)

        if isinstance(data, dict):
            self.chat_title = data.get("chat_title")
            raw_items = data.get("messages", [])
            if isinstance(raw_items, list):
                return self._build_messages(raw_items, chat_id)
            return []
        if isinstance(data, list):
            return self._build_messages(data, chat_id)
        return []

    @staticmethod
    def _strip_markdown_fence(text: str) -> str:
        """Remove optional markdown ```json fence from the response."""
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.lstrip("`").strip()
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()
        return stripped

    @staticmethod
    def _first_json_value(text: str) -> Any | None:
        """Return the first JSON object or array found in text, or None."""
        for pattern in (r"\{.*\}", r"\[.*\]"):
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    continue
        return None

    @staticmethod
    def _build_messages(raw_items: list[Any], chat_id: str | None) -> list[dict[str, Any]]:
        """Convert raw message dicts into the canonical message format."""
        messages: list[dict[str, Any]] = []
        for idx, msg in enumerate(raw_items):
            if not isinstance(msg, dict):
                continue
            sender_name = msg.get("sender_name") or msg.get("sender_id") or "对方"
            sender_id = "me" if sender_name == "我" else sender_name
            messages.append(
                {
                    "message_id": f"img_{idx}",
                    "chat_id": chat_id or "imported",
                    "sender_id": sender_id,
                    "sender_name": sender_name,
                    "content": msg.get("content", ""),
                    "timestamp": msg.get(
                        "timestamp", datetime.now(timezone.utc).isoformat()
                    ),
                    "message_type": "text",
                    "is_group": False,
                }
            )
        return messages

def get_parser(file_path: Path) -> ChatHistoryParser:
    """Return the appropriate parser for a file based on its extension."""
    suffix = file_path.suffix.lower()
    if suffix in {".txt"}:
        return TextParser()
    if suffix in {".json", ".jsonl"}:
        return JsonParser()
    if suffix in {".csv"}:
        return CsvParser()
    if suffix in {".pdf"}:
        return PdfParser()
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif"}:
        return ImageParser()
    raise ValueError(f"Unsupported file format: {suffix}")
