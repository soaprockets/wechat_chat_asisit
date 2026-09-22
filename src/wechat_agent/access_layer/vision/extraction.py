"""Extract structured chat messages from a screenshot using a vision LLM."""

from __future__ import annotations

import json
import re
from typing import Any, cast

import structlog

from wechat_agent.access_layer.vision.models import ExtractedChat, ExtractedMessage, VisionTarget
from wechat_agent.llm.providers import LLMProvider, get_provider

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a WeChat chat window parser. Look at the screenshot and extract the "
    "chat title at the top and the latest visible messages (at most 10). "
    "Green bubbles are from the user (sender_name = '我'); white/gray bubbles are "
    "from the contact. Output ONLY a compact JSON object with this shape:\n"
    '{"chat_title": "...", "messages": [{"sender_name": "...", "content": "..."}]}\n'
    "Do not include markdown, explanations, or any other text."
)

USER_PROMPT = (
    "Parse this WeChat chat screenshot. Identify the chat title and the most recent "
    "messages (no more than 10). Return only the compact JSON object."
)


class VisionExtractor:
    """Use a multimodal LLM to turn a chat screenshot into structured messages."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm or get_provider()

    def extract(self, image_b64: str, targets: list[VisionTarget]) -> ExtractedChat | None:
        """Extract messages from a base64 PNG and filter by configured targets."""
        try:
            response = self._llm.generate(
                USER_PROMPT,
                system=SYSTEM_PROMPT,
                temperature=0.0,
                images=[image_b64],
                max_tokens=2048,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("vision_extraction_failed", error=str(exc))
            return None

        data = self._parse_json(response)
        if data is None:
            return None

        chat_title = data.get("chat_title", "").strip()
        if not chat_title:
            logger.debug("vision_extraction_no_title")
            return None

        if targets and not self._match_target(chat_title, targets):
            logger.debug("vision_extraction_title_mismatch", chat_title=chat_title)
            return None

        raw_messages = data.get("messages", [])
        if not isinstance(raw_messages, list):
            return None

        messages: list[ExtractedMessage] = []
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            sender_name = (msg.get("sender_name") or "对方").strip()
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            sender_id = "me" if sender_name == "我" else sender_name
            messages.append(
                ExtractedMessage(
                    sender_name=sender_name,
                    sender_id=sender_id,
                    content=content,
                )
            )

        if not messages:
            return None
        return ExtractedChat(chat_title=chat_title, messages=messages)

    @staticmethod
    def _parse_json(response: str) -> dict[str, Any] | None:
        """Best-effort extraction of a JSON object from the LLM response."""
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            return None
        try:
            return cast(dict[str, Any], json.loads(match.group(0)))
        except json.JSONDecodeError:
            logger.debug("vision_extraction_json_parse_failed", response=response[:200])
            return None

    @staticmethod
    def _match_target(chat_title: str, targets: list[VisionTarget]) -> bool:
        """Return True if the chat title matches any configured target."""
        title_lower = chat_title.lower()
        return any(target.friend_name.lower() in title_lower for target in targets)
