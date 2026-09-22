"""Tests for VisionExtractor."""

from __future__ import annotations

from wechat_agent.access_layer.vision.extraction import VisionExtractor
from wechat_agent.access_layer.vision.models import ExtractedMessage, VisionTarget


class FakeVisionLLM:
    """Fake LLM that returns a canned vision-parsing response."""

    name = "fake-vision"

    def __init__(self, response: str) -> None:
        self.response = response

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        images: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return self.response


def test_extract_basic_message() -> None:
    response = (
        '{"chat_title": "沐 yao", "messages": ['
        '{"sender_name": "沐 yao", "content": "在吗"},'
        '{"sender_name": "我", "content": "在"}'
        ']}'
    )
    extractor = VisionExtractor(llm=FakeVisionLLM(response))
    result = extractor.extract("ignored", [VisionTarget(friend_name="沐 yao", chat_id="muyao")])

    assert result is not None
    assert result.chat_title == "沐 yao"
    assert result.messages == [
        ExtractedMessage(sender_name="沐 yao", sender_id="沐 yao", content="在吗"),
        ExtractedMessage(sender_name="我", sender_id="me", content="在"),
    ]


def test_extract_markdown_wrapped() -> None:
    response = (
        '```json\n'
        '{"chat_title": "Alice", "messages": [{"sender_name": "Alice", "content": "hello"}]}\n'
        '```'
    )
    extractor = VisionExtractor(llm=FakeVisionLLM(response))
    result = extractor.extract("ignored", [VisionTarget(friend_name="Alice", chat_id="alice")])

    assert result is not None
    assert result.chat_title == "Alice"
    assert result.messages[0].content == "hello"


def test_extract_no_match_target() -> None:
    response = '{"chat_title": "Bob", "messages": [{"sender_name": "Bob", "content": "hi"}]}'
    extractor = VisionExtractor(llm=FakeVisionLLM(response))
    result = extractor.extract("ignored", [VisionTarget(friend_name="Alice", chat_id="alice")])

    assert result is None


def test_extract_self_message_only() -> None:
    response = '{"chat_title": "Alice", "messages": [{"sender_name": "我", "content": "ok"}]}'
    extractor = VisionExtractor(llm=FakeVisionLLM(response))
    result = extractor.extract("ignored", [VisionTarget(friend_name="Alice", chat_id="alice")])

    assert result is not None
    assert result.messages[0].sender_id == "me"


def test_extract_empty_content_ignored() -> None:
    response = (
        '{"chat_title": "Alice", "messages": ['
        '{"sender_name": "Alice", "content": ""},'
        '{"sender_name": "Alice", "content": "real"}'
        ']}'
    )
    extractor = VisionExtractor(llm=FakeVisionLLM(response))
    result = extractor.extract("ignored", [VisionTarget(friend_name="Alice", chat_id="alice")])

    assert result is not None
    assert len(result.messages) == 1
    assert result.messages[0].content == "real"


def test_extract_malformed_json_returns_none() -> None:
    extractor = VisionExtractor(llm=FakeVisionLLM("not json"))
    result = extractor.extract("ignored", [])
    assert result is None


def test_extract_no_targets_accepts_any() -> None:
    response = '{"chat_title": "Anyone", "messages": [{"sender_name": "Anyone", "content": "?"}]}'
    extractor = VisionExtractor(llm=FakeVisionLLM(response))
    result = extractor.extract("ignored", [])

    assert result is not None
    assert result.chat_title == "Anyone"
