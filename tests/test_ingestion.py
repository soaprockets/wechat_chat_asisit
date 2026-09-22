"""Tests for chat history ingestion parsers."""

import json
from pathlib import Path
from typing import Any

import pytest

from wechat_agent.ingestion import ChatHistoryImporter, get_parser
from wechat_agent.ingestion.parsers import ImageParser
from tests.conftest import FakeLLM


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    path = tmp_path / "chat.txt"
    path.write_text("Alice: hello\n我: hi\n", encoding="utf-8")
    return path


@pytest.fixture
def sample_json(tmp_path: Path) -> Path:
    path = tmp_path / "chat.json"
    messages = [
        {"sender_id": "friend_001", "sender_name": "Alice", "content": "周末有空吗？"},
        {"sender_id": "me", "sender_name": "我", "content": "有空"},
    ]
    path.write_text(json.dumps(messages, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    path = tmp_path / "chat.csv"
    path.write_text("sender_id,sender_name,content\nfriend_001,Alice,hello\nme,我,hi\n", encoding="utf-8")
    return path


def test_text_parser(sample_txt: Path) -> None:
    parser = get_parser(sample_txt)
    messages = parser.parse(sample_txt, chat_id="c1")
    assert len(messages) == 2
    assert messages[0]["sender_name"] == "Alice"
    assert messages[0]["content"] == "hello"
    assert messages[1]["sender_name"] == "我"


def test_json_parser(sample_json: Path) -> None:
    parser = get_parser(sample_json)
    messages = parser.parse(sample_json, chat_id="c1")
    assert len(messages) == 2
    assert messages[0]["chat_id"] == "c1"
    assert messages[0]["sender_name"] == "Alice"
    assert messages[1]["sender_id"] == "me"


def test_csv_parser(sample_csv: Path) -> None:
    parser = get_parser(sample_csv)
    messages = parser.parse(sample_csv, chat_id="c1")
    assert len(messages) == 2
    assert messages[0]["sender_id"] == "friend_001"
    assert messages[1]["content"] == "hi"


def test_importer_builds_profile(sample_json: Path, tmp_path: Path) -> None:
    importer = ChatHistoryImporter()
    profile = importer.import_file(sample_json, chat_id="friend_test")
    assert profile["chat_id"] == "friend_test"
    assert profile["relationship"]["conversation_count"] == 2


def test_unsupported_format(tmp_path: Path) -> None:
    path = tmp_path / "chat.xyz"
    path.write_text("data", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file format"):
        get_parser(path)


def test_image_parser_extracts_chat_title_and_messages(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """ImageParser should capture the chat title at the top of the screenshot."""
    parser = ImageParser()
    monkeypatch.setattr(parser, "_llm", FakeLLM(
        '{"chat_title": "Alice", "messages": ['
        '{"sender_name": "Alice", "content": "hello"},'
        '{"sender_name": "我", "content": "hi"}'
        "]}"
    ))
    monkeypatch.setattr(ImageParser, "_encode_image", staticmethod(lambda _path: "fake"))

    messages = parser.parse(tmp_path / "fake.png", chat_id="c1")

    assert parser.chat_title == "Alice"
    assert len(messages) == 2
    assert messages[0]["sender_name"] == "Alice"
    assert messages[1]["sender_id"] == "me"


def test_image_parser_falls_back_to_array_format(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """ImageParser should still handle the legacy array-only LLM response."""
    parser = ImageParser()
    monkeypatch.setattr(parser, "_llm", FakeLLM(
        '[{"sender_name": "Bob", "content": "ok"}]'
    ))
    monkeypatch.setattr(ImageParser, "_encode_image", staticmethod(lambda _path: "fake"))

    messages = parser.parse(tmp_path / "fake.png", chat_id="c1")

    assert parser.chat_title is None
    assert len(messages) == 1
    assert messages[0]["sender_name"] == "Bob"
    assert messages[0]["content"] == "ok"
