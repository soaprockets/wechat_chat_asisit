"""Shared test fixtures."""

from datetime import datetime, timezone
from typing import Any

import pytest

from wechat_agent.access_layer.mock_gateway import MockGateway
from wechat_agent.agents.auto_reply import AutoReplyAgent
from wechat_agent.agents.profile_builder import ProfileBuilderAgent
from wechat_agent.agents.safety_guard import SafetyGuardAgent
from wechat_agent.agents.session_manager import SessionManagerAgent
from wechat_agent.llm.providers import LLMProvider
from wechat_agent.memory.profile_store import JsonProfileStore
from wechat_agent.memory.session_store import InMemorySessionStore
from wechat_agent.models.message import WeChatMessage


class FakeLLM(LLMProvider):
    """A fake LLM that returns canned responses."""

    name = "fake"

    def __init__(self, response: str = "好的") -> None:
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


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def session_store() -> InMemorySessionStore:
    return InMemorySessionStore()


@pytest.fixture
def profile_store(tmp_path) -> JsonProfileStore:
    return JsonProfileStore(base_path=str(tmp_path / "profiles"))


@pytest.fixture
def sample_message() -> WeChatMessage:
    return WeChatMessage(
        message_id="msg_001",
        chat_id="friend_001",
        sender_id="friend_001",
        sender_name="Alice",
        content="周末有空吃饭吗？",
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_profile() -> dict[str, Any]:
    return {
        "chat_id": "friend_001",
        "chat_type": "private",
        "user_name": "我",
        "my_style": {
            "tone": "casual",
            "emoji_freq": 0.5,
            "avg_sentence_len": 12,
            "catchphrases": ["哈哈哈", "确实"],
            "punctuation_habits": ["~", "！"],
            "response_delay_avg": 180,
        },
        "friend_style": {
            "tone": "casual",
            "topics": ["生活"],
            "active_hours": [12, 18],
        },
        "relationship": {
            "intimacy": 0.7,
            "last_active": "2026-09-20T10:00:00+00:00",
            "conversation_count": 50,
        },
        "metadata": {
            "version": 1,
            "confidence": 0.8,
            "created_at": "2026-09-21T10:00:00+00:00",
        },
    }


@pytest.fixture
def mock_gateway() -> MockGateway:
    return MockGateway()


@pytest.fixture
def session_manager_agent(session_store) -> SessionManagerAgent:
    return SessionManagerAgent(store=session_store)


@pytest.fixture
def auto_reply_agent(fake_llm) -> AutoReplyAgent:
    return AutoReplyAgent(llm=fake_llm)


@pytest.fixture
def safety_guard_agent() -> SafetyGuardAgent:
    return SafetyGuardAgent(keywords=["转账", "密码"])


@pytest.fixture
def profile_builder_agent(profile_store) -> ProfileBuilderAgent:
    return ProfileBuilderAgent(store=profile_store)
