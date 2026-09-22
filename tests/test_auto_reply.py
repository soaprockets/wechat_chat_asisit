"""Tests for the auto reply agent."""

from wechat_agent.agents.auto_reply import AutoReplyAgent
from wechat_agent.models.message import WeChatMessage
from wechat_agent.models.state import AgentState


def test_generate_reply_uses_profile(auto_reply_agent, sample_profile, sample_message) -> None:
    state: AgentState = {
        "message": sample_message,
        "profile": sample_profile,
        "chat_history": [],
    }
    result = auto_reply_agent.generate(state)
    assert result["candidate_reply"] == "好的"


def test_prompt_contains_profile_info(fake_llm, sample_profile) -> None:
    agent = AutoReplyAgent(llm=fake_llm)
    msg = WeChatMessage(
        message_id="m1",
        chat_id="friend_001",
        sender_id="friend_001",
        sender_name="Alice",
        content="周末有空吗？",
    )
    state: AgentState = {"message": msg, "profile": sample_profile, "chat_history": []}
    result = agent.generate(state)
    assert result["candidate_reply"] == "好的"
