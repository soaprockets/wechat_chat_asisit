"""Tests for the safety guard agent."""

from wechat_agent.agents.safety_guard import SafetyGuardAgent
from wechat_agent.models.message import WeChatMessage
from wechat_agent.models.state import AgentState


def test_keyword_block(safety_guard_agent) -> None:
    msg = WeChatMessage(
        message_id="m1",
        chat_id="c1",
        sender_id="friend",
        sender_name="Friend",
        content="借钱",
    )
    state: AgentState = {"message": msg, "candidate_reply": "把密码发我"}
    result = safety_guard_agent.check(state)
    assert result["safety_result"]["decision"] == "BLOCK"
    assert "密码" in result["safety_result"]["reason"]


def test_decide_send_passes_when_safe() -> None:
    guard = SafetyGuardAgent(keywords=[])
    state: AgentState = {
        "message": WeChatMessage(
            message_id="m1",
            chat_id="c1",
            sender_id="friend",
            sender_name="Friend",
            content="hi",
        ),
        "candidate_reply": "好的",
        "safety_result": {"decision": "PASS", "reason": "safe"},
    }
    result = guard.decide_send(state)
    assert result.get("sent_reply") == "好的"


def test_decide_send_blocks_when_unsafe() -> None:
    guard = SafetyGuardAgent(keywords=[])
    state: AgentState = {
        "message": WeChatMessage(
            message_id="m1",
            chat_id="c1",
            sender_id="friend",
            sender_name="Friend",
            content="hi",
        ),
        "candidate_reply": "test",
        "safety_result": {"decision": "BLOCK", "reason": "unsafe"},
    }
    result = guard.decide_send(state)
    assert result.get("sent_reply") is None
    assert "blocked" in result.get("error", "")
