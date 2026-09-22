"""Tests for the session manager agent."""

from datetime import datetime, timezone

from wechat_agent.models.message import WeChatMessage
from wechat_agent.models.state import AgentState


def test_load_session_empty(session_manager_agent) -> None:
    state: AgentState = {
        "message": WeChatMessage(
            message_id="m1",
            chat_id="c1",
            sender_id="friend",
            sender_name="Friend",
            content="hi",
        )
    }
    result = session_manager_agent.load_session(state)
    assert result["chat_history"] == []


def test_save_turn_with_reply(session_manager_agent) -> None:
    msg = WeChatMessage(
        message_id="m1",
        chat_id="c1",
        sender_id="friend",
        sender_name="Friend",
        content="hi",
        timestamp=datetime.now(timezone.utc),
    )
    state: AgentState = {"message": msg, "sent_reply": "hello"}
    session_manager_agent.save_turn(state)

    history = session_manager_agent._store.get_history("c1")
    assert len(history) == 2
    assert history[0]["content"] == "hi"
    assert history[1]["content"] == "hello"
    assert history[1]["sender_id"] == "me"


