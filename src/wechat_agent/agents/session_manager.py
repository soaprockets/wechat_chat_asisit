"""Session manager agent: load and persist short-term conversation state."""

from typing import Any

import structlog

from wechat_agent.memory.session_store import SessionStore, get_session_store
from wechat_agent.models.message import WeChatMessage
from wechat_agent.models.state import AgentState

logger = structlog.get_logger(__name__)


class SessionManagerAgent:
    """Manage short-term chat session history."""

    def __init__(self, store: SessionStore | None = None) -> None:
        self._store = store or get_session_store()

    def load_session(self, state: AgentState) -> AgentState:
        """Load recent history into state."""
        message = state["message"]
        history = self._store.get_history(message.chat_id)
        logger.debug("session_loaded", chat_id=message.chat_id, turns=len(history))
        return {**state, "chat_history": history}

    def save_turn(self, state: AgentState) -> AgentState:
        """Persist the current message and final reply to session store."""
        message = state["message"]
        reply = state.get("sent_reply")

        self._store.add_message(message.chat_id, message.to_dict())
        if reply:
            reply_message = WeChatMessage(
                message_id=f"{message.message_id}_reply",
                chat_id=message.chat_id,
                sender_id="me",
                sender_name="me",
                content=reply,
                timestamp=message.timestamp,
                is_group=message.is_group,
            )
            self._store.add_message(message.chat_id, reply_message.to_dict())
            logger.debug("session_turn_saved", chat_id=message.chat_id)
        return state


def format_history_for_prompt(history: list[dict[str, Any]], max_turns: int = 10) -> str:
    """Format stored history as a prompt-ready conversation string."""
    lines = []
    for item in history[-max_turns:]:
        sender = item.get("sender_name", "对方")
        if item.get("sender_id") == "me":
            sender = "我"
        lines.append(f"{sender}: {item.get('content', '')}")
    return "\n".join(lines)
