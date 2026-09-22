"""LangGraph state schema for the WeChat agent."""

from typing import Any, TypedDict

from wechat_agent.models.message import WeChatMessage


class AgentState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    # Input
    message: WeChatMessage

    # Context loaded by agents
    chat_history: list[dict[str, Any]]
    profile: dict[str, Any] | None

    # Intermediate outputs
    candidate_reply: str | None
    safety_result: dict[str, Any] | None

    # Final output
    sent_reply: str | None

    # Error / early exit reason
    error: str | None
