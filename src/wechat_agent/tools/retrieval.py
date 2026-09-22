"""Retrieval tools for LangGraph conditional edges."""

from wechat_agent.models.state import AgentState


def is_safe(state: AgentState) -> str:
    """LangGraph conditional edge: did safety check pass?"""
    if state.get("error"):
        return "error"
    result = state.get("safety_result")
    return "safe" if result and result.get("decision") == "PASS" else "unsafe"
