"""LangGraph StateGraph definition for the WeChat agent."""

from typing import Any, cast

import structlog
from langgraph.graph import END, StateGraph

from wechat_agent.access_layer.base_gateway import BaseGateway
from wechat_agent.agents.auto_reply import AutoReplyAgent
from wechat_agent.agents.profile_builder import ProfileBuilderAgent
from wechat_agent.agents.safety_guard import SafetyGuardAgent
from wechat_agent.agents.session_manager import SessionManagerAgent
from wechat_agent.models.state import AgentState
from wechat_agent.tools.retrieval import is_safe

logger = structlog.get_logger(__name__)


class WeChatAgentGraph:
    """Builds and runs the LangGraph for WeChat auto-reply."""

    def __init__(
        self,
        gateway: BaseGateway,
        profile_builder: ProfileBuilderAgent | None = None,
        auto_reply: AutoReplyAgent | None = None,
        safety_guard: SafetyGuardAgent | None = None,
        session_manager: SessionManagerAgent | None = None,
    ) -> None:
        self._gateway = gateway
        self._profile_builder = profile_builder or ProfileBuilderAgent()
        self._auto_reply = auto_reply or AutoReplyAgent()
        self._safety_guard = safety_guard or SafetyGuardAgent()
        self._session_manager = session_manager or SessionManagerAgent()
        self._graph = self._build_graph()

    def _load_profile(self, state: AgentState) -> AgentState:
        """Load profile for the current chat, creating a default one if missing."""
        chat_id = state["message"].chat_id
        profile = self._profile_builder.get_or_create_profile(chat_id)
        return {**state, "profile": profile}

    def _build_graph(self) -> Any:
        """Construct the StateGraph."""
        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node("load_profile", self._load_profile)
        workflow.add_node("load_session", self._session_manager.load_session)
        workflow.add_node("generate_reply", self._auto_reply.generate)
        workflow.add_node("safety_check", self._safety_guard.check)
        workflow.add_node("decide_send", self._safety_guard.decide_send)
        workflow.add_node("send_reply", self._send_reply)
        workflow.add_node("save_turn", self._session_manager.save_turn)

        # Edges
        workflow.set_entry_point("load_profile")
        workflow.add_edge("load_profile", "load_session")
        workflow.add_edge("load_session", "generate_reply")
        workflow.add_conditional_edges(
            "generate_reply",
            self._has_error,
            {
                "ok": "safety_check",
                "error": END,
            },
        )
        workflow.add_edge("safety_check", "decide_send")
        workflow.add_conditional_edges(
            "decide_send",
            is_safe,
            {
                "safe": "send_reply",
                "unsafe": END,
                "error": END,
            },
        )
        workflow.add_edge("send_reply", "save_turn")
        workflow.add_edge("save_turn", END)

        return workflow.compile()

    @staticmethod
    def _has_error(state: AgentState) -> str:
        """Route to error endpoint if state contains an error."""
        return "error" if state.get("error") else "ok"

    def _send_reply(self, state: AgentState) -> AgentState:
        """Send the final reply via the gateway."""
        reply = state.get("sent_reply")
        if reply:
            self._gateway.send_message(state["message"].chat_id, reply)
            logger.info("reply_sent", chat_id=state["message"].chat_id, reply=reply)
        return state

    def run(self, state: AgentState) -> AgentState:
        """Run the graph with the given initial state."""
        return cast(AgentState, self._graph.invoke(state))
