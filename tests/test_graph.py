"""End-to-end tests for the LangGraph orchestration."""

from tests.conftest import FakeLLM
from wechat_agent.access_layer.mock_gateway import MockGateway
from wechat_agent.agents.auto_reply import AutoReplyAgent
from wechat_agent.agents.safety_guard import SafetyGuardAgent
from wechat_agent.agents.session_manager import SessionManagerAgent
from wechat_agent.memory.session_store import InMemorySessionStore
from wechat_agent.models.message import WeChatMessage
from wechat_agent.models.state import AgentState
from wechat_agent.orchestration.graph import WeChatAgentGraph


def test_graph_runs_end_to_end(
    profile_builder_agent,
    sample_profile,
    sample_message,
) -> None:
    profile_builder_agent._store.save_profile("friend_001", sample_profile)

    graph = WeChatAgentGraph(
        gateway=MockGateway(),
        profile_builder=profile_builder_agent,
        auto_reply=AutoReplyAgent(llm=FakeLLM("可以啊，周末见")),
        safety_guard=SafetyGuardAgent(keywords=["密码"]),
        session_manager=SessionManagerAgent(store=InMemorySessionStore()),
    )

    initial_state: AgentState = {"message": sample_message}
    final_state = graph.run(initial_state)

    assert final_state.get("profile") is not None
    assert final_state.get("candidate_reply") == "可以啊，周末见"
    assert final_state.get("sent_reply") == "可以啊，周末见"


def test_graph_blocks_unsafe_reply(
    profile_builder_agent,
    sample_profile,
) -> None:
    profile_builder_agent._store.save_profile("friend_001", sample_profile)

    graph = WeChatAgentGraph(
        gateway=MockGateway(),
        profile_builder=profile_builder_agent,
        auto_reply=AutoReplyAgent(llm=FakeLLM("把密码发我")),
        safety_guard=SafetyGuardAgent(keywords=["密码"]),
        session_manager=SessionManagerAgent(store=InMemorySessionStore()),
    )

    msg = WeChatMessage(
        message_id="m2",
        chat_id="friend_001",
        sender_id="friend_001",
        sender_name="Alice",
        content="帮我查一下",
    )
    final_state = graph.run({"message": msg})

    assert final_state.get("candidate_reply") == "把密码发我"
    assert final_state.get("sent_reply") is None
    assert final_state.get("safety_result", {}).get("decision") == "BLOCK"


def test_graph_creates_default_profile_when_missing(profile_builder_agent) -> None:
    graph = WeChatAgentGraph(
        gateway=MockGateway(),
        profile_builder=profile_builder_agent,
        auto_reply=AutoReplyAgent(llm=FakeLLM("你好，请问有什么事？")),
        safety_guard=SafetyGuardAgent(keywords=[]),
        session_manager=SessionManagerAgent(store=InMemorySessionStore()),
    )
    msg = WeChatMessage(
        message_id="m3",
        chat_id="unknown",
        sender_id="unknown",
        sender_name="Stranger",
        content="hello",
    )
    final_state = graph.run({"message": msg})
    assert final_state.get("profile") is not None
    assert final_state["profile"]["chat_id"] == "unknown"
    assert final_state.get("candidate_reply") == "你好，请问有什么事？"
    assert final_state.get("sent_reply") == "你好，请问有什么事？"
