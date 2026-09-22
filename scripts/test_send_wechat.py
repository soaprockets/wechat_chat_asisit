"""WeChat Agent mock end-to-end test script.

Runs the full LangGraph agent pipeline against simulated incoming messages.
No real WeChat client or database is touched.

Usage:
    python scripts/test_send_wechat.py
    python scripts/test_send_wechat.py --mock-delay 1.0
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from datetime import datetime, timezone

from wechat_agent.access_layer.mock_gateway import MockGateway
from wechat_agent.models.message import MessageType, WeChatMessage
from wechat_agent.orchestration.graph import WeChatAgentGraph

# Predefined mock conversation scenario.
MOCK_SCENARIO: list[dict[str, str]] = [
    {"chat_id": "friend_001", "sender_name": "Alice", "content": "周末有空吃饭吗？"},
    {"chat_id": "friend_001", "sender_name": "Alice", "content": "那下周呢？"},
    {"chat_id": "friend_002", "sender_name": "Bob", "content": "在吗？借我两千块钱急用。"},
    {"chat_id": "group_001", "sender_name": "Carol", "content": "大家晚上好呀~", "is_group": "true"},
]


def _mock_messages(scenario: list[dict[str, str]] | None = None) -> Iterator[WeChatMessage]:
    """Yield simulated incoming messages."""
    items = scenario or MOCK_SCENARIO
    for idx, item in enumerate(items):
        chat_id = item["chat_id"]
        sender = item["sender_name"]
        yield WeChatMessage(
            message_id=f"mock_{idx}_{int(time.time() * 1000)}",
            chat_id=chat_id,
            sender_id=sender,
            sender_name=sender,
            content=item["content"],
            timestamp=datetime.now(timezone.utc),
            message_type=MessageType.TEXT,
            is_group=item.get("is_group", "false").lower() == "true",
        )


def _run_mock(scenario: list[dict[str, str]] | None = None, delay: float = 0.5) -> None:
    """Run the Agent graph against simulated messages."""
    gateway = MockGateway()
    graph = WeChatAgentGraph(gateway=gateway)

    print("=== Mock mode: running simulated messages ===\n")
    for msg in _mock_messages(scenario):
        print(f"[{msg.chat_id}] {msg.sender_name}: {msg.content}")
        result = graph.run({"message": msg})

        reply = result.get("candidate_reply")
        safety = result.get("safety_result", {})
        sent = result.get("sent_reply")

        print(f"  -> AI reply: {reply}")
        print(f"  -> Safety: {safety.get('decision')} ({safety.get('reason')})")
        print(f"  -> Sent: {sent}\n")
        time.sleep(delay)

    print("Mock run complete.")


def main() -> None:
    """Entry point for the mock WeChat auto-reply test script."""
    parser = argparse.ArgumentParser(description="WeChat Agent mock end-to-end test")
    parser.add_argument(
        "--mock-delay",
        type=float,
        default=0.5,
        help="Seconds between mock messages",
    )
    args = parser.parse_args()

    _run_mock(delay=args.mock_delay)


if __name__ == "__main__":
    main()
