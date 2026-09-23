"""End-to-end vision gateway test for a single friend.

This script bypasses Redis by using an in-memory session store and the JSON
profile store, so it can be run locally without any extra services.

If --friend-name is omitted, it is read from the generated profile.

Usage:
    python scripts/test_vision_friend.py --chat-id "friend_001"
    python scripts/test_vision_friend.py --chat-id "friend_001" --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import structlog

from wechat_agent.access_layer.vision.gateway import VisionGateway
from wechat_agent.agents.auto_reply import AutoReplyAgent
from wechat_agent.agents.profile_builder import ProfileBuilderAgent
from wechat_agent.agents.safety_guard import SafetyGuardAgent
from wechat_agent.agents.session_manager import SessionManagerAgent
from wechat_agent.config import Settings, settings
from wechat_agent.memory.profile_store import JsonProfileStore
from wechat_agent.memory.session_store import InMemorySessionStore
from wechat_agent.models.message import WeChatMessage
from wechat_agent.models.state import AgentState
from wechat_agent.orchestration.graph import WeChatAgentGraph


def _configure_logging(log_level: str = "INFO") -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _resolve_friend_name(chat_id: str, friend_name: str | None) -> str:
    """Return explicit friend name or read it from the stored profile."""
    if friend_name:
        return friend_name
    store = JsonProfileStore(base_path="./data/profiles")
    profile = store.get_profile(chat_id)
    if profile:
        name = profile.get("friend_name")
        if name:
            print(f"Auto-detected friend name from profile: {name}")
            return str(name)
    print(
        f"Warning: could not detect friend name for '{chat_id}'. "
        f"Falling back to chat id as window title.",
        file=sys.stderr,
    )
    return chat_id


def _build_settings(
    friend_name: str,
    chat_id: str,
    dry_run: bool,
    poll_interval: float,
    window_title: str,
) -> Settings:
    overrides: dict[str, object] = {
        "gateway_type": "vision",
        "vision_targets": json.dumps(
            [{"friend_name": friend_name, "chat_id": chat_id}],
            ensure_ascii=False,
        ),
        "vision_dry_run": dry_run,
        "vision_poll_interval": poll_interval,
        "vision_window_title_regex": window_title,
    }
    return settings.model_copy(update=overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision gateway end-to-end test")
    parser.add_argument(
        "--friend-name",
        default=None,
        help="Chat title used to match the conversation in the screenshot "
        "(default: read from profile's friend_name)",
    )
    parser.add_argument(
        "--chat-id",
        required=True,
        help="Stable chat ID used to load the friend's profile",
    )
    parser.add_argument(
        "--window-title",
        default=None,
        help="Regex to locate the WeChat window (default: WeChat|微信)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not physically send replies",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between screenshot polls (default: 2.0)",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=3,
        help="Number of polling ticks to run before stopping (default: 3)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level",
    )
    args = parser.parse_args()

    friend_name = _resolve_friend_name(args.chat_id, args.friend_name)
    # macOS WeChat window titles are usually just "微信", not the chat name.
    # Default to the global WeChat window regex; the extractor will still only
    # react when the screenshot's chat_title matches the configured friend_name.
    window_title = args.window_title or settings.vision_window_title_regex

    _configure_logging(args.log_level)

    vision_settings = _build_settings(
        friend_name=friend_name,
        chat_id=args.chat_id,
        dry_run=args.dry_run,
        poll_interval=args.poll_interval,
        window_title=window_title,
    )

    gateway = VisionGateway(settings_obj=vision_settings)
    graph = WeChatAgentGraph(
        gateway,
        profile_builder=ProfileBuilderAgent(store=JsonProfileStore(base_path="./data/profiles")),
        session_manager=SessionManagerAgent(store=InMemorySessionStore()),
        auto_reply=AutoReplyAgent(),
        safety_guard=SafetyGuardAgent(),
    )

    def on_message(message: WeChatMessage) -> None:
        print(f"\n[EXTRACTED] {message.sender_name}: {message.content}")
        state: AgentState = {"message": message}
        final = graph.run(state)
        if final.get("error"):
            print(f"[ERROR]     {final['error']}")
            return
        print(f"[REPLY]     {final.get('candidate_reply')!r}")
        safety = final.get("safety_result") or {}
        print(f"[SAFETY]    {safety.get('decision')} ({safety.get('reason')})")
        sent_label = "[WOULD SEND]" if args.dry_run else "[SENT]"
        print(f"{sent_label}      {final.get('sent_reply')!r}")

    gateway.start(on_message)
    print(
        f"Vision test started (dry_run={args.dry_run}, ticks={args.ticks}). "
        f"Make sure the WeChat chat window for '{friend_name}' is open."
    )

    try:
        for i in range(args.ticks):
            print(f"\n--- Tick {i + 1}/{args.ticks} ---")
            gateway._tick()
            if i < args.ticks - 1:
                time.sleep(args.poll_interval)
    finally:
        gateway.stop()
        print("\nVision test stopped.")


if __name__ == "__main__":
    main()
