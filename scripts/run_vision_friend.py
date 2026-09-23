"""Real vision gateway runner for a single friend.

This script will actually capture the WeChat window, read new messages,
generate replies, and send them via AppleScript keyboard automation.

If --friend-name is omitted, it is read from the generated profile.

Usage (dry-run, recommended first run):
    python scripts/run_vision_friend.py --chat-id "friend_001" --dry-run

Usage (real send, requires explicit confirmation):
    python scripts/run_vision_friend.py --chat-id "friend_001" --confirm-send
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
    parser = argparse.ArgumentParser(description="Vision gateway real-send runner")
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
        help="Do not physically send replies (recommended for first run)",
    )
    parser.add_argument(
        "--confirm-send",
        action="store_true",
        help="Required flag to enable real sending",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="Seconds between screenshot polls (default: 10.0)",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=3,
        help="Number of polling ticks to run, 0 = run until Ctrl+C (default: 3)",
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

    if not args.dry_run and not args.confirm_send:
        print(
            "\n⚠️  This script will REALLY send messages to WeChat.\n"
            "   To proceed, add --confirm-send or use --dry-run to test safely.\n",
            file=sys.stderr,
        )
        sys.exit(1)

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
    mode = "DRY-RUN" if args.dry_run else "AUTO-SEND"
    print(
        f"\n{mode} mode started for friend '{friend_name}'.\n"
        f"Make sure the WeChat chat window is open and visible.\n"
        "Press Ctrl+C to stop."
    )

    try:
        if args.ticks > 0:
            for i in range(args.ticks):
                print(f"\n--- Tick {i + 1}/{args.ticks} ---")
                gateway._tick()
                if i < args.ticks - 1:
                    time.sleep(args.poll_interval)
        else:
            tick = 1
            while True:
                print(f"\n--- Tick {tick} ---")
                gateway._tick()
                tick += 1
                time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        gateway.stop()
        print("Vision runner stopped.")


if __name__ == "__main__":
    main()
