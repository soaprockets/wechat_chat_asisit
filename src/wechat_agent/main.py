"""CLI entry point for the WeChat agent."""

import json
from datetime import datetime, timezone

import structlog
import typer

from wechat_agent.access_layer.base_gateway import BaseGateway
from wechat_agent.access_layer.mock_gateway import MockGateway
from wechat_agent.config import Settings, settings
from wechat_agent.memory.profile_store import JsonProfileStore
from wechat_agent.models.message import WeChatMessage
from wechat_agent.models.state import AgentState
from wechat_agent.orchestration.graph import WeChatAgentGraph

app = typer.Typer(help="WeChat chat替身 multi-agent CLI")
logger = structlog.get_logger(__name__)


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


@app.command()
def simulate(
    chat_id: str = typer.Argument(..., help="Chat ID to simulate"),
    message: str = typer.Argument(..., help="Incoming message text"),
    sender_id: str = typer.Option("friend_001", help="Sender ID"),
    sender_name: str = typer.Option("好友", help="Sender name"),
) -> None:
    """Simulate a single inbound message and print the agent's reply."""
    _configure_logging()
    gateway = MockGateway()
    graph = WeChatAgentGraph(gateway)

    msg = WeChatMessage(
        message_id=f"mock_{datetime.now(timezone.utc).isoformat()}",
        chat_id=chat_id,
        sender_id=sender_id,
        sender_name=sender_name,
        content=message,
    )

    initial_state: AgentState = {"message": msg}
    final_state = graph.run(initial_state)

    typer.echo("--- Result ---")
    typer.echo(f"Profile found: {final_state.get('profile') is not None}")
    typer.echo(f"Candidate reply: {final_state.get('candidate_reply')}")
    typer.echo(f"Safety result: {final_state.get('safety_result')}")
    typer.echo(f"Sent reply: {final_state.get('sent_reply')}")
    if final_state.get("error"):
        typer.echo(f"Error: {final_state['error']}", err=True)


def _resolve_friend_name(chat_id: str, friend_name: str | None) -> str:
    """Return explicit friend name or read it from the stored profile."""
    if friend_name:
        return friend_name
    store = JsonProfileStore(base_path=settings.profile_store_path)
    profile = store.get_profile(chat_id)
    if profile:
        name = profile.get("friend_name")
        if name:
            return str(name)
    return chat_id


def _build_vision_settings(
    friend_name: str | None,
    chat_id: str | None,
    window_title: str | None,
    dry_run: bool,
) -> Settings:
    """Return settings with CLI overrides applied for the vision gateway."""
    overrides: dict[str, object] = {}
    if friend_name and chat_id:
        overrides["vision_targets"] = json.dumps(
            [{"friend_name": friend_name, "chat_id": chat_id}],
            ensure_ascii=False,
        )
        # Default the window title regex to the friend name so we target the
        # chat window instead of accidentally matching a settings dialog.
        overrides["vision_window_title_regex"] = window_title or friend_name
    elif window_title:
        overrides["vision_window_title_regex"] = window_title
    if dry_run:
        overrides["vision_dry_run"] = True
    if overrides:
        return settings.model_copy(update=overrides)
    return settings


@app.command()
def run_gateway(
    gateway_type: str = typer.Option(
        None,
        help="Gateway type: mock or vision (defaults to WECHAT_GATEWAY env)",
    ),
    dry_run: bool = typer.Option(False, help="Vision: do not physically send replies"),
    friend_name: str = typer.Option(
        None,
        help="Vision: chat title for matching the conversation (default: read from profile)",
    ),
    chat_id: str = typer.Option(None, help="Vision: stable chat ID for this target"),
    window_title: str = typer.Option(
        None, help="Vision: regex to locate the WeChat window"
    ),
) -> None:
    """Start the configured gateway and handle incoming messages."""
    _configure_logging()
    gateway_type = gateway_type or settings.gateway_type

    gateway: BaseGateway
    if gateway_type == "vision":
        from wechat_agent.access_layer.vision.gateway import VisionGateway

        if not chat_id:
            typer.echo("Error: --chat-id is required for vision gateway.", err=True)
            raise typer.Exit(1)

        resolved_friend_name = _resolve_friend_name(chat_id, friend_name)
        vision_settings = _build_vision_settings(
            resolved_friend_name, chat_id, window_title, dry_run
        )
        gateway = VisionGateway(settings_obj=vision_settings)
    else:
        gateway = MockGateway()

    graph = WeChatAgentGraph(gateway)

    def on_message(message: WeChatMessage) -> None:
        logger.info("incoming_message", chat_id=message.chat_id, content=message.content)
        initial_state: AgentState = {"message": message}
        graph.run(initial_state)

    gateway.start(on_message)
    typer.echo(f"Gateway started (type={gateway_type}). Press Ctrl+C to stop.")
    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        gateway.stop()
        typer.echo("Gateway stopped.")


@app.command()
def vision(
    chat_id: str = typer.Option(..., help="Stable chat ID for this target"),
    friend_name: str = typer.Option(
        None, help="Chat title for matching the conversation (default: read from profile)"
    ),
    window_title: str = typer.Option(
        None, help="Regex to locate the WeChat window"
    ),
    dry_run: bool = typer.Option(False, help="Do not physically send replies"),
    poll_interval: float = typer.Option(
        None, help="Polling interval in seconds (overrides VISION_POLL_INTERVAL)"
    ),
) -> None:
    """Start the vision gateway for a specific friend (shortcut)."""
    _configure_logging()
    from wechat_agent.access_layer.vision.gateway import VisionGateway

    resolved_friend_name = _resolve_friend_name(chat_id, friend_name)
    overrides: dict[str, object] = {
        "gateway_type": "vision",
        "vision_targets": json.dumps(
            [{"friend_name": resolved_friend_name, "chat_id": chat_id}],
            ensure_ascii=False,
        ),
    }
    # Default window title regex to the friend name to target the chat window.
    overrides["vision_window_title_regex"] = window_title or resolved_friend_name
    if dry_run:
        overrides["vision_dry_run"] = True
    if poll_interval is not None:
        overrides["vision_poll_interval"] = poll_interval

    vision_settings = settings.model_copy(update=overrides)
    gateway: BaseGateway = VisionGateway(settings_obj=vision_settings)
    graph = WeChatAgentGraph(gateway)

    def on_message(message: WeChatMessage) -> None:
        logger.info("incoming_message", chat_id=message.chat_id, content=message.content)
        initial_state: AgentState = {"message": message}
        graph.run(initial_state)

    gateway.start(on_message)
    mode = "dry-run" if dry_run else "auto-send"
    typer.echo(f"Vision gateway started ({mode}). Press Ctrl+C to stop.")
    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        gateway.stop()
        typer.echo("Vision gateway stopped.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
