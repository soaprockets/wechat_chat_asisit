"""Mock gateway for local development and tests."""

from collections.abc import Callable
from typing import Any

import structlog

from wechat_agent.access_layer.base_gateway import BaseGateway
from wechat_agent.models.message import WeChatMessage

logger = structlog.get_logger(__name__)


class MockGateway(BaseGateway):
    """A fake gateway that prints instead of sending real messages."""

    def __init__(self) -> None:
        self._handler: Callable[[WeChatMessage], Any] | None = None

    def start(self, on_message: Callable[[WeChatMessage], Any]) -> None:
        self._handler = on_message
        logger.info("mock_gateway_started")

    def send_message(self, chat_id: str, content: str) -> bool:
        logger.info("mock_send_message", chat_id=chat_id, content=content)
        return True

    def stop(self) -> None:
        logger.info("mock_gateway_stopped")

    def inject_message(self, message: WeChatMessage) -> Any:
        """Simulate an inbound message (used by CLI and tests)."""
        if self._handler is None:
            raise RuntimeError("Gateway not started")
        return self._handler(message)
