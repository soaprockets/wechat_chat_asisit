"""Abstract WeChat gateway interface."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from wechat_agent.models.message import WeChatMessage


class BaseGateway(ABC):
    """Abstract interface for WeChat message gateways."""

    @abstractmethod
    def start(self, on_message: Callable[[WeChatMessage], Any]) -> None:
        """Start the gateway and register a message handler."""

    @abstractmethod
    def send_message(self, chat_id: str, content: str) -> bool:
        """Send a message to a chat. Return True on success."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the gateway."""
