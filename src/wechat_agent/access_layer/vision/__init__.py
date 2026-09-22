"""Vision-based WeChat gateway."""

from wechat_agent.access_layer.vision.backends.factory import get_backend
from wechat_agent.access_layer.vision.extraction import VisionExtractor
from wechat_agent.access_layer.vision.gateway import VisionGateway
from wechat_agent.access_layer.vision.models import VisionTarget, WindowInfo

__all__ = [
    "VisionGateway",
    "VisionExtractor",
    "get_backend",
    "VisionTarget",
    "WindowInfo",
]
