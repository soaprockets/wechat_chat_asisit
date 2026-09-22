"""Auto reply agent: generate a human-like reply based on profile and context."""

from typing import Any

import structlog

from wechat_agent.agents.session_manager import format_history_for_prompt
from wechat_agent.llm.router import LLMRouter, get_default_router
from wechat_agent.models.state import AgentState

logger = structlog.get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = """你是用户的微信聊天替身。请根据画像和对话历史生成回复。"""

REPLY_PROMPT_TEMPLATE = """你是 {user_name} 的微信聊天替身。请根据以下信息生成回复：

【你的语言风格】
- 语气：{my_tone}
- 常用口头禅：{catchphrases}
- emoji 频率：{emoji_freq}
- 平均句长：{avg_sentence_len} 字

【对方画像】
- 对方语气：{friend_tone}
- 关系亲密度：{intimacy}
- 常用话题：{topics}

【最近对话历史】
{chat_history}

【当前消息】
{current_message}

【当前时间】
{current_time}

要求：
1. 回复要符合你的语言风格，不要有 AI 味
2. 根据亲密度调整语气正式程度
3. 回复长度控制在 1-3 句话
4. 如果对方在深夜发消息，语气要更随意
"""


class AutoReplyAgent:
    """Generate a candidate reply using profile and session context."""

    def __init__(self, llm: LLMRouter | None = None) -> None:
        self._llm = llm or get_default_router()

    def generate(self, state: AgentState) -> AgentState:
        """Generate a reply candidate and update state."""
        message = state["message"]
        profile = state.get("profile") or {}
        history = state.get("chat_history", [])

        prompt = self._build_prompt(message.content, profile, history)
        try:
            reply = self._llm.generate(prompt, system=DEFAULT_SYSTEM_PROMPT, temperature=0.7)
            logger.debug("reply_generated", chat_id=message.chat_id, reply=reply)
            return {**state, "candidate_reply": reply.strip()}
        except Exception as exc:  # noqa: BLE001
            logger.error("reply_generation_failed", chat_id=message.chat_id, error=str(exc))
            return {**state, "error": f"reply_generation_failed: {exc}"}

    def _build_prompt(
        self,
        current_message: str,
        profile: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> str:
        """Build the prompt from profile and history."""
        my_style = profile.get("my_style", {})
        friend_style = profile.get("friend_style", {})
        relationship = profile.get("relationship", {})

        return REPLY_PROMPT_TEMPLATE.format(
            user_name=profile.get("user_name", "我"),
            my_tone=my_style.get("tone", "casual"),
            catchphrases=", ".join(my_style.get("catchphrases", [])),
            emoji_freq=my_style.get("emoji_freq", 0.5),
            avg_sentence_len=my_style.get("avg_sentence_len", 12),
            friend_tone=friend_style.get("tone", "casual"),
            intimacy=relationship.get("intimacy", 0.5),
            topics=", ".join(friend_style.get("topics", [])),
            chat_history=format_history_for_prompt(history) or "（无）",
            current_message=current_message,
            current_time=self._current_time_label(),
        )

    @staticmethod
    def _current_time_label() -> str:
        """Return a simple time-of-day label."""
        from datetime import datetime

        hour = datetime.now().hour
        if hour < 6:
            return "深夜"
        if hour < 12:
            return "上午"
        if hour < 18:
            return "下午"
        return "晚上"
