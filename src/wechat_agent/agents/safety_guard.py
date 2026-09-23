"""Safety guard agent: detect sensitive content and decide whether to block."""

from typing import Any

import structlog

from wechat_agent.config import settings
from wechat_agent.llm.providers import get_provider
from wechat_agent.models.state import AgentState

logger = structlog.get_logger(__name__)

SAFETY_SYSTEM_PROMPT = """你是一个内容安全审核助手。判断以下回复是否适合通过微信自动发送。
只回答 "PASS" 或 "BLOCK"，并给出一个简短理由。"""

SAFETY_PROMPT_TEMPLATE = """请判断这条自动生成的微信回复是否安全：

对方消息：
{incoming_message}

拟发送回复：
{candidate_reply}

判断维度：
1. 是否包含敏感个人信息泄露（身份证、地址、密码等）
2. 是否涉及金钱交易、转账、验证码等高风险内容
3. 是否可能对关系造成损害（评价他人、传播八卦、冲突性言论）
4. 是否符合当前对话上下文

只输出 JSON：{{"decision": "PASS" 或 "BLOCK", "reason": "..."}}
"""


class SafetyGuardAgent:
    """Check generated replies for safety issues."""

    def __init__(self, keywords: list[str] | None = None) -> None:
        self._keywords = keywords or settings.safety_keyword_list

    def check(self, state: AgentState) -> AgentState:
        """Run safety checks on the candidate reply."""
        candidate = state.get("candidate_reply")
        if not candidate:
            result = {"decision": "BLOCK", "reason": "empty candidate reply"}
            logger.warning("safety_blocked_empty_candidate")
            return {**state, "safety_result": result}

        # 1. Keyword check
        blocked_keyword = self._match_keyword(candidate)
        if blocked_keyword:
            result = {"decision": "BLOCK", "reason": f"命中敏感词: {blocked_keyword}"}
            logger.warning("safety_blocked_by_keyword", keyword=blocked_keyword)
            return {**state, "safety_result": result}

        # 2. LLM self-check
        result = self._llm_check(state["message"].content, candidate)
        logger.debug("safety_check_completed", decision=result.get("decision"))
        return {**state, "safety_result": result}

    def decide_send(self, state: AgentState) -> AgentState:
        """Determine the final sent_reply based on safety_result."""
        result = state.get("safety_result")
        candidate = state.get("candidate_reply")

        if not result or result.get("decision") != "PASS":
            reason = result.get("reason", "unknown") if result else "no_safety_result"
            logger.warning("reply_blocked", reason=reason)
            return {**state, "sent_reply": None, "error": f"blocked: {reason}"}

        return {**state, "sent_reply": candidate}

    def _match_keyword(self, text: str) -> str | None:
        """Return first matched keyword, or None."""
        for keyword in self._keywords:
            if keyword in text:
                return keyword
        return None

    def _llm_check(self, incoming_message: str, candidate_reply: str) -> dict[str, Any]:
        """Use a fast LLM to evaluate safety."""
        prompt = SAFETY_PROMPT_TEMPLATE.format(
            incoming_message=incoming_message,
            candidate_reply=candidate_reply,
        )
        try:
            # If no safety LLM is explicitly configured, reuse the main LLM
            # settings (including its API type) instead of defaulting to openai.
            if settings.safety_llm_base_url or settings.safety_llm_api_key or settings.safety_llm_model:
                provider = get_provider(
                    base_url=settings.safety_llm_base_url,
                    api_key=settings.safety_llm_api_key,
                    model=settings.safety_llm_model,
                    api_type=settings.safety_llm_api_type,
                )
            else:
                provider = get_provider()
            response = provider.generate(prompt, system=SAFETY_SYSTEM_PROMPT, temperature=0.0)
            return self._parse_response(response)
        except Exception as exc:  # noqa: BLE001
            logger.warning("safety_llm_check_failed", error=str(exc))
            # Fail open: if safety LLM fails, allow the reply but log it.
            return {"decision": "PASS", "reason": f"safety_llm_failed: {exc}"}

    @staticmethod
    def _parse_response(response: str) -> dict[str, Any]:
        """Parse LLM safety response; default to BLOCK if ambiguous."""
        import json
        import re

        # Try to extract JSON from the response.
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                decision = data.get("decision", "BLOCK")
                return {
                    "decision": "PASS" if decision == "PASS" else "BLOCK",
                    "reason": data.get("reason", "no reason"),
                }
            except json.JSONDecodeError:
                pass

        # Fallback: simple text check.
        if "PASS" in response and "BLOCK" not in response:
            return {"decision": "PASS", "reason": "text fallback"}
        return {"decision": "BLOCK", "reason": "ambiguous safety response"}
