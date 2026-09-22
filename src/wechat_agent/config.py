"""Application configuration loaded from environment variables."""

import json

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pydantic settings for the WeChat agent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Unified domestic LLM configuration
    # Defaults are empty so tests can import settings without a configured .env.
    llm_base_url: str = Field(default="", alias="LLM_BASE_URL")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_model: str = Field(default="", alias="LLM_MODEL")
    # openai | anthropic
    llm_api_type: str = Field(default="openai", alias="LLM_API_TYPE")

    # Optional fallback LLM (same unified schema)
    fallback_llm_base_url: str | None = Field(default=None, alias="FALLBACK_LLM_BASE_URL")
    fallback_llm_api_key: str | None = Field(default=None, alias="FALLBACK_LLM_API_KEY")
    fallback_llm_model: str | None = Field(default=None, alias="FALLBACK_LLM_MODEL")
    fallback_llm_api_type: str = Field(default="openai", alias="FALLBACK_LLM_API_TYPE")

    # Safety LLM (uses the same unified schema)
    safety_llm_base_url: str | None = Field(default=None, alias="SAFETY_LLM_BASE_URL")
    safety_llm_api_key: str | None = Field(default=None, alias="SAFETY_LLM_API_KEY")
    safety_llm_model: str | None = Field(default=None, alias="SAFETY_LLM_MODEL")
    safety_llm_api_type: str = Field(default="openai", alias="SAFETY_LLM_API_TYPE")

    # Embeddings (same unified schema)
    embedding_base_url: str | None = Field(default=None, alias="EMBEDDING_BASE_URL")
    embedding_api_key: str | None = Field(default=None, alias="EMBEDDING_API_KEY")
    embedding_model: str = Field(default="embedding-2", alias="EMBEDDING_MODEL")

    # Memory
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    chroma_db_path: str = Field(default="./data/chroma", alias="CHROMA_DB_PATH")
    profile_store_type: str = Field(default="json", alias="PROFILE_STORE_TYPE")
    profile_store_path: str = Field(default="./data/profiles", alias="PROFILE_STORE_PATH")

    # Safety
    safety_keywords: str = Field(
        default="转账,密码,验证码,身份证,银行卡,地址",
        alias="SAFETY_KEYWORDS",
    )

    # Gateway selection
    gateway_type: str = Field(default="mock", alias="WECHAT_GATEWAY")  # mock | vision

    # Vision gateway settings (used when gateway_type == "vision")
    vision_poll_interval: float = Field(default=2.0, alias="VISION_POLL_INTERVAL")
    vision_window_title_regex: str = Field(
        default="WeChat|微信", alias="VISION_WINDOW_TITLE_REGEX"
    )
    vision_targets: str = Field(default="", alias="VISION_TARGETS")
    vision_send_enabled: bool = Field(default=True, alias="VISION_SEND_ENABLED")
    vision_dry_run: bool = Field(default=False, alias="VISION_DRY_RUN")
    vision_change_threshold: int = Field(default=5, alias="VISION_CHANGE_THRESHOLD")
    vision_image_max_width: int = Field(default=1280, alias="VISION_IMAGE_MAX_WIDTH")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def vision_target_list(self) -> list[dict[str, str]]:
        """Parse VISION_TARGETS JSON into a list of target dicts."""
        if not self.vision_targets:
            return []
        try:
            data = json.loads(self.vision_targets)
            if not isinstance(data, list):
                return []
            return [
                {"friend_name": str(item["friend_name"]), "chat_id": str(item["chat_id"])}
                for item in data
                if isinstance(item, dict) and "friend_name" in item and "chat_id" in item
            ]
        except (json.JSONDecodeError, TypeError):
            return []

    @property
    def safety_keyword_list(self) -> list[str]:
        """Return the safety keyword list."""
        return [kw.strip() for kw in self.safety_keywords.split(",") if kw.strip()]


settings = Settings()
