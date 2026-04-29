"""
config.py — Centralized settings for Zupwell WhatsApp Bot
==========================================================
All values loaded from .env or environment variables.

Required:
    AZURE_OPENAI_LLM_KEY        — Azure OpenAI key for LLM
    AZURE_OPENAI_EMB_KEY        — Azure OpenAI key for embeddings
    AZURE_LLM_ENDPOINT          — Azure OpenAI endpoint for LLM
    AZURE_EMB_ENDPOINT          — Azure OpenAI endpoint for embeddings
    AZURE_LLM_DEPLOYMENT_41_MINI — Azure deployment name for LLM (gpt-4.1-mini)
    AZURE_EMB_DEPLOYMENT        — Azure deployment name for embeddings
    TWILIO_ACCOUNT_SID          — Twilio Account SID
    TWILIO_AUTH_TOKEN           — Twilio Auth Token
    TWILIO_WHATSAPP_NUMBER      — Your Twilio WhatsApp number

Optional:
    AZURE_LLM_API_VERSION       — API version for LLM (default: 2025-01-01-preview)
    AZURE_EMB_API_VERSION       — API version for embeddings (default: 2024-12-01-preview)
    REDIS_URL                   — Redis connection URL (default: redis://localhost:6379/0)
    CHROMA_PATH                 — Path for ChromaDB storage (default: ./chroma_db)
    LOG_LEVEL                   — Logging level (default: INFO)
    BOT_NAME                    — Bot display name (default: Zupwell Support)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Azure OpenAI — LLM ────────────────────────────────────────────────────
    AZURE_OPENAI_LLM_KEY:         str = ""
    AZURE_LLM_ENDPOINT:           str = ""
    AZURE_LLM_API_VERSION:        str = "2025-01-01-preview"
    AZURE_LLM_DEPLOYMENT_41_MINI: str = "gpt-4.1-mini"

    # ── Azure OpenAI — Embeddings ─────────────────────────────────────────────
    AZURE_OPENAI_EMB_KEY:         str = ""
    AZURE_EMB_ENDPOINT:           str = ""
    AZURE_EMB_API_VERSION:        str = "2024-12-01-preview"
    AZURE_EMB_DEPLOYMENT:         str = "text-embedding-3-large"

    # ── Twilio WhatsApp ────────────────────────────────────────────────────────
    TWILIO_ACCOUNT_SID:       str = ""
    TWILIO_AUTH_TOKEN:        str = ""
    TWILIO_WHATSAPP_NUMBER:   str = ""

    # ── Storage ────────────────────────────────────────────────────────────────
    REDIS_URL:   str = "redis://localhost:6379/0"
    CHROMA_PATH: str = ""

    # ── App ────────────────────────────────────────────────────────────────────
    BOT_NAME:    str = "Zupwell Support"
    LOG_LEVEL:   str = "INFO"
    APP_ENV:     str = "development"
    ADMIN_KEY:   str = "zupwell-admin-2024"

    def model_post_init(self, __context):
        if not self.CHROMA_PATH:
            self.CHROMA_PATH = str(Path(__file__).parent.parent / "chroma_db")
        Path(self.CHROMA_PATH).mkdir(parents=True, exist_ok=True)


settings = Settings()
