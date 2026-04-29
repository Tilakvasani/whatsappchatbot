"""
config.py — Centralized settings for Zupwell WhatsApp Bot
==========================================================
All values loaded from .env or environment variables.

Required:
    OPENAI_API_KEY          — OpenAI API key for LLM + embeddings
    TWILIO_ACCOUNT_SID      — Twilio Account SID
    TWILIO_AUTH_TOKEN       — Twilio Auth Token
    TWILIO_WHATSAPP_NUMBER  — Your Twilio WhatsApp number (e.g. whatsapp:+14155238886)

Optional:
    REDIS_URL               — Redis connection URL (default: redis://localhost:6379/0)
    CHROMA_PATH             — Path for ChromaDB storage (default: ./chroma_db)
    OPENAI_MODEL            — LLM model name (default: gpt-4o-mini)
    OPENAI_EMBED_MODEL      — Embedding model (default: text-embedding-3-small)
    LOG_LEVEL               — Logging level (default: INFO)
    BOT_NAME                — Bot display name (default: Zupwell Support)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── OpenAI ─────────────────────────────────────────────────────────────────
    OPENAI_API_KEY:       str = ""
    OPENAI_MODEL:         str = "gpt-4o-mini"
    OPENAI_EMBED_MODEL:   str = "text-embedding-3-small"

    # ── Twilio WhatsApp ────────────────────────────────────────────────────────
    TWILIO_ACCOUNT_SID:       str = ""
    TWILIO_AUTH_TOKEN:        str = ""
    TWILIO_WHATSAPP_NUMBER:   str = ""   # e.g. "whatsapp:+14155238886"

    # ── Storage ────────────────────────────────────────────────────────────────
    REDIS_URL:   str = "redis://localhost:6379/0"
    CHROMA_PATH: str = ""

    # ── App ────────────────────────────────────────────────────────────────────
    BOT_NAME:    str = "Zupwell Support"
    LOG_LEVEL:   str = "INFO"
    APP_ENV:     str = "development"

    def model_post_init(self, __context):
        if not self.CHROMA_PATH:
            self.CHROMA_PATH = str(Path(__file__).parent.parent / "chroma_db")
        Path(self.CHROMA_PATH).mkdir(parents=True, exist_ok=True)


settings = Settings()
