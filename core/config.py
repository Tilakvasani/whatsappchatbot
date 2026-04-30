"""
config.py — Zupwell Bot Settings
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Azure OpenAI — LLM ────────────────────────────────────────────────────
    AZURE_OPENAI_LLM_KEY:         str = ""
    AZURE_LLM_ENDPOINT:           str = ""
    AZURE_LLM_API_VERSION:        str = "2025-01-01-preview"
    AZURE_LLM_DEPLOYMENT_41_MINI: str = "gpt-4.1-mini"

    # ── Azure OpenAI — Embeddings ─────────────────────────────────────────────
    AZURE_OPENAI_EMB_KEY:  str = ""
    AZURE_EMB_ENDPOINT:    str = ""
    AZURE_EMB_API_VERSION: str = "2024-12-01-preview"
    AZURE_EMB_DEPLOYMENT:  str = "text-embedding-3-large"

    # ── Notion ────────────────────────────────────────────────────────────────
    NOTION_TOKEN:       str = ""
    NOTION_DATABASE_ID: str = ""

    # ── Twilio WhatsApp ───────────────────────────────────────────────────────
    TWILIO_ACCOUNT_SID:     str = ""
    TWILIO_AUTH_TOKEN:      str = ""
    TWILIO_WHATSAPP_NUMBER: str = ""

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql://user:pass@localhost:5432/zupwell"

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    CHROMA_PATH: str = ""

    # ── File uploads (local path served at /uploads) ──────────────────────────
    UPLOAD_DIR:     str = "./uploads"
    UPLOAD_MAX_MB:  int = 10

    # ── JWT (same secret as ps5 backend) ─────────────────────────────────────
    JWT_SECRET:    str = "your-jwt-secret-here"
    JWT_ALGORITHM: str = "HS256"

    # ── App ───────────────────────────────────────────────────────────────────
    BOT_NAME:     str = "Zupwell Support"
    ADMIN_KEY:    str = "zupwell-admin-2024"
    APP_ENV:      str = "development"
    LOG_LEVEL:    str = "INFO"
    CORS_ORIGINS: str = "http://localhost:3000"

    def model_post_init(self, __context):
        if not self.CHROMA_PATH:
            self.CHROMA_PATH = str(Path(__file__).parent.parent / "chroma_db")
        Path(self.CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        Path(self.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


settings = Settings()
