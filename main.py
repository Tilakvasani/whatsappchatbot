"""
main.py — Zupwell Unified Bot — FastAPI Application
=====================================================
Serves both:
  - WhatsApp webhook  → /whatsapp/webhook  (Twilio)
  - Website chat API  → /chat/message      (Next.js widget)
  - Admin endpoints   → /admin/*           (protected)
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.logger import logger
from core.vector import get_collection
from rag.ingest import ingest_notion
from tickets.tickets import get_pool
from whatsapp.webhook import router as wa_router
from website.chat_api import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting %s...", settings.BOT_NAME)

    # ChromaDB check
    try:
        col = get_collection()
        count = col.count()
        if count == 0:
            logger.warning("⚠️  ChromaDB empty — run: python scripts/ingest_notion.py")
        else:
            logger.info("✅ ChromaDB ready — %d chunks", count)
    except Exception as e:
        logger.error("❌ ChromaDB error: %s", e)

    # PostgreSQL — ensure ticket table exists
    try:
        await get_pool()
        logger.info("✅ PostgreSQL ticket table ready")
    except Exception as e:
        logger.error("❌ DB pool error: %s", e)

    # Config check
    missing = [
        k for k, v in {
            "AZURE_OPENAI_LLM_KEY": settings.AZURE_OPENAI_LLM_KEY,
            "AZURE_LLM_ENDPOINT":   settings.AZURE_LLM_ENDPOINT,
            "AZURE_OPENAI_EMB_KEY": settings.AZURE_OPENAI_EMB_KEY,
            "AZURE_EMB_ENDPOINT":   settings.AZURE_EMB_ENDPOINT,
            "NOTION_TOKEN":         settings.NOTION_TOKEN,
            "NOTION_DATABASE_ID":   settings.NOTION_DATABASE_ID,
            "DATABASE_URL":         settings.DATABASE_URL,
            "TWILIO_ACCOUNT_SID":   settings.TWILIO_ACCOUNT_SID,
        }.items() if not v
    ]
    if missing:
        logger.error("❌ Missing .env vars: %s", ", ".join(missing))
    else:
        logger.info("✅ All config present. Bot is ready! 🎉")

    yield
    logger.info("👋 Shutting down.")


app = FastAPI(
    title="Zupwell Bot API",
    description="Unified AI support bot for WhatsApp + Website",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(wa_router)
app.include_router(chat_router)


# ── Admin endpoints ────────────────────────────────────────────────────────────

def _check_admin(x_admin_key: str = Header(...)):
    if x_admin_key != settings.ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    return True


@app.post("/admin/ingest")
async def admin_ingest(force: bool = False, _=Depends(_check_admin)):
    """Re-ingest Notion docs into ChromaDB. Pass ?force=true to re-embed all."""
    logger.info("Admin triggered Notion ingest (force=%s)", force)
    result = await ingest_notion(force=force)
    return {"status": "success", **result}


@app.get("/admin/stats")
async def admin_stats(_=Depends(_check_admin)):
    try:
        count = get_collection().count()
    except Exception:
        count = -1
    return {
        "bot_name":        settings.BOT_NAME,
        "env":             settings.APP_ENV,
        "chroma_chunks":   count,
        "llm_deployment":  settings.AZURE_LLM_DEPLOYMENT_41_MINI,
        "embed_deployment": settings.AZURE_EMB_DEPLOYMENT,
        "notion_db":       settings.NOTION_DATABASE_ID,
    }


@app.get("/")
async def root():
    return {
        "name":            settings.BOT_NAME,
        "status":          "running",
        "whatsapp_webhook": "/whatsapp/webhook",
        "chat_api":        "/chat/message",
        "docs":            "/docs",
    }
