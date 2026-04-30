"""
main.py — Zupwell Bot — FastAPI Entry Point
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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
        count = get_collection().count()
        if count == 0:
            logger.warning("⚠️  ChromaDB empty — run: python scripts/ingest_notion.py")
        else:
            logger.info("✅ ChromaDB: %d chunks loaded", count)
    except Exception as e:
        logger.error("❌ ChromaDB: %s", e)

    # PostgreSQL
    try:
        await get_pool()
        logger.info("✅ PostgreSQL: ticket table ready")
    except Exception as e:
        logger.error("❌ PostgreSQL: %s", e)

    # Config check
    required = {
        "AZURE_OPENAI_LLM_KEY": settings.AZURE_OPENAI_LLM_KEY,
        "AZURE_LLM_ENDPOINT":   settings.AZURE_LLM_ENDPOINT,
        "AZURE_OPENAI_EMB_KEY": settings.AZURE_OPENAI_EMB_KEY,
        "AZURE_EMB_ENDPOINT":   settings.AZURE_EMB_ENDPOINT,
        "NOTION_TOKEN":         settings.NOTION_TOKEN,
        "NOTION_DATABASE_ID":   settings.NOTION_DATABASE_ID,
        "DATABASE_URL":         settings.DATABASE_URL,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        logger.error("❌ Missing .env vars: %s", ", ".join(missing))
    else:
        logger.info("✅ All config present — bot is ready! 🎉")

    yield
    logger.info("👋 Shutting down.")


app = FastAPI(
    title="Zupwell Bot API",
    description="AI support bot — WhatsApp + Website",
    version="3.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded photos as static files
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

app.include_router(wa_router)
app.include_router(chat_router)


# ── Admin ─────────────────────────────────────────────────────────────────────

def _admin(x_admin_key: str = Header(...)):
    if x_admin_key != settings.ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    return True


@app.post("/admin/ingest")
async def admin_ingest(force: bool = False, _=Depends(_admin)):
    """Re-sync Notion → ChromaDB. ?force=true re-embeds everything."""
    result = await ingest_notion(force=force)
    return {"status": "success", **result}


@app.get("/admin/stats")
async def admin_stats(_=Depends(_admin)):
    try:
        count = get_collection().count()
    except Exception:
        count = -1
    return {
        "bot_name":    settings.BOT_NAME,
        "env":         settings.APP_ENV,
        "chroma_docs": count,
        "llm":         settings.AZURE_LLM_DEPLOYMENT_41_MINI,
        "embed":       settings.AZURE_EMB_DEPLOYMENT,
        "notion_db":   settings.NOTION_DATABASE_ID,
    }


@app.get("/")
async def root():
    return {
        "name":              settings.BOT_NAME,
        "version":           "3.0",
        "whatsapp_webhook":  "/whatsapp/webhook",
        "chat_api":          "/chat/message",
        "photo_upload":      "/chat/upload-photo",
        "ticket_status":     "/chat/ticket/{ref}",
        "docs":              "/docs",
    }