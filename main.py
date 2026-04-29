"""
main.py — Zupwell WhatsApp Bot — FastAPI Application
======================================================
Entry point for the Zupwell WhatsApp Bot server.

Startup sequence:
  1. FastAPI app initialization
  2. ChromaDB collection check (warns if empty — run ingest first)
  3. Twilio + Azure OpenAI config validation

Routes:
  POST /whatsapp/webhook  — Twilio webhook (main bot endpoint)
  GET  /whatsapp/health   — Health check
  POST /admin/ingest      — Trigger FAQ re-ingestion (protected)
  GET  /admin/stats       — Bot stats (protected)
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.logger import logger
from core.vector import get_collection
from rag.ingest import ingest_faqs
from whatsapp.webhook import router as whatsapp_router


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("🚀 Starting %s...", settings.BOT_NAME)

    # Check ChromaDB — warn if empty (user needs to run ingest)
    try:
        collection = get_collection()
        count = collection.count()
        if count == 0:
            logger.warning(
                "⚠️  ChromaDB is empty! Run 'python scripts/ingest_faqs.py' "
                "to populate the FAQ knowledge base before the bot can answer questions."
            )
        else:
            logger.info("✅ ChromaDB ready — %d FAQ chunks loaded.", count)
    except Exception as e:
        logger.error("❌ ChromaDB check failed: %s", e)

    # Validate Twilio + Azure config
    missing = []
    if not settings.TWILIO_ACCOUNT_SID:
        missing.append("TWILIO_ACCOUNT_SID")
    if not settings.TWILIO_AUTH_TOKEN:
        missing.append("TWILIO_AUTH_TOKEN")
    if not settings.TWILIO_WHATSAPP_NUMBER:
        missing.append("TWILIO_WHATSAPP_NUMBER")
    if not settings.AZURE_OPENAI_LLM_KEY:
        missing.append("AZURE_OPENAI_LLM_KEY")
    if not settings.AZURE_LLM_ENDPOINT:
        missing.append("AZURE_LLM_ENDPOINT")
    if not settings.AZURE_OPENAI_EMB_KEY:
        missing.append("AZURE_OPENAI_EMB_KEY")
    if not settings.AZURE_EMB_ENDPOINT:
        missing.append("AZURE_EMB_ENDPOINT")

    if missing:
        logger.error("❌ Missing required env vars: %s", ", ".join(missing))
    else:
        logger.info("✅ Twilio & Azure OpenAI config looks good.")

    logger.info("✅ %s is ready!", settings.BOT_NAME)
    yield
    logger.info("👋 Shutting down %s.", settings.BOT_NAME)


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=f"{settings.BOT_NAME} API",
    description="WhatsApp Bot for Zupwell Health Supplements",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(whatsapp_router)


# ── Admin endpoints ────────────────────────────────────────────────────────────

def verify_admin_key(x_admin_key: str = Header(...)):
    """Simple admin key check — set ADMIN_KEY in your .env."""
    if x_admin_key != settings.ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    return True


@app.post("/admin/ingest")
async def admin_ingest(
    force: bool = False,
    _: bool = Depends(verify_admin_key),
):
    """
    Trigger FAQ re-ingestion into ChromaDB.
    Pass ?force=true to re-embed all FAQs (even if already present).

    Requires header: X-Admin-Key: <your-admin-key>
    """
    logger.info("Admin triggered FAQ ingest (force=%s)", force)
    result = await ingest_faqs(force=force)
    return {
        "status": "success",
        "message": f"Ingested {result['ingested']} FAQs, skipped {result['skipped']}, total: {result['total']}",
        **result,
    }


@app.get("/admin/stats")
async def admin_stats(_: bool = Depends(verify_admin_key)):
    """
    Bot stats — FAQ count in ChromaDB, config status.
    Requires header: X-Admin-Key: <your-admin-key>
    """
    try:
        collection = get_collection()
        faq_count = collection.count()
    except Exception:
        faq_count = -1

    return {
        "bot_name":         settings.BOT_NAME,
        "environment":      settings.APP_ENV,
        "faq_chunks":       faq_count,
        "llm_deployment":   settings.AZURE_LLM_DEPLOYMENT_41_MINI,
        "embed_deployment": settings.AZURE_EMB_DEPLOYMENT,
        "llm_endpoint":     settings.AZURE_LLM_ENDPOINT,
        "twilio_configured": bool(settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN),
        "azure_llm_configured": bool(settings.AZURE_OPENAI_LLM_KEY and settings.AZURE_LLM_ENDPOINT),
        "azure_emb_configured": bool(settings.AZURE_OPENAI_EMB_KEY and settings.AZURE_EMB_ENDPOINT),
    }


@app.get("/")
async def root():
    return {
        "name":    settings.BOT_NAME,
        "status":  "running",
        "docs":    "/docs",
        "webhook": "/whatsapp/webhook",
    }
