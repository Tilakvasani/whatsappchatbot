"""
webhook.py — Twilio WhatsApp Webhook (LangGraph powered)
=========================================================
Same structure as original — returns TwiML immediately, processes in background.
Now routes through the LangGraph agent instead of calling RAG directly.

Session state (ticket_stage, ticket_draft) is persisted in Redis
between turns so multi-turn ticket collection works across messages.
"""

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, Request, BackgroundTasks, Response
from twilio.rest import Client as TwilioClient

from agent.runner import run_agent
from core.config import settings
from core.logger import logger
from whatsapp.session import clear_session, _get_redis, _session_key

router = APIRouter(prefix="/whatsapp", tags=["whatsapp"])

_twilio_client: Optional[TwilioClient] = None


def get_twilio_client() -> TwilioClient:
    global _twilio_client
    if _twilio_client is None:
        _twilio_client = TwilioClient(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
    return _twilio_client


# ── WhatsApp message sender ───────────────────────────────────────────────────

def _send_wa(to: str, body: str) -> None:
    try:
        client = get_twilio_client()
        msg = client.messages.create(
            from_=settings.TWILIO_WHATSAPP_NUMBER,
            to=to,
            body=body,
        )
        logger.info("WA sent to %s | SID: %s", to[-4:], msg.sid)
    except Exception as e:
        logger.error("WA send failed to %s: %s", to, e)


async def _send_wa_async(to: str, body: str) -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _send_wa, to, body)


def _split_message(text: str, max_len: int = 1500) -> list[str]:
    """Split long messages on paragraph boundaries."""
    if len(text) <= max_len:
        return [text]
    parts, current = [], ""
    for para in text.split("\n\n"):
        if len(current) + len(para) + 2 <= max_len:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                parts.append(current)
            current = para
    if current:
        parts.append(current)
    return parts or [text[:max_len]]


# ── Ticket flow state in Redis ────────────────────────────────────────────────

TICKET_STATE_TTL = 3600  # 1 hour


async def _get_ticket_state(phone: str) -> dict:
    """Load ticket stage/draft from Redis for this phone number."""
    try:
        redis = _get_redis()
        key = f"zupwell:ticket_state:{phone.replace('whatsapp:', '')}"
        raw = await redis.get(key)
        return json.loads(raw) if raw else {"stage": "idle", "draft": {}}
    except Exception:
        return {"stage": "idle", "draft": {}}


async def _save_ticket_state(phone: str, stage: str, draft: dict) -> None:
    try:
        redis = _get_redis()
        key = f"zupwell:ticket_state:{phone.replace('whatsapp:', '')}"
        await redis.set(key, json.dumps({"stage": stage, "draft": draft}), ex=TICKET_STATE_TTL)
    except Exception as e:
        logger.warning("Failed to save ticket state: %s", e)


async def _clear_ticket_state(phone: str) -> None:
    try:
        redis = _get_redis()
        key = f"zupwell:ticket_state:{phone.replace('whatsapp:', '')}"
        await redis.delete(key)
    except Exception:
        pass


# ── Background processing ─────────────────────────────────────────────────────

async def process_and_reply(from_number: str, message_body: str, profile_name: str) -> None:
    """Core background task — runs agent, sends WhatsApp reply."""
    text = message_body.strip()

    # Handle reset command
    if text.lower() in {"reset", "clear", "restart"}:
        await clear_session(from_number)
        await _clear_ticket_state(from_number)
        await _send_wa_async(from_number, "✅ *Chat reset!* Let's start fresh. How can I help you? 😊")
        return

    # Load persisted ticket state
    ticket_state = await _get_ticket_state(from_number)

    # Typing indicator for longer questions
    if len(text) > 40:
        await _send_wa_async(from_number, "⏳ Looking into that for you...")

    # Run the LangGraph agent
    result = await run_agent(
        user_input=text,
        session_id=from_number,
        source="whatsapp",
        user_name=profile_name if profile_name != "Customer" else None,
        user_phone=from_number.replace("whatsapp:", ""),
        ticket_stage=ticket_state.get("stage", "idle"),
        ticket_draft=ticket_state.get("draft", {}),
    )

    # Persist updated ticket state
    await _save_ticket_state(
        from_number,
        result.get("ticket_stage", "idle"),
        result.get("ticket_draft", {}),
    )

    # Send reply (split if needed)
    parts = _split_message(result["response"])
    for part in parts:
        await _send_wa_async(from_number, part)
        if len(parts) > 1:
            await asyncio.sleep(0.5)


# ── Webhook endpoint ──────────────────────────────────────────────────────────

@router.post("/webhook")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Twilio webhook endpoint.
    Returns empty TwiML immediately → processes in background.
    """
    form = await request.form()
    from_number  = form.get("From", "")
    message_body = form.get("Body", "")
    profile_name = form.get("ProfileName", "Customer")

    if not from_number or not message_body:
        return Response(content='<?xml version="1.0"?><Response></Response>', media_type="text/xml")

    logger.info("WA IN from %s (%s): '%s...'", from_number[-4:], profile_name, message_body[:50])

    background_tasks.add_task(process_and_reply, from_number, message_body, profile_name)

    return Response(content='<?xml version="1.0"?><Response></Response>', media_type="text/xml")


@router.get("/health")
async def health():
    return {"status": "ok", "bot": settings.BOT_NAME}
