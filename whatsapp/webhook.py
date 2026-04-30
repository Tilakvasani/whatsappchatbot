"""
webhook.py — Twilio WhatsApp Webhook (LangGraph powered)
=========================================================
Handles text messages AND media (photos for damaged product tickets).
Twilio sends MediaUrl0, MediaContentType0 etc. in the form body.
We download the image, save it locally, and pass the URL to the agent.
"""

import asyncio
import json
import mimetypes
import uuid
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Request, Response
from twilio.rest import Client as TwilioClient

from agent.runner import run_agent
from core.config import settings
from core.logger import logger
from whatsapp.session import _get_redis, clear_session

router = APIRouter(prefix="/whatsapp", tags=["whatsapp"])

_twilio_client: Optional[TwilioClient] = None
UPLOAD_DIR = Path(settings.UPLOAD_DIR)


def get_twilio() -> TwilioClient:
    global _twilio_client
    if _twilio_client is None:
        _twilio_client = TwilioClient(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
    return _twilio_client


# ── Media download ────────────────────────────────────────────────────────────

async def _download_twilio_media(media_url: str, content_type: str) -> Optional[str]:
    """
    Download a Twilio media URL (photo from WhatsApp) and save locally.
    Returns the local file URL path (e.g. /uploads/abc123.jpg) or None on failure.
    Twilio requires HTTP Basic Auth with Account SID + Auth Token.
    """
    try:
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ".jpg"
        ext = ext.replace(".jpe", ".jpg")
        filename = f"{uuid.uuid4().hex}{ext}"
        filepath = UPLOAD_DIR / filename

        auth = (settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(media_url, auth=auth, follow_redirects=True)
            resp.raise_for_status()
            filepath.write_bytes(resp.content)

        url = f"/uploads/{filename}"
        logger.info("Downloaded WA media → %s (%d bytes)", filename, len(resp.content))
        return url
    except Exception as e:
        logger.error("Failed to download Twilio media %s: %s", media_url, e)
        return None


async def _collect_media_urls(form: dict) -> list[str]:
    """Extract all media URLs from Twilio form data and download them."""
    urls = []
    i = 0
    while True:
        media_url  = form.get(f"MediaUrl{i}")
        media_type = form.get(f"MediaContentType{i}", "image/jpeg")
        if not media_url:
            break
        local_url = await _download_twilio_media(media_url, media_type)
        if local_url:
            urls.append(local_url)
        i += 1
    return urls


# ── Ticket state persistence (Redis) ─────────────────────────────────────────

async def _get_ticket_state(phone: str) -> dict:
    try:
        redis = _get_redis()
        key = f"zupwell:wa_ticket:{phone.replace('whatsapp:','')}"
        raw = await redis.get(key)
        return json.loads(raw) if raw else {"stage": "idle", "draft": {}}
    except Exception:
        return {"stage": "idle", "draft": {}}


async def _save_ticket_state(phone: str, stage: str, draft: dict) -> None:
    try:
        redis = _get_redis()
        key = f"zupwell:wa_ticket:{phone.replace('whatsapp:','')}"
        await redis.set(key, json.dumps({"stage": stage, "draft": draft}), ex=3600)
    except Exception as e:
        logger.warning("Failed to save WA ticket state: %s", e)


# ── WhatsApp sender ───────────────────────────────────────────────────────────

def _send_wa(to: str, body: str) -> None:
    try:
        msg = get_twilio().messages.create(
            from_=settings.TWILIO_WHATSAPP_NUMBER,
            to=to,
            body=body,
        )
        logger.info("WA sent → %s | SID: %s", to[-4:], msg.sid)
    except Exception as e:
        logger.error("WA send failed → %s: %s", to, e)


async def _send_wa_async(to: str, body: str) -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _send_wa, to, body)


def _split(text: str, max_len: int = 1500) -> list[str]:
    if len(text) <= max_len:
        return [text]
    parts, cur = [], ""
    for para in text.split("\n\n"):
        if len(cur) + len(para) + 2 <= max_len:
            cur = (cur + "\n\n" + para).strip()
        else:
            if cur:
                parts.append(cur)
            cur = para
    if cur:
        parts.append(cur)
    return parts or [text[:max_len]]


# ── Background task ───────────────────────────────────────────────────────────

async def process_and_reply(
    from_number:  str,
    message_body: str,
    profile_name: str,
    photo_urls:   list,
) -> None:
    text = (message_body or "").strip()

    # Reset command
    if text.lower() in {"reset", "clear", "restart"}:
        await clear_session(from_number)
        redis = _get_redis()
        await redis.delete(f"zupwell:wa_ticket:{from_number.replace('whatsapp:','')}")
        await _send_wa_async(from_number, "✅ Chat cleared! How can I help you? 😊")
        return

    # Typing hint for longer messages
    if len(text) > 50 or photo_urls:
        await _send_wa_async(from_number, "⏳ Got it, looking into that...")

    ticket_state = await _get_ticket_state(from_number)

    result = await run_agent(
        user_input=text or "(Customer sent a photo)",
        session_id=from_number,
        source="whatsapp",
        user_name=profile_name if profile_name not in {"Customer", ""} else None,
        user_phone=from_number.replace("whatsapp:", ""),
        photo_urls=photo_urls,
        ticket_stage=ticket_state.get("stage", "idle"),
        ticket_draft=ticket_state.get("draft", {}),
    )

    await _save_ticket_state(
        from_number,
        result.get("ticket_stage", "idle"),
        result.get("ticket_draft", {}),
    )

    parts = _split(result["response"])
    for part in parts:
        await _send_wa_async(from_number, part)
        if len(parts) > 1:
            await asyncio.sleep(0.4)


# ── Webhook endpoint ──────────────────────────────────────────────────────────

@router.post("/webhook")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Twilio WhatsApp webhook.
    Returns empty TwiML immediately — processing happens in background.
    Handles both text messages and media (photos).
    """
    form = await request.form()
    form_dict    = dict(form)
    from_number  = form_dict.get("From", "")
    message_body = form_dict.get("Body", "")
    profile_name = form_dict.get("ProfileName", "Customer")
    num_media    = int(form_dict.get("NumMedia", "0"))

    if not from_number:
        return Response(
            content='<?xml version="1.0"?><Response></Response>',
            media_type="text/xml"
        )

    logger.info(
        "WA IN | %s (%s) | text='%s...' | media=%d",
        from_number[-4:], profile_name, message_body[:40], num_media
    )

    # Collect media asynchronously
    photo_urls = []
    if num_media > 0:
        photo_urls = await _collect_media_urls(form_dict)

    background_tasks.add_task(
        process_and_reply, from_number, message_body, profile_name, photo_urls
    )

    return Response(
        content='<?xml version="1.0"?><Response></Response>',
        media_type="text/xml"
    )


@router.get("/health")
async def health():
    return {"status": "ok", "bot": settings.BOT_NAME}
