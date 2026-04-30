"""
website/chat_api.py — Website Chat REST API
============================================
Endpoints:
  POST /chat/message        Send a message → get bot reply
  POST /chat/upload-photo   Upload a photo for a ticket
  GET  /chat/history        Session conversation history
  POST /chat/reset          Clear session
  GET  /chat/ticket/{ref}   Check ticket status (public)
  GET  /chat/admin/tickets  List all tickets (admin)

Auth: reads JWT Bearer token from Authorization header.
Same JWT secret as ps5 backend — logged-in user auto-identified.
"""

import mimetypes
import uuid
from pathlib import Path
from typing import Optional

import jwt
from fastapi import APIRouter, Depends, Header, HTTPException, UploadFile, File
from pydantic import BaseModel

from agent.runner import run_agent
from core.config import settings
from core.logger import logger
from tickets.tickets import get_ticket, list_tickets
from whatsapp.session import (
    clear_session, get_session_messages, _get_redis
)

router  = APIRouter(prefix="/chat", tags=["chat"])
UPLOAD_DIR = Path(settings.UPLOAD_DIR)
MAX_BYTES  = settings.UPLOAD_MAX_MB * 1024 * 1024

ALLOWED_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/webp", "image/heic", "image/heif",
}


# ── Models ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:    str
    session_id: str
    photo_urls: list[str] = []   # URLs returned by /chat/upload-photo


class ChatResponse(BaseModel):
    response:   str
    intent:     str
    confidence: str
    ticket_ref: Optional[str] = None
    session_id: str


class UploadResponse(BaseModel):
    url:      str
    filename: str


# ── Auth ──────────────────────────────────────────────────────────────────────

def _decode_jwt(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
    except Exception:
        return None


def _get_user(authorization: Optional[str] = Header(None)) -> dict:
    user = {"user_id": None, "name": None, "email": None}
    if not authorization or not authorization.startswith("Bearer "):
        return user
    payload = _decode_jwt(authorization.removeprefix("Bearer ").strip())
    if payload:
        user["user_id"] = payload.get("userId") or payload.get("id")
        user["name"]    = payload.get("name")   or payload.get("username")
        user["email"]   = payload.get("email")
    return user


def _web_sid(session_id: str) -> str:
    return f"web:{session_id}"


# ── Ticket state helpers ──────────────────────────────────────────────────────

import json

async def _get_ticket_state(session_id: str) -> dict:
    try:
        redis = _get_redis()
        raw = await redis.get(f"zupwell:web_ticket:{session_id}")
        return json.loads(raw) if raw else {"stage": "idle", "draft": {}}
    except Exception:
        return {"stage": "idle", "draft": {}}


async def _save_ticket_state(session_id: str, stage: str, draft: dict) -> None:
    try:
        redis = _get_redis()
        await redis.set(
            f"zupwell:web_ticket:{session_id}",
            json.dumps({"stage": stage, "draft": draft}),
            ex=3600,
        )
    except Exception as e:
        logger.warning("Ticket state save failed: %s", e)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/upload-photo", response_model=UploadResponse)
async def upload_photo(file: UploadFile = File(...)):
    """
    Upload a photo for a damaged product ticket.
    Returns a URL that the frontend passes in the next /chat/message call
    via the photo_urls field.

    Accepts: JPEG, PNG, WEBP, HEIC (max 10 MB by default)
    """
    content_type = file.content_type or "image/jpeg"
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{content_type}'. Allowed: JPEG, PNG, WEBP, HEIC"
        )

    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size is {settings.UPLOAD_MAX_MB} MB."
        )

    ext      = mimetypes.guess_extension(content_type) or ".jpg"
    ext      = ext.replace(".jpe", ".jpg")
    filename = f"{uuid.uuid4().hex}{ext}"
    (UPLOAD_DIR / filename).write_bytes(data)

    url = f"/uploads/{filename}"
    logger.info("Photo uploaded: %s (%d bytes)", filename, len(data))
    return UploadResponse(url=url, filename=filename)


@router.post("/message", response_model=ChatResponse)
async def chat_message(
    body: ChatRequest,
    user: dict = Depends(_get_user),
):
    """
    Process one chat message and return the bot's reply.

    Send photo_urls (from /chat/upload-photo) alongside the message
    when the customer is reporting a damaged product.
    """
    sid = _web_sid(body.session_id)
    ticket_state = await _get_ticket_state(body.session_id)

    result = await run_agent(
        user_input=body.message,
        session_id=sid,
        source="website",
        user_id=user.get("user_id"),
        user_name=user.get("name"),
        user_email=user.get("email"),
        photo_urls=body.photo_urls or [],
        ticket_stage=ticket_state.get("stage", "idle"),
        ticket_draft=ticket_state.get("draft", {}),
    )

    await _save_ticket_state(
        body.session_id,
        result.get("ticket_stage", "idle"),
        result.get("ticket_draft", {}),
    )

    ticket_ref = None
    if result.get("ticket_created"):
        ticket_ref = result["ticket_created"].get("ticket_ref")

    return ChatResponse(
        response=result["response"],
        intent=result["intent"],
        confidence=result["confidence"],
        ticket_ref=ticket_ref,
        session_id=body.session_id,
    )


@router.get("/history")
async def chat_history(session_id: str):
    """Return the conversation history for a session."""
    messages = await get_session_messages(_web_sid(session_id))
    return {"session_id": session_id, "messages": messages}


@router.post("/reset")
async def chat_reset(session_id: str):
    """Clear conversation history and ticket state."""
    await clear_session(_web_sid(session_id))
    try:
        redis = _get_redis()
        await redis.delete(f"zupwell:web_ticket:{session_id}")
    except Exception:
        pass
    return {"status": "reset", "session_id": session_id}


@router.get("/ticket/{ticket_ref}")
async def ticket_status(ticket_ref: str):
    """Public endpoint — customer checks their ticket status."""
    ticket = await get_ticket(ticket_ref)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return {
        "ticket_ref": ticket["ticket_ref"],
        "subject":    ticket["subject"],
        "status":     ticket["status"],
        "category":   ticket["category"],
        "has_photos": bool(ticket.get("photo_urls")),
        "created_at": ticket["created_at"],
    }


@router.get("/admin/tickets")
async def admin_tickets(
    status: Optional[str] = None,
    source: Optional[str] = None,
    limit:  int = 50,
    x_admin_key: Optional[str] = Header(None),
):
    """List all tickets — admin only. Pass X-Admin-Key header."""
    if x_admin_key != settings.ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    tickets = await list_tickets(status=status, source=source, limit=limit)
    return {"tickets": tickets, "count": len(tickets)}
