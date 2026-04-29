"""
webhook.py — Twilio WhatsApp Webhook Handler
=============================================
Handles incoming WhatsApp messages from Twilio, processes them via RAG,
and sends replies using the Twilio REST API.

Architecture:
  - FastAPI endpoint receives Twilio POST webhook
  - Returns empty TwiML immediately (avoids Twilio's 15s timeout)
  - Background task handles the actual RAG + reply

Special commands:
  hi / hello / start       → Welcome message
  help / menu              → Show available help topics
  reset / clear / restart  → Clear conversation history
  about / who are you      → About the bot
"""

import asyncio
from fastapi import APIRouter, Request, BackgroundTasks, Response
from twilio.rest import Client as TwilioClient

from core.config import settings
from core.logger import logger
from rag.rag_service import answer_question, split_for_whatsapp
from whatsapp.session import clear_session

router = APIRouter(prefix="/whatsapp", tags=["whatsapp"])

# ── Twilio client ─────────────────────────────────────────────────────────────
_twilio_client: TwilioClient | None = None


def get_twilio_client() -> TwilioClient:
    """Return the shared Twilio client singleton."""
    global _twilio_client
    if _twilio_client is None:
        _twilio_client = TwilioClient(
            settings.TWILIO_ACCOUNT_SID,
            settings.TWILIO_AUTH_TOKEN,
        )
    return _twilio_client


# ── Static responses ──────────────────────────────────────────────────────────

WELCOME_MESSAGE = """\
👋 *Welcome to Zupwell Support!*

Hi there! I'm your Zupwell assistant. I can help you with:

• 🛍️ Products & ingredients
• 📦 Orders & payments
• 🚚 Shipping & delivery
• 🔄 Returns & refunds
• ℹ️ About Zupwell

Just type your question and I'll be happy to help! 😊

_Type *help* to see example questions._"""

HELP_MESSAGE = """\
🤖 *Zupwell Support — Example Questions*

You can ask me things like:

*Products:*
• What products does Zupwell offer?
• Are Zupwell supplements FSSAI approved?
• What are the benefits of the electrolytes?

*Orders & Shipping:*
• How do I place an order?
• How long does delivery take?
• How can I track my order?

*Returns & Refunds:*
• What is the return policy?
• How do I request a refund?

*Contact:*
• How can I contact Zupwell?

📧 Email: info@zupwell.com
📱 Call/WhatsApp: +91 6355466208

_Just type your question and I'll answer! 😊_"""

RESET_MESSAGE = """\
✅ *Conversation reset!*

Your chat history has been cleared. Let's start fresh! 😊

How can I help you today?"""

ABOUT_MESSAGE = """\
🌟 *About This Bot*

I'm the Zupwell virtual support assistant, powered by AI.

I'm trained on Zupwell's FAQs and can answer questions about:
• Products, ingredients & usage
• Orders, payments & tracking
• Shipping, delivery & returns
• Brand information & policies

For complex queries, I'll always connect you with our team:
📧 info@zupwell.com
📱 +91 6355466208

_Ask away — I'm here to help! 💪_"""


# ── Intent detection ──────────────────────────────────────────────────────────

def _detect_special_command(text: str) -> str | None:
    """
    Check if the message is a special bot command.

    Returns:
        Command name or None.
    """
    t = text.strip().lower()

    greetings = {"hi", "hello", "hey", "hii", "helo", "namaste", "start", "kem cho"}
    if t in greetings or t.startswith("hi ") or t.startswith("hello "):
        return "welcome"

    if t in {"help", "menu", "options", "commands", "?"}:
        return "help"

    if t in {"reset", "clear", "restart", "new chat", "start over", "clr"}:
        return "reset"

    if t in {"about", "who are you", "what are you", "bot info"}:
        return "about"

    return None


# ── Message sender ────────────────────────────────────────────────────────────

def send_whatsapp_message(to: str, body: str) -> None:
    """
    Send a WhatsApp message via Twilio REST API.
    Synchronous — called from background task thread.

    Args:
        to:   Recipient number in whatsapp:+XXXXXXXXXXX format.
        body: Message text.
    """
    try:
        client = get_twilio_client()
        message = client.messages.create(
            from_=settings.TWILIO_WHATSAPP_NUMBER,
            to=to,
            body=body,
        )
        logger.info("Sent WA message to %s | SID: %s", to[-4:], message.sid)
    except Exception as e:
        logger.error("Failed to send WhatsApp message to %s: %s", to, e)


async def send_whatsapp_message_async(to: str, body: str) -> None:
    """Async wrapper — runs Twilio's sync client in a thread pool."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, send_whatsapp_message, to, body)


# ── Background processing ─────────────────────────────────────────────────────

async def process_and_reply(from_number: str, message_text: str) -> None:
    """
    Core background task:
      1. Detect special commands.
      2. Run RAG to get an answer.
      3. Send reply via Twilio.

    Args:
        from_number:  Sender's WhatsApp number (whatsapp:+XXXXXXXXXX).
        message_text: The incoming message text.
    """
    text = message_text.strip()

    # ── 1. Handle special commands ────────────────────────────────────────────
    command = _detect_special_command(text)

    if command == "welcome":
        await send_whatsapp_message_async(from_number, WELCOME_MESSAGE)
        return

    if command == "help":
        await send_whatsapp_message_async(from_number, HELP_MESSAGE)
        return

    if command == "reset":
        await clear_session(from_number)
        await send_whatsapp_message_async(from_number, RESET_MESSAGE)
        return

    if command == "about":
        await send_whatsapp_message_async(from_number, ABOUT_MESSAGE)
        return

    # ── 2. Send "typing" indicator ────────────────────────────────────────────
    # Optionally send a brief thinking message for longer queries
    if len(text) > 50:
        await send_whatsapp_message_async(from_number, "⏳ Looking that up for you...")

    # ── 3. Run RAG ────────────────────────────────────────────────────────────
    try:
        result = await answer_question(
            question=text,
            phone_number=from_number,
        )
        answer = result["answer"]

        # Log low-confidence answers for monitoring
        if result["confidence"] == "low":
            logger.warning(
                "Low confidence answer for: '%s...' from %s",
                text[:50], from_number[-4:]
            )

    except Exception as e:
        logger.error("RAG failed for %s: %s", from_number[-4:], e)
        answer = (
            "Sorry, something went wrong on my end! 😅\n\n"
            "Please contact us directly:\n"
            "📧 info@zupwell.com\n"
            "📱 +91 6355466208"
        )

    # ── 4. Split and send (handles long responses) ────────────────────────────
    parts = split_for_whatsapp(answer)
    for part in parts:
        await send_whatsapp_message_async(from_number, part)
        if len(parts) > 1:
            await asyncio.sleep(0.5)  # small gap between split messages


# ── Webhook endpoint ──────────────────────────────────────────────────────────

@router.post("/webhook")
async def whatsapp_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Twilio WhatsApp webhook endpoint.

    Twilio sends a POST with form data including:
        From: whatsapp:+XXXXXXXXXX
        Body: the message text
        ProfileName: sender's WhatsApp display name (optional)

    Returns an empty TwiML response immediately (avoids Twilio's 15s timeout).
    The actual processing happens in a BackgroundTask.
    """
    form_data = await request.form()

    from_number  = form_data.get("From", "")
    message_body = form_data.get("Body", "")
    profile_name = form_data.get("ProfileName", "Customer")

    if not from_number or not message_body:
        logger.warning("Received empty webhook — From: %s | Body: %s",
                        from_number, message_body)
        return Response(
            content='<?xml version="1.0"?><Response></Response>',
            media_type="text/xml",
        )

    logger.info(
        "Incoming WA from %s (%s): '%s...'",
        from_number[-4:], profile_name, message_body[:50]
    )

    # Queue the processing as a background task and respond immediately
    background_tasks.add_task(process_and_reply, from_number, message_body)

    # Return empty TwiML — Twilio is satisfied, no timeout
    return Response(
        content='<?xml version="1.0"?><Response></Response>',
        media_type="text/xml",
    )


@router.get("/health")
async def health():
    """Health check for the WhatsApp bot service."""
    return {"status": "ok", "bot": settings.BOT_NAME}
