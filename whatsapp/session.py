"""
session.py — Redis-based conversation history for WhatsApp sessions
====================================================================
Mirrors docForge's _get_history() and _save_turn() pattern.

Key format:  zupwell:session:{phone_number}
Value:       JSON list of {role, content} dicts (OpenAI message format)
TTL:         24 hours (resets on each new message)
Max turns:   20 turns kept (40 messages) to keep context window manageable
"""

import json
from redis.asyncio import Redis
from core.config import settings
from core.logger import logger

_redis: Redis | None = None

SESSION_PREFIX = "zupwell:session:"
SESSION_TTL    = 86400   # 24 hours
MAX_MESSAGES   = 40      # 20 turns (user + assistant each = 2)


def _get_redis() -> Redis:
    """Return (or create) the Redis async client."""
    global _redis
    if _redis is None:
        _redis = Redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis


def _session_key(phone_number: str) -> str:
    """Build the Redis key for a phone number's session."""
    # Strip the whatsapp: prefix if present
    clean = phone_number.replace("whatsapp:", "").strip()
    return f"{SESSION_PREFIX}{clean}"


async def get_history(phone_number: str, last_n: int = 8) -> str:
    """
    Retrieve conversation history from Redis and format for LLM prompt.
    Mirrors docForge's _get_history() — returns formatted string.

    Args:
        phone_number: The WhatsApp phone number (session ID).
        last_n:       Number of most recent turns to include.

    Returns:
        Formatted history string, or "" if no history.
    """
    try:
        redis = _get_redis()
        key   = _session_key(phone_number)
        raw   = await redis.get(key)

        if not raw:
            return ""

        messages = json.loads(raw)
        # Take the last N pairs (each turn = 2 messages)
        recent = messages[-(last_n * 2):]

        if not recent:
            return ""

        lines = ["Previous conversation:"]
        for msg in recent:
            role    = msg.get("role", "")
            content = (msg.get("content") or "")[:300]
            if role == "user":
                lines.append(f"Customer: {content}")
            elif role == "assistant":
                lines.append(f"Bot: {content[:200]}...")
        return "\n".join(lines) + "\n\n"

    except Exception as e:
        logger.warning("Failed to read session for %s: %s", phone_number[-4:], e)
        return ""


async def save_turn(phone_number: str, question: str, answer: str) -> None:
    """
    Save a conversation turn to Redis.
    Mirrors docForge's _save_turn() — deduplication + rolling window.

    Args:
        phone_number: The WhatsApp phone number.
        question:     The customer's message.
        answer:       The bot's response.
    """
    try:
        redis = _get_redis()
        key   = _session_key(phone_number)
        raw   = await redis.get(key)

        messages: list[dict] = json.loads(raw) if raw else []

        # Deduplication: don't save the same question twice in a row
        if (len(messages) >= 2
                and messages[-2].get("role") == "user"
                and messages[-2].get("content") == question):
            return

        messages.append({"role": "user",      "content": question})
        messages.append({"role": "assistant",  "content": answer})

        # Rolling window — keep last MAX_MESSAGES only
        messages = messages[-MAX_MESSAGES:]

        await redis.set(key, json.dumps(messages), ex=SESSION_TTL)

    except Exception as e:
        logger.warning("Failed to save session for %s: %s", phone_number[-4:], e)


async def clear_session(phone_number: str) -> bool:
    """
    Delete the conversation history for a phone number.
    Used when customer sends 'reset' or 'start over'.

    Returns:
        True if cleared, False if nothing to clear.
    """
    try:
        redis = _get_redis()
        key   = _session_key(phone_number)
        result = await redis.delete(key)
        return result > 0
    except Exception as e:
        logger.warning("Failed to clear session for %s: %s", phone_number[-4:], e)
        return False


async def get_session_messages(phone_number: str) -> list[dict]:
    """Return the raw message list for a session (used in admin/debug)."""
    try:
        redis = _get_redis()
        key   = _session_key(phone_number)
        raw   = await redis.get(key)
        return json.loads(raw) if raw else []
    except Exception:
        return []
