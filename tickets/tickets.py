"""
tickets.py — Support Ticket Service (PostgreSQL via asyncpg)
=============================================================
Creates and manages support tickets in the same PostgreSQL DB
as the ps5/Zupwell backend (shared DB, separate table).

Table: support_tickets
  - Auto-created on first use (no migration needed)
  - Links to users table via user_id (nullable for anonymous/WhatsApp users)
"""

import asyncio
from datetime import datetime
from typing import Optional
import asyncpg
from core.config import settings
from core.logger import logger

_pool: Optional[asyncpg.Pool] = None

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS support_tickets (
    id            SERIAL PRIMARY KEY,
    ticket_ref    VARCHAR(20)  UNIQUE NOT NULL,
    user_id       INTEGER      REFERENCES users(id) ON DELETE SET NULL,
    source        VARCHAR(20)  NOT NULL DEFAULT 'website',   -- 'website' | 'whatsapp'
    name          VARCHAR(200),
    email         VARCHAR(200),
    phone         VARCHAR(50),
    subject       VARCHAR(500) NOT NULL,
    description   TEXT         NOT NULL,
    category      VARCHAR(100) DEFAULT 'general',
    status        VARCHAR(50)  NOT NULL DEFAULT 'open',      -- open | in_progress | resolved | closed
    priority      VARCHAR(20)  NOT NULL DEFAULT 'normal',    -- low | normal | high | urgent
    created_at    TIMESTAMP    NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMP    NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_tickets_status    ON support_tickets(status);
CREATE INDEX IF NOT EXISTS idx_tickets_user_id   ON support_tickets(user_id);
CREATE INDEX IF NOT EXISTS idx_tickets_source    ON support_tickets(source);
CREATE INDEX IF NOT EXISTS idx_tickets_created   ON support_tickets(created_at DESC);
"""


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            settings.DATABASE_URL,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
        # Ensure table exists
        async with _pool.acquire() as conn:
            await conn.execute(CREATE_TABLE_SQL)
        logger.info("✅ Ticket DB pool ready")
    return _pool


def _generate_ref() -> str:
    """Generate a human-friendly ticket reference like ZUP-20241230-4F2A."""
    import random, string
    now = datetime.utcnow().strftime("%Y%m%d")
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"ZUP-{now}-{suffix}"


def _classify_priority(subject: str, description: str) -> str:
    """Auto-classify ticket priority from subject/description keywords."""
    text = (subject + " " + description).lower()
    urgent_kw = ["urgent", "asap", "immediately", "critical", "emergency", "not working"]
    high_kw = ["damaged", "wrong product", "not received", "fraud", "charged twice"]
    if any(k in text for k in urgent_kw):
        return "urgent"
    if any(k in text for k in high_kw):
        return "high"
    return "normal"


def _classify_category(subject: str, description: str) -> str:
    """Auto-classify ticket category."""
    text = (subject + " " + description).lower()
    if any(k in text for k in ["order", "cancel", "purchase", "buy"]):
        return "order"
    if any(k in text for k in ["ship", "deliver", "track", "dispatch", "courier"]):
        return "shipping"
    if any(k in text for k in ["refund", "return", "money back", "exchange"]):
        return "refund"
    if any(k in text for k in ["product", "ingredient", "dose", "side effect", "expire"]):
        return "product"
    if any(k in text for k in ["payment", "upi", "card", "charged", "invoice"]):
        return "payment"
    if any(k in text for k in ["account", "login", "password", "profile"]):
        return "account"
    return "general"


async def create_ticket(
    subject: str,
    description: str,
    source: str = "website",
    name: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    user_id: Optional[int] = None,
) -> dict:
    """
    Create a new support ticket in PostgreSQL.

    Args:
        subject:     Short summary of the issue (from agent extraction).
        description: Full issue description from the conversation.
        source:      'website' or 'whatsapp'
        name:        Customer name (from auth session or collected by bot).
        email:       Customer email.
        phone:       Customer phone (WhatsApp number).
        user_id:     Logged-in user ID (from JWT, None for anonymous).

    Returns:
        Dict with ticket_ref, id, status, created_at.
    """
    pool = await get_pool()

    ticket_ref = _generate_ref()
    priority   = _classify_priority(subject, description)
    category   = _classify_category(subject, description)

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO support_tickets
                (ticket_ref, user_id, source, name, email, phone,
                 subject, description, category, priority)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
            RETURNING id, ticket_ref, status, priority, category, created_at
            """,
            ticket_ref, user_id, source, name, email, phone,
            subject, description, category, priority,
        )

    result = dict(row)
    result["created_at"] = result["created_at"].isoformat()

    logger.info(
        "🎫 Ticket created: %s | source=%s | priority=%s | category=%s",
        ticket_ref, source, priority, category,
    )
    return result


async def get_ticket(ticket_ref: str) -> Optional[dict]:
    """Fetch a ticket by its reference number."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM support_tickets WHERE ticket_ref = $1", ticket_ref
        )
    if not row:
        return None
    result = dict(row)
    for k in ("created_at", "updated_at"):
        if result.get(k):
            result[k] = result[k].isoformat()
    return result


async def list_tickets(
    status: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """List tickets with optional filters — for admin panel."""
    pool = await get_pool()
    conditions = []
    params = []
    i = 1

    if status:
        conditions.append(f"status = ${i}")
        params.append(status)
        i += 1
    if source:
        conditions.append(f"source = ${i}")
        params.append(source)
        i += 1

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    params.append(limit)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT * FROM support_tickets {where} ORDER BY created_at DESC LIMIT ${i}",
            *params,
        )

    results = []
    for row in rows:
        r = dict(row)
        for k in ("created_at", "updated_at"):
            if r.get(k):
                r[k] = r[k].isoformat()
        results.append(r)
    return results
