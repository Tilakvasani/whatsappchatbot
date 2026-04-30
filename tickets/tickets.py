"""
tickets.py — Support Ticket Service (PostgreSQL / asyncpg)
==========================================================
Supports photo_urls for damaged product tickets.
Table is auto-created on first use — no migration needed.
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
    id          SERIAL       PRIMARY KEY,
    ticket_ref  VARCHAR(20)  UNIQUE NOT NULL,
    user_id     INTEGER,
    source      VARCHAR(20)  NOT NULL DEFAULT 'website',
    name        VARCHAR(200),
    email       VARCHAR(200),
    phone       VARCHAR(50),
    subject     VARCHAR(500) NOT NULL,
    description TEXT         NOT NULL,
    category    VARCHAR(100) DEFAULT 'general',
    status      VARCHAR(50)  NOT NULL DEFAULT 'open',
    priority    VARCHAR(20)  NOT NULL DEFAULT 'normal',
    photo_urls  TEXT[]       DEFAULT '{}',
    created_at  TIMESTAMP    NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMP    NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_tickets_status  ON support_tickets(status);
CREATE INDEX IF NOT EXISTS idx_tickets_user    ON support_tickets(user_id);
CREATE INDEX IF NOT EXISTS idx_tickets_source  ON support_tickets(source);
CREATE INDEX IF NOT EXISTS idx_tickets_created ON support_tickets(created_at DESC);
"""


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            settings.DATABASE_URL, min_size=2, max_size=10, command_timeout=30
        )
        async with _pool.acquire() as conn:
            await conn.execute(CREATE_TABLE_SQL)
        logger.info("✅ Ticket DB pool ready")
    return _pool


def _make_ref() -> str:
    import random, string
    now    = datetime.utcnow().strftime("%Y%m%d")
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"ZUP-{now}-{suffix}"


def _priority(subject: str, desc: str) -> str:
    text = (subject + " " + desc).lower()
    if any(k in text for k in ["urgent","asap","critical","emergency","immediately"]):
        return "urgent"
    if any(k in text for k in ["damaged","wrong product","not received","fraud","charged twice"]):
        return "high"
    return "normal"


def _category(subject: str, desc: str) -> str:
    text = (subject + " " + desc).lower()
    if any(k in text for k in ["order","cancel","purchase","buy"]):        return "order"
    if any(k in text for k in ["ship","deliver","track","dispatch"]):      return "shipping"
    if any(k in text for k in ["refund","return","money back","exchange"]): return "refund"
    if any(k in text for k in ["damage","broken","defect","crack","leak"]): return "damaged_product"
    if any(k in text for k in ["product","ingredient","dose","expire"]):    return "product"
    if any(k in text for k in ["payment","upi","card","charged","invoice"]): return "payment"
    if any(k in text for k in ["account","login","password"]):              return "account"
    return "general"


async def create_ticket(
    subject:     str,
    description: str,
    source:      str = "website",
    name:        Optional[str]  = None,
    email:       Optional[str]  = None,
    phone:       Optional[str]  = None,
    user_id:     Optional[int]  = None,
    photo_urls:  Optional[list] = None,
) -> dict:
    pool       = await get_pool()
    ref        = _make_ref()
    priority   = _priority(subject, description)
    category   = _category(subject, description)
    photo_list = photo_urls or []

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO support_tickets
              (ticket_ref, user_id, source, name, email, phone,
               subject, description, category, priority, photo_urls)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
            RETURNING id, ticket_ref, status, priority, category, photo_urls, created_at
            """,
            ref, user_id, source, name, email, phone,
            subject, description, category, priority, photo_list,
        )

    result = dict(row)
    result["created_at"] = result["created_at"].isoformat()
    logger.info("🎫 %s created | %s | priority=%s | photos=%d",
                ref, category, priority, len(photo_list))
    return result


async def get_ticket(ticket_ref: str) -> Optional[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM support_tickets WHERE ticket_ref=$1", ticket_ref
        )
    if not row:
        return None
    r = dict(row)
    for k in ("created_at", "updated_at"):
        if r.get(k):
            r[k] = r[k].isoformat()
    return r


async def list_tickets(
    status: Optional[str] = None,
    source: Optional[str] = None,
    limit:  int = 50,
) -> list[dict]:
    pool = await get_pool()
    conds, params, i = [], [], 1
    if status:
        conds.append(f"status=${i}"); params.append(status); i += 1
    if source:
        conds.append(f"source=${i}"); params.append(source); i += 1
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    params.append(limit)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT * FROM support_tickets {where} ORDER BY created_at DESC LIMIT ${i}",
            *params,
        )
    results = []
    for row in rows:
        r = dict(row)
        for k in ("created_at","updated_at"):
            if r.get(k): r[k] = r[k].isoformat()
        results.append(r)
    return results
