"""
rag_service.py — RAG-powered FAQ answering for Zupwell WhatsApp Bot
=====================================================================
Adapted from docForge's rag_service.py.
Keeps the same core logic:
  - ChromaDB vector search for retrieval
  - LLM answer generation with conversation history
  - Answer caching via Redis
  - Confidence scoring

Simplified for WhatsApp FAQ use case:
  - Single tool: answer_question() (equivalent to docForge's tool_search)
  - WhatsApp-friendly response format (plain text, short, no heavy markdown)
  - Session managed per phone number
"""

import asyncio
import hashlib
import json

from core.config import settings
from core.llm import get_embedding, chat_completion
from core.logger import logger
from core.vector import get_collection
from whatsapp.session import get_history, save_turn

# ── Tuning constants ──────────────────────────────────────────────────────────
MIN_SCORE     = 0.25     # minimum cosine similarity to include a chunk
TOP_K         = 5        # number of chunks to retrieve
TTL_ANSWER    = 3600     # cache FAQ answers for 1 hour
MAX_CONTEXT   = 3000     # max chars of context to send to LLM
MAX_WA_LENGTH = 1500     # max WhatsApp message length before splitting

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are Zupwell's friendly WhatsApp customer support assistant.
Zupwell is a premium health supplements brand from Ahmedabad, India.
You help customers with questions about Zupwell products, orders, shipping, returns, and general brand information.

RULES — follow every rule strictly:

1. ANSWER FROM CONTEXT ONLY
   - Use ONLY the FAQ context provided. Do not make up information.
   - If the context partially answers the question, give the partial answer and say what is missing.
   - If there is absolutely no relevant info, say so politely and direct them to contact support.

2. TONE & STYLE
   - Be friendly, warm, and concise — this is WhatsApp, not an essay.
   - Use a helpful, professional tone like a knowledgeable customer support agent.
   - Keep responses under 250 words where possible.
   - Use simple formatting: bullet points (•), line breaks. Avoid heavy markdown.

3. FORMAT BY QUESTION TYPE
   - Simple fact → Direct 1-2 sentence answer.
   - Process question → Short numbered steps.
   - Yes/No question → Start with Yes or No, then brief explanation.
   - Contact questions → Always include: Email: info@zupwell.com | WhatsApp: +91 6355466208

4. WHEN NOT IN CONTEXT
   - Say: "I don't have that specific detail right now. Please contact us at info@zupwell.com or WhatsApp +91 6355466208 — our team will be happy to help! 😊"

5. CLOSING
   - End most responses with a brief helpful closing or "Is there anything else I can help you with? 😊"
   - Do NOT add this closer if the conversation already feels concluded.
"""

ANSWER_PROMPT = """\
{history}

FAQ Context:
{context}

Customer Question: {question}

Answer (friendly, concise, WhatsApp-appropriate):"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cache_key(question: str) -> str:
    """Deterministic cache key for a question (same pattern as docForge)."""
    raw = json.dumps({"q": question.strip().lower()}, sort_keys=True)
    return f"zupwell:faq:answer:{hashlib.md5(raw.encode()).hexdigest()}"


def _build_context(chunks: list[dict]) -> str:
    """Build a readable context string from retrieved chunks."""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[{i}] {c['title']} ({c['category']})\n{c['text']}"
        )
    return "\n\n".join(parts)[:MAX_CONTEXT]


def _confidence(chunks: list[dict]) -> str:
    """Score confidence based on top chunk similarity (same logic as docForge)."""
    if not chunks:
        return "low"
    top = chunks[0]["score"]
    if top >= 0.65:
        return "high"
    if top >= 0.40:
        return "medium"
    return "low"


# ── Retrieval ─────────────────────────────────────────────────────────────────

async def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Vector search against ChromaDB for the given query.
    Mirrors docForge's _retrieve() function.

    Returns a list of chunk dicts sorted by score descending.
    """
    collection = get_collection()

    count = collection.count()
    if count == 0:
        logger.warning("ChromaDB collection is empty — please run ingest first.")
        return []

    # Get embedding for the query
    query_emb = await get_embedding(query)

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    docs      = results.get("documents", [[]])[0]
    metas     = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        # Convert cosine distance to similarity score (same formula as docForge)
        score = round(1 - dist / 2, 4)
        if score < MIN_SCORE:
            continue
        chunks.append({
            "score":    score,
            "faq_id":   meta.get("faq_id", ""),
            "category": meta.get("category", ""),
            "title":    meta.get("title", ""),
            "text":     doc,
        })

    # Sort by score descending
    chunks.sort(key=lambda x: x["score"], reverse=True)
    logger.info(
        "Retrieved %d chunks for query '%s...' (top score: %.3f)",
        len(chunks), query[:40], chunks[0]["score"] if chunks else 0
    )
    return chunks


# ── Main answer function ──────────────────────────────────────────────────────

async def answer_question(question: str, phone_number: str) -> dict:
    """
    Main RAG Q&A function — equivalent to docForge's tool_search().

    1. Retrieve relevant FAQ chunks from ChromaDB.
    2. Build context + conversation history.
    3. Call LLM to generate a WhatsApp-friendly answer.
    4. Save turn to Redis session.

    Args:
        question:     The customer's message text.
        phone_number: The customer's WhatsApp phone number (used as session ID).

    Returns:
        {
            "answer":     str,   # the response to send
            "confidence": str,   # "high" / "medium" / "low"
            "chunks":     list,  # retrieved chunks (for debugging)
        }
    """
    # ── 1. Retrieve relevant chunks ───────────────────────────────────────────
    chunks = await retrieve(question)

    if not chunks:
        answer = (
            "I'm sorry, I don't have that information right now. 😔\n\n"
            "Please contact our team directly:\n"
            "📧 Email: info@zupwell.com\n"
            "📱 WhatsApp: +91 6355466208\n\n"
            "We're happy to help! 😊"
        )
        return {"answer": answer, "confidence": "low", "chunks": []}

    # ── 2. Build context and history ──────────────────────────────────────────
    context = _build_context(chunks)
    history = await get_history(phone_number)

    # ── 3. Build messages for LLM ─────────────────────────────────────────────
    user_content = ANSWER_PROMPT.format(
        history=history,
        context=context,
        question=question,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    # ── 4. Generate answer ────────────────────────────────────────────────────
    try:
        answer = await chat_completion(messages, temperature=0.3, max_tokens=400)
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        answer = (
            "Sorry, I'm having a bit of trouble right now! 😅\n"
            "Please reach out to us directly:\n"
            "📧 info@zupwell.com\n"
            "📱 +91 6355466208"
        )

    # ── 5. Save turn to Redis session ─────────────────────────────────────────
    await save_turn(phone_number, question, answer)

    confidence = _confidence(chunks)
    logger.info(
        "Answered question from %s | confidence: %s | chunks used: %d",
        phone_number[-4:], confidence, len(chunks)
    )

    return {
        "answer":     answer,
        "confidence": confidence,
        "chunks":     chunks,
    }


def split_for_whatsapp(text: str, max_len: int = MAX_WA_LENGTH) -> list[str]:
    """
    Split a long response into multiple WhatsApp messages if needed.
    Splits on paragraph boundaries where possible.

    Args:
        text:    The full response text.
        max_len: Maximum characters per message.

    Returns:
        List of message strings (usually just one).
    """
    if len(text) <= max_len:
        return [text]

    parts = []
    paragraphs = text.split("\n\n")
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_len:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                parts.append(current)
            current = para

    if current:
        parts.append(current)

    return parts or [text[:max_len]]
