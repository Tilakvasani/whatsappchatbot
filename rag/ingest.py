"""
ingest.py — Notion → Chunks → Embeddings → ChromaDB
=====================================================
Pipeline:
  1. Fetch all pages from your Notion database (products, docs, policies etc.)
  2. Extract full page content block-by-block (same approach as docForge)
  3. Chunk text into ~500-token pieces with 50-token overlap
  4. Embed each chunk via Azure OpenAI text-embedding-3-large
  5. Upsert into ChromaDB (idempotent — safe to re-run anytime)

How to add content to the bot:
  - Add a new page to your Notion database
  - Run:  python scripts/ingest_notion.py
  - Done! The bot will now know about it.
"""

import asyncio
import hashlib
import re
from typing import Optional

import httpx

from core.config import settings
from core.llm import get_embedding
from core.logger import logger
from core.vector import get_collection

# ── Chunking constants ────────────────────────────────────────────────────────
CHUNK_SIZE    = 1800   # chars per chunk (~450 tokens)
CHUNK_OVERLAP = 200    # chars of overlap between chunks
BATCH_SIZE    = 8      # embed this many chunks at once

NOTION_API = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"


# ── Notion API helpers ────────────────────────────────────────────────────────

def _headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


async def _get_all_pages(client: httpx.AsyncClient) -> list[dict]:
    """Fetch all pages from the Notion database (handles pagination)."""
    pages = []
    cursor: Optional[str] = None

    while True:
        body: dict = {"page_size": 100}
        if cursor:
            body["start_cursor"] = cursor

        resp = await client.post(
            f"{NOTION_API}/databases/{settings.NOTION_DATABASE_ID}/query",
            headers=_headers(),
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()
        pages.extend(data.get("results", []))

        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")

    logger.info("Fetched %d pages from Notion database", len(pages))
    return pages


async def _get_page_blocks(client: httpx.AsyncClient, page_id: str) -> list[dict]:
    """Recursively fetch all content blocks for a Notion page."""
    blocks = []
    cursor: Optional[str] = None

    while True:
        params: dict = {"page_size": 100}
        if cursor:
            params["start_cursor"] = cursor

        resp = await client.get(
            f"{NOTION_API}/blocks/{page_id}/children",
            headers=_headers(),
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        blocks.extend(results)

        # Recursively fetch children for toggle/column/synced blocks
        for block in results:
            if block.get("has_children") and block.get("type") not in (
                "child_page", "child_database"
            ):
                children = await _get_page_blocks(client, block["id"])
                blocks.extend(children)

        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")

    return blocks


def _extract_rich_text(rich_text_list: list) -> str:
    """Pull plain text from Notion rich_text array."""
    return "".join(rt.get("plain_text", "") for rt in rich_text_list)


def _block_to_text(block: dict) -> str:
    """Convert a single Notion block to a plain text string."""
    btype = block.get("type", "")
    content = block.get(btype, {})

    if btype in ("paragraph", "quote", "callout"):
        return _extract_rich_text(content.get("rich_text", []))

    if btype in ("heading_1", "heading_2", "heading_3"):
        text = _extract_rich_text(content.get("rich_text", []))
        return f"\n{'#' * int(btype[-1])} {text}\n" if text else ""

    if btype in ("bulleted_list_item", "numbered_list_item", "to_do"):
        text = _extract_rich_text(content.get("rich_text", []))
        return f"• {text}" if text else ""

    if btype == "code":
        text = _extract_rich_text(content.get("rich_text", []))
        lang = content.get("language", "")
        return f"[Code - {lang}]: {text}" if text else ""

    if btype == "divider":
        return "---"

    if btype == "table_row":
        cells = content.get("cells", [])
        return " | ".join(_extract_rich_text(cell) for cell in cells)

    return ""


def _page_title(page: dict) -> str:
    """Extract the title from a Notion page object."""
    props = page.get("properties", {})
    for prop in props.values():
        if prop.get("type") == "title":
            return _extract_rich_text(prop.get("title", []))
    return "Untitled"


def _page_metadata(page: dict) -> dict:
    """Extract useful metadata fields from a page."""
    props = page.get("properties", {})
    meta: dict = {
        "notion_id": page.get("id", ""),
        "title": _page_title(page),
        "url": page.get("url", ""),
    }

    # Try to grab a 'Category' or 'Tags' select/multi-select property
    for key, prop in props.items():
        if prop.get("type") == "select" and prop.get("select"):
            meta["category"] = prop["select"].get("name", "")
            break
        if prop.get("type") == "multi_select":
            tags = [t.get("name", "") for t in prop.get("multi_select", [])]
            if tags:
                meta["tags"] = ", ".join(tags)

    return meta


def _chunk_text(text: str, title: str) -> list[str]:
    """
    Split text into overlapping chunks.
    Prepend the page title to every chunk so context is never lost.
    """
    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text.strip())

    if not text:
        return []

    # If short enough, return as single chunk
    if len(text) <= CHUNK_SIZE:
        return [f"[{title}]\n{text}"]

    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]

        # Try to split on paragraph boundary
        last_para = chunk.rfind("\n\n")
        if last_para > CHUNK_SIZE // 2:
            chunk = chunk[:last_para]

        chunks.append(f"[{title}]\n{chunk.strip()}")
        start += len(chunk) - CHUNK_OVERLAP
        if start >= len(text):
            break

    return chunks


def _chunk_id(notion_id: str, chunk_index: int) -> str:
    """Deterministic ChromaDB document ID for a chunk."""
    raw = f"{notion_id}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── Main ingest function ──────────────────────────────────────────────────────

async def ingest_notion(force: bool = False) -> dict:
    """
    Full Notion → ChromaDB ingest pipeline.

    Args:
        force: Re-embed all chunks even if already present in ChromaDB.

    Returns:
        {"pages": int, "chunks_ingested": int, "chunks_skipped": int, "total": int}
    """
    if not settings.NOTION_TOKEN or not settings.NOTION_DATABASE_ID:
        raise ValueError(
            "NOTION_TOKEN and NOTION_DATABASE_ID must be set in .env"
        )

    collection = get_collection()

    # Get existing chunk IDs for delta sync
    existing_ids: set[str] = set()
    if not force:
        try:
            existing = collection.get(include=[])
            existing_ids = set(existing.get("ids", []))
        except Exception:
            pass

    total_pages = 0
    total_ingested = 0
    total_skipped = 0

    async with httpx.AsyncClient(timeout=30) as client:
        pages = await _get_all_pages(client)

        for page in pages:
            page_id = page.get("id", "")
            meta = _page_metadata(page)
            title = meta["title"]

            logger.info("Processing page: '%s' (%s)", title, page_id[:8])

            # Fetch all content blocks
            try:
                blocks = await _get_page_blocks(client, page_id)
            except Exception as e:
                logger.error("Failed to fetch blocks for '%s': %s", title, e)
                continue

            # Convert blocks to text
            lines = []
            for block in blocks:
                text = _block_to_text(block)
                if text:
                    lines.append(text)

            full_text = "\n".join(lines)
            if not full_text.strip():
                logger.warning("Page '%s' has no extractable text — skipping", title)
                continue

            # Chunk the text
            chunks = _chunk_text(full_text, title)
            if not chunks:
                continue

            # Embed and upsert in batches
            batch_ids, batch_docs, batch_embeddings, batch_metas = [], [], [], []

            for idx, chunk_text in enumerate(chunks):
                cid = _chunk_id(page_id, idx)

                if cid in existing_ids and not force:
                    total_skipped += 1
                    continue

                embedding = await get_embedding(chunk_text)

                batch_ids.append(cid)
                batch_docs.append(chunk_text)
                batch_embeddings.append(embedding)
                batch_metas.append({
                    "notion_id": page_id,
                    "title":     title,
                    "category":  meta.get("category", ""),
                    "tags":      meta.get("tags", ""),
                    "url":       meta.get("url", ""),
                    "chunk_idx": idx,
                })

                # Upsert when batch is full
                if len(batch_ids) >= BATCH_SIZE:
                    collection.upsert(
                        ids=batch_ids,
                        documents=batch_docs,
                        embeddings=batch_embeddings,
                        metadatas=batch_metas,
                    )
                    total_ingested += len(batch_ids)
                    batch_ids, batch_docs, batch_embeddings, batch_metas = [], [], [], []
                    await asyncio.sleep(0.3)

            # Flush remaining
            if batch_ids:
                collection.upsert(
                    ids=batch_ids,
                    documents=batch_docs,
                    embeddings=batch_embeddings,
                    metadatas=batch_metas,
                )
                total_ingested += len(batch_ids)

            total_pages += 1
            logger.info(
                "  ✅ '%s' → %d chunks ingested", title, len(chunks) - total_skipped
            )
            await asyncio.sleep(0.2)  # be polite to Notion API

    total_in_db = collection.count()
    logger.info(
        "Ingest complete — pages: %d | ingested: %d | skipped: %d | total in DB: %d",
        total_pages, total_ingested, total_skipped, total_in_db,
    )

    return {
        "pages":           total_pages,
        "chunks_ingested": total_ingested,
        "chunks_skipped":  total_skipped,
        "total":           total_in_db,
    }


if __name__ == "__main__":
    asyncio.run(ingest_notion(force=True))
