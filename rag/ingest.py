"""
ingest.py — FAQ Ingestion: JSON → Embeddings → ChromaDB
=========================================================
Adapted from docForge's ingest_service.py.

Pipeline:
  1. Load FAQs from data/zupwell_faqs.json
  2. Generate embeddings via OpenAI
  3. Upsert into ChromaDB with metadata

Idempotent: uses FAQ id as ChromaDB document ID, safe to re-run.
"""

import asyncio
import json
import hashlib
from pathlib import Path

from core.config import settings
from core.llm import get_embedding
from core.logger import logger
from core.vector import get_collection

FAQ_FILE = Path(__file__).parent.parent / "data" / "zupwell_faqs.json"
BATCH_SIZE = 10  # embed this many FAQs at once


async def ingest_faqs(force: bool = False) -> dict:
    """
    Load FAQs from JSON, embed them, and upsert into ChromaDB.

    Args:
        force: If True, re-ingest all FAQs even if already present.

    Returns:
        {"ingested": int, "skipped": int, "total": int}
    """
    collection = get_collection()

    # Load FAQ data
    if not FAQ_FILE.exists():
        raise FileNotFoundError(f"FAQ file not found: {FAQ_FILE}")

    with open(FAQ_FILE, "r", encoding="utf-8") as f:
        faqs = json.load(f)

    logger.info("Loaded %d FAQs from %s", len(faqs), FAQ_FILE)

    # Check what's already in the collection (for delta sync like docForge)
    existing_ids: set[str] = set()
    if not force:
        try:
            existing = collection.get(include=[])
            existing_ids = set(existing.get("ids", []))
        except Exception:
            pass

    ingested = 0
    skipped  = 0

    # Process in batches
    for i in range(0, len(faqs), BATCH_SIZE):
        batch = faqs[i : i + BATCH_SIZE]
        batch_ids        = []
        batch_documents  = []
        batch_embeddings = []
        batch_metadatas  = []

        for faq in batch:
            faq_id = faq["id"]

            # Build the full text to embed (title + content gives better retrieval)
            text_to_embed = f"{faq['title']}\n{faq['content']}"
            doc_id = faq_id  # use faq id as chroma doc id (idempotent)

            if doc_id in existing_ids and not force:
                skipped += 1
                continue

            # Generate embedding
            embedding = await get_embedding(text_to_embed)

            batch_ids.append(doc_id)
            batch_documents.append(text_to_embed)
            batch_embeddings.append(embedding)
            batch_metadatas.append({
                "faq_id":   faq_id,
                "category": faq.get("category", ""),
                "title":    faq.get("title", ""),
            })

        if batch_ids:
            collection.upsert(
                ids=batch_ids,
                documents=batch_documents,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
            )
            ingested += len(batch_ids)
            logger.info(
                "Ingested batch %d-%d (%d docs)",
                i + 1, i + len(batch), len(batch_ids)
            )

        # Small delay to avoid rate limiting
        if i + BATCH_SIZE < len(faqs):
            await asyncio.sleep(0.5)

    total = collection.count()
    logger.info(
        "Ingest complete — ingested: %d, skipped: %d, total in DB: %d",
        ingested, skipped, total
    )

    return {"ingested": ingested, "skipped": skipped, "total": total}


if __name__ == "__main__":
    asyncio.run(ingest_faqs(force=True))
