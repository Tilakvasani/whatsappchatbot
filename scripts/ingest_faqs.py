"""
scripts/ingest_faqs.py
========================
Run this script once to populate ChromaDB with Zupwell FAQ embeddings.
Must be run from the project root directory.

Usage:
    python scripts/ingest_faqs.py          # delta sync (only new FAQs)
    python scripts/ingest_faqs.py --force  # re-embed all FAQs

This is equivalent to docForge's POST /api/rag/ingest endpoint,
but as a standalone CLI script.
"""

import sys
import asyncio
from pathlib import Path

# Ensure the project root is in the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.ingest import ingest_faqs
from core.logger import logger


async def main():
    force = "--force" in sys.argv or "-f" in sys.argv

    if force:
        logger.info("🔄 Running FULL re-ingest (force mode)...")
    else:
        logger.info("🔄 Running delta sync ingest...")

    try:
        result = await ingest_faqs(force=force)
        print("\n" + "=" * 50)
        print("✅ Ingest Complete!")
        print(f"   Ingested : {result['ingested']} FAQ chunks")
        print(f"   Skipped  : {result['skipped']} (already in DB)")
        print(f"   Total    : {result['total']} chunks in ChromaDB")
        print("=" * 50)
        print("\n🚀 Your Zupwell WhatsApp bot is ready to answer questions!\n")
    except Exception as e:
        print(f"\n❌ Ingest failed: {e}")
        print("\nMake sure:")
        print("  1. Your .env file has OPENAI_API_KEY set")
        print("  2. data/zupwell_faqs.json exists")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
