"""
scripts/ingest_notion.py
=========================
Run this from the project root to sync Notion docs → ChromaDB.

Usage:
    python scripts/ingest_notion.py          # delta sync (new/changed only)
    python scripts/ingest_notion.py --force  # re-embed everything
"""
import sys, asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.ingest import ingest_notion
from core.logger import logger


async def main():
    force = "--force" in sys.argv or "-f" in sys.argv
    print(f"\n🔄 {'Full re-ingest' if force else 'Delta sync'} from Notion...\n")
    try:
        result = await ingest_notion(force=force)
        print("=" * 50)
        print("✅  Ingest complete!")
        print(f"   Pages processed : {result['pages']}")
        print(f"   Chunks ingested : {result['chunks_ingested']}")
        print(f"   Chunks skipped  : {result['chunks_skipped']}")
        print(f"   Total in DB     : {result['total']}")
        print("=" * 50)
        print("\n🚀 Bot is ready!\n")
    except Exception as e:
        print(f"\n❌ Ingest failed: {e}")
        print("Check: NOTION_TOKEN + NOTION_DATABASE_ID in .env")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
