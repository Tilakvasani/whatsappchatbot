"""
vector.py — ChromaDB client factory
"""
from typing import Optional
import chromadb
from core.config import settings

COLLECTION_NAME = "zupwell_notion_docs"

_client_instance: Optional[chromadb.Client] = None
_collection_instance = None


def get_chroma_client() -> chromadb.Client:
    global _client_instance
    if _client_instance is None:
        _client_instance = chromadb.PersistentClient(path=settings.CHROMA_PATH)
    return _client_instance


def get_collection():
    global _collection_instance
    if _collection_instance is None:
        client = get_chroma_client()
        _collection_instance = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection_instance
