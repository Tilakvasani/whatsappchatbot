"""
retriever.py — Vector retrieval from ChromaDB
==============================================
Pure retrieval layer — no LLM calls here.
The LangGraph agent calls this to get relevant context chunks.
"""
from core.llm import get_embedding
from core.logger import logger
from core.vector import get_collection

MIN_SCORE = 0.22   # minimum cosine similarity
TOP_K = 6          # chunks to retrieve
MAX_CONTEXT_CHARS = 4000


async def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Search ChromaDB for chunks relevant to the query.

    Returns:
        List of chunk dicts sorted by score desc:
        [{"score", "title", "category", "text", "url", "notion_id"}, ...]
    """
    collection = get_collection()
    count = collection.count()

    if count == 0:
        logger.warning("ChromaDB is empty — run ingest first!")
        return []

    query_emb = await get_embedding(query)

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
    ):
        score = round(1 - dist / 2, 4)
        if score < MIN_SCORE:
            continue
        chunks.append({
            "score":     score,
            "title":     meta.get("title", ""),
            "category":  meta.get("category", ""),
            "tags":      meta.get("tags", ""),
            "url":       meta.get("url", ""),
            "notion_id": meta.get("notion_id", ""),
            "text":      doc,
        })

    chunks.sort(key=lambda x: x["score"], reverse=True)

    if chunks:
        logger.info(
            "Retrieved %d chunks | query='%s...' | top_score=%.3f",
            len(chunks), query[:40], chunks[0]["score"]
        )

    return chunks


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block for the LLM."""
    parts = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c['title']}"
        if c.get("category"):
            header += f" — {c['category']}"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n".join(parts)[:MAX_CONTEXT_CHARS]


def confidence_level(chunks: list[dict]) -> str:
    """Score retrieval confidence from top chunk similarity."""
    if not chunks:
        return "none"
    top = chunks[0]["score"]
    if top >= 0.60:
        return "high"
    if top >= 0.38:
        return "medium"
    return "low"
