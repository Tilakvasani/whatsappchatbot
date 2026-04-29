"""
llm.py — OpenAI LLM client factory
====================================
Provides a singleton OpenAI async client for use across the bot.
Uses the same singleton pattern as docForge's llm.py.
"""

from openai import AsyncOpenAI
from core.config import settings

_client: AsyncOpenAI | None = None


def get_openai_client() -> AsyncOpenAI:
    """Return the shared AsyncOpenAI singleton."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


async def chat_completion(messages: list[dict], temperature: float = 0.2,
                          max_tokens: int = 800) -> str:
    """
    Call OpenAI chat completion and return the response text.

    Args:
        messages:    List of {role, content} dicts (OpenAI format).
        temperature: Sampling temperature (lower = more deterministic).
        max_tokens:  Max tokens in response (kept low for WhatsApp-sized answers).

    Returns:
        The assistant's response as a plain string.
    """
    client = get_openai_client()
    response = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


async def get_embedding(text: str) -> list[float]:
    """
    Get an embedding vector for the given text using OpenAI.
    Same approach as docForge's AzureOpenAIEmbeddings but with standard OpenAI.

    Args:
        text: The text to embed.

    Returns:
        A list of floats representing the embedding vector.
    """
    client = get_openai_client()
    response = await client.embeddings.create(
        model=settings.OPENAI_EMBED_MODEL,
        input=text.replace("\n", " "),
    )
    return response.data[0].embedding
