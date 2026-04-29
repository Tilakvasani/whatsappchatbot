"""
llm.py — Azure OpenAI LLM + Embedding client factory
=====================================================
Uses two separate AsyncAzureOpenAI clients:
  - LLM client  → AZURE_LLM_ENDPOINT  + AZURE_OPENAI_LLM_KEY
  - EMB client  → AZURE_EMB_ENDPOINT  + AZURE_OPENAI_EMB_KEY
"""

from typing import Optional
from openai import AsyncAzureOpenAI
from core.config import settings

# ── Singletons ────────────────────────────────────────────────────────────────
_llm_client: Optional[AsyncAzureOpenAI] = None
_emb_client: Optional[AsyncAzureOpenAI] = None


def get_llm_client() -> AsyncAzureOpenAI:
    """Return the shared Azure OpenAI LLM client singleton."""
    global _llm_client
    if _llm_client is None:
        _llm_client = AsyncAzureOpenAI(
            api_key=settings.AZURE_OPENAI_LLM_KEY,
            azure_endpoint=settings.AZURE_LLM_ENDPOINT,
            api_version=settings.AZURE_LLM_API_VERSION,
        )
    return _llm_client


def get_emb_client() -> AsyncAzureOpenAI:
    """Return the shared Azure OpenAI Embeddings client singleton."""
    global _emb_client
    if _emb_client is None:
        _emb_client = AsyncAzureOpenAI(
            api_key=settings.AZURE_OPENAI_EMB_KEY,
            azure_endpoint=settings.AZURE_EMB_ENDPOINT,
            api_version=settings.AZURE_EMB_API_VERSION,
        )
    return _emb_client


async def chat_completion(messages: list, temperature: float = 0.2,
                          max_tokens: int = 800) -> str:
    """
    Call Azure OpenAI chat completion and return the response text.
    """
    client = get_llm_client()
    response = await client.chat.completions.create(
        model=settings.AZURE_LLM_DEPLOYMENT_41_MINI,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


async def get_embedding(text: str) -> list:
    """
    Get an embedding vector for the given text using Azure OpenAI.
    """
    client = get_emb_client()
    response = await client.embeddings.create(
        model=settings.AZURE_EMB_DEPLOYMENT,
        input=text.replace("\n", " "),
    )
    return response.data[0].embedding