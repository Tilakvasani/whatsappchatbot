"""
runner.py — Unified Agent Runner
=================================
Single function used by both WhatsApp webhook and website API.
"""
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage

from agent.graph import get_graph, AgentState
from core.logger import logger
from whatsapp.session import get_session_messages, save_turn


async def run_agent(
    user_input:   str,
    session_id:   str,
    source:       str = "website",
    user_id:      Optional[int]  = None,
    user_name:    Optional[str]  = None,
    user_email:   Optional[str]  = None,
    user_phone:   Optional[str]  = None,
    photo_urls:   Optional[list] = None,
    ticket_stage: str  = "idle",
    ticket_draft: Optional[dict] = None,
) -> dict:
    """
    Run one agent turn. Returns:
    {
      "response":       str,
      "intent":         str,
      "confidence":     str,
      "ticket_created": dict | None,
      "ticket_stage":   str,
      "ticket_draft":   dict,
    }
    """
    # Load history from Redis
    raw_msgs = await get_session_messages(session_id)
    messages = []
    for m in raw_msgs[-16:]:
        if m.get("role") == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))

    initial: AgentState = {
        "session_id":       session_id,
        "source":           source,
        "user_input":       user_input,
        "response":         "",
        "photo_urls":       photo_urls or [],
        "user_id":          user_id,
        "user_name":        user_name,
        "user_email":       user_email,
        "user_phone":       user_phone,
        "messages":         messages,
        "retrieved_chunks": [],
        "context":          "",
        "confidence":       "none",
        "intent":           "",
        "next_node":        "",
        "ticket_stage":     ticket_stage,
        "ticket_draft":     ticket_draft or {},
        "ticket_result":    {},
    }

    try:
        final = await get_graph().ainvoke(initial)
    except Exception as e:
        logger.error("Graph error for %s: %s", session_id[-6:], e)
        final = {
            **initial,
            "response": (
                "Sorry, something went wrong! 😅\n"
                "Please contact us at info@zupwell.com or +91 6355466208"
            ),
            "intent": "error",
        }

    response      = final.get("response", "")
    ticket_result = final.get("ticket_result") or None

    await save_turn(session_id, user_input, response)

    return {
        "response":       response,
        "intent":         final.get("intent", "answer"),
        "confidence":     final.get("confidence", "none"),
        "ticket_created": ticket_result,
        "ticket_stage":   final.get("ticket_stage", "idle"),
        "ticket_draft":   final.get("ticket_draft", {}),
    }
