"""
runner.py — Unified Agent Runner
=================================
Single function `run_agent()` used by both WhatsApp webhook and website API.
Manages LangGraph state init and history injection from Redis.
"""

import json
from typing import Optional
# ✅ Correct
from langchain_core.messages import HumanMessage, AIMessage

from agent.graph import get_graph, AgentState
from core.logger import logger
from whatsapp.session import get_session_messages, save_turn


async def run_agent(
    user_input: str,
    session_id: str,
    source: str = "website",   # "website" | "whatsapp"
    user_id: Optional[int] = None,
    user_name: Optional[str] = None,
    user_email: Optional[str] = None,
    user_phone: Optional[str] = None,
    # carry over stateful ticket flow from previous turn
    ticket_stage: str = "idle",
    ticket_draft: Optional[dict] = None,
) -> dict:
    """
    Run the LangGraph agent for one user turn.

    Returns:
        {
          "response": str,          # message to send back to user
          "intent": str,            # what the agent classified this as
          "confidence": str,        # rag confidence level
          "ticket_created": dict,   # ticket data if one was created, else None
          "ticket_stage": str,      # updated ticket stage (pass back next turn)
          "ticket_draft": dict,     # updated draft (pass back next turn)
        }
    """
    # Load message history from Redis
    raw_messages = await get_session_messages(session_id)
    messages = []
    for m in raw_messages[-(12):]  :   # last 6 turns
        if m.get("role") == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))

    # Build initial state
    initial_state: AgentState = {
        "session_id":      session_id,
        "source":          source,
        "user_input":      user_input,
        "response":        "",
        "user_id":         user_id,
        "user_name":       user_name,
        "user_email":      user_email,
        "user_phone":      user_phone,
        "messages":        messages,
        "retrieved_chunks": [],
        "context":         "",
        "confidence":      "none",
        "intent":          "",
        "ticket_stage":    ticket_stage,
        "ticket_draft":    ticket_draft or {},
        "ticket_result":   {},
        "next_node":       "",
    }

    graph = get_graph()

    try:
        final_state = await graph.ainvoke(initial_state)
    except Exception as e:
        logger.error("Graph execution failed for session %s: %s", session_id[-6:], e)
        final_state = {
            **initial_state,
            "response": (
                "Sorry, something went wrong! 😅\n"
                "Please contact us at info@zupwell.com or +91 6355466208"
            ),
            "intent": "error",
        }

    response     = final_state.get("response", "")
    ticket_result = final_state.get("ticket_result") or None

    # Save turn to Redis
    await save_turn(session_id, user_input, response)

    logger.info(
        "Agent done | session=%s | intent=%s | confidence=%s | ticket=%s",
        session_id[-6:],
        final_state.get("intent", "?"),
        final_state.get("confidence", "?"),
        ticket_result.get("ticket_ref") if ticket_result else "none",
    )

    return {
        "response":       response,
        "intent":         final_state.get("intent", "faq"),
        "confidence":     final_state.get("confidence", "none"),
        "ticket_created": ticket_result,
        "ticket_stage":   final_state.get("ticket_stage", "idle"),
        "ticket_draft":   final_state.get("ticket_draft", {}),
    }
