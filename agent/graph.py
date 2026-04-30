"""
graph.py — LangGraph Agentic Workflow — Zupwell Bot v3
=======================================================

KEY DESIGN: One Master LLM call per turn (fast, no double roundtrip).
The LLM is given the full system prompt + context + history and returns
a structured JSON decision that drives routing.

Graph:
  START
    │
    ▼
  [master_agent]   ← One LLM call. Reads context, decides everything.
    │
    ├── intent=answer    ──► END              (FAQ answered)
    ├── intent=greeting  ──► [handle_greeting] → END
    ├── intent=ticket    ──► [collect_ticket_info]
    │                              │
    │                         complete? ──► [create_ticket] → END
    │                         need more? ──► END (await next msg)
    │
    └── intent=escalate  ──► [escalate_node] → END

Photo flow (WhatsApp):
  Twilio sends MediaUrl0 in webhook form → stored on ticket as photo_urls array
"""

import json
import re
from typing import Annotated, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from core.llm import chat_completion
from core.logger import logger
from rag.retriever import retrieve, build_context, confidence_level
from tickets.tickets import create_ticket


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    session_id:   str
    source:       str          # "whatsapp" | "website"
    user_input:   str
    response:     str
    photo_urls:   list         # images attached to this turn (damaged product etc.)

    # User identity
    user_id:    Optional[int]
    user_name:  Optional[str]
    user_email: Optional[str]
    user_phone: Optional[str]

    # Conversation memory
    messages: Annotated[list, add_messages]

    # RAG
    retrieved_chunks: list
    context:          str
    confidence:       str

    # Routing
    intent:       str    # answer | greeting | ticket | escalate
    next_node:    str

    # Ticket multi-turn state
    ticket_stage: str    # idle | collecting | created
    ticket_draft: dict
    ticket_result: dict


# ─────────────────────────────────────────────────────────────────────────────
# MASTER SYSTEM PROMPT
# One prompt, does everything. The LLM reads this + context + history
# and returns structured JSON that drives the entire workflow.
# ─────────────────────────────────────────────────────────────────────────────

MASTER_SYSTEM_PROMPT = """\
You are Zara, Zupwell's intelligent customer support AI.
Zupwell is a premium health supplements brand based in Ahmedabad, Gujarat, India.
Founded by Parag Hirpara, Zupwell creates science-backed, delicious health supplements
for the modern, fast-paced lifestyle.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR JOB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Read the customer's message + the knowledge base context provided.
Then return a single JSON object (no markdown, no explanation, just JSON).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT — always return this exact JSON shape:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "intent": "<answer|greeting|ticket|escalate>",
  "answer": "<your reply to the customer>",
  "ticket_fields": {
    "subject": null,
    "description": null,
    "name": null,
    "email": null,
    "phone": null
  },
  "needs_more_info": false,
  "next_question": null
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTENT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• answer    → Customer asked a question and context has enough info to answer
• greeting  → Hi/hello/hey/help/menu/start/namaste or reset commands
• ticket    → Customer reports a problem: damaged product, wrong order, not received,
              refund request, complaint, or any issue that needs team follow-up
• escalate  → Customer is very angry, mentions legal action, demands human NOW

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANSWERING RULES (intent=answer)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Answer ONLY from the knowledge base context. Never invent facts.
2. If context partially answers: give what you know, say what you don't.
3. If context has NO relevant info: set intent=answer, answer with:
   "I don't have that detail right now. Please contact us at
    📧 info@zupwell.com | 📱 +91 6355466208 — our team will help! 😊"
4. Be warm, friendly, conversational — not robotic.
5. WhatsApp source: plain text, bullet points with •, max 250 words.
6. Website source: same but can be slightly more detailed.
7. End most answers with "Is there anything else I can help you with? 😊"
   unless the conversation is clearly wrapping up.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TICKET RULES (intent=ticket)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Extract ticket fields from the customer's message into ticket_fields.
Put null for anything not mentioned.

If you have enough to create the ticket (description + any contact info):
  • needs_more_info = false
  • Fill ticket_fields as completely as possible
  • answer = warm confirmation that you're creating the ticket

If critical info is missing (need description or any contact):
  • needs_more_info = true
  • next_question = the ONE most important missing piece to ask for
  • answer = your question to the customer (friendly, warm)

For damaged product tickets: let the customer know they can share a photo
by saying "If you have a photo of the damaged product, please share it —
it will help our team process your request faster."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ESCALATION RULES (intent=escalate)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Be empathetic and sincere, never defensive.
• Apologize genuinely for their experience.
• Promise a senior team member will reach out within 2 hours.
• Give direct contacts: info@zupwell.com | +91 6355466208

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Never reveal you are using a knowledge base, vector search, or AI system.
• Never mention "ChromaDB", "embeddings", "LangGraph", or any tech internals.
• You are Zara from Zupwell — that's all the customer needs to know.
• Always use the customer's name if you know it.
• For GST, legal, regulatory questions: answer from context if available,
  otherwise refer to info@zupwell.com
"""

MASTER_USER_TEMPLATE = """\
━━ KNOWLEDGE BASE CONTEXT ━━
{context}

━━ CONVERSATION HISTORY ━━
{history}

━━ CURRENT TURN ━━
Customer name: {name}
Source: {source}
Customer message: {message}
{photo_note}
━━ TICKET IN PROGRESS (if any) ━━
{ticket_draft}

Return ONLY the JSON object. No markdown. No explanation."""

GREETING_RESPONSES = {
    "default": """\
👋 *Welcome to Zupwell Support!*

Hi {name}! I'm Zara, your Zupwell AI assistant. I'm here to help you with:

• 🧴 Products, ingredients & benefits
• 💰 Pricing & availability  
• 📦 Orders, payments & tracking
• 🚚 Shipping & delivery
• 🔄 Returns & refunds
• 🎫 Raising a support ticket

Just ask me anything — I'm happy to help! 😊""",

    "help": """\
🤖 *Here's what I can help with:*

*Products & Info*
• "What electrolytes does Zupwell offer?"
• "What are the ingredients?"
• "Is it FSSAI approved?"

*Orders & Delivery*
• "How long does delivery take?"
• "How do I track my order?"
• "Can I cancel my order?"

*Issues & Support*
• "I received a damaged product"
• "My order hasn't arrived"
• "I want a refund"

Just type your question! 😊
📧 info@zupwell.com | 📱 +91 6355466208""",

    "reset": "✅ Chat cleared! How can I help you today? 😊"
}


# ── Node: Master Agent ────────────────────────────────────────────────────────

async def master_agent(state: AgentState) -> AgentState:
    """
    Single LLM call that does everything:
    retrieves context, classifies intent, generates answer or ticket action.
    """
    # ── 1. Retrieve relevant context ──────────────────────────────────────────
    chunks = await retrieve(state["user_input"])
    context = build_context(chunks) if chunks else "No relevant knowledge base content found."
    conf = confidence_level(chunks)

    # ── 2. Build history string ───────────────────────────────────────────────
    history = _format_history(state.get("messages", []), n=8)

    # ── 3. Photo note ─────────────────────────────────────────────────────────
    photos = state.get("photo_urls") or []
    photo_note = f"Customer shared {len(photos)} photo(s) with this message.\n" if photos else ""

    # ── 4. Ticket draft context ───────────────────────────────────────────────
    draft = state.get("ticket_draft") or {}
    ticket_draft_str = json.dumps(draft, ensure_ascii=False) if draft else "None"

    # ── 5. Build user message ─────────────────────────────────────────────────
    user_content = MASTER_USER_TEMPLATE.format(
        context=context,
        history=history or "No previous conversation.",
        name=state.get("user_name") or "Customer",
        source=state.get("source", "website"),
        message=state["user_input"],
        photo_note=photo_note,
        ticket_draft=ticket_draft_str,
    )

    # ── 6. Call LLM ───────────────────────────────────────────────────────────
    raw = await chat_completion(
        messages=[
            {"role": "system", "content": MASTER_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.25,
        max_tokens=600,
    )

    # ── 7. Parse JSON response ────────────────────────────────────────────────
    parsed = _safe_parse_json(raw)

    intent         = parsed.get("intent", "answer")
    answer         = parsed.get("answer", "")
    ticket_fields  = parsed.get("ticket_fields") or {}
    needs_more     = parsed.get("needs_more_info", False)

    logger.info(
        "MasterAgent | intent=%s | confidence=%s | needs_more=%s | session=%s",
        intent, conf, needs_more, state["session_id"][-6:]
    )

    # ── 8. Handle greeting fast-path ──────────────────────────────────────────
    msg = state["user_input"].strip().lower()
    if msg in {"hi","hello","hey","hii","start","kem cho","namaste","helo"}:
        intent = "greeting"
    elif msg in {"help","menu","?","options"}:
        intent = "greeting"
        answer = "help"
    elif msg in {"reset","clear","restart"}:
        intent = "greeting"
        answer = "reset"

    # ── 9. Merge ticket fields into draft ─────────────────────────────────────
    new_draft = dict(draft)
    for field, val in ticket_fields.items():
        if val and not new_draft.get(field):
            new_draft[field] = val
    # Pre-fill from auth session
    if not new_draft.get("name")  and state.get("user_name"):
        new_draft["name"]  = state["user_name"]
    if not new_draft.get("email") and state.get("user_email"):
        new_draft["email"] = state["user_email"]
    if not new_draft.get("phone") and state.get("user_phone"):
        new_draft["phone"] = state["user_phone"]
    # Attach photos to draft
    if photos:
        existing_photos = new_draft.get("photo_urls") or []
        new_draft["photo_urls"] = existing_photos + photos

    # ── 10. Determine next node ───────────────────────────────────────────────
    if intent == "greeting":
        next_node = "handle_greeting"
    elif intent == "ticket":
        if needs_more:
            next_node = "end"   # ask question, wait for next message
        else:
            next_node = "create_ticket"
    elif intent == "escalate":
        next_node = "escalate_node"
    else:
        next_node = "end"

    ticket_stage = "collecting" if (intent == "ticket" and needs_more) else state.get("ticket_stage", "idle")

    return {
        **state,
        "retrieved_chunks": chunks,
        "context":          context,
        "confidence":       conf,
        "intent":           intent,
        "response":         answer,
        "next_node":        next_node,
        "ticket_draft":     new_draft,
        "ticket_stage":     ticket_stage,
    }


# ── Node: Handle Greeting ─────────────────────────────────────────────────────

async def handle_greeting(state: AgentState) -> AgentState:
    name = state.get("user_name") or "there"
    cmd  = state.get("response", "").strip().lower()

    if cmd in {"help", "menu"}:
        response = GREETING_RESPONSES["help"]
    elif cmd in {"reset", "clear"}:
        response = GREETING_RESPONSES["reset"]
    else:
        response = GREETING_RESPONSES["default"].format(name=name)

    return {**state, "response": response}


# ── Node: Create Ticket ───────────────────────────────────────────────────────

async def create_ticket_node(state: AgentState) -> AgentState:
    """Write ticket to PostgreSQL with photo_urls if present."""
    draft = state.get("ticket_draft") or {}

    # Auto-generate subject if missing
    if not draft.get("subject") and draft.get("description"):
        draft["subject"] = draft["description"][:80].strip() + (
            "..." if len(draft["description"]) > 80 else ""
        )

    try:
        result = await create_ticket(
            subject=draft.get("subject", "Customer support request"),
            description=draft.get("description", state["user_input"]),
            source=state.get("source", "website"),
            name=draft.get("name"),
            email=draft.get("email"),
            phone=draft.get("phone") or state.get("user_phone"),
            user_id=state.get("user_id"),
            photo_urls=draft.get("photo_urls") or [],
        )

        ref = result["ticket_ref"]
        cat = result.get("category", "general")

        has_photos = bool(draft.get("photo_urls"))
        photo_line = "\n📷 Your photo has been attached to the ticket." if has_photos else ""

        response = (
            f"✅ *Support ticket raised successfully!*\n\n"
            f"🎫 Ticket ID: *{ref}*\n"
            f"📋 Category: {cat.title()}\n"
            f"⏱️ Expected response: within 24 hours{photo_line}\n\n"
            f"Our team will reach out to you soon.\n"
            f"For urgent help: 📧 info@zupwell.com | 📱 +91 6355466208\n\n"
            f"Is there anything else I can help you with? 😊"
        )

        logger.info("Ticket created: %s | photos=%d", ref, len(draft.get("photo_urls") or []))

        return {
            **state,
            "response":      response,
            "ticket_result": result,
            "ticket_stage":  "created",
            "ticket_draft":  {},
        }

    except Exception as e:
        logger.error("Ticket creation failed: %s", e)
        return {
            **state,
            "response": (
                "I'm sorry, I couldn't raise the ticket right now. 😔\n\n"
                "Please contact us directly:\n"
                "📧 info@zupwell.com\n"
                "📱 +91 6355466208\n\n"
                "Our team will help you right away!"
            ),
            "ticket_stage": "idle",
        }


# ── Node: Escalate ────────────────────────────────────────────────────────────

async def escalate_node(state: AgentState) -> AgentState:
    """Human handoff — the master agent already wrote the answer."""
    return state  # response already set by master_agent


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_master(state: AgentState) -> str:
    node = state.get("next_node", "end")
    return node if node != "end" else END


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("master_agent",    master_agent)
    g.add_node("handle_greeting", handle_greeting)
    g.add_node("create_ticket",   create_ticket_node)
    g.add_node("escalate_node",   escalate_node)

    g.set_entry_point("master_agent")

    g.add_conditional_edges("master_agent", route_after_master, {
        "handle_greeting": "handle_greeting",
        "create_ticket":   "create_ticket",
        "escalate_node":   "escalate_node",
        END:               END,
    })

    g.add_edge("handle_greeting", END)
    g.add_edge("create_ticket",   END)
    g.add_edge("escalate_node",   END)

    return g.compile()


_compiled_graph = None

def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_history(messages: list, n: int = 8) -> str:
    recent = messages[-(n * 2):]
    lines = []
    for m in recent:
        role    = "Customer" if getattr(m, "type", "") == "human" else "Zara"
        content = str(getattr(m, "content", m.get("content", "") if isinstance(m, dict) else ""))[:400]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _safe_parse_json(raw: str) -> dict:
    """Parse LLM JSON output safely — strip markdown fences if present."""
    try:
        text = raw.strip()
        # Strip ```json ... ``` fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)
    except Exception:
        logger.warning("Failed to parse LLM JSON: %s", raw[:200])
        # Fallback: treat whole raw as the answer
        return {
            "intent": "answer",
            "answer": raw.strip(),
            "ticket_fields": {},
            "needs_more_info": False,
            "next_question": None,
        }
