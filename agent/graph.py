"""
graph.py — LangGraph Agentic Workflow for Zupwell Bot
======================================================

Graph nodes (each is an async function that takes/returns AgentState):

  START
    │
    ▼
  [classify_intent]          Decide: greeting / faq / ticket / escalate
    │
    ├──► [handle_greeting]   Send welcome / help message
    │
    ├──► [retrieve_docs]     Vector search ChromaDB
    │         │
    │         ▼
    │    [generate_answer]   LLM generates answer from retrieved docs
    │         │
    │         ▼
    │    [check_confidence]  High/medium → END | Low → ask to raise ticket?
    │         │
    │         ├── high/medium ──► END
    │         └── low ──────────► [offer_ticket]
    │
    ├──► [collect_ticket_info]  Multi-turn: gather name/email/issue
    │         │
    │         ▼
    │    [create_ticket]      Write to PostgreSQL, confirm to user
    │
    └──► [escalate]          Immediate human handoff message

All state flows via TypedDict AgentState — clean, inspectable, easy to extend.
"""

import json
from typing import Annotated, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from core.config import settings
from core.llm import chat_completion
from core.logger import logger
from rag.retriever import retrieve, build_context, confidence_level
from tickets.tickets import create_ticket


# ── State definition ──────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Core
    session_id:     str                    # phone number or website session id
    source:         str                    # "whatsapp" | "website"
    user_input:     str                    # latest user message
    response:       str                    # final bot response to send

    # User identity (from JWT for website, collected for WhatsApp)
    user_id:        Optional[int]
    user_name:      Optional[str]
    user_email:     Optional[str]
    user_phone:     Optional[str]

    # Conversation history for context
    messages:       Annotated[list, add_messages]

    # RAG
    retrieved_chunks:  list
    context:           str
    confidence:        str

    # Intent + routing
    intent:         str   # greeting | faq | ticket | escalate | collect_info
    ticket_stage:   str   # idle | collecting | confirming | created
    ticket_draft:   dict  # accumulated ticket fields before creation
    ticket_result:  dict  # created ticket data

    # Flow control
    next_node:      str


# ── System prompts ────────────────────────────────────────────────────────────

CLASSIFY_PROMPT = """\
You are an intent classifier for Zupwell's customer support bot.
Zupwell is a health supplements brand from India.

Classify the user's message into EXACTLY ONE of these intents:

- greeting    : hi, hello, hey, start, help, menu, namaste, reset
- faq         : any question about products, pricing, ingredients, orders, shipping, returns, policies
- ticket      : wants to report an issue, complaint, damaged product, refund request, wrong order, not received
- escalate    : very angry, abusive, mentions legal action, demands to talk to human immediately
- collect_info: user is providing their name/email/phone/issue details (follow-up in ticket collection flow)

User message: "{message}"
Conversation context (last 3 turns): {history}

Reply with ONLY the intent word, nothing else."""

ANSWER_PROMPT = """\
You are Zupwell's friendly AI support assistant.
Zupwell is a premium health supplements brand from Ahmedabad, India.

Customer info:
  Name: {name}
  Source: {source}

Relevant knowledge base content:
{context}

Conversation so far:
{history}

Customer's question: {question}

RULES:
1. Answer ONLY from the provided knowledge base content. Never make up facts.
2. Be warm, concise, and helpful — like a great customer support agent.
3. For WhatsApp: plain text, bullet points with •, keep under 250 words.
4. For website: you may use slightly richer formatting.
5. If the content partially answers: give what you know, acknowledge the gap.
6. End with "Is there anything else I can help you with? 😊" unless the conversation feels concluded.
7. Never reveal that you're using a knowledge base or vector search.

Answer:"""

NO_ANSWER_PROMPT = """\
You are Zupwell's friendly support assistant.
The knowledge base didn't have a clear answer for: "{question}"

Write a SHORT, warm message (2-3 sentences max) that:
1. Acknowledges you don't have that specific info right now.
2. Offers to raise a support ticket so the team can help personally.
3. Mentions: email info@zupwell.com | WhatsApp +91 6355466208

Source: {source}
Customer name: {name}"""

COLLECT_INFO_PROMPT = """\
You are collecting information to raise a support ticket for a Zupwell customer.

Already collected:
{collected}

Customer's latest message: "{message}"

Extract any of these from the message (leave null if not mentioned):
- name: customer's full name
- email: email address
- phone: phone number
- subject: one-line summary of the issue (max 100 chars)
- description: detailed description of the issue

Also determine: is the information complete enough to create the ticket?
Required fields: either (name OR email OR phone) AND description.

Reply with ONLY valid JSON:
{{
  "extracted": {{"name": null, "email": null, "phone": null, "subject": null, "description": null}},
  "is_complete": false,
  "next_question": "What specific issue are you experiencing?"
}}"""

TICKET_CONFIRM_PROMPT = """\
A support ticket is about to be raised for a Zupwell customer.

Ticket details:
  Name: {name}
  Email: {email}
  Phone: {phone}
  Subject: {subject}
  Description: {description}
  Source: {source}

Write a short, warm confirmation message (3-4 sentences) that:
1. Confirms the ticket has been raised with reference number {ticket_ref}.
2. Says the team will respond within 24 hours.
3. Gives contact details for urgent queries: info@zupwell.com | +91 6355466208"""

ESCALATE_PROMPT = """\
A Zupwell customer seems very upset or wants immediate human help.
Customer name: {name}
Their message: "{message}"

Write a short, empathetic message (3 sentences) that:
1. Sincerely apologizes for their experience.
2. Tells them a senior team member will reach out within 2 hours.
3. Gives direct contact: info@zupwell.com | +91 6355466208

Be warm, not robotic."""

GREETING_RESPONSE = """\
👋 *Welcome to Zupwell Support!*

Hi {name}! I'm your Zupwell AI assistant. I can help you with:

• 🧴 Products, ingredients & pricing
• 📦 Orders & payments
• 🚚 Shipping & delivery
• 🔄 Returns & refunds
• 🎫 Raising a support ticket

Just ask me anything — I'm here to help! 😊

_Type *help* to see example questions._"""

HELP_RESPONSE = """\
🤖 *Zupwell Support — What can I help with?*

Try asking:
• *"What are the ingredients in your electrolytes?"*
• *"How long does delivery take?"*
• *"What is your return policy?"*
• *"My order hasn't arrived — I need help"*
• *"I received a damaged product"*

For anything else, type your question and I'll do my best! 😊

📧 info@zupwell.com | 📱 +91 6355466208"""


# ── Node functions ────────────────────────────────────────────────────────────

async def classify_intent(state: AgentState) -> AgentState:
    """Classify the user's message intent using LLM."""
    msg = state["user_input"].strip().lower()

    # Fast-path: hardcoded command detection (saves an LLM call)
    greetings = {"hi", "hello", "hey", "hii", "start", "kem cho", "namaste"}
    help_cmds  = {"help", "menu", "?", "options"}
    reset_cmds = {"reset", "clear", "restart"}

    if msg in greetings or msg.startswith("hi ") or msg.startswith("hello "):
        return {**state, "intent": "greeting", "next_node": "handle_greeting"}
    if msg in help_cmds:
        return {**state, "intent": "greeting", "next_node": "handle_greeting", "user_input": "help"}
    if msg in reset_cmds:
        return {**state, "intent": "greeting", "next_node": "handle_greeting", "user_input": "reset"}

    # If we're mid-ticket collection, stay in that flow
    if state.get("ticket_stage") in ("collecting", "confirming"):
        return {**state, "intent": "collect_info", "next_node": "collect_ticket_info"}

    # LLM classification for everything else
    history = _format_history(state.get("messages", []), n=3)
    intent_raw = await chat_completion(
        messages=[{
            "role": "user",
            "content": CLASSIFY_PROMPT.format(
                message=state["user_input"],
                history=history or "None",
            )
        }],
        temperature=0.0,
        max_tokens=10,
    )

    intent = intent_raw.strip().lower()
    if intent not in ("greeting", "faq", "ticket", "escalate", "collect_info"):
        intent = "faq"   # default to FAQ if unexpected

    node_map = {
        "greeting":     "handle_greeting",
        "faq":          "retrieve_docs",
        "ticket":       "collect_ticket_info",
        "escalate":     "escalate",
        "collect_info": "collect_ticket_info",
    }

    logger.info("Intent: %s | msg='%s...'", intent, state["user_input"][:40])
    return {**state, "intent": intent, "next_node": node_map[intent]}


async def handle_greeting(state: AgentState) -> AgentState:
    """Return welcome or help message."""
    name = state.get("user_name") or "there"
    msg  = state["user_input"].strip().lower()

    if "help" in msg or "menu" in msg:
        response = HELP_RESPONSE
    elif "reset" in msg or "clear" in msg:
        response = f"✅ Chat reset! How can I help you, {name}? 😊"
    else:
        response = GREETING_RESPONSE.format(name=name)

    return {**state, "response": response}


async def retrieve_docs(state: AgentState) -> AgentState:
    """Retrieve relevant chunks from ChromaDB."""
    chunks = await retrieve(state["user_input"])
    context = build_context(chunks)
    conf = confidence_level(chunks)
    return {**state, "retrieved_chunks": chunks, "context": context, "confidence": conf}


async def generate_answer(state: AgentState) -> AgentState:
    """Generate an LLM answer using retrieved context."""
    history = _format_history(state.get("messages", []), n=6)
    name = state.get("user_name") or "Customer"

    answer = await chat_completion(
        messages=[{
            "role": "user",
            "content": ANSWER_PROMPT.format(
                name=name,
                source=state.get("source", "website"),
                context=state["context"],
                history=history or "None",
                question=state["user_input"],
            )
        }],
        temperature=0.3,
        max_tokens=450,
    )
    return {**state, "response": answer}


async def check_confidence(state: AgentState) -> AgentState:
    """If low confidence, offer ticket. Otherwise pass through."""
    if state["confidence"] == "none" or (
        state["confidence"] == "low" and not state.get("retrieved_chunks")
    ):
        name = state.get("user_name") or "there"
        fallback = await chat_completion(
            messages=[{
                "role": "user",
                "content": NO_ANSWER_PROMPT.format(
                    question=state["user_input"],
                    source=state.get("source", "website"),
                    name=name,
                )
            }],
            temperature=0.3,
            max_tokens=150,
        )
        return {**state, "response": fallback, "next_node": "offer_ticket"}

    return {**state, "next_node": "end"}


async def offer_ticket(state: AgentState) -> AgentState:
    """Append a ticket-offer prompt to the fallback response."""
    offer = (
        "\n\n🎫 Would you like me to raise a support ticket so our team can help you personally? "
        "Just say *'yes raise a ticket'* or *'create ticket'* and I'll get that sorted for you!"
    )
    # Check if user just said yes to the ticket offer
    user_msg = state["user_input"].lower()
    if any(k in user_msg for k in ["yes", "ticket", "raise", "create", "sure", "ok"]):
        return {**state, "intent": "ticket", "next_node": "collect_ticket_info"}

    return {**state, "response": state["response"] + offer}


async def collect_ticket_info(state: AgentState) -> AgentState:
    """
    Multi-turn node: collect name/email/issue details from the user.
    Runs until all required fields are gathered, then creates the ticket.
    """
    draft = dict(state.get("ticket_draft") or {})

    # Pre-fill from authenticated user session
    if not draft.get("name")  and state.get("user_name"):
        draft["name"] = state["user_name"]
    if not draft.get("email") and state.get("user_email"):
        draft["email"] = state["user_email"]
    if not draft.get("phone") and state.get("user_phone"):
        draft["phone"] = state["user_phone"]

    # If this is the very first ticket message, the user_input might BE the issue
    if not draft.get("description") and len(state["user_input"]) > 10:
        # Use LLM to extract structured info from the message
        collected_str = json.dumps({k: v for k, v in draft.items() if v}, ensure_ascii=False)
        extraction_raw = await chat_completion(
            messages=[{
                "role": "user",
                "content": COLLECT_INFO_PROMPT.format(
                    collected=collected_str or "Nothing yet",
                    message=state["user_input"],
                )
            }],
            temperature=0.0,
            max_tokens=300,
        )
        try:
            extraction = json.loads(extraction_raw.strip())
        except Exception:
            extraction = {"extracted": {}, "is_complete": False, "next_question": ""}

        # Merge extracted fields (don't overwrite existing)
        for field, val in (extraction.get("extracted") or {}).items():
            if val and not draft.get(field):
                draft[field] = val

        is_complete = extraction.get("is_complete", False)
        next_q = extraction.get("next_question", "")
    else:
        is_complete = False
        next_q = ""

    # Check if we have the minimum required: (name or email or phone) + description
    has_contact = any([draft.get("name"), draft.get("email"), draft.get("phone")])
    has_issue   = bool(draft.get("description") or draft.get("subject"))

    if has_contact and has_issue:
        # Ready to create! Route to create_ticket node
        if not draft.get("subject") and draft.get("description"):
            # Auto-generate subject from first 80 chars of description
            draft["subject"] = draft["description"][:80].strip() + (
                "..." if len(draft["description"]) > 80 else ""
            )
        return {
            **state,
            "ticket_draft": draft,
            "ticket_stage": "confirming",
            "next_node": "create_ticket",
        }

    # Need more info — build a collecting message
    name = draft.get("name") or state.get("user_name") or "there"

    if state.get("ticket_stage") != "collecting":
        # First time entering ticket flow
        opening = (
            f"I'll raise a support ticket for you, {name}! 🎫\n\n"
            "I just need a couple of details to get this to our team.\n\n"
        )
    else:
        opening = ""

    if next_q:
        ask = next_q
    elif not has_issue:
        ask = "Could you please describe the issue you're experiencing in detail?"
    elif not has_contact:
        ask = "What's the best email or phone number for our team to reach you on?"
    else:
        ask = "Is there anything else you'd like to add about the issue?"

    response = opening + ask

    return {
        **state,
        "ticket_draft": draft,
        "ticket_stage": "collecting",
        "response": response,
    }


async def create_ticket_node(state: AgentState) -> AgentState:
    """Create the ticket in PostgreSQL and send confirmation."""
    draft = state.get("ticket_draft", {})

    try:
        result = await create_ticket(
            subject=draft.get("subject", "Customer support request"),
            description=draft.get("description", state["user_input"]),
            source=state.get("source", "website"),
            name=draft.get("name"),
            email=draft.get("email"),
            phone=draft.get("phone") or state.get("user_phone"),
            user_id=state.get("user_id"),
        )

        confirm = await chat_completion(
            messages=[{
                "role": "user",
                "content": TICKET_CONFIRM_PROMPT.format(
                    name=draft.get("name") or "Customer",
                    email=draft.get("email") or "—",
                    phone=draft.get("phone") or state.get("user_phone") or "—",
                    subject=draft.get("subject", ""),
                    description=draft.get("description", ""),
                    ticket_ref=result["ticket_ref"],
                    source=state.get("source", "website"),
                )
            }],
            temperature=0.3,
            max_tokens=200,
        )

        return {
            **state,
            "response": confirm,
            "ticket_result": result,
            "ticket_stage": "created",
            "ticket_draft": {},   # clear draft
        }

    except Exception as e:
        logger.error("Ticket creation failed: %s", e)
        return {
            **state,
            "response": (
                "I'm sorry, I had trouble raising the ticket right now. 😔\n\n"
                "Please contact us directly:\n"
                "📧 info@zupwell.com\n"
                "📱 +91 6355466208\n\n"
                "Our team will help you right away!"
            ),
            "ticket_stage": "idle",
        }


async def escalate(state: AgentState) -> AgentState:
    """Send an empathetic human-handoff message."""
    name = state.get("user_name") or "there"
    response = await chat_completion(
        messages=[{
            "role": "user",
            "content": ESCALATE_PROMPT.format(
                name=name,
                message=state["user_input"],
            )
        }],
        temperature=0.4,
        max_tokens=150,
    )
    return {**state, "response": response}


# ── Routing functions ─────────────────────────────────────────────────────────

def route_after_classify(state: AgentState) -> str:
    return state.get("next_node", "retrieve_docs")


def route_after_confidence(state: AgentState) -> str:
    return state.get("next_node", "end")


def route_after_offer(state: AgentState) -> str:
    return state.get("next_node", END)


def route_after_collect(state: AgentState) -> str:
    return state.get("next_node", END)


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and compile the LangGraph workflow."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify_intent",      classify_intent)
    graph.add_node("handle_greeting",      handle_greeting)
    graph.add_node("retrieve_docs",        retrieve_docs)
    graph.add_node("generate_answer",      generate_answer)
    graph.add_node("check_confidence",     check_confidence)
    graph.add_node("offer_ticket",         offer_ticket)
    graph.add_node("collect_ticket_info",  collect_ticket_info)
    graph.add_node("create_ticket",        create_ticket_node)
    graph.add_node("escalate",             escalate)

    # Entry point
    graph.set_entry_point("classify_intent")

    # Edges
    graph.add_conditional_edges("classify_intent", route_after_classify, {
        "handle_greeting":     "handle_greeting",
        "retrieve_docs":       "retrieve_docs",
        "collect_ticket_info": "collect_ticket_info",
        "escalate":            "escalate",
    })

    graph.add_edge("handle_greeting",  END)
    graph.add_edge("retrieve_docs",    "generate_answer")
    graph.add_edge("generate_answer",  "check_confidence")

    graph.add_conditional_edges("check_confidence", route_after_confidence, {
        "offer_ticket": "offer_ticket",
        "end":          END,
    })

    graph.add_conditional_edges("offer_ticket", route_after_offer, {
        "collect_ticket_info": "collect_ticket_info",
        END:                   END,
    })

    graph.add_conditional_edges("collect_ticket_info", route_after_collect, {
        "create_ticket": "create_ticket",
        END:             END,
    })

    graph.add_edge("create_ticket", END)
    graph.add_edge("escalate",      END)

    return graph.compile()


# ── Singleton compiled graph ──────────────────────────────────────────────────
_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


# ── Helper ────────────────────────────────────────────────────────────────────

def _format_history(messages: list, n: int = 6) -> str:
    """Format last N message pairs for prompt injection."""
    recent = messages[-(n * 2):]
    lines = []
    for m in recent:
        role = "Customer" if m.get("type") == "human" else "Bot"
        content = str(m.get("content", ""))[:300]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
