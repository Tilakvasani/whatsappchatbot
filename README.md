# 🌿 Zupwell WhatsApp FAQ Bot

A production-ready WhatsApp chatbot for Zupwell health supplements — answers customer FAQs using AI-powered semantic search (RAG).

**Built with the same core logic as docForge AI:**
- ChromaDB vector search for relevant FAQ retrieval
- OpenAI LLM for natural, WhatsApp-friendly answers
- Redis for per-customer conversation history
- FastAPI + Twilio for the WhatsApp integration

---

## 🏗️ Architecture

```
Customer on WhatsApp
        ↓
    Twilio API
        ↓
POST /whatsapp/webhook  (FastAPI)
        ↓
  Background Task
        ↓
  ┌─────────────────────────┐
  │  1. Session History     │ ← Redis
  │  2. Vector Search       │ ← ChromaDB (FAQ embeddings)
  │  3. LLM Answer          │ ← OpenAI GPT-4o-mini
  └─────────────────────────┘
        ↓
  Twilio REST API → Customer
```

---

## 📁 Project Structure

```
zupwell-whatsapp-bot/
├── main.py                    ← FastAPI app entry point
├── requirements.txt
├── .env.example               ← Copy to .env and fill in values
├── Dockerfile
├── docker-compose.yml
│
├── core/
│   ├── config.py              ← All settings (OpenAI, Twilio, Redis, ChromaDB)
│   ├── llm.py                 ← OpenAI client factory
│   ├── vector.py              ← ChromaDB client factory
│   └── logger.py              ← Logging setup
│
├── rag/
│   ├── rag_service.py         ← Core RAG: retrieve + answer (from docForge)
│   └── ingest.py              ← FAQ ingestion pipeline
│
├── whatsapp/
│   ├── webhook.py             ← Twilio webhook handler + message logic
│   └── session.py             ← Redis conversation history (per phone number)
│
├── data/
│   └── zupwell_faqs.json      ← Zupwell FAQ knowledge base (edit to update)
│
└── scripts/
    └── ingest_faqs.py         ← CLI script to populate ChromaDB
```

---

## 🚀 Quick Start

### Step 1 — Install dependencies

```bash
cd zupwell-whatsapp-bot
pip install -r requirements.txt
```

### Step 2 — Set up environment

```bash
cp .env.example .env
# Edit .env and fill in:
#   OPENAI_API_KEY
#   TWILIO_ACCOUNT_SID
#   TWILIO_AUTH_TOKEN
#   TWILIO_WHATSAPP_NUMBER
```

### Step 3 — Start Redis (using Docker)

```bash
docker run -d -p 6379:6379 redis:7-alpine
```

### Step 4 — Ingest FAQs into ChromaDB

**Run this ONCE before starting the bot.** This embeds all FAQs and stores them in ChromaDB.

```bash
python scripts/ingest_faqs.py
```

You should see:
```
✅ Ingest Complete!
   Ingested : 30 FAQ chunks
   Total    : 30 chunks in ChromaDB
🚀 Your Zupwell WhatsApp bot is ready!
```

### Step 5 — Start the bot

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The bot is running at: `http://localhost:8000`

---

## 📱 Connect to WhatsApp (Twilio Setup)

### Option A — Sandbox (Free Testing)

1. Go to [Twilio Console](https://console.twilio.com) → Messaging → Try it out → Send a WhatsApp message
2. Note your sandbox number: `+1 415 523 8886`
3. Set your `.env`:
   ```
   TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
   ```
4. Expose your local server with ngrok:
   ```bash
   ngrok http 8000
   ```
5. In Twilio Console → Sandbox settings, set:
   - **When a message comes in**: `https://your-ngrok-url.ngrok.io/whatsapp/webhook`
   - Method: `HTTP POST`
6. To use the sandbox, customers send this message to `+14155238886`:
   ```
   join <your-sandbox-keyword>
   ```

### Option B — Production WhatsApp Business

1. Apply for a WhatsApp Business Account via Meta Business Manager
2. Connect your approved number to Twilio
3. Update `.env` with your production number:
   ```
   TWILIO_WHATSAPP_NUMBER=whatsapp:+91XXXXXXXXXX
   ```
4. Deploy to a public server (see deployment section)
5. Set webhook URL in Twilio Console

---

## 💬 Bot Commands

| Customer types | Bot responds |
|---|---|
| `hi`, `hello`, `hey` | Welcome message with menu |
| `help`, `menu` | Example questions |
| `reset`, `clear` | Clear conversation history |
| `about` | About the bot |
| Any question | AI-powered FAQ answer |

---

## 🔧 Updating FAQs

Edit `data/zupwell_faqs.json` and add new FAQ entries:

```json
{
  "id": "product_007",
  "category": "Products",
  "title": "Does Zupwell offer protein supplements?",
  "content": "Zupwell is currently developing protein supplements..."
}
```

Then re-run the ingest:

```bash
python scripts/ingest_faqs.py        # only ingest new ones
python scripts/ingest_faqs.py --force  # re-embed all FAQs
```

Or trigger via the admin API:
```bash
curl -X POST "http://localhost:8000/admin/ingest?force=true" \
     -H "X-Admin-Key: zupwell-admin-2024"
```

---

## 🛠️ Admin API

| Endpoint | Method | Description |
|---|---|---|
| `/whatsapp/webhook` | POST | Twilio webhook (main bot) |
| `/whatsapp/health` | GET | Health check |
| `/admin/ingest` | POST | Re-ingest FAQs |
| `/admin/stats` | GET | Bot stats |
| `/docs` | GET | Swagger API docs |

Admin endpoints require header: `X-Admin-Key: <your-admin-key>`

---

## 🐳 Docker Deployment

```bash
# Build and start everything (bot + redis)
docker-compose up -d

# Run FAQ ingest inside the container
docker-compose exec bot python scripts/ingest_faqs.py

# View logs
docker-compose logs -f bot
```

---

## ☁️ Deploy to Cloud (Render / Railway)

### Render (Recommended — free tier available)

1. Push to GitHub
2. New Web Service → Connect your repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables in Render dashboard
6. Add Redis: New → Redis → copy the URL to `REDIS_URL`
7. After deploy, run ingest:
   ```bash
   # In Render shell or locally pointing to prod
   CHROMA_PATH=/opt/render/project/src/chroma_db python scripts/ingest_faqs.py
   ```

### Environment Variables for Production

```bash
OPENAI_API_KEY=sk-...
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
TWILIO_WHATSAPP_NUMBER=whatsapp:+...
REDIS_URL=redis://...  # from your cloud Redis
CHROMA_PATH=/app/chroma_db
APP_ENV=production
ADMIN_KEY=<strong-random-key>
```

---

## 🔍 How It Works (RAG Flow)

1. **Customer sends message** → Twilio forwards to `/whatsapp/webhook`
2. **Webhook returns 200 immediately** (avoids Twilio's 15-second timeout)
3. **Background task runs:**
   - Checks for special commands (hi, help, reset)
   - Gets conversation history from Redis
   - Embeds the question using OpenAI `text-embedding-3-small`
   - Searches ChromaDB for top-5 most similar FAQ chunks
   - Sends context + history + question to GPT-4o-mini
   - Gets a short, WhatsApp-friendly answer
4. **Reply sent** via Twilio REST API
5. **Turn saved** to Redis (24h TTL, 20-turn rolling window)

---

## 📊 Key Configuration

| Setting | Default | Description |
|---|---|---|
| `MIN_SCORE` | 0.25 | Minimum similarity score to use a FAQ chunk |
| `TOP_K` | 5 | Number of FAQ chunks to retrieve |
| `MAX_WA_LENGTH` | 1500 | Max chars per WhatsApp message |
| `SESSION_TTL` | 86400 | Session expiry (24 hours) |
| `MAX_MESSAGES` | 40 | Max messages kept in history (20 turns) |

---

## 📞 Zupwell Contact Info

The bot is configured with Zupwell's actual contact details:
- **Email**: info@zupwell.com
- **WhatsApp/Phone**: +91 6355466208
- **Location**: Ahmedabad, Gujarat, India

Update these in `data/zupwell_faqs.json` if they change, then re-run ingest.

---

## 🐛 Troubleshooting

**Bot is not answering:**
- Check if FAQs are ingested: `GET /admin/stats`
- Check logs: `uvicorn main:app --log-level debug`

**ChromaDB is empty:**
- Run: `python scripts/ingest_faqs.py --force`

**Twilio webhook not receiving:**
- Make sure your server is publicly accessible (use ngrok for local dev)
- Check Twilio console for webhook errors

**Redis connection refused:**
- Make sure Redis is running: `docker run -d -p 6379:6379 redis:7-alpine`
