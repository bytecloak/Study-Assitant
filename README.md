# 🎓 AI Study Assistant

A Retrieval-Augmented Generation (RAG) study tool that turns your PDFs, YouTube lectures, and web articles into **summaries, flashcards, and quizzes** — with a progress dashboard that tracks what you actually know.

Built with **Streamlit**, **LangChain**, **Google Gemini**, and a local **ChromaDB** vector store.

---

## ✨ Features

| Tab | What it does |
|---|---|
| 📥 **Ingest** | Upload PDFs, or paste YouTube / website URLs. Content is scraped, chunked, embedded, and stored locally in ChromaDB. |
| 📋 **Summary** | Generates a structured study guide for any topic — key points, key terms, a detailed summary, suggested questions, and prerequisites. Falls back to a map-reduce strategy for broad topics that exceed the context budget. |
| 🃏 **Flashcards** | Generates a deck of front/back flashcards (with optional examples) grounded in your ingested material. Flip through them in the UI, or export to CSV / Anki-compatible text. |
| 📝 **Quiz** | Generates multiple-choice quizzes at a chosen difficulty, with an optional "faithfulness check" that flags questions not well-supported by the retrieved context. Scores are broken down by difficulty. |
| 📊 **Progress** | Tracks quiz history, computes a running performance profile, and surfaces strong vs. weak topics so you know what to revisit. |

> Every generation (summary, flashcards, quiz) is **retrieval-grounded** — nothing comes from the model's general knowledge alone, it's built from chunks retrieved out of your own ingested documents.

---

## 🧠 How it works

```
 PDF / YouTube / Web URL
          │
          ▼
 ┌─────────────────┐   chunk + embed   ┌──────────────────┐
 │   ingestion/*    │ ─────────────────▶│     ChromaDB      │
 │   (loaders)      │                   │   (vector_store)  │
 └─────────────────┘                    └──────────────────┘
                                                   │
                                   similarity search (top-k)
                                                   │
                                                   ▼
                                        ┌──────────────────────┐
                                        │       chains/*        │
                                        │  summary / flashcard /│
                                        │  quiz generation      │
                                        │  (Gemini + Pydantic   │
                                        │   structured output)  │
                                        └──────────────────────┘
                                                   │
                                                   ▼
                                             Streamlit UI
```

1. **Ingestion** (`ingestion/`) — Each loader (`pdf_loader.py`, `youtube_loader.py`, `web_loader.py`) extracts raw text, splits it into overlapping chunks with `RecursiveCharacterTextSplitter`, and attaches metadata (`source`, `doc_type`, page/timestamp, etc.)
2. **Storage & retrieval** (`vector_store/store_manager.py`) — Chunks are embedded with Gemini's `gemini-embedding-001` and persisted in a local ChromaDB collection, with metadata filtering support (e.g. "only my PDFs")
3. **Generation** (`chains/`) — Each chain retrieves relevant chunks for a topic, builds a prompt with strict JSON formatting instructions, calls Gemini, and parses the response into a validated Pydantic model (`models/schemas.py`) via `PydanticOutputParser` + `OutputFixingParser` (auto-retries on malformed JSON)
4. **UI** (`app.py`) — Streamlit renders everything, keeps state in `st.session_state`, and tracks quiz results into a `PerformanceProfile`

---

## 🚀 Setup

```bash
git clone <your-repo-url>
cd ai-study-assistant
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Add your Gemini API key to `.env`:

```env
GOOGLE_API_KEY=your_key_here
```

## ▶️ Run

```bash
streamlit run app.py
```

---

## 🛠️ Tech stack

Streamlit · LangChain · Google Gemini · ChromaDB · Pydantic
