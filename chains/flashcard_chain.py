"""
chains/flashcard_chain.py
--------------------------
Generates flashcard decks grounded in the user's study materials,
using Google's Gemini via ChatGoogleGenerativeAI.

Pipeline:
    1. Retrieve top-k relevant chunks from the vector store
    2. Format chunks into a context block
    3. Build a prompt with structured JSON format instructions
    4. Call Gemini (gemini-2.0-flash)
    5. Parse the response into a validated FlashcardDeck object
    6. Return deck + metadata ready to render in Streamlit

Why Gemini for this chain?
---------------------------
Flashcard generation is a high-frequency, low-latency operation —
users may generate decks for many topics in one session.
Gemini 2.0 Flash is fast, free-tier generous, and more than capable
of structured flashcard generation from retrieved context.

NOTE on model name:
    The correct model string is "gemini-2.0-flash" (not "gemini-3-flash-preview"
    which does not exist yet in the stable API as of April 2026).
    To upgrade, change GENERATION_MODEL to the new model string — nothing
    else in this file needs to change.

Install dependencies:
    pip install langchain langchain-google-genai google-generativeai python-dotenv
"""

import os
import json
from typing import Optional
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers import OutputFixingParser

from models.schemas import FlashcardDeck, Flashcard, get_flashcard_parser
from vector_store.store_manager import get_store
load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Gemini 2.0 Flash — fast, free-tier generous, great for structured output.
# To upgrade: change this string to the new model ID (e.g. "gemini-2.5-flash").
GENERATION_MODEL = "gemini-2.0-flash"

# Number of chunks to retrieve.
# Flashcards need broad coverage, not pinpoint precision —
# higher k ensures enough concepts are covered across the topic.
RETRIEVAL_K = 8

# Slight creativity helps vary card phrasing, but we stay factual.
GENERATION_TEMPERATURE = 0.2

# Max output tokens — flashcard JSON is compact, 2048 is plenty for 15 cards.
MAX_OUTPUT_TOKENS = 2048


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

FLASHCARD_PROMPT_TEMPLATE = """
You are an expert study coach and educator. Your task is to create a set of
high-quality flashcards based STRICTLY on the study material provided below.

RULES:
- Every flashcard MUST be grounded in the provided context only.
- Do NOT use any external knowledge.
- Front of card: a clear concept, term, or thought-provoking question.
- Back of card: a concise, accurate explanation or answer (2–4 sentences max).
- Example field: add a concrete example or analogy ONLY when it genuinely
  aids understanding. Leave null if it would be forced or unnecessary.
- Do NOT create duplicate cards or cards that test the same concept twice.
- Vary the card types — mix definitions, "how does X work", "what is the
  difference between X and Y", and application-style questions.
- Use plain, student-friendly language. Avoid jargon unless it IS the term
  being defined.
- Assign a topic_tag to each card using lowercase_underscore format
  (e.g. "gradient_descent", "attention_mechanism").

TOPIC: {topic}
NUMBER OF CARDS: {num_cards}
SOURCE DOCUMENTS: {source_hint}

STUDY MATERIAL (retrieved context):
--------------------------------------
{context}
--------------------------------------

{format_instructions}

Generate the flashcard deck now.
Return ONLY the JSON object. No preamble, no explanation, no markdown fences.
""".strip()


# ---------------------------------------------------------------------------
# Helper: Format retrieved chunks into context string
# ---------------------------------------------------------------------------

def _format_context(docs) -> tuple[str, str]:
    """
    Format retrieved Document chunks into a labelled context string
    and a source hint string.

    Args:
        docs: List of LangChain Document objects from retrieval.

    Returns:
        Tuple of (context_string, source_hint_string).
    """
    context_parts = []
    sources = set()

    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page_label", "")
        ts     = doc.metadata.get("timestamp", "")

        if page:
            loc = f"{source}, {page}"
        elif ts:
            loc = f"{source} [{ts}]"
        else:
            loc = source

        context_parts.append(f"[{i+1}] ({loc})\n{doc.page_content}")
        sources.add(source)

    context_str = "\n\n".join(context_parts)
    source_hint = ", ".join(sorted(sources))
    return context_str, source_hint


# ---------------------------------------------------------------------------
# Helper: Build the Gemini LLM instance
# ---------------------------------------------------------------------------

def _build_llm() -> ChatGoogleGenerativeAI:
    """
    Instantiate and return the Gemini LLM.

    Reads GOOGLE_API_KEY from environment (via .env).
    Raises EnvironmentError early if the key is missing.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not found in environment.\n"
            "Add it to your .env file: GOOGLE_API_KEY=your_key_here\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )

    return ChatGoogleGenerativeAI(
        model       = GENERATION_MODEL,
        temperature = GENERATION_TEMPERATURE,
        max_tokens  = MAX_OUTPUT_TOKENS,
        google_api_key = api_key,
    )


# ---------------------------------------------------------------------------
# Helper: Strip markdown fences from LLM output
# ---------------------------------------------------------------------------

def _clean_llm_output(raw: str) -> str:
    """
    Remove markdown code fences that Gemini sometimes wraps around JSON.

    Gemini occasionally returns:
        ```json
        { ... }
        ```
    even when told not to. This strips those fences before parsing.

    Args:
        raw: Raw string output from the LLM.

    Returns:
        Cleaned string with fences removed.
    """
    raw = raw.strip()

    # Remove ```json ... ``` or ``` ... ```
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Remove first line (``` or ```json) and last line (```)
        lines = lines[1:] if lines[0].startswith("```") else lines
        lines = lines[:-1] if lines and lines[-1].strip() == "```" else lines
        raw = "\n".join(lines).strip()

    return raw


# ---------------------------------------------------------------------------
# Core Function: generate_flashcards
# ---------------------------------------------------------------------------

def generate_flashcards(
    topic: str,
    num_cards: int = 10,
    doc_type: Optional[str] = None,
    source: Optional[str] = None,
) -> dict:
    """
    Generate a flashcard deck grounded in the user's study materials.

    Full pipeline:
        1. Retrieve relevant chunks from ChromaDB
        2. Format chunks into a labelled context block
        3. Build prompt with Pydantic format instructions
        4. Call Gemini 2.0 Flash
        5. Clean and parse the response into a FlashcardDeck object

    Args:
        topic:    The topic or concept to generate flashcards for.
                  Should correspond to content in the ingested documents.
        num_cards: Number of flashcards to generate (clamped to 5–20).
        doc_type: Optional filter — "pdf", "youtube", or "website".
                  Pass None to search across all ingested sources.
        source:   Optional filter — exact source document name.
                  Use store.list_sources() to find available names.

    Returns:
        Dict with keys:
            "deck"        : FlashcardDeck object (validated Pydantic model)
            "context"     : The raw context string used for generation
            "chunks_used" : Number of chunks retrieved from the vector store
            "model"       : Model name used for generation

    Raises:
        ValueError:    If no documents are found for the given topic,
                       or if the topic string is empty.
        RuntimeError:  If the LLM call or output parsing fails.
        EnvironmentError: If GOOGLE_API_KEY is not set.

    Example:
        >>> result = generate_flashcards("attention mechanism", num_cards=8)
        >>> deck = result["deck"]
        >>> print(deck.card_count)
        8
        >>> for card in deck.cards:
        ...     print(card.front)
        ...     print(card.back)
    """
    # ------------------------------------------------------------------
    # Validate and clamp inputs
    # ------------------------------------------------------------------
    topic = topic.strip()
    if not topic:
        raise ValueError("Topic cannot be empty.")

    num_cards = max(5, min(num_cards, 20))  # clamp to 5–20

    print(f"[Flashcard Chain] Generating {num_cards} flashcards on: '{topic}'")
    print(f"[Flashcard Chain] Model: {GENERATION_MODEL}")

    # ------------------------------------------------------------------
    # Step 1 — Retrieve relevant chunks
    # ------------------------------------------------------------------
    store = get_store()
    docs  = store.retrieve(
        query    = topic,
        k        = RETRIEVAL_K,
        doc_type = doc_type,
        source   = source,
    )

    if not docs:
        raise ValueError(
            f"No relevant content found for topic: '{topic}'.\n"
            f"Make sure you've ingested documents that cover this topic.\n"
            f"Use store.list_sources() to see what's available."
        )

    print(f"[Flashcard Chain] Retrieved {len(docs)} chunks")

    # ------------------------------------------------------------------
    # Step 2 — Format context and source hint
    # ------------------------------------------------------------------
    context, source_hint = _format_context(docs)

    # ------------------------------------------------------------------
    # Step 3 — Build prompt with format instructions
    # ------------------------------------------------------------------
    parser = get_flashcard_parser()

    prompt_template = PromptTemplate(
        template=FLASHCARD_PROMPT_TEMPLATE,
        input_variables=["topic", "num_cards", "source_hint", "context"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
    )

    prompt = prompt_template.format(
        topic       = topic,
        num_cards   = num_cards,
        source_hint = source_hint,
        context     = context,
    )

    # ------------------------------------------------------------------
    # Step 4 — Call Gemini
    # ------------------------------------------------------------------
    llm = _build_llm()

    print(f"[Flashcard Chain] Calling {GENERATION_MODEL}...")
    try:
        response   = llm.invoke(prompt)
        raw_output = response.content
    except Exception as e:
        raise RuntimeError(
            f"Gemini API call failed: {e}\n"
            f"Check your GOOGLE_API_KEY and network connection."
        )

    # ------------------------------------------------------------------
    # Step 5 — Clean and parse output
    # ------------------------------------------------------------------
    print("[Flashcard Chain] Parsing response...")
    cleaned_output = _clean_llm_output(raw_output)

    try:
        deck: FlashcardDeck = parser.parse(cleaned_output)
        # Attach source hint to the deck metadata
        deck.source_hint = source_hint
        print(f"[Flashcard Chain] Successfully parsed {deck.card_count} cards")

    except Exception as e:
        raise RuntimeError(
            f"Failed to parse flashcard deck from LLM response.\n"
            f"Error: {e}\n"
            f"Raw output (first 500 chars):\n{raw_output[:500]}"
        )

    return {
        "deck"        : deck,
        "context"     : context,
        "chunks_used" : len(docs),
        "model"       : GENERATION_MODEL,
    }


# ---------------------------------------------------------------------------
# Utility: Export helpers (called from app.py)
# ---------------------------------------------------------------------------

def deck_to_csv(deck: FlashcardDeck) -> str:
    """
    Convert a FlashcardDeck to a CSV string for download in Streamlit.

    Columns: topic, front, back, example, topic_tag

    Args:
        deck: A validated FlashcardDeck object.

    Returns:
        CSV string with header row.

    Example in app.py:
        csv_str = deck_to_csv(deck)
        st.download_button(
            label="Download Flashcards (CSV)",
            data=csv_str,
            file_name=f"{deck.topic}_flashcards.csv",
            mime="text/csv",
        )
    """
    import io
    import csv

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["topic", "front", "back", "example", "topic_tag"]
    )
    writer.writeheader()
    writer.writerows(deck.to_csv_rows())
    return output.getvalue()


def deck_to_anki_format(deck: FlashcardDeck) -> str:
    """
    Export the deck as a tab-separated text file compatible with Anki import.

    Anki's "Basic" card type expects: Front[TAB]Back
    The example (if present) is appended to the back with a line break.

    Args:
        deck: A validated FlashcardDeck object.

    Returns:
        Tab-separated string, one card per line.

    Anki import instructions:
        File → Import → select .txt file → Fields separated by Tab
        Note type: Basic | Deck: your deck name
    """
    lines = []
    for card in deck.cards:
        back = card.back
        if card.example:
            back += f"\n\nExample: {card.example}"
        lines.append(f"{card.front}\t{back}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick test — run directly to verify
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Flashcard Chain Quick Test ===\n")
    print("Requires: GOOGLE_API_KEY in .env + documents ingested into vector store.\n")

    topic = input("Enter a topic (e.g. 'neural networks'): ").strip()
    if not topic:
        topic = "machine learning"

    try:
        result = generate_flashcards(
            topic     = topic,
            num_cards = 5,
        )

        deck = result["deck"]

        print(f"\n=== Flashcard Deck: {deck.topic} ===")
        print(f"Source: {deck.source_hint}")
        print(f"Cards: {deck.card_count}")
        print(f"Model: {result['model']}\n")

        for i, card in enumerate(deck.cards):
            print(f"--- Card {i+1} [{card.topic_tag}] ---")
            print(f"  Front:   {card.front}")
            print(f"  Back:    {card.back}")
            if card.example:
                print(f"  Example: {card.example}")
            print()

        # Test CSV export
        csv_output = deck_to_csv(deck)
        print(f"--- CSV Export (first 300 chars) ---")
        print(csv_output[:300])

        # Test Anki export
        anki_output = deck_to_anki_format(deck)
        print(f"\n--- Anki Export (first 300 chars) ---")
        print(anki_output[:300])

    except EnvironmentError as e:
        print(f"Environment error: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except RuntimeError as e:
        print(f"Runtime error: {e}")