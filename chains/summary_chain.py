"""
chains/summary_chain.py
------------------------
Generates structured topic summaries grounded in the user's study materials,
using Google's Gemini via ChatGoogleGenerativeAI.

Pipeline:
    1. Retrieve top-k relevant chunks from the vector store
       (broader retrieval than quiz/flashcard chains — more context needed)
    2. If chunks exceed context budget, apply a two-stage map-reduce summary
    3. Format chunks into a labelled context block
    4. Build a prompt with structured JSON format instructions
    5. Call Gemini (gemini-2.5-flash
)
    6. Parse response into a validated TopicSummary object
    7. Return summary + metadata ready to render in Streamlit

Why a map-reduce fallback?
---------------------------
For broad topics like "neural networks" or "machine learning basics",
retrieval may return very large chunks that approach the model's context
limit. The map-reduce strategy:
    - MAP:    Summarise each chunk individually into a mini-summary
    - REDUCE: Feed all mini-summaries into the final structured prompt
This prevents context overflow while preserving coverage across all chunks.

Install dependencies:
    pip install langchain langchain-google-genai google-generativeai python-dotenv
"""

import os
from typing import Optional
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from models.schemas import TopicSummary, get_summary_parser
from vector_store.store_manager import get_store

load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GENERATION_MODEL     = "gemini-2.5-flash"

# Summary needs wide coverage — retrieve more chunks than quiz/flashcard.
RETRIEVAL_K          = 10

# Character threshold for triggering map-reduce.
# If total context exceeds this, we compress chunks first.
# ~12,000 chars ≈ ~3,000 tokens — safely within Gemini Flash's context window
# while leaving room for the prompt and output.
MAP_REDUCE_THRESHOLD = 12_000

# Temperature settings
GENERATION_TEMPERATURE = 0.2   # main summary — low for factual accuracy
MAP_TEMPERATURE        = 0.0   # map stage — fully deterministic compression

MAX_OUTPUT_TOKENS = 3000        # summaries can be longer than quizzes/flashcards


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

# Main structured summary prompt (used when context fits in one call)
SUMMARY_PROMPT_TEMPLATE = """
You are an expert educator creating a comprehensive study guide.
Your task is to produce a structured summary of the topic below,
based STRICTLY on the provided study material.

RULES:
- Use ONLY information from the provided context. No external knowledge.
- key_points: 3–7 concise bullet-point takeaways. Each should be a complete,
  standalone insight — not just a topic label.
- detailed_summary: 2–4 clear paragraphs explaining the topic as if to a
  student encountering it for the first time. Use plain language.
- key_terms: identify the most important technical terms and define each
  in 1–2 plain-English sentences.
- suggested_questions: 3–5 questions the student should be able to answer
  after studying this topic. These feed into quiz generation later.
- prerequisites: topics the student needs before tackling this one.
  Leave as an empty list [] if there are no clear prerequisites in the material.
- Write as a knowledgeable teacher, not as a document extractor.
  Synthesise and explain, don't just paraphrase sentences from the text.

TOPIC: {topic}
SOURCE DOCUMENTS: {source_hint}

STUDY MATERIAL (retrieved context):
--------------------------------------
{context}
--------------------------------------

{format_instructions}

Generate the structured summary now.
Return ONLY the JSON object. No preamble, no explanation, no markdown fences.
""".strip()


# Map stage prompt — compress one chunk into a mini-summary
MAP_PROMPT_TEMPLATE = """
You are a concise study assistant. Summarise the key points from the
passage below in 3–5 bullet points. Focus on facts, definitions,
and concepts directly relevant to the topic: "{topic}".
Ignore irrelevant content. Be brief and factual.

PASSAGE:
{chunk}

Return ONLY the bullet points. No preamble.
""".strip()


# Reduce stage prompt — structured summary from compressed mini-summaries
REDUCE_PROMPT_TEMPLATE = """
You are an expert educator creating a comprehensive study guide.
The material below is a set of condensed notes on: "{topic}".
Use these notes to produce a structured summary.

RULES:
- Synthesise across all notes — don't just repeat them.
- Use ONLY the information in the notes. No external knowledge.
- key_points: 3–7 concise, standalone bullet-point takeaways.
- detailed_summary: 2–4 paragraphs explaining the topic clearly.
- key_terms: important technical terms with plain-English definitions.
- suggested_questions: 3–5 questions the student should be able to answer.
- prerequisites: topics needed before this one, or [] if none apparent.

TOPIC: {topic}
SOURCE DOCUMENTS: {source_hint}

CONDENSED NOTES:
--------------------------------------
{condensed_context}
--------------------------------------

{format_instructions}

Generate the structured summary now.
Return ONLY the JSON object. No preamble, no explanation, no markdown fences.
""".strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_llm(temperature: float = GENERATION_TEMPERATURE) -> ChatGoogleGenerativeAI:
    """
    Instantiate and return the Gemini LLM.

    Args:
        temperature: Generation temperature. Use 0.0 for deterministic
                     compression (map stage), 0.2 for final generation.

    Raises:
        EnvironmentError: If GOOGLE_API_KEY is not set.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not found in environment.\n"
            "Add it to your .env file: GOOGLE_API_KEY=your_key_here\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )
    return ChatGoogleGenerativeAI(
        model          = GENERATION_MODEL,
        temperature    = temperature,
        max_tokens     = MAX_OUTPUT_TOKENS,
        google_api_key = api_key,
    )


def _format_context(docs) -> tuple[str, str]:
    """
    Format retrieved Document chunks into a labelled context string
    and a comma-separated source hint string.

    Args:
        docs: List of LangChain Document objects.

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


def _clean_llm_output(raw: str) -> str:
    """
    Strip markdown code fences that Gemini sometimes wraps around JSON.

    Args:
        raw: Raw LLM output string.

    Returns:
        Cleaned string ready for JSON parsing.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        lines = lines[:-1] if lines and lines[-1].strip() == "```" else lines
        raw = "\n".join(lines).strip()
    return raw


# ---------------------------------------------------------------------------
# Map-Reduce Strategy
# ---------------------------------------------------------------------------

def _map_chunks(docs: list, topic: str, llm: ChatGoogleGenerativeAI) -> str:
    """
    MAP stage: Compress each chunk into a mini-summary independently.

    Each chunk is summarised in isolation. This means even chunks that
    only partially overlap with the topic still contribute their relevant
    content without polluting the final context with noise.

    Args:
        docs:  Retrieved Document objects.
        topic: The topic being summarised (used to focus compression).
        llm:   The LLM instance for compression calls.

    Returns:
        A single string of concatenated mini-summaries, one per chunk.
    """
    print(f"[Summary Chain] MAP stage: compressing {len(docs)} chunks...")
    mini_summaries = []

    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page_label", "")
        loc    = f"{source}, {page}" if page else source

        map_prompt = MAP_PROMPT_TEMPLATE.format(
            topic = topic,
            chunk = doc.page_content,
        )

        try:
            response = llm.invoke(map_prompt)
            summary  = response.content.strip()
            mini_summaries.append(f"[From: {loc}]\n{summary}")
            print(f"[Summary Chain]   Compressed chunk {i+1}/{len(docs)}")
        except Exception as e:
            # If one chunk fails, skip it rather than aborting everything
            print(f"[Summary Chain]   Skipping chunk {i+1} — {e}")

    return "\n\n".join(mini_summaries)


def _run_map_reduce(
    docs: list,
    topic: str,
    source_hint: str,
    parser,
    llm: ChatGoogleGenerativeAI,
) -> TopicSummary:
    """
    Full map-reduce pipeline for large context situations.

    Steps:
        1. MAP  — compress each chunk individually
        2. REDUCE — build final structured summary from compressed notes

    Args:
        docs:        Retrieved Document objects.
        topic:       The topic being summarised.
        source_hint: Comma-separated source names for metadata.
        parser:      The OutputFixingParser for TopicSummary.
        llm:         The generation LLM (used for reduce stage).

    Returns:
        Validated TopicSummary object.
    """
    # MAP — compress each chunk with a cold (temperature=0) LLM
    map_llm          = _build_llm(temperature=MAP_TEMPERATURE)
    condensed_context = _map_chunks(docs, topic, map_llm)

    print(f"[Summary Chain] REDUCE stage: generating structured summary...")

    # REDUCE — build the final structured summary
    reduce_template = PromptTemplate(
        template=REDUCE_PROMPT_TEMPLATE,
        input_variables=["topic", "source_hint", "condensed_context"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
    )

    reduce_prompt = reduce_template.format(
        topic             = topic,
        source_hint       = source_hint,
        condensed_context = condensed_context,
    )

    response   = llm.invoke(reduce_prompt)
    raw_output = _clean_llm_output(response.content)
    return parser.parse(raw_output)


# ---------------------------------------------------------------------------
# Core Function
# ---------------------------------------------------------------------------

def generate_summary(
    topic: str,
    doc_type: Optional[str] = None,
    source: Optional[str] = None,
    force_map_reduce: bool = False,
) -> dict:
    """
    Generate a structured topic summary grounded in the user's study materials.

    Automatically chooses between single-pass and map-reduce strategies
    based on the size of the retrieved context:
        - Small context  (< 12,000 chars) → single LLM call (fast)
        - Large context  (≥ 12,000 chars) → map-reduce (thorough, avoids overflow)

    Args:
        topic:            The topic to summarise. Should correspond to
                          content in the ingested documents.
        doc_type:         Optional filter — "pdf", "youtube", or "website".
        source:           Optional filter — exact source document name.
        force_map_reduce: Set True to always use map-reduce regardless of
                          context size. Useful for testing or very dense topics.

    Returns:
        Dict with keys:
            "summary"      : TopicSummary object (validated Pydantic model)
            "context"      : Raw context string used for generation
            "strategy"     : "single_pass" or "map_reduce"
            "chunks_used"  : Number of chunks retrieved
            "model"        : Model name used

    Raises:
        ValueError:       If topic is empty or no documents found.
        RuntimeError:     If LLM call or parsing fails.
        EnvironmentError: If GOOGLE_API_KEY is not set.

    Example:
        >>> result = generate_summary("backpropagation")
        >>> summary = result["summary"]
        >>> print(summary.topic)
        'backpropagation'
        >>> for point in summary.key_points:
        ...     print("-", point)
        >>> for term in summary.key_terms:
        ...     print(term.term, ":", term.definition)
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    topic = topic.strip()
    if not topic:
        raise ValueError("Topic cannot be empty.")

    print(f"[Summary Chain] Generating summary for: '{topic}'")
    print(f"[Summary Chain] Model: {GENERATION_MODEL}")

    # ------------------------------------------------------------------
    # Step 1 — Retrieve chunks (broader than quiz/flashcard)
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
            f"Make sure you've ingested documents covering this topic."
        )

    print(f"[Summary Chain] Retrieved {len(docs)} chunks")

    # ------------------------------------------------------------------
    # Step 2 — Format context
    # ------------------------------------------------------------------
    context, source_hint = _format_context(docs)
    context_size         = len(context)
    print(f"[Summary Chain] Context size: {context_size:,} characters")

    # ------------------------------------------------------------------
    # Step 3 — Choose strategy
    # ------------------------------------------------------------------
    use_map_reduce = force_map_reduce or (context_size >= MAP_REDUCE_THRESHOLD)
    strategy       = "map_reduce" if use_map_reduce else "single_pass"
    print(f"[Summary Chain] Strategy: {strategy}")

    # ------------------------------------------------------------------
    # Step 4 — Build LLM and parser
    # ------------------------------------------------------------------
    llm    = _build_llm(temperature=GENERATION_TEMPERATURE)
    parser = get_summary_parser()

    # ------------------------------------------------------------------
    # Step 5 — Generate summary
    # ------------------------------------------------------------------
    try:
        if use_map_reduce:
            summary = _run_map_reduce(docs, topic, source_hint, parser, llm)

        else:
            # Single-pass — context fits comfortably, one LLM call
            prompt_template = PromptTemplate(
                template=SUMMARY_PROMPT_TEMPLATE,
                input_variables=["topic", "source_hint", "context"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                },
            )

            prompt     = prompt_template.format(
                topic       = topic,
                source_hint = source_hint,
                context     = context,
            )

            print(f"[Summary Chain] Calling {GENERATION_MODEL} (single pass)...")
            response   = llm.invoke(prompt)
            raw_output = _clean_llm_output(response.content)
            summary    = parser.parse(raw_output)

    except Exception as e:
        raise RuntimeError(
            f"Summary generation failed for topic '{topic}'.\n"
            f"Error: {e}"
        )

    # Attach source hint to the summary object
    summary.source_hint = source_hint

    print(
        f"[Summary Chain] Done — "
        f"{len(summary.key_points)} key points, "
        f"{len(summary.key_terms)} key terms, "
        f"{len(summary.suggested_questions)} suggested questions"
    )

    return {
        "summary"     : summary,
        "context"     : context,
        "strategy"    : strategy,
        "chunks_used" : len(docs),
        "model"       : GENERATION_MODEL,
    }


# ---------------------------------------------------------------------------
# Utility: Feed summary back into quiz/flashcard chains
# ---------------------------------------------------------------------------

def get_suggested_quiz_topics(summary: TopicSummary) -> list[str]:
    """
    Extract a list of quiz-ready topic strings from a TopicSummary.

    These come from two places:
        1. suggested_questions   — direct quiz candidates
        2. key_terms             — each term can anchor a definition question

    The returned list can be passed directly as the `topic` arg
    to generate_quiz() or generate_flashcards() for fine-grained
    follow-up generation.

    Args:
        summary: A validated TopicSummary object.

    Returns:
        List of topic strings the user can select to generate targeted quizzes.

    Example:
        >>> topics = get_suggested_quiz_topics(summary)
        >>> result = generate_quiz(topics[0], num_questions=5)
    """
    topics = []

    # The summary's main topic itself
    topics.append(summary.topic)

    # Each key term is a quiz-worthy sub-topic
    for kt in summary.key_terms:
        topics.append(kt.term)

    # Suggested questions can be used directly as quiz topics
    for q in summary.suggested_questions:
        topics.append(q)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in topics:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique.append(t)

    return unique


# ---------------------------------------------------------------------------
# Quick test — run directly to verify
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Summary Chain Quick Test ===\n")
    print("Requires: GOOGLE_API_KEY in .env + documents ingested.\n")

    topic = input("Enter a topic to summarise (e.g. 'gradient descent'): ").strip()
    if not topic:
        topic = "machine learning"
    try:
        result  = generate_summary(topic)
        summary = result["summary"]

        print(f"\n{'='*60}")
        print(f"TOPIC:    {summary.topic}")
        print(f"SOURCE:   {summary.source_hint}")
        print(f"STRATEGY: {result['strategy']}")
        print(f"MODEL:    {result['model']}")
        print(f"{'='*60}\n")

        print("KEY POINTS:")
        for point in summary.key_points:
            print(f"  • {point}")

        print(f"\nDETAILED SUMMARY:")
        print(summary.detailed_summary)

        print(f"\nKEY TERMS ({len(summary.key_terms)}):")
        for kt in summary.key_terms:
            print(f"  {kt.term}: {kt.definition}")

        print(f"\nSUGGESTED QUESTIONS ({len(summary.suggested_questions)}):")
        for i, q in enumerate(summary.suggested_questions, 1):
            print(f"  {i}. {q}")

        if summary.prerequisites:
            print(f"\nPREREQUISITES:")
            for p in summary.prerequisites:
                print(f"  • {p}")

        print(f"\nSUGGESTED QUIZ TOPICS:")
        quiz_topics = get_suggested_quiz_topics(summary)
        for t in quiz_topics[:5]:
            print(f"  → {t}")

    except EnvironmentError as e:
        print(f"Environment error: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except RuntimeError as e:
        print(f"Runtime error: {e}")