"""
chains/quiz_chain.py
--------------------
Generates multiple-choice quizzes grounded in the user's study materials.

Pipeline:
    1. Retrieve top-k relevant chunks from the vector store
    2. Format chunks into a context block
    3. Build a prompt with strict JSON format instructions
    4. Call the LLM (GPT-4o-mini)
    5. Parse the response into a validated Quiz object
    6. Optionally run a faithfulness check (Phase 3)

The chain is intentionally kept as a plain function rather than a
LangChain LCEL chain object — this makes it easier to read, debug,
and explain in an interview. You can refactor to LCEL later if needed.

Install dependencies:
    pip install langchain langchain-openai openai python-dotenv
"""

import os
from typing import Optional
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from models.schemas import Quiz, DifficultyLevel, get_quiz_parser
from vector_store.store_manager import get_store

load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Model used for quiz generation.
# gpt-4o-mini is cheap (~$0.15/1M input tokens) and more than capable
# of structured quiz generation from retrieved context.
GENERATION_MODEL = "gemini-2.0-flash"

# Model used for the faithfulness check (Phase 3).
# Same model is fine — it's a simple yes/no verification task.
FAITHFULNESS_MODEL = "gemini-2.0-flash"

# Number of chunks to retrieve per topic query.
# 6 gives the LLM enough material to generate varied questions
# without flooding the context window.
RETRIEVAL_K = 6

# Temperature for generation.
# 0.3 gives slightly varied question phrasing while staying factual.
# Use 0.0 if you want fully deterministic output.
GENERATION_TEMPERATURE = 0.3

# Temperature for faithfulness check — always 0 (purely analytical task).
FAITHFULNESS_TEMPERATURE = 0.0


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

QUIZ_PROMPT_TEMPLATE = """
You are an expert educator and quiz designer. Your task is to generate a
multiple-choice quiz based STRICTLY on the study material provided below.

RULES:
- Every question MUST be answerable using ONLY the provided context.
- Do NOT use any external knowledge. If the context doesn't cover a concept
  well enough to write a question, skip it.
- Each question must have EXACTLY 4 options.
- EXACTLY one option per question must be correct.
- The explanation must clearly state WHY the correct answer is right,
  referencing the source material.
- Vary the difficulty across questions if possible.
- Do NOT repeat questions or rephrase the same concept twice.
- Write questions that test understanding, not just memorisation.
  Avoid questions like "According to the text, what is X?" — instead
  ask "What is X?" and test whether they understood the concept.

TOPIC: {topic}
NUMBER OF QUESTIONS: {num_questions}
TARGET DIFFICULTY: {difficulty}
SOURCE DOCUMENTS: {source_hint}

STUDY MATERIAL (retrieved context):
--------------------------------------
{context}
--------------------------------------

{format_instructions}

Generate the quiz now. Return ONLY the JSON object, no preamble or explanation.
""".strip()


FAITHFULNESS_PROMPT_TEMPLATE = """
You are a strict fact-checker. Your job is to verify whether each quiz
question and its marked correct answer is directly supported by the
provided context.

For each question, respond with:
- "supported"   if the question AND correct answer are clearly in the context
- "unsupported" if the question or correct answer cannot be verified from context

CONTEXT:
--------------------------------------
{context}
--------------------------------------

QUIZ QUESTIONS TO VERIFY:
{questions_summary}

Respond ONLY with a JSON array of objects, one per question, in this format:
[
  {{"question_index": 0, "verdict": "supported",   "note": "brief reason"}},
  {{"question_index": 1, "verdict": "unsupported", "note": "brief reason"}},
  ...
]

Return ONLY the JSON array. No preamble.
""".strip()


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def _format_context(docs) -> tuple[str, str]:
    """
    Format retrieved Document chunks into a clean context string
    for the prompt, and a source hint string for metadata.

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

        # Build a human-readable source tag for each chunk
        if page:
            loc = f"{source}, {page}"
        elif ts:
            loc = f"{source} [{ts}]"
        else:
            loc = source

        context_parts.append(f"[{i+1}] ({loc})\n{doc.page_content}")
        sources.add(source)

    context_str   = "\n\n".join(context_parts)
    source_hint   = ", ".join(sorted(sources))
    return context_str, source_hint


def _run_faithfulness_check(
    context: str,
    quiz: Quiz,
    llm: ChatGoogleGenerativeAI,
) -> list[dict]:
    """
    Verify that each quiz question is grounded in the retrieved context.

    Sends a second LLM call asking it to check whether each question
    and its correct answer can be found in the context. Questions
    flagged as "unsupported" should be highlighted in the UI or removed.

    Args:
        context: The formatted context string used for generation.
        quiz:    The Quiz object to verify.
        llm:     The ChatGoogleGenerativeAI instance to use for checking.

    Returns:
        List of dicts with keys: question_index, verdict, note.
        verdict is either "supported" or "unsupported".
    """
    import json

    # Build a compact summary of questions for the faithfulness prompt
    questions_summary = "\n".join([
        f"Q{i}: {q.question} → Correct: {q.get_correct_option().option}"
        for i, q in enumerate(quiz.questions)
    ])

    prompt = FAITHFULNESS_PROMPT_TEMPLATE.format(
        context=context,
        questions_summary=questions_summary,
    )

    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Strip markdown fences if the model wrapped it
    raw = raw.strip("```json").strip("```").strip()

    try:
        verdicts = json.loads(raw)
        return verdicts
    except json.JSONDecodeError:
        # If parsing fails, treat all questions as supported
        # (better to show a potentially weak question than crash)
        print("[Quiz Chain] Faithfulness check parse failed — skipping.")
        return [
            {"question_index": i, "verdict": "supported", "note": "check skipped"}
            for i in range(len(quiz.questions))
        ]


def generate_quiz(
    topic: str,
    num_questions: int = 5,
    difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE,
    doc_type: Optional[str] = None,
    source: Optional[str] = None,
    run_faithfulness_check: bool = False,
) -> dict:
    """
    Generate a multiple-choice quiz grounded in the user's study materials.

    Full pipeline:
        1. Retrieve relevant chunks from ChromaDB
        2. Format chunks into context
        3. Build prompt with format instructions
        4. Call GPT-4o-mini
        5. Parse response into a Quiz object
        6. (Optional) Run faithfulness check

    Args:
        topic:                  The topic or concept to quiz on.
                                Should match something in the ingested docs.
        num_questions:          How many MCQs to generate (1–10).
        difficulty:             Target difficulty level.
        doc_type:               Optional filter — "pdf", "youtube", "website".
        source:                 Optional filter — specific source document name.
        run_faithfulness_check: If True, runs a second LLM call to verify
                                that each question is grounded in context.
                                Adds ~1–2 seconds but improves reliability.

    Returns:
        Dict with keys:
            "quiz"        : Quiz object (validated Pydantic model)
            "context"     : The raw context string used for generation
            "verdicts"    : List of faithfulness verdicts (empty if check skipped)
            "chunks_used" : Number of chunks retrieved

    Raises:
        ValueError: If no documents are found for the topic.
        RuntimeError: If LLM call or parsing fails after retry.

    Example:
        >>> result = generate_quiz("backpropagation", num_questions=5)
        >>> quiz = result["quiz"]
        >>> print(quiz.total_marks)
        5
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    num_questions = max(1, min(num_questions, 10))  # clamp to 1–10

    print(f"[Quiz Chain] Generating {num_questions}-question quiz on: '{topic}'")
    print(f"[Quiz Chain] Difficulty: {difficulty.value}")

    # ------------------------------------------------------------------
    # Step 1 — Retrieve context
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
            f"Make sure you've ingested documents that cover this topic."
        )

    print(f"[Quiz Chain] Retrieved {len(docs)} chunks")

    # ------------------------------------------------------------------
    # Step 2 — Format context
    # ------------------------------------------------------------------
    context, source_hint = _format_context(docs)

    # ------------------------------------------------------------------
    # Step 3 — Build prompt
    # ------------------------------------------------------------------
    parser = get_quiz_parser()

    prompt_template = PromptTemplate(
        template=QUIZ_PROMPT_TEMPLATE,
        input_variables=["topic", "num_questions", "difficulty",
                         "source_hint", "context"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
    )

    prompt = prompt_template.format(
        topic         = topic,
        num_questions = num_questions,
        difficulty    = difficulty.value,
        source_hint   = source_hint,
        context       = context,
    )

    # ------------------------------------------------------------------
    # Step 4 — Call LLM
    # ------------------------------------------------------------------
    llm = ChatGoogleGenerativeAI(
        model       = GENERATION_MODEL,
        temperature = GENERATION_TEMPERATURE,
        google_api_key = os.getenv("GOOGLE_API_KEY"),
    )

    print(f"[Quiz Chain] Calling {GENERATION_MODEL}...")
    response = llm.invoke(prompt)
    raw_output = response.content

    # ------------------------------------------------------------------
    # Step 5 — Parse into Quiz object
    # ------------------------------------------------------------------
    print("[Quiz Chain] Parsing response...")
    try:
        quiz: Quiz = parser.parse(raw_output)
        # Attach source hint to the quiz object
        quiz.source_hint = source_hint
        print(f"[Quiz Chain] Successfully parsed {quiz.total_marks} questions")

    except Exception as e:
        raise RuntimeError(
            f"Failed to parse quiz from LLM response: {e}\n"
            f"Raw output:\n{raw_output[:500]}"
        )

    # ------------------------------------------------------------------
    # Step 6 — Optional faithfulness check
    # ------------------------------------------------------------------
    verdicts = []
    if run_faithfulness_check:
        print("[Quiz Chain] Running faithfulness check...")
        faithfulness_llm = ChatGoogleGenerativeAI(
            model          = FAITHFULNESS_MODEL,
            temperature    = FAITHFULNESS_TEMPERATURE,
            google_api_key = os.getenv("google_api_key"),
        )
        verdicts = _run_faithfulness_check(context, quiz, faithfulness_llm)

        supported_count = sum(
            1 for v in verdicts if v.get("verdict") == "supported"
        )
        print(f"[Quiz Chain] Faithfulness: "
              f"{supported_count}/{len(verdicts)} questions supported by context")

    return {
        "quiz"        : quiz,
        "context"     : context,
        "verdicts"    : verdicts,
        "chunks_used" : len(docs),
    }


def score_quiz(quiz: Quiz, user_answers: list[int]) -> dict:
    """
    Score a completed quiz and return a detailed result dict.

    This is a thin wrapper around Quiz.score() that also computes
    per-difficulty breakdowns useful for Phase 4 performance tracking.

    Args:
        quiz:         The Quiz object that was presented to the user.
        user_answers: List of 0-based indices of the user's selected
                      option for each question, in question order.

    Returns:
        Extended result dict from Quiz.score() with an additional
        "difficulty_breakdown" key showing scores per difficulty level.

    Example:
        >>> result = score_quiz(quiz, user_answers=[0, 2, 1, 3, 0])
        >>> print(result["percentage"])
        80.0
        >>> print(result["difficulty_breakdown"])
        {"beginner": {"correct": 2, "total": 2},
         "intermediate": {"correct": 1, "total": 2},
         "advanced": {"correct": 1, "total": 1}}
    """
    result = quiz.score(user_answers)

    # Compute per-difficulty breakdown
    breakdown: dict[str, dict] = {}
    for r in result["results"]:
        d = r["difficulty"].value if hasattr(r["difficulty"], "value") else r["difficulty"]
        if d not in breakdown:
            breakdown[d] = {"correct": 0, "total": 0}
        breakdown[d]["total"] += 1
        if r["is_correct"]:
            breakdown[d]["correct"] += 1

    result["difficulty_breakdown"] = breakdown
    return result


# ---------------------------------------------------------------------------
# Quick test — run directly to verify
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Quiz Chain Quick Test ===\n")
    print("NOTE: Requires documents to be ingested into the vector store.")
    print("      Run ingestion first, then test this chain.\n")

    topic = input("Enter a topic to quiz on (e.g. 'gradient descent'): ").strip()
    if not topic:
        topic = "machine learning"

    try:
        result = generate_quiz(
            topic                  = topic,
            num_questions          = 3,
            difficulty             = DifficultyLevel.INTERMEDIATE,
            run_faithfulness_check = True,
        )

        quiz     = result["quiz"]
        verdicts = result["verdicts"]

        print(f"\n=== Quiz: {quiz.topic} ===")
        print(f"Source: {quiz.source_hint}")
        print(f"Questions: {quiz.total_marks}\n")

        for i, q in enumerate(quiz.questions):
            verdict_label = ""
            if verdicts:
                v = next((v for v in verdicts if v["question_index"] == i), None)
                if v and v["verdict"] == "unsupported":
                    verdict_label = " ⚠️ [UNSUPPORTED]"

            print(f"Q{i+1}{verdict_label}: {q.question}")
            for j, opt in enumerate(q.options):
                marker = "✓" if opt.is_correct else " "
                print(f"  [{marker}] {chr(65+j)}. {opt.option}")
            print(f"  Explanation: {q.explanation}")
            print(f"  Difficulty: {q.difficulty.value} | Tag: {q.topic_tag}\n")

        # Test scoring
        print("--- Scoring test (all correct answers) ---")
        correct_answers = [q.get_correct_index() for q in quiz.questions]
        score_result = score_quiz(quiz, correct_answers)
        print(f"Score: {score_result['score']}/{score_result['total']} "
              f"({score_result['percentage']}%)")
        print(f"Difficulty breakdown: {score_result['difficulty_breakdown']}")

    except ValueError as e:
        print(f"Error: {e}")
    except RuntimeError as e:
        print(f"Runtime error: {e}")