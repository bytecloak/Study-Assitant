"""
models/schemas.py
-----------------
Pydantic models that define the structured output contracts for every
LLM generation in the Study Assistant.

Why structured outputs?
------------------------
By default, LLMs return free-form text. For a study tool, we need
consistent, machine-readable structures so we can:
    - Render quiz questions as interactive UI components
    - Flip flashcards programmatically
    - Export quizzes as PDFs or CSVs
    - Track quiz scores per topic
    - Feed performance data into the recommendation system (Phase 4)

How these are used:
--------------------
Each schema is paired with a LangChain PydanticOutputParser in the
corresponding chain file. The parser:
    1. Calls parser.get_format_instructions() → injects JSON schema
       into the prompt so the LLM knows exactly what to produce
    2. Calls parser.parse(llm_output) → validates and deserialises
       the LLM's response into a typed Python object

If the LLM produces malformed JSON, the OutputFixingParser (also
defined here) automatically retries with the error message attached.

Schema hierarchy:
-----------------
    Quiz
    └── List[MCQuestion]
        └── question: str
        └── options: List[MCQOption]
            └── option: str
            └── is_correct: bool
        └── explanation: str
        └── difficulty: DifficultyLevel
        └── topic_tag: str

    FlashcardDeck
    └── List[Flashcard]
        └── front: str
        └── back: str
        └── example: Optional[str]
        └── topic_tag: str

    TopicSummary
    └── topic: str
    └── key_points: List[str]
    └── detailed_summary: str
    └── key_terms: List[KeyTerm]
    └── suggested_questions: List[str]
    └── prerequisites: List[str]

    PerformanceReport   (Phase 4 — feeds recommendation system)
    └── strong_topics: List[str]
    └── weak_topics: List[str]
    └── recommended_query: List[str]

    ResourceRecommendation  (Phase 4)
    └── title: str
    └── url: str
    └── resource_type: ResourceType
    └── reason: str
    └── difficulty: DifficultyLevel
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared Enums
# ---------------------------------------------------------------------------

class DifficultyLevel(str, Enum):
    """
    Difficulty level for quiz questions and resource recommendations.
    Inherits from str so it serialises cleanly to JSON ("beginner" not
    "<DifficultyLevel.BEGINNER: 'beginner'>").
    """
    BEGINNER     = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED     = "advanced"


class ResourceType(str, Enum):
    """Type of external resource returned by the recommendation system."""
    VIDEO   = "video"
    ARTICLE = "article"
    PAPER   = "paper"
    COURSE  = "course"
    DOCS    = "documentation"


# ---------------------------------------------------------------------------
# Quiz Schemas
# ---------------------------------------------------------------------------

class MCQOption(BaseModel):
    """
    A single answer option in a multiple-choice question.

    Fields:
        option:     The answer text shown to the user.
        is_correct: Whether this option is the correct answer.
                    Exactly one option per question should be True.
    """
    option:     str  = Field(..., description="The answer option text.")
    is_correct: bool = Field(..., description="True if this is the correct answer.")


class MCQuestion(BaseModel):
    """
    A single multiple-choice question grounded in retrieved document context.

    Fields:
        question:    The question stem shown to the user.
        options:     List of 4 answer options (exactly one correct).
        explanation: Why the correct answer is right — shown after answering.
                     This is the most valuable learning moment.
        difficulty:  Estimated difficulty of this question.
        topic_tag:   The sub-topic this question tests (e.g. "backpropagation").
                     Used for per-topic score tracking in Phase 3.
    """
    question:    str            = Field(..., description="The question to ask the user.")
    options:     List[MCQOption] = Field(..., description="Exactly 4 answer options.")
    explanation: str            = Field(
        ...,
        description=(
            "A clear explanation of why the correct answer is right, "
            "referencing the source material. Shown after the user answers."
        )
    )
    difficulty:  DifficultyLevel = Field(
        default=DifficultyLevel.INTERMEDIATE,
        description="Estimated difficulty of this question."
    )
    topic_tag:   str = Field(
        ...,
        description="The specific sub-topic or concept this question tests."
    )

    @field_validator("options")
    @classmethod
    def must_have_exactly_four_options(cls, v: List[MCQOption]) -> List[MCQOption]:
        """Enforce exactly 4 options per question."""
        if len(v) != 4:
            raise ValueError(
                f"Each question must have exactly 4 options, got {len(v)}."
            )
        return v

    @field_validator("options")
    @classmethod
    def must_have_exactly_one_correct(cls, v: List[MCQOption]) -> List[MCQOption]:
        """Enforce exactly one correct answer per question."""
        correct_count = sum(1 for opt in v if opt.is_correct)
        if correct_count != 1:
            raise ValueError(
                f"Each question must have exactly 1 correct option, "
                f"got {correct_count}."
            )
        return v

    def get_correct_option(self) -> MCQOption:
        """Return the correct MCQOption for this question."""
        return next(opt for opt in self.options if opt.is_correct)

    def get_correct_index(self) -> int:
        """Return the 0-based index of the correct option."""
        return next(i for i, opt in enumerate(self.options) if opt.is_correct)


class Quiz(BaseModel):
    """
    A complete quiz grounded in the user's ingested study materials.

    Fields:
        topic:       The topic or concept the quiz covers.
        source_hint: The document(s) the questions were drawn from.
        questions:   List of MCQuestion objects.
        total_marks: Total number of questions (auto-computed).
    """
    topic:       str           = Field(..., description="The topic this quiz covers.")
    source_hint: str           = Field(
        default="",
        description="The source document(s) used to generate this quiz."
    )
    questions:   List[MCQuestion] = Field(
        ...,
        description="List of multiple-choice questions."
    )

    @property
    def total_marks(self) -> int:
        """Total number of questions in the quiz."""
        return len(self.questions)

    def score(self, user_answers: List[int]) -> dict:
        """
        Score a completed quiz.

        Args:
            user_answers: List of 0-based indices of the user's selected
                          option for each question.

        Returns:
            Dict with score, total, percentage, and per-question results.

        Example:
            >>> result = quiz.score([0, 2, 1, 3, 0])
            >>> print(result["percentage"])
            80.0
        """
        if len(user_answers) != len(self.questions):
            raise ValueError(
                f"Expected {len(self.questions)} answers, "
                f"got {len(user_answers)}."
            )

        results = []
        correct_count = 0

        for i, (question, user_idx) in enumerate(
            zip(self.questions, user_answers)
        ):
            is_correct = (user_idx == question.get_correct_index())
            if is_correct:
                correct_count += 1

            results.append({
                "question_index" : i,
                "topic_tag"      : question.topic_tag,
                "difficulty"     : question.difficulty,
                "user_answer"    : question.options[user_idx].option,
                "correct_answer" : question.get_correct_option().option,
                "is_correct"     : is_correct,
                "explanation"    : question.explanation,
            })

        return {
            "topic"      : self.topic,
            "score"      : correct_count,
            "total"      : self.total_marks,
            "percentage" : round(correct_count / self.total_marks * 100, 1),
            "results"    : results,
        }


# ---------------------------------------------------------------------------
# Flashcard Schemas
# ---------------------------------------------------------------------------

class Flashcard(BaseModel):
    """
    A single flashcard for active recall practice.

    Fields:
        front:     The concept, term, or question on the front of the card.
        back:      The definition, explanation, or answer on the back.
        example:   Optional concrete example to make the concept stick.
        topic_tag: The sub-topic this card covers. Used for filtering.
    """
    front:     str           = Field(
        ...,
        description="The term, concept, or question on the front of the card."
    )
    back:      str           = Field(
        ...,
        description="The definition or explanation on the back of the card."
    )
    example:   Optional[str] = Field(
        default=None,
        description=(
            "An optional concrete example or analogy to make the concept "
            "easier to remember. Leave null if not applicable."
        )
    )
    topic_tag: str           = Field(
        ...,
        description="The specific concept or sub-topic this card covers."
    )


class FlashcardDeck(BaseModel):
    """
    A complete deck of flashcards for a given topic.

    Fields:
        topic:  The topic this deck covers.
        cards:  List of Flashcard objects.
        source_hint: Document(s) the cards were generated from.
    """
    topic:       str            = Field(..., description="The topic this deck covers.")
    cards:       List[Flashcard] = Field(..., description="List of flashcards.")
    source_hint: str            = Field(
        default="",
        description="The source document(s) used to generate this deck."
    )

    @property
    def card_count(self) -> int:
        """Total number of cards in the deck."""
        return len(self.cards)

    def to_csv_rows(self) -> List[dict]:
        """
        Convert the deck to a list of flat dicts for CSV export.

        Columns: topic, front, back, example, topic_tag

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame(deck.to_csv_rows())
            >>> df.to_csv("flashcards.csv", index=False)
        """
        return [
            {
                "topic"     : self.topic,
                "front"     : card.front,
                "back"      : card.back,
                "example"   : card.example or "",
                "topic_tag" : card.topic_tag,
            }
            for card in self.cards
        ]


# ---------------------------------------------------------------------------
# Topic Summary Schema
# ---------------------------------------------------------------------------

class KeyTerm(BaseModel):
    """
    A key term or concept extracted from the study material.

    Fields:
        term:       The technical term or concept name.
        definition: A concise, plain-English definition.
    """
    term:       str = Field(..., description="The key term or concept.")
    definition: str = Field(..., description="A concise definition in plain English.")


class TopicSummary(BaseModel):
    """
    A structured summary of a topic, grounded in the user's documents.

    Fields:
        topic:               The topic being summarised.
        key_points:          3–7 bullet-point takeaways (shown first).
        detailed_summary:    A paragraph-form explanation of the topic.
        key_terms:           Important terms and their definitions.
        suggested_questions: Questions the user should be able to answer
                             after studying this topic. Feeds into quiz generation.
        prerequisites:       Topics the user should understand first.
        source_hint:         Document(s) the summary was drawn from.
    """
    topic:               str          = Field(..., description="The topic being summarised.")
    key_points:          List[str]    = Field(
        ...,
        description="3 to 7 concise bullet-point takeaways from the topic."
    )
    detailed_summary:    str          = Field(
        ...,
        description=(
            "A 2–4 paragraph explanation of the topic in plain language, "
            "as if explaining to a student encountering it for the first time."
        )
    )
    key_terms:           List[KeyTerm] = Field(
        ...,
        description="Important technical terms and their definitions."
    )
    suggested_questions: List[str]    = Field(
        ...,
        description=(
            "3–5 questions a student should be able to answer after "
            "studying this topic. These will be used to generate quiz questions."
        )
    )
    prerequisites:       List[str]    = Field(
        default_factory=list,
        description=(
            "Topics or concepts the user should understand before studying this. "
            "Leave empty if there are no prerequisites."
        )
    )
    source_hint:         str          = Field(
        default="",
        description="The source document(s) used to generate this summary."
    )

    @field_validator("key_points")
    @classmethod
    def must_have_key_points(cls, v: List[str]) -> List[str]:
        """Enforce at least 3 key points."""
        if len(v) < 3:
            raise ValueError(
                f"TopicSummary must have at least 3 key points, got {len(v)}."
            )
        return v


# ---------------------------------------------------------------------------
# Phase 4 — Performance & Recommendation Schemas
# ---------------------------------------------------------------------------

class QuizPerformanceEntry(BaseModel):
    """
    A record of a user's quiz attempt on a specific topic.
    Accumulated in st.session_state across multiple quiz sessions.

    Fields:
        topic:      The quiz topic.
        score:      Number of correct answers.
        total:      Total number of questions.
        percentage: Score as a percentage.
        weak_tags:  topic_tags where the user answered incorrectly.
    """
    topic:      str       = Field(..., description="The quiz topic.")
    score:      int       = Field(..., description="Number of correct answers.")
    total:      int       = Field(..., description="Total questions attempted.")
    percentage: float     = Field(..., description="Score as a percentage (0–100).")
    weak_tags:  List[str] = Field(
        default_factory=list,
        description="topic_tags where the user got questions wrong."
    )


class PerformanceProfile(BaseModel):
    """
    Aggregated performance profile built from all quiz sessions.
    This is the primary input to the recommendation chain in Phase 4.

    Fields:
        strong_topics:    Topics where avg score >= 70%.
        weak_topics:      Topics where avg score < 70%.
        syllabus_topics:  All topics extracted from the syllabus.
        untested_topics:  Syllabus topics not yet quizzed.
        history:          Full list of individual quiz attempts.
        inferred_level:   LLM-inferred overall level based on performance.
    """
    strong_topics:   List[str]                  = Field(default_factory=list)
    weak_topics:     List[str]                  = Field(default_factory=list)
    syllabus_topics: List[str]                  = Field(default_factory=list)
    untested_topics: List[str]                  = Field(default_factory=list)
    history:         List[QuizPerformanceEntry] = Field(default_factory=list)
    inferred_level:  str                        = Field(
        default="intermediate",
        description=(
            "Overall learning level inferred from quiz performance. "
            "One of: beginner, intermediate, advanced."
        )
    )

    def add_quiz_result(self, quiz_result: dict) -> None:
        """
        Update the profile with the result dict returned by Quiz.score().

        Args:
            quiz_result: The dict returned by quiz.score(user_answers).
        """
        entry = QuizPerformanceEntry(
            topic      = quiz_result["topic"],
            score      = quiz_result["score"],
            total      = quiz_result["total"],
            percentage = quiz_result["percentage"],
            weak_tags  = [
                r["topic_tag"]
                for r in quiz_result["results"]
                if not r["is_correct"]
            ],
        )
        self.history.append(entry)
        self._recompute()

    def _recompute(self) -> None:
        """Recompute strong/weak topic lists from full history."""
        topic_scores: dict[str, List[float]] = {}

        for entry in self.history:
            if entry.topic not in topic_scores:
                topic_scores[entry.topic] = []
            topic_scores[entry.topic].append(entry.percentage)

        self.strong_topics = [
            t for t, scores in topic_scores.items()
            if sum(scores) / len(scores) >= 70
        ]
        self.weak_topics = [
            t for t, scores in topic_scores.items()
            if sum(scores) / len(scores) < 70
        ]
        self.untested_topics = [
            t for t in self.syllabus_topics
            if t not in topic_scores
        ]


class SearchQuery(BaseModel):
    """
    A targeted search query generated by the LLM for finding study resources.
    Used internally by the recommendation chain.
    """
    query:     str           = Field(..., description="The search query string.")
    intent:    str           = Field(
        ...,
        description="What this query is trying to find (e.g. 'beginner tutorial on backpropagation')."
    )
    topic:     str           = Field(..., description="The weak topic this query targets.")
    preferred: ResourceType  = Field(
        default=ResourceType.VIDEO,
        description="Preferred resource type for this query."
    )


class SearchQueryList(BaseModel):
    """Wrapper so the LLM returns a list of SearchQuery objects."""
    queries: List[SearchQuery] = Field(
        ...,
        description="List of 3–5 targeted search queries for weak topics."
    )


class ResourceRecommendation(BaseModel):
    """
    A single recommended external resource for the user.

    Fields:
        title:         Display title of the resource.
        url:           Full URL.
        resource_type: video / article / paper / course / documentation.
        reason:        One sentence explaining why this was recommended.
        difficulty:    Estimated difficulty level.
        topic:         The weak topic this resource addresses.
    """
    title:         str           = Field(..., description="Title of the resource.")
    url:           str           = Field(..., description="Full URL to the resource.")
    resource_type: ResourceType  = Field(..., description="Type of resource.")
    reason:        str           = Field(
        ...,
        description="One sentence explaining why this resource is recommended for this user."
    )
    difficulty:    DifficultyLevel = Field(
        ...,
        description="Estimated difficulty level of this resource."
    )
    topic:         str           = Field(
        ...,
        description="The weak topic this resource helps address."
    )


class RecommendationList(BaseModel):
    """
    The final list of ranked resource recommendations returned to the UI.

    Fields:
        recommendations: Top 5 resources, ranked by relevance.
        profile_summary: One-sentence summary of the user's performance
                         shown at the top of the recommendations tab.
    """
    recommendations: List[ResourceRecommendation] = Field(
        ...,
        description="Top 5 recommended resources, ranked by relevance."
    )
    profile_summary: str = Field(
        ...,
        description=(
            "A one-sentence summary of the user's current performance "
            "shown as a header in the recommendations UI."
        )
    )


# ---------------------------------------------------------------------------
# Parser Factory Helpers
# ---------------------------------------------------------------------------

def get_quiz_parser():
    """
    Return a PydanticOutputParser for the Quiz schema,
    wrapped in an OutputFixingParser for automatic error recovery.

    The OutputFixingParser catches malformed JSON from the LLM and
    sends it back with the error message, asking for a corrected version.
    This dramatically improves reliability without any extra code in the chain.

    Usage in quiz_chain.py:
        parser = get_quiz_parser()
        format_instructions = parser.get_format_instructions()
        # inject format_instructions into your prompt template
        quiz: Quiz = parser.parse(llm_output)
    """
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_classic.output_parsers import OutputFixingParser
    from langchain_google_genai import ChatGoogleGenerativeAI 

    base_parser = PydanticOutputParser(pydantic_object=Quiz)
    fixing_llm  = ChatGoogleGenerativeAI (model="gemini-3-flash-preview", temperature=0)
    return OutputFixingParser.from_llm(parser=base_parser, llm=fixing_llm)


def get_flashcard_parser():
    """Return a fixing parser for the FlashcardDeck schema."""
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_classic.output_parsers import OutputFixingParser
    from langchain_google_genai import ChatGoogleGenerativeAI 

    base_parser = PydanticOutputParser(pydantic_object=FlashcardDeck)
    fixing_llm  = ChatGoogleGenerativeAI (model="gemini-3-flash-preview", temperature=0)
    return OutputFixingParser.from_llm(parser=base_parser, llm=fixing_llm)


def get_summary_parser():
    """Return a fixing parser for the TopicSummary schema."""
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_classic.output_parsers import OutputFixingParser
    from langchain_google_genai import ChatGoogleGenerativeAI 

    base_parser = PydanticOutputParser(pydantic_object=TopicSummary)
    fixing_llm  = ChatGoogleGenerativeAI (model="gemini-3-flash-preview", temperature=0)
    return OutputFixingParser.from_llm(parser=base_parser, llm=fixing_llm)


def get_search_query_parser():
    """Return a fixing parser for the SearchQueryList schema (Phase 4)."""
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_classic.output_parsers import OutputFixingParser
    from langchain_google_genai import ChatGoogleGenerativeAI 

    base_parser = PydanticOutputParser(pydantic_object=SearchQueryList)
    fixing_llm  = ChatGoogleGenerativeAI (model="gemini-3-flash-preview", temperature=0)
    return OutputFixingParser.from_llm(parser=base_parser, llm=fixing_llm)


def get_recommendation_parser():
    """Return a fixing parser for the RecommendationList schema (Phase 4)."""
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_classic.output_parsers import OutputFixingParser
    from langchain_google_genai import ChatGoogleGenerativeAI 

    base_parser = PydanticOutputParser(pydantic_object=RecommendationList)
    fixing_llm  = ChatGoogleGenerativeAI (model="gemini-3-flash-preview", temperature=0)
    return OutputFixingParser.from_llm(parser=base_parser, llm=fixing_llm)


# ---------------------------------------------------------------------------
# Quick test — run directly to verify schema validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=== Schema Validation Tests ===\n")

    # Test: Valid Quiz
    quiz = Quiz(
        topic="Gradient Descent",
        questions=[
            MCQuestion(
                question="What does gradient descent minimise?",
                options=[
                    MCQOption(option="The loss function",   is_correct=True),
                    MCQOption(option="The learning rate",   is_correct=False),
                    MCQOption(option="The number of epochs", is_correct=False),
                    MCQOption(option="The batch size",      is_correct=False),
                ],
                explanation="Gradient descent is an optimisation algorithm that iteratively updates model parameters to minimise the loss function.",
                difficulty=DifficultyLevel.BEGINNER,
                topic_tag="gradient_descent",
            )
        ]
    )
    print(f"✅ Quiz created: '{quiz.topic}' ({quiz.total_marks} question(s))")

    # Test: Score a quiz
    result = quiz.score(user_answers=[0])
    print(f"✅ Quiz scored: {result['score']}/{result['total']} ({result['percentage']}%)")

    # Test: FlashcardDeck
    deck = FlashcardDeck(
        topic="Neural Networks",
        cards=[
            Flashcard(
                front="What is a neuron in a neural network?",
                back="A computational unit that takes inputs, applies weights, adds a bias, and passes the result through an activation function.",
                example="Like a biological neuron, it fires (activates) when its inputs are strong enough.",
                topic_tag="neurons",
            )
        ]
    )
    print(f"✅ FlashcardDeck created: '{deck.topic}' ({deck.card_count} card(s))")
    print(f"✅ CSV rows: {deck.to_csv_rows()}")

    # Test: TopicSummary
    summary = TopicSummary(
        topic="Backpropagation",
        key_points=[
            "Computes gradients via the chain rule",
            "Propagates error backwards through the network",
            "Required for training deep neural networks",
        ],
        detailed_summary="Backpropagation is the algorithm used to train neural networks by computing the gradient of the loss function with respect to each weight.",
        key_terms=[
            KeyTerm(term="Chain rule", definition="A calculus rule for computing the derivative of a composite function."),
        ],
        suggested_questions=[
            "What is the role of the chain rule in backpropagation?",
            "How does backpropagation differ from forward propagation?",
        ],
    )
    print(f"✅ TopicSummary created: '{summary.topic}' "
          f"({len(summary.key_points)} key points, "
          f"{len(summary.key_terms)} key terms)")

    # Test: PerformanceProfile
    profile = PerformanceProfile(syllabus_topics=["Gradient Descent", "Backpropagation"])
    profile.add_quiz_result(result)
    print(f"✅ PerformanceProfile updated: "
          f"strong={profile.strong_topics}, weak={profile.weak_topics}")

    # Test: Validator — wrong number of options
    print("\n--- Validator Tests ---")
    try:
        bad_question = MCQuestion(
            question="Test?",
            options=[MCQOption(option="A", is_correct=True)],  # only 1 option
            explanation="Test",
            topic_tag="test",
        )
    except Exception as e:
        print(f"✅ Caught invalid MCQuestion (1 option): {type(e).__name__}")

    # Test: Validator — two correct options
    try:
        bad_question = MCQuestion(
            question="Test?",
            options=[
                MCQOption(option="A", is_correct=True),
                MCQOption(option="B", is_correct=True),   # two correct — invalid
                MCQOption(option="C", is_correct=False),
                MCQOption(option="D", is_correct=False),
            ],
            explanation="Test",
            topic_tag="test",
        )
    except Exception as e:
        print(f"✅ Caught invalid MCQuestion (2 correct): {type(e).__name__}")

    print("\n=== All schema tests passed ===")