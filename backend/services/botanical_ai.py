"""
Botanical AI service.

Answers user questions about a specific plant by grounding an LLM call
in the trusted plant data from plant_service.py. The LLM is instructed
to rely on the supplied context rather than inventing botanical facts.

If LLM_API_KEY is not configured, or the LLM call fails/times out, this
falls back to a deterministic, context-derived answer so the feature
keeps working (mocking/fallback rule).
"""

import logging
import os

from services import plant_service

logger = logging.getLogger(__name__)

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "15"))

SYSTEM_PROMPT = (
    "You are the Botanical Guide inside Native Flora AI, a biodiversity "
    "field app. You answer questions about ONE specific plant using ONLY "
    "the 'Plant context' block provided below. "
    "Explain things simply and accessibly, focus on ecological importance "
    "and conservation, and keep answers concise (2-5 sentences unless the "
    "question needs more detail). "
    "If the context does not contain the information needed to answer, "
    "say plainly that you don't have that information rather than "
    "guessing or inventing facts. Do not contradict the provided context."
)

_client = None


def _get_client():
    """Lazily construct the Anthropic client so import-time failures
    (e.g. missing package/key during local dev) don't crash the app."""
    global _client
    if _client is None and LLM_API_KEY:
        import anthropic

        _client = anthropic.Anthropic(api_key=LLM_API_KEY)
    return _client


def _fallback_answer(context: str, question: str) -> str:
    """Deterministic answer used when no LLM is configured or the call fails."""
    return (
        "I don't have live AI access right now, so here's what I know about "
        f"this plant from verified data:\n\n{context}\n\n"
        "Try asking again shortly for a more tailored answer to: "
        f"\"{question}\""
    )


def answer_question(plant_id: str, question: str) -> str:
    """
    Build grounded context for plant_id and answer `question` using the
    configured LLM. Returns a plain-text answer; never raises for LLM
    failures (falls back to a safe canned response instead).

    Callers are responsible for validating that plant_id exists and that
    question is non-empty before calling this.
    """
    context = plant_service.build_botanical_context(plant_id)
    if context is None:
        raise ValueError(f"Unknown plant_id: {plant_id}")

    client = _get_client()
    if client is None:
        logger.info("LLM_API_KEY not configured; using fallback answer.")
        return _fallback_answer(context, question)

    try:
        message = client.messages.create(
            model=LLM_MODEL,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Plant context:\n{context}\n\n"
                        f"User question: {question}"
                    ),
                }
            ],
            timeout=LLM_TIMEOUT_SECONDS,
        )
        text_blocks = [
            block.text for block in message.content if getattr(block, "type", None) == "text"
        ]
        answer = "\n".join(text_blocks).strip()
        return answer or _fallback_answer(context, question)
    except Exception:
        logger.exception("LLM call failed for plant_id=%s", plant_id)
        return _fallback_answer(context, question)
