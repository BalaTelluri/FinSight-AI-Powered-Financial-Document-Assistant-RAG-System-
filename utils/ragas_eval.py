from __future__ import annotations
from typing import List
from groq import Groq

# ── RAGAS-style evaluation using LLM-as-judge ─────────────────────────────────
# We use Groq LLM to score each metric (0.0 to 1.0)
# This is the same approach used by the RAGAS framework

FAITHFULNESS_PROMPT = """You are evaluating an AI answer for faithfulness.

FAITHFULNESS means: Is every claim in the answer supported by the context?
Score from 0.0 to 1.0:
- 1.0 = Every claim is directly supported by context
- 0.5 = Some claims supported, some not
- 0.0 = Answer contains claims not in context (hallucination)

Context:
{context}

Answer:
{answer}

Respond with ONLY a number between 0.0 and 1.0. Nothing else."""

RELEVANCY_PROMPT = """You are evaluating an AI answer for relevancy.

ANSWER RELEVANCY means: Does the answer directly address the question asked?
Score from 0.0 to 1.0:
- 1.0 = Answer perfectly addresses the question
- 0.5 = Answer partially addresses the question
- 0.0 = Answer is off-topic or does not address the question

Question:
{question}

Answer:
{answer}

Respond with ONLY a number between 0.0 and 1.0. Nothing else."""

CONTEXT_PRECISION_PROMPT = """You are evaluating retrieved context for precision.

CONTEXT PRECISION means: Is the retrieved context relevant to the question?
Score from 0.0 to 1.0:
- 1.0 = All retrieved context is highly relevant to the question
- 0.5 = Some context is relevant, some is not
- 0.0 = Retrieved context is not relevant to the question

Question:
{question}

Retrieved Context:
{context}

Respond with ONLY a number between 0.0 and 1.0. Nothing else."""


def _get_score(client: Groq, prompt: str) -> float:
    """Ask Groq to return a score."""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10
        )
        score_text = response.choices[0].message.content.strip()
        score = float(score_text)
        return round(min(max(score, 0.0), 1.0), 2)
    except:
        return 0.0


def evaluate_ragas(
    client: Groq,
    question: str,
    answer: str,
    context: str
) -> dict:
    """
    Run all 3 RAGAS metrics.
    Returns dict with scores and overall quality.
    """
    faithfulness = _get_score(
        client,
        FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    )
    relevancy = _get_score(
        client,
        RELEVANCY_PROMPT.format(question=question, answer=answer)
    )
    precision = _get_score(
        client,
        CONTEXT_PRECISION_PROMPT.format(question=question, context=context)
    )

    # Overall score = weighted average
    overall = round((faithfulness * 0.4 + relevancy * 0.4 + precision * 0.2), 2)

    return {
        "faithfulness":      round(faithfulness * 100, 1),
        "answer_relevancy":  round(relevancy * 100, 1),
        "context_precision": round(precision * 100, 1),
        "overall":           round(overall * 100, 1)
    }
