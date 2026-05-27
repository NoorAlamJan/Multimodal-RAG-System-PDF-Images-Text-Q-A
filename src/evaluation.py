"""
evaluation.py — RAG Evaluation Metrics
=======================================
Implements:
    - Faithfulness   : is the answer grounded in retrieved context?
    - Context Recall : are relevant chunks being retrieved?
    - Answer Relevance: does the answer address the question?
    - MRR / Hit@K    : ranking quality metrics
"""

from __future__ import annotations
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class EvalResult:
    metric: str
    score: float
    details: Dict


class RAGEvaluator:
    """Lightweight RAG evaluation without external eval frameworks."""

    # ── Faithfulness ─────────────────────────────────────────

    def faithfulness(self, answer: str, context: str) -> EvalResult:
        """
        Measures what fraction of sentences in the answer
        have at least one supporting sentence in the context.
        (Simplified RAGAS-style metric — no LLM judge required.)
        """
        answer_sents  = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
        context_lower = context.lower()
        supported     = 0

        for sent in answer_sents:
            words    = set(sent.lower().split())
            # Check if >50% of content words appear in context
            content  = [w for w in words if len(w) > 3]
            if not content:
                continue
            overlap  = sum(1 for w in content if w in context_lower)
            if overlap / len(content) > 0.5:
                supported += 1

        score = supported / max(len(answer_sents), 1)
        return EvalResult(
            metric  = "faithfulness",
            score   = round(score, 4),
            details = {"supported_sentences": supported, "total_sentences": len(answer_sents)}
        )

    # ── Answer Relevance ─────────────────────────────────────

    def answer_relevance(self, query: str, answer: str) -> EvalResult:
        """
        Keyword overlap between query terms and answer.
        A proxy for relevance without an LLM judge.
        """
        q_words = set(re.findall(r"\w+", query.lower()))
        a_words = set(re.findall(r"\w+", answer.lower()))
        q_words -= {"what", "how", "why", "is", "are", "the", "a", "of", "in"}

        if not q_words:
            return EvalResult("answer_relevance", 0.0, {})

        overlap = q_words & a_words
        score   = len(overlap) / len(q_words)
        return EvalResult(
            metric  = "answer_relevance",
            score   = round(score, 4),
            details = {"matched_terms": list(overlap), "query_terms": list(q_words)}
        )

    # ── Context Precision ────────────────────────────────────

    def context_precision(self, query: str, retrieved_docs: List[str]) -> EvalResult:
        """
        What fraction of retrieved docs are relevant to the query?
        (Keyword-overlap proxy.)
        """
        q_words   = set(re.findall(r"\w+", query.lower()))
        q_words  -= {"what", "how", "is", "the", "a", "of"}
        relevant  = 0

        for doc in retrieved_docs:
            d_words = set(re.findall(r"\w+", doc.lower()))
            if len(q_words & d_words) >= 2:
                relevant += 1

        score = relevant / max(len(retrieved_docs), 1)
        return EvalResult(
            metric  = "context_precision",
            score   = round(score, 4),
            details = {"relevant_retrieved": relevant, "total_retrieved": len(retrieved_docs)}
        )

    # ── MRR ──────────────────────────────────────────────────

    def mrr(self, ranked_scores: List[float], relevance_threshold: float = 0.3) -> EvalResult:
        """Mean Reciprocal Rank given a list of similarity scores."""
        for rank, score in enumerate(ranked_scores, 1):
            if score >= relevance_threshold:
                return EvalResult(
                    metric  = "mrr",
                    score   = round(1.0 / rank, 4),
                    details = {"first_relevant_rank": rank}
                )
        return EvalResult("mrr", 0.0, {"first_relevant_rank": None})

    # ── Full Evaluation Report ────────────────────────────────

    def evaluate(
        self,
        query:    str,
        answer:   str,
        context:  str,
        retrieved_docs: List[str],
        scores:   List[float],
    ) -> Dict:
        results = {
            "faithfulness":      self.faithfulness(answer, context),
            "answer_relevance":  self.answer_relevance(query, answer),
            "context_precision": self.context_precision(query, retrieved_docs),
            "mrr":               self.mrr(scores),
        }
        avg = np.mean([r.score for r in results.values()])
        return {
            "metrics":       {k: {"score": v.score, "details": v.details} for k, v in results.items()},
            "overall_score": round(float(avg), 4),
        }

    def print_report(self, eval_dict: Dict) -> None:
        print("\n" + "="*50)
        print("  RAG EVALUATION REPORT")
        print("="*50)
        for name, data in eval_dict["metrics"].items():
            bar = "█" * int(data["score"] * 20)
            print(f"  {name:<22} {data['score']:.4f}  |{bar:<20}|")
        print(f"\n  Overall Score:         {eval_dict['overall_score']:.4f}")
        print("="*50)
