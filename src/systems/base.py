"""
Abstract base class for all QA systems.
Every system (LC, Simple RAG, Advanced RAG) implements the same interface
so experiment runners are fully system-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QAResult:
    """
    Standardized output from any QA system.
    All fields are populated before evaluation metrics are added.
    """
    # Inputs
    question_id:    str = ""
    question:       str = ""
    verdict_id:     str = ""

    # System output
    answer:         str = ""
    retrieved_chunks: list[str] = field(default_factory=list)  # empty for LC
    context_used:   str = ""   # full context string injected into prompt

    # System metadata
    condition:      str = ""   # "lc" | "simple_rag" | "advanced_rag" | ablation variant
    model:          str = ""
    chunk_size:     int = 0
    top_k:          int = 0

    # Efficiency
    input_tokens:   int = 0
    cost_usd:       float = 0.0
    latency_s:      float = 0.0

    # Evaluation metrics (populated by batch_evaluator)
    faithfulness:           float | None = None
    hallucination_rate:     float | None = None
    hallucination_flag:     bool  | None = None
    bertscore_f1:           float | None = None
    context_precision:      float | None = None
    context_recall:         float | None = None
    legal_accuracy:         int   | None = None   # 0/1/2 from human review

    # Extra metadata for ablation experiments
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }


class QASystem(ABC):
    """
    Abstract base class for all QA systems.

    Usage:
        system = LongContextSystem(model="gemini-2.5-flash")
        result = system.query(
            question="Apa amar putusan dalam perkara ini?",
            verdict_id="1_PUU-XIX_2021",
            question_id="q001",
        )
    """

    @abstractmethod
    def query(
        self,
        question: str,
        verdict_id: str,
        question_id: str = "",
    ) -> QAResult:
        """
        Answer a question about a specific MK verdict.

        Args:
            question:    Natural language question in Indonesian.
            verdict_id:  File stem matching the processed verdict.
            question_id: QA pair identifier for logging.

        Returns:
            QAResult with answer, context, and efficiency metrics populated.
        """
        ...

    @property
    @abstractmethod
    def condition_name(self) -> str:
        """Short identifier for this system condition, e.g. 'lc' or 'simple_rag'."""
        ...
