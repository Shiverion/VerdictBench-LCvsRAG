from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


AnnotationStatus = Literal["accepted", "modified", "rejected"]


class AnnotatorCreate(BaseModel):
    annotator_id: str = Field(min_length=2, max_length=64)
    display_name: str = Field(min_length=2, max_length=120)


class BootstrapRequest(BaseModel):
    dataset_path: str = "data/qa_dataset/qa_drafts_raw.jsonl"
    annotator_ids: list[str] = Field(min_length=2)
    assignments_per_item: int = Field(default=2, ge=2)
    seed: int = 42
    reset_existing: bool = False


class AnnotationItem(BaseModel):
    question_id: str
    verdict_id: str
    question: str
    gold_answer: str
    gold_paragraphs: list[str]
    question_type: str
    source_status: str | None = None


class AssignmentView(BaseModel):
    assignment_id: int
    annotator_id: str
    item: AnnotationItem


class AnnotationSubmit(BaseModel):
    assignment_id: int
    annotator_id: str
    status: AnnotationStatus
    edited_question: str | None = None
    edited_gold_answer: str | None = None
    edited_gold_paragraphs: list[str] | None = None
    notes: str | None = None


class ExportRequest(BaseModel):
    output_dir: str = "data/qa_dataset/annotation_exports"


class AnnotatorSummary(BaseModel):
    annotator_id: str
    display_name: str
    assigned: int
    completed: int


class AgreementSummary(BaseModel):
    question_id: str
    verdict_id: str
    votes: dict[str, int]
    consensus_status: AnnotationStatus | None
    is_tie: bool


class AdminSummary(BaseModel):
    total_items: int
    total_assignments: int
    completed_assignments: int
    completed_items: int
    cohen_kappa: float | None
    annotators: list[AnnotatorSummary]
    status_counts: dict[str, int]
    agreement_items: list[AgreementSummary]
