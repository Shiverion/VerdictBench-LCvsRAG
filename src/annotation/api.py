from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.annotation.db import session
from src.annotation.schemas import AnnotatorCreate, AnnotationSubmit, BootstrapRequest, ExportRequest
from src.annotation.service import (
    bootstrap_annotation_project,
    build_admin_summary,
    create_annotator,
    ensure_ready,
    export_review_outputs,
    get_next_assignment,
    list_annotators,
    submit_annotation,
)


def create_app() -> FastAPI:
    app = FastAPI(title="VerdictBench Annotation API", version="0.1.0")

    default_origins = "http://localhost:5173,http://127.0.0.1:5173"
    allowed_origins = [
        origin.strip()
        for origin in os.getenv("ANNOTATION_ALLOWED_ORIGINS", default_origins).split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def startup() -> None:
        with session() as conn:
            ensure_ready(conn)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/annotators")
    def annotators() -> list[dict]:
        with session() as conn:
            return list_annotators(conn)

    @app.post("/api/annotators")
    def create_annotator_endpoint(payload: AnnotatorCreate) -> dict:
        with session() as conn:
            create_annotator(conn, payload)
            return {"ok": True, "annotator": payload.model_dump()}

    @app.post("/api/admin/bootstrap")
    def bootstrap(payload: BootstrapRequest) -> dict:
        with session() as conn:
            try:
                return bootstrap_annotation_project(conn, payload)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/admin/summary")
    def summary() -> dict:
        with session() as conn:
            return build_admin_summary(conn).model_dump()

    @app.post("/api/admin/export")
    def export(payload: ExportRequest) -> dict:
        with session() as conn:
            return export_review_outputs(conn, payload.output_dir)

    @app.get("/api/tasks/next/{annotator_id}")
    def next_task(annotator_id: str) -> dict:
        with session() as conn:
            assignment = get_next_assignment(conn, annotator_id)
            return {"assignment": assignment.model_dump() if assignment else None}

    @app.post("/api/annotations")
    def annotate(payload: AnnotationSubmit) -> dict:
        with session() as conn:
            try:
                submit_annotation(
                    conn,
                    assignment_id=payload.assignment_id,
                    annotator_id=payload.annotator_id,
                    status=payload.status,
                    edited_question=payload.edited_question,
                    edited_gold_answer=payload.edited_gold_answer,
                    edited_gold_paragraphs=payload.edited_gold_paragraphs,
                    notes=payload.notes,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            assignment = get_next_assignment(conn, payload.annotator_id)
            return {"ok": True, "next_assignment": assignment.model_dump() if assignment else None}

    return app


app = create_app()
