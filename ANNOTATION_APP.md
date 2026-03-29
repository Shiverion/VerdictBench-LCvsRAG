# Annotation App

This repository now includes a web annotation system for two-annotator QA review.

## What it does

- imports QA items from JSONL
- stores assignments and annotations in SQLite
- assigns each item to exactly 2 annotators
- keeps assignments randomized but load-balanced
- supports `accepted`, `modified`, and `rejected` decisions
- computes per-item agreement summaries
- computes Cohen's kappa from completed 2-rater items

## Backend

The backend lives in [`src/annotation`](src/annotation) and exposes a FastAPI app.

Run it with:

```bash
uv run uvicorn src.annotation.api:app --reload
```

Default API URL:

```text
http://127.0.0.1:8000
```

Environment variables:

- `ANNOTATION_DB_PATH`: SQLite file path
- `ANNOTATION_ALLOWED_ORIGINS`: comma-separated frontend origins

## Frontend

The React app lives in [`annotation-ui/`](annotation-ui).

Run it with:

```bash
cd annotation-ui
npm install
npm run dev
```

Default frontend URL:

```text
http://localhost:5173
```

If the backend runs somewhere else, set:

```bash
VITE_API_BASE=http://127.0.0.1:8000
```

## Suggested workflow

1. Start the FastAPI backend.
2. Start the React frontend.
3. Build a shared overlap set for IAA if needed:

```bash
uv run python scripts/build_iaa_subset.py
```

This writes:

```text
data/qa_dataset/qa_pairs_iaa_30.jsonl
```

4. Open the admin tab and bootstrap the project with:
   - dataset path
   - two annotator IDs
   - `assignments_per_item = 2`

For the paper workflow, use:

```text
data/qa_dataset/qa_pairs_iaa_30.jsonl
```

5. Share annotator IDs with reviewers.
6. Let annotators work through their queue.
7. Use the admin summary to monitor:
   - completed assignments
   - completed overlap items
   - Cohen's kappa
   - per-item agreement outcomes
8. Export the results through the admin API:

```text
POST /api/admin/export
```

This writes:

- `consensus.jsonl`
- `disagreements.jsonl`

## Local consensus export

The simplest local path is:

```bash
uv run python scripts/annotation_iaa_report.py
```

That will:

- read the current annotation database
- export:
  - `data/qa_dataset/annotation_exports/consensus.jsonl`
  - `data/qa_dataset/annotation_exports/disagreements.jsonl`
- print a compact agreement summary including Cohen's kappa

To change the export directory:

```bash
uv run python scripts/annotation_iaa_report.py --output-dir data/qa_dataset/my_export_dir
```

## Current scope

This first version is intentionally narrow:

- local auth is by annotator ID only
- storage is SQLite
- agreement is surfaced in the backend summary view
- disagreements are surfaced but not yet adjudicated through a dedicated UI

The next likely steps are:

- proper login/authentication
- adjudication UI for ties
- export of consensus and disagreement labels back to JSONL
- deployment to a managed Postgres-backed environment

## Render deployment

This repo now includes [`render.yaml`](render.yaml) for a two-service Render deployment:

- `verdictbench-annotation-api`: FastAPI web service
- `verdictbench-annotation-ui`: React static site

### Why the backend uses a paid plan

The current backend stores annotation state in SQLite. For a real deployment, that requires persistent disk storage. The Render blueprint therefore uses a disk-backed web service instead of a stateless free-tier instance.

### Required post-create settings

After creating the services in Render:

1. Copy the frontend URL into `ANNOTATION_ALLOWED_ORIGINS` on the API service
2. Copy the API URL into `VITE_API_BASE` on the static site
3. Trigger a frontend redeploy after setting `VITE_API_BASE`

### Deploy flow

1. Commit and push the repo with `render.yaml`
2. Open Render Blueprint creation
3. Review the two services
4. Fill the two env vars marked as `sync: false`
5. Apply the Blueprint
6. Verify:
   - API health: `/api/health`
   - frontend can load tasks and submit annotations

### Recommended next deployment upgrade

If you want a more durable production setup later, move the annotation store from SQLite to Postgres. That removes the disk dependency and makes concurrency, backups, and multi-instance scaling cleaner.
