# Public Release Policy

This directory contains the public, privacy-minimized release artifacts for VerdictBench.

Raw verdict text, sectioned verdict JSON, reviewed QA rows, retrieved chunks, generated answers, gold evidence paragraphs, annotation databases, and per-query result JSONL are not redistributed. Those files may contain personal data present in public court documents, including names, addresses, occupations, identity-number references, or other party details.

Public reproducibility is supported through:

- aggregate result tables under `aggregate_results/`
- figures and paper tables in the repository
- `verdict_manifest.csv`, which lists verdict identifiers and non-identifying size/stratum metadata
- source code and scripts needed to rebuild the private working dataset from official MKRI sources

`sample_redacted_qa.jsonl` is intentionally withheld until each example is manually reviewed and redacted.
