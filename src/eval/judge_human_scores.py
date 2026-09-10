"""Human scores for the judge-validation sample (see judge_validation.py).

thread_id -> {groundedness, helpfulness, tone, safety}  each 1-5.

Populated by the author after reading data/judge_human_sheet.jsonl. A real
submission should have a SECOND person score these independently and report
inter-annotator agreement as well.
"""
from __future__ import annotations

HUMAN: dict[int, dict[str, int]] = {}
