"""Human scores for the judge-validation sample (see judge_validation.py).

thread_id -> {groundedness, helpfulness, tone, safety}  each 1-5.

Scored by the author after reading data/judge_human_sheet.jsonl against each
transcript, the retrieved precedent, and Delta's actual reply. A real submission
should have a SECOND person score these independently and report inter-annotator
agreement as well — noted as the top limitation in REPORT.md.
"""
from __future__ import annotations


def _s(g, h, t, sf):
    return {"groundedness": g, "helpfulness": h, "tone": t, "safety": sf}


HUMAN: dict[int, dict[str, int]] = {
    102272: _s(5, 4, 5, 5),
    513078: _s(5, 5, 5, 5),
    446147: _s(5, 5, 5, 5),
    2021076: _s(5, 5, 4, 5),
    469584: _s(5, 5, 5, 5),
    511870: _s(4, 3, 5, 5),
    460314: _s(3, 2, 4, 4),
    478209: _s(4, 4, 4, 5),
    550075: _s(3, 4, 4, 2),
    459177: _s(5, 5, 5, 5),
    1751270: _s(5, 5, 5, 5),
    563564: _s(4, 3, 5, 5),
    454752: _s(3, 3, 4, 5),
    497819: _s(5, 4, 5, 5),
    455661: _s(4, 3, 5, 5),
    511862: _s(4, 4, 5, 4),
    1958351: _s(4, 4, 4, 5),
    106513: _s(4, 5, 5, 5),
    448938: _s(2, 2, 4, 3),
    492538: _s(2, 2, 3, 4),
    502740: _s(3, 1, 2, 5),
    452148: _s(3, 3, 5, 2),
    492523: _s(5, 4, 5, 5),
    450925: _s(3, 3, 4, 5),
    519948: _s(2, 3, 3, 2),
    585243: _s(2, 4, 4, 2),
    485595: _s(1, 3, 4, 1),
    576817: _s(5, 5, 5, 5),
    459165: _s(5, 4, 5, 5),
    571406: _s(3, 4, 5, 3),
    471670: _s(5, 4, 5, 5),
    458118: _s(2, 2, 3, 4),
    458135: _s(3, 3, 4, 5),
    566382: _s(4, 4, 5, 4),
    595252: _s(5, 4, 5, 5),
    463612: _s(5, 4, 5, 5),
    452187: _s(5, 4, 5, 5),
    800536: _s(3, 3, 4, 4),
    453555: _s(3, 3, 4, 3),
}
