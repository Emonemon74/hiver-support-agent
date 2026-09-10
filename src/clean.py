"""Text normalisation shared across the pipeline."""
from __future__ import annotations

import re

_URL = re.compile(r"https?://\S+")
_MENTION = re.compile(r"@\w+")
_LEAD_HANDLE = re.compile(r"^(?:@\w+\s+)+")
_WS = re.compile(r"\s+")
_AGENT_SIG = re.compile(r"\s*\*[A-Z]{2,3}\s*$")  # Delta agents sign replies "*TCC"


def clean_text(s: str, *, drop_lead_handle: bool = True) -> str:
    if not isinstance(s, str):
        return ""
    if drop_lead_handle:
        s = _LEAD_HANDLE.sub("", s)
    s = _URL.sub("<URL>", s)
    s = _MENTION.sub("@user", s)
    s = _AGENT_SIG.sub("", s)
    return _WS.sub(" ", s).strip()
