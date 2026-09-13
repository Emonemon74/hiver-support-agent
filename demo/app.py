"""Streamlit demo for the Delta support agent.

Run locally:   streamlit run demo/app.py
Deploy:        see demo/README.md (Hugging Face Spaces, free tier)

This is a PORTFOLIO DEMO, not a production support tool: it drafts replies
using a public Twitter dataset and a free-tier open model, and it can be wrong.
It is not affiliated with, endorsed by, or connected to Delta Air Lines.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# make `src` importable when Streamlit runs this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config  # noqa: E402,F401 - side effect: loads .env before we check the key

st.set_page_config(page_title="Delta Support Agent — Demo", page_icon="✈️", layout="centered")

# --- secrets: Streamlit/HF secrets take priority over .env for a deployed Space.
#     st.secrets raises if no secrets.toml exists at all (e.g. running locally
#     with only a .env), so this must be guarded rather than a plain "in" check. ---
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

MAX_REQUESTS_PER_SESSION = 12  # protects the shared free-tier quota

st.title("✈️ Delta Support Agent — Demo")
st.caption(
    "A portfolio project: classifies a customer tweet, drafts a reply grounded in "
    "how @Delta historically resolved similar issues, and decides auto-handle vs. "
    "escalate to a human, with a reason."
)
st.warning(
    "**Not affiliated with, endorsed by, or connected to Delta Air Lines.** "
    "This is an unofficial demo built on public data for a take-home assignment "
    "and personal-portfolio purposes. Drafts are illustrative only — hallucination "
    "rate in the eval was ~18%. Do not use this for real customer contact. "
    "[Full write-up →](https://github.com/Emonemon74/hiver-support-agent)",
    icon="⚠️",
)

EXAMPLES = [
    "My flight got cancelled and I need to get to Atlanta tonight for a funeral",
    "Huge thanks to the crew on DL245 today, best service ever!",
    "Does Delta fly nonstop from JFK to Austin?",
    "You charged me $950 for an earlier flight with empty seats?? Unacceptable.",
    "why won't the app let me change my seat even though there are open seats",
]

if "history" not in st.session_state:
    st.session_state.history = []

# a widget's session_state key can't be reassigned after that widget is created
# in the same run, so an example click stages the text here and it's applied
# BEFORE the text_area below is instantiated on the resulting rerun.
if "pending_prefill" in st.session_state:
    st.session_state["message_input"] = st.session_state.pop("pending_prefill")

message = st.text_area(
    "Customer tweet", placeholder="Type a message to @Delta...", height=90,
    key="message_input",
)
submitted = st.button("Run agent", type="primary")

st.caption("Try an example:")
cols = st.columns(len(EXAMPLES))
for c, ex in zip(cols, EXAMPLES):
    if c.button(ex[:18] + "…", key=ex, help=ex):
        st.session_state["pending_prefill"] = ex
        st.rerun()

if submitted:
    if not message.strip():
        st.error("Type a message first.")
    elif len(st.session_state.history) >= MAX_REQUESTS_PER_SESSION:
        st.error(
            f"Demo limit reached ({MAX_REQUESTS_PER_SESSION} runs per session) — "
            "this protects a shared free API quota. Refresh the page to reset, "
            "or run it yourself: the repo is fully reproducible."
        )
    elif not os.environ.get("GROQ_API_KEY"):
        st.error(
            "No GROQ_API_KEY configured for this demo. Set one for free at "
            "console.groq.com and add it as a secret (see demo/README.md)."
        )
    else:
        with st.spinner("Classifying → retrieving precedent → drafting → routing..."):
            try:
                from src.agent.pipeline import run

                result = run(message.strip())
                st.session_state.history.append(result)
            except Exception as e:  # noqa: BLE001 - surface any failure to the visitor
                st.error(f"Something went wrong: {e}")
                result = None

        if result:
            badge = "🔴 ESCALATE" if result.decision == "escalate" else "🟢 AUTO-HANDLE"
            st.subheader(badge)
            st.write(f"**Reason:** {result.reason}  \n*(trigger: `{result.trigger}`)*")

            c1, c2 = st.columns(2)
            c1.metric("Intent", result.intent, f"{result.intent_confidence:.0%} confidence")
            c2.metric("Reply grounded", "Yes" if result.reply_grounded else "No — abstained")

            st.markdown("**Drafted reply:**")
            st.info(result.reply)

            with st.expander("Retrieved precedent (what the draft was grounded in)"):
                for p in result.precedent:
                    st.markdown(f"— *{p['opening']}*")
                    st.markdown(f"  → {p['reply']}  \n  `similarity {p['score']:.2f}`")

st.divider()
st.caption(
    f"{len(st.session_state.history)}/{MAX_REQUESTS_PER_SESSION} runs used this session · "
    "Models: Groq gpt-oss (classify/draft/route) · local embeddings for retrieval · "
    "[full eval + report](https://github.com/Emonemon74/hiver-support-agent/blob/main/REPORT.md)"
)
