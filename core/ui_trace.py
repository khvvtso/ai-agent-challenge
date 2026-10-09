"""Streamlit renderer for a Trace: a step timeline with timings, tokens and errors."""
from __future__ import annotations

import json

import streamlit as st

from core.trace import Trace

ICONS = {"generation": "🧠", "tool": "🔧", "retriever": "🔎", "evaluator": "⚖️", "guardrail": "🛡️",
         "chain": "⛓️", "agent": "🤖", "span": "•"}


def _fmt(v) -> str:
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, indent=2, default=str)[:6000]
    except Exception:
        return str(v)[:6000]


def render_trace(trace: Trace, title: str = "Trace") -> None:
    """Do not call inside an st.expander (Streamlit forbids nested expanders)."""
    if not trace:
        return
    llm_calls = sum(1 for s in trace.spans if s.kind == "generation")
    errors = sum(1 for s in trace.spans if s.error)
    st.markdown(f"**{title}** · `{trace.id}` · {len(trace.spans)} spans · {llm_calls} LLM calls · "
                f"{trace.total_tokens:,} tokens · {trace.total_ms/1000:.1f}s" + (f" · ⚠️ {errors} errors" if errors else ""))
    if trace.scores:
        st.caption("Scores: " + " · ".join(f"{k} = {v}" for k, v in trace.scores.items()))
    if trace.lf_url:
        st.markdown(f"[Open full trace in Langfuse ↗]({trace.lf_url})")
    for s in trace.spans:
        tokens = s.metadata.get("usage", {}).get("total")
        label = (f"{' ' * s.depth}{ICONS.get(s.kind, '•')} {s.name} — {s.duration_ms:,.0f} ms"
                 + (f" · {tokens} tok" if tokens else "") + (" · ❌" if s.error else ""))
        with st.expander(label):
            if s.error:
                st.error(s.error)
            meta = {k: v for k, v in s.metadata.items()}
            if meta:
                st.caption("metadata")
                st.code(_fmt(meta), language="json")
            if s.input is not None:
                st.caption("input")
                st.code(_fmt(s.input), language="json")
            if s.output is not None:
                st.caption("output")
                st.code(_fmt(s.output), language="json")
