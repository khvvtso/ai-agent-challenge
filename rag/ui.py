"""Streamlit tab for Part 1: the AI Opportunity Copilot."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from core import llm
from rag import pipeline
from rag.ingest import DOCS, Index

EXAMPLES = [
    "What are the top 5 use cases by expected value?",
    "Which Finance use cases are blocked by poor data readiness?",
    "What do we need before deploying maintenance request triage, and which models can it use?",
    "And what would it cost per year to run?",
    "What is our Azure budget for next year?",
]


@st.cache_resource(show_spinner="Building hybrid index (row chunks + bge-small embeddings + BM25)...")
def get_index() -> Index:
    return Index()


def _render_trace(trace):
    try:
        from core.ui_trace import render_trace

        render_trace(trace)
    except ImportError:
        for s in trace.spans:
            st.write(f"{'  ' * s.depth}- **{s.name}** ({s.kind}) {s.duration_ms:.0f} ms")


def _render_answer(r: dict):
    res, ver, plan = r["result"], r["verification"], r["plan"]
    st.markdown(res.get("answer", ""))
    if r["extras"].get("table") is not None:
        st.caption(f"Structured query executed by pandas: {r['extras']['table_desc']}")
        st.dataframe(r["extras"]["table"], hide_index=True, width="stretch")
    cols = st.columns(4)
    cols[0].metric("Intent", plan.get("intent", "-"))
    cols[1].metric("Confidence", res.get("confidence", "-"))
    if ver:
        cols[2].metric("Numeric grounding", f"{ver.get('numeric_grounding', 0):.0%}")
        j = ver.get("judge")
        cols[3].metric("Judge: supported", f"{j['probabilities'].get('supported', 0):.0%}" if j else "-")
    if res.get("citations"):
        st.caption("Sources: " + " · ".join(f"`{c}`" for c in res["citations"]))
    if ver.get("unsupported_numbers"):
        st.warning(f"Numbers not found in sources: {', '.join(ver['unsupported_numbers'])}")
    if ver.get("unknown_citations"):
        st.warning(f"Citations not in retrieved context: {', '.join(ver['unknown_citations'])}")

    with st.expander("1 · Query understanding (rewrite + routing)"):
        st.json(plan)
    with st.expander(f"2 · Retrieved context ({len(r['context'])} records)"):
        if r["context"]:
            st.dataframe(pd.DataFrame(r["context"])[["id", "via", "score", "dense", "bm25", "file", "text"]],
                         hide_index=True, width="stretch")
    if r["extras"].get("decisions"):
        with st.expander("3 · Governance decision (DMN table, not LLM)"):
            for g in r["extras"]["decisions"]:
                st.json(g)
    with st.expander("4 · Grounding verification"):
        st.json(ver)
    if st.toggle(f"5 · Show trace · {r['trace'].total_ms:.0f} ms · {r['trace'].total_tokens} tokens", key=f"tr-{r['trace'].id}"):
        with st.container(border=True):
            if r["trace"].lf_url:
                st.markdown(f"[Open in Langfuse]({r['trace'].lf_url})")
            _render_trace(r["trace"])


def _chat(index: Index):
    if not llm.available():
        st.info("No LLM key configured: retrieval runs fully; answers show the retrieved records. Add GEMINI_API_KEY to enable generation.")
    st.session_state.setdefault("rag_msgs", [])
    for m in st.session_state.rag_msgs:
        with st.chat_message(m["role"]):
            if m["role"] == "assistant" and "run" in m:
                _render_answer(m["run"])
            else:
                st.markdown(m["content"])

    st.caption("Try:")
    picked = None
    bcols = st.columns(len(EXAMPLES))
    for i, ex in enumerate(EXAMPLES):
        if bcols[i].button(ex, key=f"ex{i}", width="stretch"):
            picked = ex
    q = st.chat_input("Ask about Halden Living's AI use cases, data, controls or models") or picked
    c1, _ = st.columns([1, 5])
    if c1.button("Clear chat"):
        st.session_state.rag_msgs = []
        st.rerun()
    if q:
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.rag_msgs]
        st.session_state.rag_msgs.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Understanding → retrieving → generating → verifying..."):
                r = pipeline.answer(index, q, history)
            _render_answer(r)
        st.session_state.rag_msgs.append({"role": "assistant", "content": r["result"].get("answer", ""), "run": r})


def _kb(index: Index):
    st.markdown("""**Ingestion:** each structured file is loaded with pandas; every **row becomes one chunk**,
serialised as `column: value` pairs prefixed with the document title so the chunk carries its own schema.
One summary chunk per document supports routing. Chunks are embedded locally with **bge-small-en-v1.5**
and indexed in **BM25**; retrieval fuses both with Reciprocal Rank Fusion. Derived fields
(`p_success`, `expected_value_gbp`) are computed deterministically at ingestion.""")
    m = st.columns(3)
    m[0].metric("Documents", len(DOCS))
    m[1].metric("Chunks", len(index.chunks))
    m[2].metric("Embedding dims", index.vectors.shape[1])
    name = st.selectbox("Document", list(DOCS), format_func=lambda n: f"{DOCS[n]['title']} ({DOCS[n]['file']})")
    st.dataframe(index.tables[name], hide_index=True, width="stretch")
    with st.expander("Example chunk"):
        st.code(next(c.text for c in index.chunks if c.doc == name and c.meta["key"] != "_summary"))


def _eval(index: Index):
    cases = json.loads((Path(__file__).resolve().parent.parent / "tests" / "rag_eval.json").read_text())
    st.markdown("Labelled test set. **Retrieval recall** = expected source records retrieved; "
                "**answer match** = expected facts present in the answer (needs an LLM key).")
    if st.button(f"Run eval ({len(cases)} questions)"):
        rows = []
        bar = st.progress(0.0)
        for i, c in enumerate(cases):
            r = pipeline.answer(index, c["q"], [])
            got = {h["id"] for h in r["context"]} | set(r["result"].get("citations") or [])
            recall = (len(set(c["expect_ids"]) & got) / len(c["expect_ids"])) if c["expect_ids"] else 1.0
            ans = r["result"].get("answer", "").replace(",", "")
            if c.get("expect_not_in_kb"):
                match = bool(r["result"].get("not_in_kb")) or r["plan"].get("intent") == "out_of_scope"
            else:
                match = all(t.replace(",", "").lower() in ans.lower() for t in c["expect_text"])
            rows.append({"question": c["q"], "expected_intent": c["intent"], "intent": r["plan"].get("intent"),
                         "retrieval_recall": round(recall, 2), "answer_match": match,
                         "numeric_grounding": r["verification"].get("numeric_grounding"),
                         "latency_ms": round(r["trace"].total_ms), "tokens": r["trace"].total_tokens})
            bar.progress((i + 1) / len(cases))
        df = pd.DataFrame(rows)
        m = st.columns(4)
        m[0].metric("Retrieval recall", f"{df.retrieval_recall.mean():.0%}")
        m[1].metric("Answer match", f"{df.answer_match.mean():.0%}")
        m[2].metric("Intent accuracy", f"{(df.intent == df.expected_intent).mean():.0%}")
        m[3].metric("Avg latency", f"{df.latency_ms.mean():.0f} ms")
        st.dataframe(df, hide_index=True, width="stretch")


def render():
    st.subheader("Part 1 · AI Opportunity Copilot (RAG)")
    st.caption("Knowledge base: a fictional UK student-accommodation operator, Halden Living, mid-way through an "
               "AI Assess engagement. 4 structured documents: use-case register, data sources, governance controls, model catalog.")
    index = get_index()
    t1, t2, t3 = st.tabs(["💬 Ask", "📚 Knowledge base & ingestion", "🧪 Evaluation"])
    with t1:
        _chat(index)
    with t2:
        _kb(index)
    with t3:
        _eval(index)
