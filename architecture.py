"""Architecture tab: diagrams, design principles, limitations."""
import streamlit as st

RAG_DOT = """
digraph { rankdir=LR; node [shape=box style="rounded,filled" fillcolor="#eef2ff" fontname="Helvetica" fontsize=11];
  q [label="User question\\n+ chat history"]; u [label="Query understanding (LLM)\\nrewrite · intent · search queries\\naggregate spec"];
  h [label="Hybrid retrieval\\nbge-small + BM25 (RRF)\\npinned IDs · entity match"]; a [label="Structured query\\npandas (no LLM maths)"];
  m [label="Multi-hop FK expansion\\nuse case → data source →\\ncontrols → models"]; d [label="Governance decision table\\n(DMN-style YAML)" fillcolor="#fef3c7"];
  g [label="Grounded generation (LLM)\\ncite record ids · JSON"]; v [label="Verification\\nnumeric grounding +\\njudge (supported?)" fillcolor="#dcfce7"];
  o [label="Answer + table +\\ncitations + trace"];
  q->u; u->h [label="lookup"]; u->a [label="aggregate"]; u->m [label="multi_hop"]; m->d; h->g; a->g; d->g; g->v; v->o; }
"""

AGENT_DOT = """
digraph { rankdir=LR; node [shape=box style="rounded,filled" fillcolor="#eef2ff" fontname="Helvetica" fontsize=11];
  p [label="Planner (LLM ReAct loop)\\n+ deterministic fallback plan" fillcolor="#e0e7ff"];
  c [label="Calendar\\n(Google Calendar API)"]; e [label="Email\\n(Gmail API read/send)"];
  x [label="Extract (LLM → schema)\\nevery field cites a source quote"]; gr [label="Grounding guard\\nvalue must appear in source" fillcolor="#dcfce7"];
  n [label="Normalise (rules)\\nnames · E.164 · emails · companies"]; dd [label="Deduplicate\\nemail key + fuzzy name"];
  w [label="Verify (web)\\nCompanies House · search · domain"]; dt [label="Record status decision table\\n+ human-review EV rule" fillcolor="#fef3c7"];
  cr [label="Critic\\ninvariants · loop back once" fillcolor="#dcfce7"]; out [label="Cleansed attendee table\\nCSV · audit log · summary email"];
  p->c; p->e; c->x; e->x; x->gr; gr->n; n->dd; dd->w; w->dt; dt->cr; cr->out; cr->p [style=dashed label="fix"]; }
"""


def render():
    st.subheader("Architecture")
    st.markdown("""
**Design principle — the LLM interprets, rules decide.** LLMs handle language: rewriting questions, extracting
fields from messy email. Deterministic components make decisions: pandas for arithmetic, DMN-style decision tables
for policy and record status, validators for data quality. Every decision is reproducible, testable and logged
with the rule id that fired.
""")
    st.markdown("#### Part 1 · RAG pipeline")
    st.graphviz_chart(RAG_DOT, width="stretch")
    st.markdown("#### Part 2 · Agent pipeline")
    st.graphviz_chart(AGENT_DOT, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
#### How the requirements are met
- **Ingestion & chunking:** row-level chunks with schema context + document summary chunks
- **Retrieval:** hybrid dense + BM25 with RRF, ID pinning, FK expansion
- **Query transformation:** history-aware rewrite, intent routing, aggregate → pandas spec
- **Traceability:** every fact cites a record id; retrieved context, scores and route are visible
- **Grounding:** numbers in the answer must exist in sources; judge scores support
- **Agent tools:** Calendar, Gmail (read + send), Companies House API, web search
- **Data quality:** schema extraction, normalisation, dedupe, verification, audit trail
- **Observability:** local step trace in-app + Langfuse traces, token usage and scores
- **Resilience:** retry/backoff, provider fallback, mock data when an integration is missing
""")
    with c2:
        st.markdown("""
#### Limitations & next steps
- **Decision models:** the `Judge` interface is LLM-backed today (verbalised probabilities). Next: a calibrated
  decision model — Cloudflare Clef (open weights, self-hostable) or OpenAI Decisions API — so thresholds and
  human-review routing rest on calibrated probabilities.
- **Evaluation:** grow the labelled set; run it in CI and as Langfuse datasets per prompt/model version
- **Retrieval:** add a cross-encoder reranker; a real vector DB (pgvector) at scale
- **Human in the loop:** approval queue for flagged records before CRM write-back
- **Governance:** free-tier APIs are for synthetic data only; production uses Vertex AI / Azure OpenAI /
  Bedrock with a DPA and UK/EU residency, pinned model versions, PII redaction before LLM calls
- **Ops:** scheduled/triggered runs (new RSVP email → incremental update), alerting on failure rates
""")
