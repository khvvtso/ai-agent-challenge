# AI Agent Technical Challenge

One Streamlit app with two parts. Both follow one design principle: **the LLM interprets, and rules decide.**
The LLM handles language: rewriting questions and extracting fields from messy email. Deterministic components
make the decisions: pandas for arithmetic, DMN-style decision tables for policy and record status, and validators
for data quality. Every decision is reproducible, unit-tested and logged with the id of the rule that fired.

| | |
|---|---|
| **Live demo** | _<Streamlit Cloud URL>_ |
| **Stack** | Python 3.12, Streamlit, Gemini via a provider-agnostic layer (Groq fallback), fastembed `bge-small-en-v1.5`, BM25, pandas, Pydantic, phonenumbers, rapidfuzz, Langfuse |
| **Integrations** | Google Calendar API, Gmail API (read + send), UK Companies House API, Tavily / DuckDuckGo web search |
| **Data** | All synthetic. The client "Halden Living" and all attendees are fictional. |

The scenario reflects a typical engagement: a mid-market UK operator (student accommodation) is part-way through an
**AI Assess** phase (Part 1), and a consultancy runs a roundtable event and needs a clean, verified lead list afterwards (Part 2).

---

## Part 1: AI Opportunity Copilot (RAG)

**Knowledge base:** 4 structured documents in `data/kb/`:
- `use_case_register.csv`: 22 use cases with £ benefit and cost, feasibility, data readiness, risk tier, PII
- `data_sources.csv`: source systems, data quality, PII, hosting region
- `governance_controls.csv`: controls by risk tier, mapped to UK GDPR, EU AI Act and ISO 42001
- `model_catalog.json`: approved models, cost, residency, PII approval, max risk tier

**Pipeline** (`rag/`):
1. **Ingestion and chunking:** one chunk per row, written as `column: value` pairs prefixed with the document title,
   so every chunk carries its own schema. There is also one summary chunk per document. Metadata (file, primary key,
   row) supports filtering and citations. Derived fields (`p_success`, `expected_value_gbp`) are computed deterministically.
2. **Query transformation:** an LLM produces a JSON plan: a standalone rewrite that uses chat history (this is how
   context is retained), an intent (`lookup | multi_hop | aggregate | out_of_scope | clarify`), search queries,
   target documents, entities, and a declarative **aggregate spec**.
3. **Routed retrieval:**
   - Hybrid dense + BM25 search fused with Reciprocal Rank Fusion. Explicit IDs (UC-01, DS-04) are pinned, because vectors are poor at matching IDs.
   - `aggregate`: the spec is executed by **pandas**, not by the LLM, so rankings and filters are exact.
   - `multi_hop`: deterministic foreign-key expansion (use case → data source → controls for its tier → models permitted by `decisions/governance.yaml`).
4. **Grounded generation:** the model may answer only from the numbered context, must cite record ids, and returns structured JSON.
5. **Verification:** every number in the answer must exist in the sources, citations must point to retrieved records,
   and a judge scores whether the answer is supported, partially supported or unsupported. All results are shown in the UI and scored in Langfuse.

**Evaluation tab:** a labelled test set (`tests/rag_eval.json`) that measures retrieval recall, answer match, intent accuracy and latency.

## Part 2: Event-to-Lead Agent

**Goal:** turn a calendar event plus messy RSVP emails into a cleansed, verified attendee list.

**Orchestration** (`agent/pipeline.py`):
- An LLM planner (a ReAct-style JSON loop) chooses each tool call.
- If no LLM is available, or the planner errors or loops, a **deterministic fallback plan** runs instead. The trace shows which path ran.

**Steps:**
1. **Calendar:** read the event and its invitees.
2. **Gmail:** search and read the RSVP emails. A judge classifies each email as an RSVP or noise, and the newsletter is ignored.
3. **Extract:** an LLM fills a schema, and **every field must quote its source**.
4. **Grounding guard:** if an extracted value doesn't appear in the cited email, it is set to null and logged as a blocked hallucination.
5. **Normalise:** deterministic rules for names (including nicknames), email syntax, UK phone numbers in E.164 format, and canonical company names.
6. **Deduplicate:** first by email key, then by fuzzy name + company match. Records are merged, and conflicting values are kept as conflicts rather than silently overwritten.
7. **Verify:** check each company against Companies House (active, dissolved or not found), check the email domain against the company's web domain, and check the person against the invite list.
8. **Decide:** `decisions/record_status.yaml` (rules R01–R10) assigns each record verified / corrected / conflict / missing_fields / unverified / declined / not_invited.
   A record goes to human review when `P(wrong) × cost_of_error > review_cost`.
9. **Critic:** checks invariants (unique emails, valid E.164 phone numbers, every verified record matched to a company). If they fail, the agent loops back once.
10. **Output:** a colour-coded table, CSV export, an audit log (before → after → rule/source), and an optional summary email sent through Gmail.

The seeded test data contains duplicates, a nickname, a typo, a missing phone number, conflicting job titles, a decline,
a plus-one, an uninvited RSVP, a dissolved company, a nonexistent company and a newsletter.

## Observability
- **In-app trace:** each step shows its kind (generation, tool, retriever or evaluator), latency, tokens, inputs and outputs, and errors (including retries).
- **Langfuse (optional):** the same spans are sent to Langfuse with model, token usage and scores (`numeric_grounding`, `judge_supported`).

## Resilience
- Retries with backoff on 429 and 5xx errors, then fallback to the second LLM provider.
- Every integration has a mock fallback (`data/mock/`), so the demo can't fail because a key is missing.
- Tool results are cached to save quota.

## Run locally
```bash
uv venv --python 3.12 && uv pip install -r requirements.txt
cp .env.example .env   # add any keys you have; all of them are optional
streamlit run app.py
pytest -q
```

**Gmail and Calendar (optional):**
1. Create a Google Cloud project, enable the Gmail and Calendar APIs, and create a Desktop OAuth client. Save it as `credentials.json`.
2. Run `python scripts/google_auth.py`, which prints the `GOOGLE_*` values for `.env` / Streamlit secrets.
3. Run `python scripts/seed_google.py` to load the synthetic mailbox and event into the demo account.

## Limitations and next steps
- **Decision models:** the `Judge` interface (`core/judge.py`) is backed by an LLM today, which gives verbalised, uncalibrated probabilities.
  It has the same shape as decision-model APIs (Cloudflare Clef, OpenAI Decisions API, Jev), so a calibrated model can be swapped in.
  That would make the confidence thresholds and human-review routing rest on real probabilities.
- **Evaluation:** a larger labelled set, run in CI and as Langfuse datasets for each prompt and model version.
- **Retrieval:** a cross-encoder reranker, and pgvector or another vector database at scale.
- **Human in the loop:** an approval queue for flagged records before writing them to a CRM. Incremental runs triggered by new emails.
- **Governance:** free-tier APIs are suitable for synthetic data only. Production would use Vertex AI, Azure OpenAI or Bedrock under a
  data processing agreement with UK/EU residency, pinned model versions, and PII redaction before any LLM call.
