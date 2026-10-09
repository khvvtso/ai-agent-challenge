"""RAG pipeline: understand -> retrieve (route) -> generate -> verify.

1. Query understanding (LLM, JSON): rewrites follow-ups into standalone questions using
   chat history, classifies intent, emits search queries and - for aggregates - a
   declarative query spec that is executed by pandas (LLMs are bad at arithmetic over chunks).
2. Retrieval: hybrid search + pinned IDs + entity matches + multi-hop FK expansion.
3. Generation: answer only from numbered context, cite chunk ids, structured JSON out.
4. Verification: deterministic numeric grounding check + judge (supported/partial/unsupported).
"""
from __future__ import annotations

import json
import re

import pandas as pd

from core import judge, llm
from core.trace import Trace
from rag import retrieve
from rag.ingest import DOCS, Chunk, Index

INTENTS = ["lookup", "multi_hop", "aggregate", "out_of_scope", "clarify"]


def schema_card(index: Index) -> str:
    lines = []
    for name, df in index.tables.items():
        cols = []
        for c in df.columns:
            vals = df[c].dropna().unique()
            if not pd.api.types.is_numeric_dtype(df[c]) and len(vals) <= 8:
                cols.append(f"{c} {{{', '.join(map(str, vals))}}}")
            else:
                cols.append(f"{c} ({df[c].dtype})")
        lines.append(f"- {name}: {DOCS[name]['about']}\n  columns: {'; '.join(cols)}")
    return "\n".join(lines)


UNDERSTAND_SYSTEM = """You translate user questions into a retrieval plan for a knowledge base of a fictional
UK student-accommodation operator, Halden Living (AI use cases, data sources, governance controls, approved models).
Tables:
{schema}

Return JSON only:
{{"standalone_question": "<question rewritten to be self-contained using chat history>",
 "intent": one of {intents},
 "search_queries": ["1-3 short semantic search queries"],
 "target_docs": ["subset of: use_cases, data_sources, controls, models"],
 "entities": ["named use cases, systems, IDs mentioned"],
 "aggregate": null or {{"table": "<table>", "filters": [{{"column": "...", "op": "eq|ne|gt|gte|lt|lte|in|contains", "value": ...}}],
                        "sort_by": "<column or null>", "ascending": false, "limit": 10, "columns": ["columns to show"],
                        "metrics": [{{"column": "<numeric column>", "fn": "sum|mean|min|max|count"}}]}},
 "clarifying_question": null or "<question to ask if the request is ambiguous>"}}
Rules: 'aggregate' for ranking/filtering/counting/totals across rows; put any total/average/count in "metrics" (computed by pandas over ALL filtered rows) (use expected_value_gbp for "priority"/"value" rankings).
'multi_hop' when the answer needs joining a use case to its data source, controls or permitted models (e.g. "what do I need to deploy X", "which model can X use").
'out_of_scope' when the KB cannot contain the answer. 'clarify' only if genuinely ambiguous."""


def understand(index: Index, question: str, history: list[dict], trace: Trace) -> dict:
    with trace.span("query_understanding", "chain", input={"question": question, "history_turns": len(history)}) as s:
        if not llm.available():
            plan = {"standalone_question": question, "intent": "lookup", "search_queries": [question],
                    "target_docs": [], "entities": [], "aggregate": None, "clarifying_question": None,
                    "fallback": "no LLM configured - heuristic plan"}
        else:
            hist = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in history[-6:]) or "(none)"
            plan = llm.complete_json(
                UNDERSTAND_SYSTEM.format(schema=schema_card(index), intents=INTENTS),
                f"CHAT HISTORY:\n{hist}\n\nUSER QUESTION: {question}", trace, "rewrite_and_route")
            if plan.get("intent") not in INTENTS:
                plan["intent"] = "lookup"
        s.update(plan)
        return plan


def run_aggregate(index: Index, spec: dict) -> tuple[pd.DataFrame, str]:
    df = index.tables[spec["table"]].copy()
    desc = []
    for f in spec.get("filters") or []:
        col, op, val = f["column"], f["op"], f["value"]
        if col not in df.columns:
            continue
        s = df[col]
        if op == "contains":
            m = s.astype(str).str.contains(str(val), case=False)
        elif op == "in":
            m = s.isin(val if isinstance(val, list) else [val])
        elif pd.api.types.is_numeric_dtype(s):
            val = float(val)
            m = {"eq": s == val, "ne": s != val, "gt": s > val, "gte": s >= val, "lt": s < val, "lte": s <= val}[op]
        else:
            m = s.astype(str).str.lower() == str(val).lower() if op == "eq" else s.astype(str).str.lower() != str(val).lower()
        df = df[m]
        desc.append(f"{col} {op} {val}")
    metrics = {}
    for m in spec.get("metrics") or []:
        col, fn = m.get("column"), m.get("fn")
        if fn == "count":
            metrics[f"count_rows"] = int(len(df))
        elif col in df.columns and fn in ("sum", "mean", "min", "max") and pd.api.types.is_numeric_dtype(df[col]):
            v = getattr(df[col], fn)()
            metrics[f"{fn}_{col}"] = round(float(v), 2) if fn == "mean" else int(v) if float(v).is_integer() else float(v)
    if spec.get("sort_by") in df.columns:
        df = df.sort_values(spec["sort_by"], ascending=bool(spec.get("ascending")))
        desc.append(f"sorted by {spec['sort_by']} {'asc' if spec.get('ascending') else 'desc'}")
    df = df.head(int(spec.get("limit") or 10))
    key = DOCS[spec["table"]]["key"]
    cols = [c for c in (spec.get("columns") or []) if c in df.columns]
    cols = [key] + [c for c in cols if c != key] if cols else list(df.columns)
    return df[cols], "; ".join(desc) or "no filters", metrics


def retrieve_context(index: Index, plan: dict, trace: Trace) -> tuple[list[dict], dict]:
    extras: dict = {}
    found: dict[str, dict] = {}

    def add(c: Chunk, score, via: str, signals=None):
        if c.id not in found:
            found[c.id] = {"id": c.id, "doc": c.doc, "file": c.meta["file"], "text": c.text, "score": score,
                           "via": via, **(signals or {})}

    with trace.span("retrieval", "retriever", input={"queries": plan.get("search_queries"), "docs": plan.get("target_docs")}) as s:
        text = " ".join([plan.get("standalone_question", "")] + (plan.get("entities") or []))
        for c in retrieve.pinned_ids(index, text):
            add(c, 99.0, "pinned-id")
        for c in retrieve.name_matches(index, plan.get("entities") or []):
            add(c, 50.0, "entity-match")
        for q in (plan.get("search_queries") or [plan.get("standalone_question")])[:3]:
            for c, score, sig in retrieve.hybrid_search(index, q, k=5, docs=plan.get("target_docs") or None):
                add(c, score, "hybrid", sig)
        s.update({"hits": [(h["id"], h["via"], h["score"]) for h in found.values()]})

    if plan.get("intent") == "aggregate" and plan.get("aggregate"):
        with trace.span("structured_query", "tool", input=plan["aggregate"]) as s:
            try:
                df, desc, metrics = run_aggregate(index, plan["aggregate"])
                extras["table"] = df
                extras["table_desc"] = desc
                extras["metrics"] = metrics
                key = DOCS[plan["aggregate"]["table"]]["key"]
                for k in df[key]:
                    c = index.by_id(f"{plan['aggregate']['table']}:{k}")
                    if c:
                        add(c, 80.0, "structured-query")
                s.update({"rows": len(df), "query": desc, "metrics": metrics})
            except Exception as e:
                s.update({"error": str(e)})

    if plan.get("intent") == "multi_hop":
        with trace.span("multi_hop_expansion", "chain") as s:
            ucs = [index.by_id(h["id"]) for h in found.values() if h["doc"] == "use_cases" and h["via"] != "hybrid"]
            if not ucs:
                ucs = [index.by_id(h["id"]) for h in found.values() if h["doc"] == "use_cases"][:1]
            gates = []
            for uc in ucs[:2]:
                related, gate = retrieve.expand_use_case(index, uc)
                for c in related:
                    add(c, 70.0, f"fk-expansion from {uc.meta['key']}")
                gates.append(gate)
            extras["decisions"] = gates
            s.update({"expanded_from": [u.meta["key"] for u in ucs[:2]], "governance_decisions": gates})

    ranked = sorted(found.values(), key=lambda h: -h["score"])[:24]
    return ranked, extras


ANSWER_SYSTEM = """You are the AI Opportunity Copilot. Answer ONLY using the CONTEXT records.
- Cite record ids in square brackets after each fact, e.g. [use_cases:UC-01].
- Copy numbers exactly as they appear in context (you may format with £ and thousands separators). Never do arithmetic yourself: use COMPUTED METRICS.
- If a GOVERNANCE DECISION is provided, treat it as authoritative policy and explain it.
- If the context does not contain the answer, say so and set not_in_kb true. Never guess.
Return JSON: {"answer": "<markdown answer, concise, use bullet points or a short table where helpful>",
 "citations": ["record ids used"], "confidence": "high|medium|low", "not_in_kb": false}"""


def generate(question: str, context: list[dict], extras: dict, trace: Trace) -> dict:
    ctx = "\n".join(f"[{h['id']}] {h['text']}" for h in context)
    if extras.get("table") is not None:
        ctx += f"\n\nSTRUCTURED QUERY RESULT ({extras['table_desc']}):\n{extras['table'].to_csv(index=False)}"
        if extras.get("metrics"):
            ctx += f"\nCOMPUTED METRICS (exact, use these instead of doing arithmetic): {json.dumps(extras['metrics'])}"
    for g in extras.get("decisions", []):
        ctx += f"\n\nGOVERNANCE DECISION (decision table '{g['table']}' v{g['version']}, rule {g['rule']}): {json.dumps({k: v for k, v in g.items() if k not in ('table', 'version')})}"
    if not llm.available():
        return {"answer": "_No LLM key configured - showing retrieved records only._\n\n" +
                          "\n".join(f"- [{h['id']}] {h['text'][:220]}" for h in context[:6]),
                "citations": [h["id"] for h in context[:6]], "confidence": "low", "not_in_kb": False}
    return llm.complete_json(ANSWER_SYSTEM, f"CONTEXT:\n{ctx}\n\nQUESTION: {question}", trace, "answer_generation")


NUM_RE = re.compile(r"(?<![\w-])£?\d[\d,]*(?:\.\d+)?%?")


def _norm_num(s: str) -> str:
    s = s.replace("£", "").replace(",", "").replace("%", "")
    return s[:-2] if s.endswith(".0") else s


def verify(answer: dict, context: list[dict], extras: dict, trace: Trace) -> dict:
    with trace.span("grounding_check", "evaluator") as s:
        cited = set(answer.get("citations") or [])
        known = {h["id"] for h in context}
        bad_citations = sorted(cited - known)
        ctx_text = " ".join(h["text"] for h in context)
        if extras.get("table") is not None:
            ctx_text += " " + extras["table"].to_csv(index=False)
        ctx_text += " " + json.dumps(extras.get("decisions", [])) + " " + json.dumps(extras.get("metrics", {}))
        ctx_nums = {_norm_num(n) for n in NUM_RE.findall(ctx_text)}
        ans_text = re.sub(r"\[[^\]]+\]", "", answer.get("answer", ""))
        ans_nums = [_norm_num(n) for n in NUM_RE.findall(ans_text)]
        ans_nums = [n for n in ans_nums if len(n.replace(".", "")) >= 2]
        unsupported_nums = sorted({n for n in ans_nums if n not in ctx_nums})
        numeric_score = 1.0 if not ans_nums else round(1 - len(unsupported_nums) / len(set(ans_nums)), 2)
        result = {"citations_valid": not bad_citations, "unknown_citations": bad_citations,
                  "numbers_checked": len(set(ans_nums)), "unsupported_numbers": unsupported_nums,
                  "numeric_grounding": numeric_score}
        if llm.available() and not answer.get("not_in_kb"):
            try:
                cited_ctx = [h for h in context if h["id"] in cited] or context[:8]
                j = judge.decide(
                    {"answer": ans_text, "evidence": [h["text"] for h in cited_ctx]},
                    {"support": {"instructions": "Is every claim in the answer supported by the evidence?",
                                 "options": {"supported": "all claims are stated in or directly computable from the evidence",
                                             "partially_supported": "some claims are not in the evidence",
                                             "unsupported": "the main claims are not in the evidence"}}},
                    trace, "judge_groundedness")
                result["judge"] = j["support"]
            except Exception as e:
                result["judge_error"] = str(e)
        s.update(result)
    trace.score("numeric_grounding", numeric_score)
    if "judge" in result:
        trace.score("judge_supported", result["judge"]["probabilities"].get("supported", 0))
    return result


def answer(index: Index, question: str, history: list[dict]) -> dict:
    trace = Trace("rag_query", input={"question": question})
    try:
        plan = understand(index, question, history, trace)
        if plan.get("intent") == "clarify" and plan.get("clarifying_question"):
            out = {"answer": plan["clarifying_question"], "citations": [], "confidence": "n/a", "clarify": True}
            trace.end(out)
            return {"plan": plan, "context": [], "extras": {}, "result": out, "verification": {}, "trace": trace}
        context, extras = retrieve_context(index, plan, trace)
        if plan.get("intent") == "out_of_scope":
            context = context[:4]
        result = generate(plan.get("standalone_question") or question, context, extras, trace)
        verification = verify(result, context, extras, trace)
        trace.end(result)
        return {"plan": plan, "context": context, "extras": extras, "result": result,
                "verification": verification, "trace": trace}
    except llm.LLMUnavailable as e:
        out = {"answer": f"LLM unavailable: {e}", "citations": [], "confidence": "n/a"}
        trace.end(out)
        return {"plan": {}, "context": [], "extras": {}, "result": out, "verification": {}, "trace": trace}
