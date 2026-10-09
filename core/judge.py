"""Judge interface: every small judgement call in the system goes through here.

Today it is backed by the configured LLM (verbalised probabilities). The interface
mirrors decision-model APIs (Cloudflare Clef, OpenAI Decisions, Jev): typed
questions in, a probability per allowed option out - so a calibrated decision model
can be dropped in as another backend without touching callers.
"""
from __future__ import annotations

import json

from core import llm
from core.trace import Trace

SYSTEM = """You are a decision model. You never write prose.
For each question, assign a probability to EVERY allowed option (probabilities sum to 1),
based only on the provided STATE. Return JSON:
{"<question_id>": {"probabilities": {"<option>": <float>, ...}, "evidence": "<short quote from STATE or ''>"}, ...}"""


def decide(state, questions: dict[str, dict], trace: Trace | None = None, name: str = "judge") -> dict:
    """questions: {id: {"instructions": str, "options": {option: description}}}
    returns {id: {"choice", "confidence", "probabilities", "evidence", "backend"}}"""
    q_text = json.dumps(
        {qid: {"instructions": q["instructions"], "options": q["options"]} for qid, q in questions.items()},
        indent=1,
    )
    state_text = state if isinstance(state, str) else json.dumps(state, indent=1, default=str)
    raw = llm.complete_json(SYSTEM, f"STATE:\n{state_text[:12000]}\n\nQUESTIONS:\n{q_text}", trace, name)
    out = {}
    for qid, q in questions.items():
        ans = raw.get(qid, {}) if isinstance(raw, dict) else {}
        probs = {o: float(ans.get("probabilities", {}).get(o, 0) or 0) for o in q["options"]}
        total = sum(probs.values()) or 1.0
        probs = {o: round(p / total, 3) for o, p in probs.items()}
        choice = max(probs, key=probs.get)
        out[qid] = {
            "choice": choice,
            "confidence": probs[choice],
            "probabilities": probs,
            "evidence": ans.get("evidence", ""),
            "backend": "llm-judge",
        }
    return out
