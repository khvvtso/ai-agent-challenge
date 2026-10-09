"""Hybrid retrieval + structured expansion.

- Dense (semantic) and BM25 (exact terms, IDs) rankings fused with Reciprocal Rank Fusion.
- Explicit IDs in the query (UC-01, DS-04, GC-08, M-02) are pinned - vectors are poor at IDs.
- Multi-hop expansion follows the tables' foreign keys deterministically:
  use case -> its data source -> controls for its risk tier -> models permitted by the
  governance decision table. The LLM never has to "remember" a join.
"""
from __future__ import annotations

import re

import numpy as np

from decisions import engine
from rag.ingest import Chunk, Index, tokenize

ID_RE = re.compile(r"\b(UC|DS|GC|M)-(\d{1,2})\b", re.I)
PREFIX_DOC = {"UC": "use_cases", "DS": "data_sources", "GC": "controls", "M": "models"}
TIER_RANK = {"Low": 0, "Medium": 1, "High": 2}


def hybrid_search(index: Index, query: str, k: int = 6, docs: list[str] | None = None) -> list[tuple[Chunk, float, dict]]:
    dense = index.vectors @ index.embed_query(query)
    sparse = index.bm25.get_scores(tokenize(query))
    mask = np.array([(not docs) or c.doc in docs for c in index.chunks])
    dense_rank = {i: r for r, i in enumerate(np.argsort(-np.where(mask, dense, -9)))}
    sparse_rank = {i: r for r, i in enumerate(np.argsort(-np.where(mask, sparse, -9)))}
    rrf = {i: 1 / (60 + dense_rank[i]) + 1 / (60 + sparse_rank[i]) for i in range(len(index.chunks)) if mask[i]}
    top = sorted(rrf, key=rrf.get, reverse=True)[:k]
    return [(index.chunks[i], round(rrf[i] * 1000, 2),
             {"dense": round(float(dense[i]), 3), "bm25": round(float(sparse[i]), 2), "via": "hybrid"}) for i in top]


def pinned_ids(index: Index, text: str) -> list[Chunk]:
    out = []
    for prefix, num in ID_RE.findall(text):
        c = index.by_id(f"{PREFIX_DOC[prefix.upper()]}:{prefix.upper()}-{int(num):02d}")
        if c:
            out.append(c)
    return out


def name_matches(index: Index, entities: list[str]) -> list[Chunk]:
    """Exact-ish match of named entities (use case / system names) against row text."""
    out = []
    for e in entities or []:
        e_l = e.lower().strip()
        if len(e_l) < 4:
            continue
        for c in index.chunks:
            row = c.meta.get("row") or {}
            name = str(row.get("name") or row.get("system") or row.get("control") or "").lower()
            if name and (e_l in name or name in e_l):
                out.append(c)
    return out


def allowed_models(index: Index, risk_tier: str, contains_pii: str) -> tuple[dict, list[dict]]:
    gate = engine.evaluate("governance", {"risk_tier": risk_tier, "contains_pii": contains_pii})
    allowed = []
    for _, m in index.tables["models"].iterrows():
        regions = [r.strip() for r in m["hosting_regions"].split(",")]
        ok_region = any(r in gate["allowed_residency"] for r in regions)
        ok_pii = contains_pii != "yes" or bool(m["approved_for_pii"])
        ok_tier = TIER_RANK[m["max_risk_tier"]] >= TIER_RANK[risk_tier]
        if ok_region and ok_pii and ok_tier and m["type"] != "embedding":
            allowed.append({"model_id": m["model_id"], "name": m["name"]})
    return gate, allowed


def expand_use_case(index: Index, uc_chunk: Chunk) -> tuple[list[Chunk], dict]:
    row = uc_chunk.meta["row"]
    related = []
    ds = index.by_id(f"data_sources:{row['primary_data_source']}")
    if ds:
        related.append(ds)
    related += [c for c in index.chunks if c.doc == "controls" and c.meta.get("row", {}).get("risk_tier") in
                [t for t, r in TIER_RANK.items() if r <= TIER_RANK[row["risk_tier"]]]]
    gate, models = allowed_models(index, row["risk_tier"], row["contains_pii"])
    related += [index.by_id(f"models:{m['model_id']}") for m in models]
    gate["use_case_id"] = row["use_case_id"]
    gate["allowed_models"] = [m["model_id"] for m in models]
    return [c for c in related if c], gate
