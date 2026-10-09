"""Ingestion: structured files -> row-level chunks -> hybrid index.

Chunking strategy: one chunk per table row (never fixed-size windows), serialised as
`column: value` pairs prefixed with the document title, so every chunk carries its own
schema context. One summary chunk per document helps document-level routing.
Each chunk keeps metadata (doc, file, primary key) for filtering and citations.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
KB = ROOT / "data" / "kb"
CACHE = ROOT / ".cache"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

DOCS = {
    "use_cases": {"file": "use_case_register.csv", "title": "AI use case register", "key": "use_case_id",
                  "about": "Candidate AI use cases with department, £ benefit and cost, feasibility, data readiness, risk tier, PII, status and owner."},
    "data_sources": {"file": "data_sources.csv", "title": "Data source inventory", "key": "source_id",
                     "about": "Source systems feeding use cases: owner, data quality score (1-5), PII, refresh frequency, hosting region."},
    "controls": {"file": "governance_controls.csv", "title": "AI governance controls", "key": "control_id",
                 "about": "Controls required per risk tier (Low/Medium/High), owner, framework reference and stage required before. Controls are cumulative: a High risk use case must also meet every Medium and Low control, and a Medium one every Low control."},
    "models": {"file": "model_catalog.json", "title": "Approved model catalog", "key": "model_id",
               "about": "Approved AI models: provider, £ cost per 1M tokens, context window, hosting regions, PII approval, max risk tier."},
}


@dataclass
class Chunk:
    id: str
    doc: str
    text: str
    meta: dict = field(default_factory=dict)


def load_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    for name, d in DOCS.items():
        path = KB / d["file"]
        if path.suffix == ".json":
            raw = json.loads(path.read_text())
            df = pd.DataFrame(raw["models"])
            df["hosting_regions"] = df["hosting_regions"].apply(lambda r: ", ".join(r))
        else:
            df = pd.read_csv(path)
        tables[name] = df
    uc = tables["use_cases"]
    # Derived, deterministic prioritisation fields (see decisions.engine.expected_value).
    uc["p_success"] = ((uc["feasibility"] * uc["data_readiness"]) / 25).round(2)
    uc["expected_value_gbp"] = (uc["annual_benefit_gbp"] * uc["p_success"] - uc["build_cost_gbp"] - uc["annual_run_cost_gbp"]).round(0).astype(int)
    return tables


def build_chunks(tables: dict[str, pd.DataFrame]) -> list[Chunk]:
    chunks = []
    for name, df in tables.items():
        d = DOCS[name]
        chunks.append(Chunk(id=f"{name}:_summary", doc=name,
                            text=f"Document: {d['title']} ({d['file']}). {d['about']} Columns: {', '.join(df.columns)}. Rows: {len(df)}.",
                            meta={"file": d["file"], "key": "_summary"}))
        for _, row in df.iterrows():
            key = str(row[d["key"]])
            body = " | ".join(f"{c}: {row[c]}" for c in df.columns)
            chunks.append(Chunk(id=f"{name}:{key}", doc=name, text=f"Document: {d['title']} | {body}",
                                meta={"file": d["file"], "key": key, "row": row.to_dict()}))
    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[0-9]+)?", text.lower())


class Index:
    """Dense (bge-small, local) + sparse (BM25) index over chunks."""

    def __init__(self):
        self.tables = load_tables()
        self.chunks = build_chunks(self.tables)
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])
        self._embedder = None
        self.vectors = self._embed_chunks()

    @property
    def embedder(self):
        if self._embedder is None:
            from fastembed import TextEmbedding

            self._embedder = TextEmbedding(EMBED_MODEL, cache_dir=str(CACHE / "fastembed"))
        return self._embedder

    def _embed_chunks(self) -> np.ndarray:
        digest = hashlib.sha1("".join(c.text for c in self.chunks).encode()).hexdigest()[:12]
        path = CACHE / f"vectors-{digest}.npy"
        if path.exists():
            return np.load(path)
        vecs = np.array(list(self.embedder.passage_embed([c.text for c in self.chunks])), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        CACHE.mkdir(exist_ok=True)
        np.save(path, vecs)
        return vecs

    def embed_query(self, q: str) -> np.ndarray:
        v = np.array(list(self.embedder.query_embed([q]))[0], dtype=np.float32)
        return v / np.linalg.norm(v)

    def by_id(self, chunk_id: str) -> Chunk | None:
        return next((c for c in self.chunks if c.id == chunk_id), None)
