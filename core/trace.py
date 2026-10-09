"""Two-layer observability.

1. A local in-memory trace (always on) that the Streamlit UI renders as a step timeline.
2. Langfuse (optional) - the same spans are mirrored when LANGFUSE_* keys are set,
   giving persistent traces, token/cost tracking, scores and eval datasets.
"""
from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from core import config

_langfuse = None


def _lf():
    global _langfuse
    if _langfuse is None and config.get("LANGFUSE_PUBLIC_KEY") and config.get("LANGFUSE_SECRET_KEY"):
        try:
            from langfuse import Langfuse

            _langfuse = Langfuse(
                public_key=config.get("LANGFUSE_PUBLIC_KEY"),
                secret_key=config.get("LANGFUSE_SECRET_KEY"),
                host=config.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            )
        except Exception:
            _langfuse = False
    return _langfuse or None


@dataclass
class Span:
    name: str
    kind: str
    input: Any = None
    output: Any = None
    metadata: dict = field(default_factory=dict)
    start: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    error: str | None = None
    depth: int = 0
    _lf_obj: Any = None

    def update(self, output: Any = None, **metadata):
        if output is not None:
            self.output = output
        self.metadata.update(metadata)


class Trace:
    """Collects spans for one user request."""

    def __init__(self, name: str, input: Any = None):
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.input = input
        self.spans: list[Span] = []
        self.scores: dict[str, float] = {}
        self._depth = 0
        self._root_cm = None
        self._root = None
        self.lf_url: str | None = None
        lf = _lf()
        if lf:
            try:
                self._root_cm = lf.start_as_current_observation(name=name, as_type="agent", input=input)
                self._root = self._root_cm.__enter__()
            except Exception:
                self._root_cm = None

    @contextmanager
    def span(self, name: str, kind: str = "span", input: Any = None, **metadata):
        s = Span(name=name, kind=kind, input=input, metadata=dict(metadata), depth=self._depth)
        self.spans.append(s)
        self._depth += 1
        lf_cm = None
        lf = _lf()
        if lf and self._root_cm:
            try:
                as_type = kind if kind in ("generation", "tool", "retriever", "evaluator", "guardrail", "chain") else "span"
                lf_cm = lf.start_as_current_observation(name=name, as_type=as_type, input=input, metadata=metadata or None)
                s._lf_obj = lf_cm.__enter__()
            except Exception:
                lf_cm = None
        try:
            yield s
        except Exception as e:
            s.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            s.duration_ms = (time.time() - s.start) * 1000
            self._depth -= 1
            if lf_cm:
                try:
                    kwargs: dict[str, Any] = {"output": s.output, "metadata": s.metadata or None}
                    if kind == "generation":
                        kwargs["model"] = s.metadata.get("model")
                        if "usage" in s.metadata:
                            kwargs["usage_details"] = s.metadata["usage"]
                    if s.error:
                        kwargs["level"] = "ERROR"
                        kwargs["status_message"] = s.error
                    s._lf_obj.update(**kwargs)
                    lf_cm.__exit__(None, None, None)
                except Exception:
                    pass

    def score(self, name: str, value: float, comment: str | None = None):
        self.scores[name] = value
        lf = _lf()
        if lf and self._root_cm:
            try:
                lf.score_current_trace(name=name, value=value, comment=comment)
            except Exception:
                pass

    def end(self, output: Any = None):
        lf = _lf()
        if lf and self._root_cm:
            try:
                self._root.update(output=output)
                try:
                    self.lf_url = lf.get_trace_url()
                except Exception:
                    pass
                self._root_cm.__exit__(None, None, None)
                lf.flush()
            except Exception:
                pass

    @property
    def total_tokens(self) -> int:
        return sum(s.metadata.get("usage", {}).get("total", 0) for s in self.spans)

    @property
    def total_ms(self) -> float:
        return sum(s.duration_ms for s in self.spans if s.depth == 0)
