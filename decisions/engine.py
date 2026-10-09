"""Tiny DMN-style decision-table engine.

Rules live in YAML (reviewable, versioned, unit-tested) - the LLM never makes the
final policy decision; it only supplies inputs. Hit policy FIRST: the first rule
whose conditions all match wins, and its id is returned for the audit trail.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

DIR = Path(__file__).parent


@lru_cache
def load(table: str) -> dict:
    return yaml.safe_load((DIR / f"{table}.yaml").read_text())


def _match(cond, value) -> bool:
    if isinstance(cond, dict):
        for op, target in cond.items():
            if op == "gte" and not (value is not None and value >= target):
                return False
            if op == "lt" and not (value is not None and value < target):
                return False
            if op == "in" and value not in target:
                return False
            if op == "not_in" and value in target:
                return False
        return True
    return value == cond


def evaluate(table: str, inputs: dict) -> dict:
    spec = load(table)
    for rule in spec["rules"]:
        if all(_match(c, inputs.get(k)) for k, c in rule.get("when", {}).items()):
            return {"table": table, "version": spec.get("version"), "rule": rule["id"],
                    "inputs": inputs, **rule["then"]}
    return {"table": table, "version": spec.get("version"), "rule": "default", "inputs": inputs, **spec["default"]}


def expected_value(annual_benefit: float, p_success: float, build_cost: float, annual_run_cost: float) -> float:
    """First-year expected value of a use case."""
    return round(annual_benefit * p_success - build_cost - annual_run_cost, 0)


def needs_human_review(p_wrong: float, cost_of_error: float, review_cost: float) -> bool:
    """Route to a human only when the expected cost of an error exceeds the cost of review."""
    return p_wrong * cost_of_error > review_cost
