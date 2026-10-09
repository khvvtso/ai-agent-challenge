"""Provider-agnostic LLM layer.

Everything in the app calls `complete_json` / `complete_text`. Swapping Gemini for
Groq (or adding Claude / Azure OpenAI) is one new `_call_*` function + LLM_PROVIDER.
Includes retry with backoff and automatic fallback to the secondary provider.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import requests

from core import config
from core.trace import Trace


class LLMUnavailable(RuntimeError):
    pass


def providers() -> list[str]:
    order = [config.get("LLM_PROVIDER", "gemini")]
    order += [p for p in ("gemini", "groq") if p not in order]
    keys = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}
    return [p for p in order if config.get(keys[p])]


def available() -> bool:
    return bool(providers())


_gemini_client = None


CACHE_DIR = Path(__file__).resolve().parent.parent / "llm_cache"  # committed: replay cache for the demo
_exhausted: dict[str, float] = {}  # model -> unix time until which it is skipped


def models_for(provider: str) -> list[str]:
    """Free-tier quotas are per model, so each provider has an ordered model chain."""
    if provider == "gemini":
        default = "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.6-flash,gemini-3.5-flash,gemini-flash-latest"
        return [m.strip() for m in config.get("GEMINI_MODELS", default).split(",") if m.strip()]
    default = "openai/gpt-oss-120b,qwen/qwen3.8-27b,openai/gpt-oss-20b"
    return [m.strip() for m in config.get("GROQ_MODELS", default).split(",") if m.strip()]


def _call_gemini(system: str, prompt: str, json_mode: bool, temperature: float, model: str):
    global _gemini_client
    from google import genai
    from google.genai import types

    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=config.get("GEMINI_API_KEY"))
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        response_mime_type="application/json" if json_mode else "text/plain",
    )
    resp = _gemini_client.models.generate_content(model=model, contents=prompt, config=cfg)
    um = resp.usage_metadata
    usage = {
        "input": getattr(um, "prompt_token_count", 0) or 0,
        "output": getattr(um, "candidates_token_count", 0) or 0,
        "total": getattr(um, "total_token_count", 0) or 0,
    }
    return resp.text or "", usage, model


def _call_groq(system: str, prompt: str, json_mode: bool, temperature: float, model: str):
    body = {
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    }
    body["max_completion_tokens"] = 8192
    if model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "low"  # hidden reasoning otherwise eats the output budget
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {config.get('GROQ_API_KEY')}"},
        json=body,
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    u = data.get("usage", {})
    usage = {"input": u.get("prompt_tokens", 0), "output": u.get("completion_tokens", 0), "total": u.get("total_tokens", 0)}
    return data["choices"][0]["message"]["content"], usage, model


_CALLERS = {"gemini": _call_gemini, "groq": _call_groq}


def _parse_json(text: str):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
        if start < 0:
            raise
        return json.loads(text[start:])


def _cache_key(system, prompt, json_mode, temperature) -> Path:
    h = hashlib.sha256(json.dumps([system, prompt, json_mode, temperature]).encode()).hexdigest()[:24]
    return CACHE_DIR / f"{h}.json"


def _retry_delay(msg: str) -> float:
    m = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", msg)
    if not m:
        return 60.0
    h, mi, s = (float(x) if x else 0.0 for x in m.groups())
    return h * 3600 + mi * 60 + s


def _complete(system, prompt, trace: Trace | None, name: str, json_mode: bool, temperature: float):
    """Try each (provider, model) target in order. Quota/overload errors move straight to the
    next target (a model that reports a long retry delay is parked); responses are cached on disk
    so repeated demo questions cost no quota."""
    provs = providers()
    if not provs:
        raise LLMUnavailable("No LLM key configured (set GEMINI_API_KEY or GROQ_API_KEY).")
    cache = _cache_key(system, prompt, json_mode, temperature)
    use_cache = (config.get("LLM_CACHE", "on") or "on").lower() != "off"
    if use_cache and cache.exists():
        hit = json.loads(cache.read_text())
        if trace:
            with trace.span(name, "generation", input={"system": system[:2000], "prompt": prompt[:6000]},
                            cache="hit") as s:
                s.update(hit["out"], usage={"input": 0, "output": 0, "total": 0}, model=hit["model"] + " (cached)")
        return hit["out"]
    last_err = None
    for provider in provs:
        for model in models_for(provider):
            if _exhausted.get(model, 0) > time.time():
                continue
            for attempt in range(2):
                try:
                    if trace:
                        with trace.span(name, "generation", input={"system": system[:2000], "prompt": prompt[:6000]},
                                        provider=provider, model=model, attempt=attempt + 1) as s:
                            text, usage, used = _CALLERS[provider](system, prompt, json_mode, temperature, model)
                            out = _parse_json(text) if json_mode else text
                            s.update(out, usage=usage, model=used)
                    else:
                        text, usage, used = _CALLERS[provider](system, prompt, json_mode, temperature, model)
                        out = _parse_json(text) if json_mode else text
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps({"model": used, "out": out}))
                    return out
                except Exception as e:
                    last_err = e
                    msg = str(e)
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                        delay = _retry_delay(msg)
                        _exhausted[model] = time.time() + delay
                        if delay <= 8 and attempt == 0:
                            time.sleep(delay + 0.5)
                            continue
                        break
                    if "503" in msg or "UNAVAILABLE" in msg or "404" in msg or "NOT_FOUND" in msg:
                        break  # overloaded / unknown model -> next target immediately
                    if attempt >= 1:
                        break
    raise LLMUnavailable(f"All LLM providers failed: {str(last_err)[:300]}")


def complete_json(system: str, prompt: str, trace: Trace | None = None, name: str = "llm", temperature: float = 0.0):
    return _complete(system, prompt, trace, name, True, temperature)


def complete_text(system: str, prompt: str, trace: Trace | None = None, name: str = "llm", temperature: float = 0.2):
    return _complete(system, prompt, trace, name, False, temperature)
