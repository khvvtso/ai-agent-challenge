"""Settings come from env vars or Streamlit secrets; every key is optional."""
import os

from dotenv import load_dotenv

load_dotenv()


def get(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    if val:
        return val
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


def status() -> dict[str, bool]:
    keys = ["GEMINI_API_KEY", "GROQ_API_KEY", "TAVILY_API_KEY", "COMPANIES_HOUSE_API_KEY",
            "LANGFUSE_PUBLIC_KEY", "GOOGLE_REFRESH_TOKEN"]
    return {k: bool(get(k)) for k in keys}
