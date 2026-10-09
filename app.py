"""AI Agent Technical Challenge - single Streamlit app for both parts."""
import streamlit as st

from core import config, llm

st.set_page_config(page_title="Agentic AI Challenge", page_icon="🧭", layout="wide")

with st.sidebar:
    st.markdown("### Configuration")
    provs = llm.providers()
    st.markdown(f"**LLM:** {' → '.join(provs) if provs else '⚠️ none (fallback mode)'}")
    labels = {"GEMINI_API_KEY": "Gemini", "GROQ_API_KEY": "Groq (fallback LLM)", "TAVILY_API_KEY": "Tavily search",
              "COMPANIES_HOUSE_API_KEY": "Companies House", "LANGFUSE_PUBLIC_KEY": "Langfuse tracing",
              "GOOGLE_REFRESH_TOKEN": "Gmail + Calendar"}
    for k, ok in config.status().items():
        st.markdown(f"{'🟢' if ok else '⚪'} {labels[k]}")
    st.caption("Every integration is optional: missing keys fall back to mock data or a secondary provider, "
               "and the trace shows which path ran.")

st.title("🧭 From pilot to production: RAG + agentic data pipeline")
st.caption("Technical challenge demo · Python · Streamlit · Gemini (provider-agnostic) · bge-small + BM25 · "
           "DMN-style decision tables · Langfuse")

tab1, tab2, tab3 = st.tabs(["Part 1 · RAG Copilot", "Part 2 · Event-to-Lead Agent", "Architecture"])

with tab1:
    from rag.ui import render as render_rag

    render_rag()

with tab2:
    try:
        from agent.ui import render as render_agent
    except ImportError as e:
        st.warning(f"Part 2 not available yet: {e}")
    else:
        render_agent()

with tab3:
    from architecture import render as render_arch

    render_arch()
