"""Streamlit tab for the Event-to-Lead agent."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from agent import pipeline, tools
from core import llm
from core.trace import Trace
from core.ui_trace import render_trace

STATUS_COLOURS = {
    "verified": "#d1fadf", "corrected": "#d1e9ff", "missing_fields": "#fef0c7", "conflict": "#fde2c8",
    "not_invited": "#e9d7fe", "unverified": "#fee4e2", "declined": "#eaecf0",
}


def _style(df: pd.DataFrame):
    def row(r):
        c = STATUS_COLOURS.get(r["status"], "")
        return [f"background-color: {c}; color: #101828" if c else ""] * len(r)
    return df.style.apply(row, axis=1)


def render() -> None:
    st.subheader("Event-to-Lead Agent")
    st.markdown(
        "Pulls the **roundtable invite** from Calendar and **RSVP emails** from Gmail, extracts attendee details, "
        "cleanses and dedupes them, verifies each company against the **UK Companies House register** and the "
        "email domain via **web search**, then applies a **decision table** to give every person a status. "
        "Every value is traceable to a source quote, and every change is in the audit log."
    )
    c1, c2, c3 = st.columns([1.2, 1.6, 1])
    source = c1.radio("Data sources", ["Auto (live where keys exist)", "Mock (bundled demo data)"],
                      help="Auto uses Google/Companies House/Tavily when keys are configured, else mock data.")
    send_to = c2.text_input("Email summary to (optional)", placeholder="organiser@example.com")
    use_planner = c3.toggle("LLM planner", value=llm.available(), disabled=not llm.available(),
                            help="LLM decides the next tool call; otherwise a fixed plan runs.")
    mode = "mock" if source.startswith("Mock") else "auto"
    live = {"Calendar/Gmail": tools.google_live() and mode == "auto",
            "Companies House": bool(tools.config.get("COMPANIES_HOUSE_API_KEY")) and mode == "auto",
            "Web search": mode == "auto", "LLM": llm.available()}
    st.caption(" · ".join(f"{'🟢' if v else '⚪'} {k}" for k, v in live.items()))

    if st.button("▶ Run agent", type="primary"):
        tools.clear_cache()
        trace = Trace("event_to_lead_agent", input={"mode": mode})
        with st.status("Agent running…", expanded=True) as status:
            result = pipeline.run(mode=mode, send_summary_to=send_to or None, trace=trace,
                                  on_step=lambda m: st.write(m), use_planner=use_planner)
            status.update(label=f"Done · orchestration: {result['orchestration']} · extractor: {result['extractor']}",
                          state="complete", expanded=False)
        st.session_state["agent_result"] = result

    result = st.session_state.get("agent_result")
    if not result:
        return

    s = result["stats"]
    cols = st.columns(6)
    for col, (label, val) in zip(cols, [("People", s["total"]), ("Verified", s["verified"] + s["corrected"]),
                                        ("Missing fields", s.get("missing_fields", 0)),
                                        ("Conflicts / unverified", s.get("conflict", 0) + s.get("unverified", 0)),
                                        ("Needs review", s["needs_review"]), ("Emails ignored", s["ignored_emails"])]):
        col.metric(label, val)
    ev = result["event"] or {}
    st.caption(f"Event: **{ev.get('summary')}** · {ev.get('start')} · {ev.get('location')} · backends: {result['backends']}")

    df = result["table"]
    statuses = st.multiselect("Filter by status", list(STATUS_COLOURS), default=[x for x in STATUS_COLOURS if x in set(df.status)])
    review_only = st.checkbox("Only rows needing human review")
    view = df[df.status.isin(statuses)]
    if review_only:
        view = view[view.human_review]
    st.dataframe(_style(view), width="stretch", hide_index=True)
    st.download_button("⬇ Download cleansed CSV", result["csv"], "attendees_verified.csv", "text/csv")

    if result["critic_summary"]:
        st.markdown("**Critic review (LLM)**")
        st.markdown(result["critic_summary"])
    if result["critic"]:
        st.error("Critic invariants failed: " + "; ".join(result["critic"]))
    else:
        st.success("Critic invariants passed: unique emails · E.164 phones · verified ⇒ active company + matching domain · "
                   "no records from ignored emails")
    if result["sent"]:
        st.info(f"Summary email: {result['sent']}")

    t1, t2, t3, t4, t5 = st.tabs(["Agent steps", "Decisions", "Audit log", "Ignored emails", "Trace"])
    with t1:
        st.dataframe(pd.DataFrame(result["steps"]), width="stretch", hide_index=True)
    with t2:
        st.caption("Rules: `decisions/record_status.yaml` (hit policy FIRST). Human review when "
                   f"P(wrong) × £{pipeline.COST_OF_BAD_LEAD:.0f} > £{pipeline.REVIEW_COST:.0f}.")
        for r in result["records"]:
            with st.expander(f"{r['name']} — {r['status']} ({r['rule_id']})"):
                st.write(r["reason"])
                st.json({"decision_inputs": r["decision_inputs"], "p_wrong": r["p_wrong"], "human_review": r["human_review"],
                         "register": {"source": r.get("register_source"), "registered_name": r.get("registered_name"),
                                      "status": r.get("company_status"), "number": r.get("company_number")},
                         "domain_check": r.get("domain_evidence"), "conflicts": r["conflicts"], "flags": r["flags"],
                         "source_quotes": r["quotes"], "sources": r["sources"]})
    with t3:
        st.dataframe(pd.DataFrame(result["audit"]), width="stretch", hide_index=True)
    with t4:
        st.dataframe(pd.DataFrame(result["ignored"]), width="stretch", hide_index=True)
    with t5:
        render_trace(result["trace"], "Agent trace")
