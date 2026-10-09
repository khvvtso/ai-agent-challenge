"""End-to-end agent run on the bundled mock data - no API keys needed."""
import pytest

from agent import pipeline
from core import llm
from decisions import engine


@pytest.fixture(scope="module")
def result():
    mp = pytest.MonkeyPatch()
    mp.setattr(llm, "available", lambda: False)  # deterministic: heuristic extractor + fixed plan
    yield pipeline.run(mode="mock")
    mp.undo()


def by_email(result, email):
    return next(r for r in result["records"] if r["email"] == email)


def test_runs_with_fixed_plan_and_passes_critic(result):
    assert result["orchestration"] == "fixed-plan"
    assert result["critic"] == []
    assert [s["tool"] for s in result["steps"]][:3] == ["get_event", "search_emails", "extract"]


def test_newsletter_ignored(result):
    assert "m11" in {i["email_id"] for i in result["ignored"]}
    assert not any("m11" in r["sources"] for r in result["records"])


def test_duplicates_merged_with_conflict(result):
    jon = [r for r in result["records"] if r["email"] == "jonathan.reed@watkinjones.com"]
    assert len(jon) == 1
    assert jon[0]["name"] == "Jonathan Reed"
    assert jon[0]["status"] == "conflict"
    assert jon[0]["conflicts"][0]["field"] == "job_title"


def test_emails_unique_and_lowercase(result):
    emails = [r["email"] for r in result["records"]]
    assert len(emails) == len(set(emails))
    assert all(e == e.lower() for e in emails)


def test_phone_e164_and_invalid_flagged(result):
    assert by_email(result, "priya.shah@serco.com")["phone"] == "+442079460123"
    dan = by_email(result, "dan.whitfield@virginmediao2.co.uk")
    assert dan["phone"] is None and any("invalid phone" in f for f in dan["flags"])


def test_company_checks(result):
    assert by_email(result, "aisha.bello@nimbusanalytica.io")["rule_id"] == "R03_company_not_found"
    assert by_email(result, "tom.hughes@brightwaterlogistics.co.uk")["rule_id"] == "R04_company_inactive"
    assert by_email(result, "priya.shah@serco.com")["status"] == "verified"


def test_statuses(result):
    assert by_email(result, "liam.oconnor@serco.com")["status"] == "declined"
    assert by_email(result, "ravi.patel@sspgroup.com")["status"] == "not_invited"
    emma = by_email(result, "emma.larsen@hsbc.co.uk")
    assert emma["name"] == "Emma Larsen" and emma["status"] == "corrected"
    hannah = by_email(result, "hannah.price@watkinjones.com")
    assert hannah["is_plus_one"] and hannah["status"] == "missing_fields"
    grace = by_email(result, "grace.chen@hsbc.co.uk")
    assert grace["rsvp_status"] == "no_response" and "phone" in grace["missing_fields"]


def test_csv_export(result):
    assert result["csv"].splitlines()[0].startswith("status,human_review,name,email")
    assert len(result["csv"].splitlines()) == len(result["records"]) + 1


def test_grounding_blocks_unsupported_values():
    rec = pipeline.Extracted(name="Priya Shah", job_title="Chief Executive", phone="020 7946 0000", source_email_id="x")
    msg = {"x": {"sender": "Priya Shah <p@serco.com>", "subject": "s", "body": "Head of Data Science\nTel: 020 7946 0123"}}
    audit = []
    from core.trace import Trace
    pipeline.grounding_check([rec], msg, audit, Trace("t"))
    assert rec.job_title is None and rec.phone is None and rec.name == "Priya Shah"
    assert {a["rule"] for a in audit} == {"hallucination_blocked"}


@pytest.mark.parametrize("inputs,rule", [
    ({"rsvp_status": "declined", "company_status": "not_found"}, "R01_declined"),
    ({"rsvp_status": "attending", "email_valid": True, "company_status": "dissolved"}, "R04_company_inactive"),
    ({"rsvp_status": "attending", "email_valid": True, "company_status": "active", "invited": False, "is_plus_one": False}, "R05_not_invited"),
    ({"rsvp_status": "attending", "email_valid": True, "company_status": "active", "invited": True, "is_plus_one": False,
      "has_conflict": False, "domain_match": True, "missing_count": 0, "was_corrected": False}, "R10_verified"),
])
def test_decision_table(inputs, rule):
    assert engine.evaluate("record_status", inputs)["rule"] == rule


def test_human_review_routing():
    assert engine.needs_human_review(0.3, 50, 5)
    assert not engine.needs_human_review(0.05, 50, 5)
