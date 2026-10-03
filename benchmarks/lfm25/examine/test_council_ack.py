"""council_ack's scenario, message builder and scorer: each able to fail."""
import json
from collections import Counter

import council_ack as ca

TABS = ["Memory", "User", "Session"]


def test_the_scenario_matches_its_preregistration():
    s = json.loads(ca.SCENARIO.read_text())
    assert [t["name"] for t in s["tabs"]] == TABS
    assert Counter(a["rules"] for a in s["actions"]) == {"allow": 7, "ask": 7, "report": 4}
    assert len(s["actions"]) == 18
    for path in ca.SPECS:
        spec = json.loads(path.read_text())
        assert spec["output_schema"]["properties"][spec["output_schema"]["required"][-1]]["enum"] == [
            "allow", "ask", "report"]


def test_acknowledgements_follow_every_user_turn():
    tab = {"messages": ["one", "two"]}
    assert ca.messages(tab, None) == [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}]
    assert ca.messages(tab, "Acknowledged.") == [
        {"role": "user", "content": "one"}, {"role": "assistant", "content": "Acknowledged."},
        {"role": "user", "content": "two"}, {"role": "assistant", "content": "Acknowledged."}]


def rows(cond, per_by_action, pooled, rules):
    return [{"condition": cond, "spec": "s", "action": str(i), "rules": r, "per": p, "pooled": q}
            for i, (p, q, r) in enumerate(zip(per_by_action, pooled, rules))]


def test_a_locked_context_fails_and_a_responsive_one_passes():
    rules = ["allow"] * 7 + ["ask"] * 7 + ["report"] * 4
    locked = rows("lone", [{t: "ask" for t in TABS}] * 18, ["ask"] * 18, rules)
    s = ca.score(locked, TABS, 18)["s"]["lone"]
    assert s["locked"]["Memory"]["share"] == 1.0 and s["pooled_matches_hint"] == 7 and s["success"] is False
    good = rows("ack", [{t: r for t in TABS} for r in rules], rules, rules)
    s = ca.score(good, TABS, 18)["s"]["ack"]
    assert s["pooled_matches_hint"] == 18 and max(v["share"] for v in s["locked"].values()) <= 7 / 18
    assert s["success"] is True
    # Responsive pool but one context still locked: fails on the lock rule alone.
    one_locked = rows("ack", [{"Memory": "ask", "User": r, "Session": r} for r in rules], rules, rules)
    assert ca.score(one_locked, TABS, 18)["s"]["ack"]["success"] is False


def test_a_later_preregistration_names_its_specs_slot0_and_conditions():
    scenario, specs, conds = ca.plan(ca.HERE.parent / "council" / "ack-v2.json")
    assert scenario == ca.SCENARIO and [p.name for p in specs] == ["council-verdict-v2.json", "council-describe-v2.json"]
    assert set(conds) == {"lone", "ack", "slot0_ack"}
    opening, ack = conds["slot0_ack"]
    assert [m["role"] for m in opening] == ["user", "assistant"] and "Acknowledged." in opening[1]["content"]
    tab = {"messages": ["fact"]}
    assert ca.messages(tab, ack, opening)[:2] == opening and ca.messages(tab, ack, opening)[2:] == [
        {"role": "user", "content": "fact"}, {"role": "assistant", "content": "Acknowledged."}]
    assert conds["lone"] == ([], None)
    for path in specs:
        assert "Amy" not in path.read_text(), "the v2 specs name no one"
