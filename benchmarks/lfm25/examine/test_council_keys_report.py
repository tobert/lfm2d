import council_keys_report as k

OPTS = ["allow", "ask", "report"]


def row(rules, probs, spec="s"):
    return {"rules": rules, "pooled_probs": probs, "action": rules + str(probs), "spec": spec,
            "probs": {"A": probs}}


def test_options_come_from_the_spec_file():
    assert k.options_of("council-describe-v3") == OPTS
    assert k.options_of("council-describe-v3-neutral") == OPTS


def test_neutral_specs_differ_from_named_only_in_their_keys():
    import json
    for named, neutral in k.PAIRS.items():
        a = json.loads((k.STATIC / f"{named}.json").read_text())
        b = json.loads((k.STATIC / f"{neutral}.json").read_text())
        pa, pb = a["output_schema"]["properties"], b["output_schema"]["properties"]
        assert list(pb) == [f"q{i + 1}" for i in range(len(pa))]
        assert list(pa.values()) == list(pb.values()), "descriptions and enums must be identical"
        assert a["system"] == b["system"] and a["input_label"] == b["input_label"]


def test_auc_counts_ties_half_and_orders_correctly():
    assert k.auc([0.9], [0.1, 0.2]) == 1.0
    assert k.auc([0.1], [0.9]) == 0.0
    assert k.auc([0.5], [0.5]) == 0.5


def test_cell_counts_a_soft_miss_and_reads_the_loud_rank():
    rows = [row("allow", [0.8, 0.1, 0.1]), row("ask", [0.6, 0.3, 0.1]), row("report", [0.1, 0.2, 0.7]),
            row("ask", [0.1, 0.8, 0.1])]
    c = k.cell(rows, OPTS)
    assert c["matches"] == 3 and c["allow_on_ask_or_report"] == 1 and c["p_loud_auc"] == 1.0


def test_cell_refuses_an_option_order_the_hints_do_not_fit():
    import pytest
    with pytest.raises(AssertionError):
        k.cell([row("allow", [1, 0, 0])], OPTS)


def test_tvd_pairs_by_action_and_is_zero_for_identical_reads():
    a = [row("ask", [0.2, 0.6, 0.2])]
    b = [dict(row("ask", [0.2, 0.6, 0.2]))]
    assert k.paired_tvd(a, b, ["A"])["A"] == {"median": 0.0, "max": 0.0}
    b = [dict(row("ask", [0.6, 0.2, 0.2]), action=a[0]["action"])]
    assert abs(k.paired_tvd(a, b, ["A"])["A"]["max"] - 0.4) < 1e-12


def test_verdict_reads_the_preregistration():
    good = {"matches": 10, "n": 18, "allow_on_ask_or_report": 0, "p_loud_auc": 1.0}
    bad = dict(good, matches=7)
    lock = {"locked": {"A": {"share": 10 / 18}}}
    cells = lambda n, m: {"council-describe-v3": {"named": n, "neutral": m}}  # noqa: E731
    score = {"council-describe-v3": {"v3_named_keys": lock, "v3_neutral_keys": lock}}
    assert k.verdict(cells(good, good), score) == "neutral keys HOLD"
    assert k.verdict(cells(good, bad), score) == "neutral keys COST"
    assert k.verdict(cells(bad, good), score).startswith("non-replication")
    locked = {"council-describe-v3": {"v3_named_keys": lock, "v3_neutral_keys": {"locked": {"A": {"share": 16 / 18}}}}}
    assert k.verdict(cells(good, good), locked) == "neutral keys COST"
