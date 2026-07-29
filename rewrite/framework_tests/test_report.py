import pytest
from conftest import a_run

from llmeval.cache_key import compute_cache_key
from llmeval.comparison.report import build_report, render_html, write_report
from llmeval.store import Store

BASE = compute_cache_key(model="m1", params={"t": 0})
CAND = compute_cache_key(model="m1", params={"t": 1})


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def graded(store, key, test_id, score):
    rid = store.add_result_row(
        test_id, key, run_id=a_run(store, key), output=f"answer for {test_id}"
    )
    store.set_grading(rid, "a1", type="rubric", score=score, passed=score >= 0.5, metric="accuracy")


def populate(store):
    graded(store, BASE, "t1", 0.5)
    graded(store, BASE, "t2", 0.5)
    graded(store, CAND, "t1", 0.7)
    graded(store, CAND, "t2", 0.7)
    store.set_verdict("t1", "cmp", winner_hash=CAND.hash, candidates=[BASE.hash, CAND.hash])
    store.set_verdict("t2", "cmp", winner_hash=CAND.hash, candidates=[BASE.hash, CAND.hash])


def configs():
    return [("base", BASE.hash), ("cand", CAND.hash)]


def test_report_data_has_metric_rows_with_delta(store):
    populate(store)
    data = build_report(store, configs(), metrics=["accuracy"], baseline_name="base", comparison_key="cmp")
    block = data["metrics"][0]
    assert block["metric"] == "accuracy"
    cand = next(r for r in block["rows"] if r["name"] == "cand")
    assert cand["mean"] == pytest.approx(0.7)
    assert cand["delta"] == pytest.approx(0.2)


def test_report_data_has_win_rates(store):
    populate(store)
    data = build_report(store, configs(), metrics=["accuracy"], baseline_name="base", comparison_key="cmp")
    assert data["winrates"]["total"] == 2
    cand = next(r for r in data["winrates"]["rows"] if r["name"] == "cand")
    assert cand["wins"] == 2


def test_render_html_contains_key_facts(store):
    populate(store)
    html = render_html(build_report(store, configs(), ["accuracy"], "base", "cmp"))
    assert "accuracy" in html
    assert "cand" in html
    assert "0.700" in html  # cand mean
    assert "t1" in html  # per-test drill-down


def test_write_report_creates_file(store, tmp_path):
    populate(store)
    out = tmp_path / "report.html"
    path = write_report(store, configs(), ["accuracy"], str(out), baseline_name="base", comparison_key="cmp")
    assert out.exists()
    assert out.read_text().strip() != ""
    assert str(out) == path


def test_html_escapes_untrusted_output(store):
    # a model answer containing markup must not be rendered as raw HTML
    rid = store.add_result_row(
        "t1", CAND, run_id=a_run(store, CAND), output="<script>alert(1)</script>"
    )
    store.set_grading(rid, "a1", type="rubric", score=1.0, passed=True, metric="accuracy")
    graded(store, BASE, "t1", 0.5)
    html = render_html(build_report(store, configs(), ["accuracy"], "base"))
    assert "<script>alert(1)</script>" not in html
