import json
import re

from scripts_repo.compare_runs import (
    CellKey,
    extract_cells,
    classify,
    diff_cells,
    summarize,
    summarize_deterministic,
    summarize_best,
    best_winner,
    diff_test_keys,
    read_eval_id,
    errored_tests,
    parse_provider_yaml,
    extract_request_info,
    resolve_endpoints,
    build_curl,
    build_curls,
    render_html,
    main,
)
from scripts_repo.tests._fixtures import (
    rubric_result,
    deterministic_result,
    make_eval_json,
)

# --- extract_cells -------------------------------------------------------


def test_extract_cells_one_per_assertion():
    ev = make_eval_json(
        [
            rubric_result(
                "prod",
                "t1",
                "research_rubrics",
                [("cover period", "Explicit", 5), ("has headers", "Comms", 2)],
                [0.8, 0.4],
            ),
        ]
    )
    cells = extract_cells(ev)
    assert len(cells) == 2
    k = CellKey(test="t1", prompt="user_only", assertion="cover period")
    assert cells[k].score == 0.8
    assert cells[k].metric == "Explicit"
    assert cells[k].weight == 5
    assert cells[k].suite == "research_rubrics"
    assert cells[k].kind == "rubric"


def test_extract_cells_includes_all_suites():
    # No allowlist anymore: a suite we never special-cased is still extracted.
    ev = make_eval_json(
        [
            rubric_result("prod", "t1", "some_new_suite", [("a", "X", 1)], [0.8]),
        ]
    )
    cells = extract_cells(ev)
    assert {c.suite for c in cells.values()} == {"some_new_suite"}


def test_extract_cells_suite_filter_still_available():
    ev = make_eval_json(
        [
            rubric_result("prod", "t1", "research_rubrics", [("a", "X", 1)], [0.8]),
            rubric_result("prod", "h1", "agentharm_refusal", [("nr", "nr", 1)], [1.0]),
        ]
    )
    cells = extract_cells(ev, suites=["research_rubrics"])
    assert {k.test for k in cells} == {"t1"}


def test_extract_cells_suiteless_test_grouped_under_no_suite():
    fact = deterministic_result("prod", "fact", None, [("icontains", "Paris")], [True])
    cells = extract_cells(make_eval_json([fact]))
    cell = next(iter(cells.values()))
    assert cell.suite == "(no suite)"
    assert cell.kind == "deterministic"
    assert cell.passed is True


def test_extract_cells_deterministic_kind_and_pass():
    ev = make_eval_json(
        [
            deterministic_result(
                "prod",
                "f1",
                "simple_facts",
                [("icontains", "Paris"), ("icontains", "London")],
                [True, False],
            ),
        ]
    )
    cells = extract_cells(ev)
    paris = cells[CellKey("f1", "user_only", "Paris")]
    london = cells[CellKey("f1", "user_only", "London")]
    assert paris.kind == "deterministic" and paris.passed is True
    assert london.kind == "deterministic" and london.passed is False


def test_extract_cells_mixed_kinds_in_one_test():
    # assert[0] python (deterministic), assert[1] llm-rubric (rubric).
    entry = {
        "provider": {"label": "prod"},
        "prompt": {"label": "user_only"},
        "testCase": {
            "description": "t1",
            "metadata": {"suite": "research_rubrics"},
            "assert": [
                {"type": "python", "value": "file://x.py"},
                {"type": "llm-rubric", "value": "quality", "metric": "Q", "weight": 1},
            ],
        },
        "gradingResult": {
            "componentResults": [
                {"score": 1.0, "pass": True},
                {"score": 0.9, "pass": True},
            ]
        },
    }
    cells = extract_cells(make_eval_json([entry]))
    py = cells[CellKey("t1", "user_only", "file://x.py")]
    rub = cells[CellKey("t1", "user_only", "quality")]
    assert py.kind == "deterministic" and py.passed is True
    assert rub.kind == "rubric" and rub.score == 0.9


def test_extract_cells_disambiguates_duplicate_rubric_text():
    ev = make_eval_json(
        [
            rubric_result(
                "prod",
                "t1",
                "research_rubrics",
                [("same text", "A", 1), ("same text", "B", 1)],
                [0.2, 0.7],
            ),
        ]
    )
    cells = extract_cells(ev)
    assert cells[CellKey("t1", "user_only", "same text")].score == 0.2
    assert cells[CellKey("t1", "user_only", "same text#1")].score == 0.7


def test_extract_cells_null_score_coerced_to_zero():
    entry = {
        "provider": {"label": "prod"},
        "prompt": {"label": "user_only"},
        "testCase": {
            "description": "t1",
            "metadata": {"suite": "research_rubrics"},
            "assert": [
                {"type": "llm-rubric", "value": "q", "metric": "Q", "weight": 1}
            ],
        },
        "gradingResult": {"componentResults": [{"score": None}]},
    }
    cells = extract_cells(make_eval_json([entry]))
    assert cells[CellKey("t1", "user_only", "q")].score == 0.0


def test_extract_cells_deterministic_pass_falls_back_to_score():
    # componentResult lacks an explicit "pass"; derive it from score >= 0.5.
    entry = {
        "provider": {"label": "prod"},
        "prompt": {"label": "user_only"},
        "testCase": {
            "description": "t1",
            "metadata": {"suite": "simple_facts"},
            "assert": [{"type": "icontains", "value": "Paris"}],
        },
        "gradingResult": {"componentResults": [{"score": 1.0}]},
    }
    cells = extract_cells(make_eval_json([entry]))
    assert cells[CellKey("t1", "user_only", "Paris")].passed is True


def test_extract_cells_falls_back_to_prompt_when_no_description():
    # The stock_prices suite gives every row the same file:// assertion and no
    # description; keyed on description alone they'd collapse onto one CellKey.
    # Fall back to the rendered prompt so distinct tests stay distinct.
    def stock(prompt):
        return {
            "provider": {"label": "prod"},
            "prompt": {
                "label": "user_only",
                "raw": json.dumps([{"role": "user", "content": prompt}]),
            },
            "testCase": {
                "vars": {"user": prompt},
                "metadata": {"suite": "stock_prices"},
                "assert": [
                    {
                        "type": "python",
                        "value": "file://assertions/assert_stock_price.py",
                    }
                ],
            },
            "gradingResult": {"componentResults": [{"score": 1.0, "pass": True}]},
        }

    ev = make_eval_json(
        [
            stock("Latest price of Arm Holdings (ARM)?"),
            stock("Latest price of HSBC (HSBA)?"),
        ]
    )
    cells = extract_cells(ev)
    assert len(cells) == 2
    assert {k.test for k in cells} == {
        "Latest price of Arm Holdings (ARM)?",
        "Latest price of HSBC (HSBA)?",
    }


def test_extract_cells_filters_by_provider_label():
    # A single unified eval file holds both providers' results for the same
    # test. Without a filter their identical CellKeys collide; the provider
    # filter splits them into the two sides.
    ev = make_eval_json(
        [
            rubric_result("prod", "t1", "research_rubrics", [("a", "X", 1)], [0.8]),
            rubric_result("dev", "t1", "research_rubrics", [("a", "X", 1)], [0.3]),
        ]
    )
    prod = extract_cells(ev, provider_label="prod")
    dev = extract_cells(ev, provider_label="dev")
    assert prod[CellKey("t1", "user_only", "a")].score == 0.8
    assert dev[CellKey("t1", "user_only", "a")].score == 0.3


def test_extract_cells_no_provider_label_includes_all():
    # Back-compat: without a label, every result counts (single-provider files).
    ev = make_eval_json(
        [
            rubric_result("prod", "t1", "research_rubrics", [("a", "X", 1)], [0.8]),
        ]
    )
    assert len(extract_cells(ev)) == 1


def test_errored_tests_filters_by_provider_label():
    ev = make_eval_json(
        [
            _provider_error_entry("t_err", "s0", label="dev"),
            rubric_result("prod", "t_err", "research_rubrics", [("a", "A", 1)], [0.9]),
        ]
    )
    assert errored_tests(ev, provider_label="dev") == {"t_err"}
    assert errored_tests(ev, provider_label="prod") == set()


def test_extract_request_info_filters_by_provider_label():
    ev = {
        "results": {
            "results": [
                {
                    "provider": {"id": "openai:chat:Prod/M", "label": "prod"},
                    "prompt": {"raw": '[{"role":"user","content":"hi"}]'},
                    "testCase": {"description": "t1", "metadata": {"suite": "s"}},
                },
                {
                    "provider": {"id": "openai:chat:Dev/M", "label": "dev"},
                    "prompt": {"raw": '[{"role":"user","content":"hi"}]'},
                    "testCase": {"description": "t1", "metadata": {"suite": "s"}},
                },
            ]
        }
    }
    info = extract_request_info(ev, provider_label="dev")
    assert info["t1"]["model"] == "Dev/M"
    assert info["t1"]["provider_label"] == "dev"


def test_build_curls_filters_by_provider_label(tmp_path):
    (tmp_path / "providers").mkdir()
    (tmp_path / "providers" / "p.yaml").write_text(_PROVIDER_YAML, encoding="utf-8")
    # _PROVIDER_YAML's label is gw_prod; a second provider shares the test desc.
    ev = {
        "config": {"providers": ["file://providers/p.yaml"]},
        "results": {
            "results": [
                {
                    "provider": {"id": "openai:chat:Q", "label": "gw_prod"},
                    "prompt": {"raw": '[{"role":"user","content":"hi"}]'},
                    "testCase": {"description": "t1", "metadata": {"suite": "s"}},
                },
                {
                    "provider": {"id": "openai:chat:Q", "label": "other"},
                    "prompt": {"raw": "[]"},
                    "testCase": {"description": "t1", "metadata": {"suite": "s"}},
                },
            ]
        },
    }
    curls = build_curls(ev, tmp_path, provider_label="gw_prod")
    assert "t1" in curls
    assert "127.0.0.1:8082" in curls["t1"]


def test_main_splits_single_file_by_provider(tmp_path):
    # One unified file with both providers; --{baseline,candidate}-provider pick
    # the two sides. prod=0.5 vs dev=0.9 on the same assertion => improved.
    ev = make_eval_json(
        [
            rubric_result(
                "prod_label", "t1", "research_rubrics", [("a", "X", 1)], [0.5]
            ),
            rubric_result(
                "dev_label", "t1", "research_rubrics", [("a", "X", 1)], [0.9]
            ),
        ]
    )
    f = _write(tmp_path, "unified.json", ev)
    out = tmp_path / "report.html"
    rc = main(
        [
            str(f),
            str(f),
            "--baseline-provider",
            "prod_label",
            "--candidate-provider",
            "dev_label",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert "1 improved" in out.read_text(encoding="utf-8")


# --- diff / classify / summary / drift -----------------------------------


def _cells(ev):
    return extract_cells(ev)


def test_classify_boundaries():
    assert classify(0.05, 0.05) == "within"
    assert classify(-0.05, 0.05) == "within"
    assert classify(0.06, 0.05) == "improved"
    assert classify(-0.06, 0.05) == "regressed"


def test_diff_improved_regressed_within():
    base = _cells(
        make_eval_json(
            [
                rubric_result(
                    "prod",
                    "t1",
                    "research_rubrics",
                    [("a", "X", 1), ("b", "X", 1), ("c", "X", 1)],
                    [0.50, 0.90, 0.50],
                ),
            ]
        )
    )
    cand = _cells(
        make_eval_json(
            [
                rubric_result(
                    "cand",
                    "t1",
                    "research_rubrics",
                    [("a", "X", 1), ("b", "X", 1), ("c", "X", 1)],
                    [0.90, 0.40, 0.52],
                ),
            ]
        )
    )
    diffs = {d.key.assertion: d for d in diff_cells(base, cand, 0.05)}
    assert diffs["a"].status == "improved"
    assert diffs["b"].status == "regressed"
    assert diffs["c"].status == "within"
    assert round(diffs["a"].delta, 2) == 0.40


def test_diff_deterministic_transitions():
    base = _cells(
        make_eval_json(
            [
                deterministic_result(
                    "prod",
                    "f1",
                    "simple_facts",
                    [("icontains", "a"), ("icontains", "b"), ("icontains", "c")],
                    [False, True, True],
                ),
            ]
        )
    )
    cand = _cells(
        make_eval_json(
            [
                deterministic_result(
                    "cand",
                    "f1",
                    "simple_facts",
                    [("icontains", "a"), ("icontains", "b"), ("icontains", "c")],
                    [True, False, True],
                ),
            ]
        )
    )
    diffs = {d.key.assertion: d for d in diff_cells(base, cand, 0.05)}
    assert diffs["a"].status == "improved"  # fail -> pass
    assert diffs["b"].status == "regressed"  # pass -> fail
    assert diffs["c"].status == "same"  # pass -> pass
    assert all(d.delta is None for d in diffs.values())
    assert diffs["a"].kind == "deterministic"


def test_diff_new_and_removed():
    base = _cells(
        make_eval_json(
            [
                rubric_result("prod", "t1", "research_rubrics", [("a", "X", 1)], [0.5]),
                rubric_result("prod", "tg", "research_rubrics", [("g", "X", 1)], [0.5]),
            ]
        )
    )
    cand = _cells(
        make_eval_json(
            [
                rubric_result("cand", "t1", "research_rubrics", [("a", "X", 1)], [0.5]),
                rubric_result("cand", "tn", "research_rubrics", [("n", "X", 1)], [0.5]),
            ]
        )
    )
    by_status = {}
    for d in diff_cells(base, cand, 0.05):
        by_status.setdefault(d.status, []).append(d.key.assertion)
    assert by_status["new"] == ["n"]
    assert by_status["removed"] == ["g"]


def test_summarize_counts_rubric_only():
    base = _cells(
        make_eval_json(
            [
                rubric_result(
                    "prod",
                    "t1",
                    "research_rubrics",
                    [("a", "X", 1), ("b", "X", 1)],
                    [0.5, 0.9],
                ),
                deterministic_result(
                    "prod", "f1", "simple_facts", [("icontains", "x")], [True]
                ),
            ]
        )
    )
    cand = _cells(
        make_eval_json(
            [
                rubric_result(
                    "cand",
                    "t1",
                    "research_rubrics",
                    [("a", "X", 1), ("b", "X", 1)],
                    [0.9, 0.4],
                ),
                deterministic_result(
                    "cand", "f1", "simple_facts", [("icontains", "x")], [False]
                ),
            ]
        )
    )
    counts = summarize(diff_cells(base, cand, 0.05))
    assert counts["improved"] == 1
    assert counts["regressed"] == 1
    assert counts["within"] == 0
    assert counts["new"] == 0
    assert counts["removed"] == 0


def test_summarize_deterministic_counts():
    base = _cells(
        make_eval_json(
            [
                deterministic_result(
                    "prod",
                    "f1",
                    "simple_facts",
                    [
                        ("icontains", "a"),
                        ("icontains", "b"),
                        ("icontains", "c"),
                        ("icontains", "d"),
                    ],
                    [False, True, True, False],
                ),
                rubric_result("prod", "t1", "research_rubrics", [("r", "X", 1)], [0.5]),
            ]
        )
    )
    cand = _cells(
        make_eval_json(
            [
                deterministic_result(
                    "cand",
                    "f1",
                    "simple_facts",
                    [
                        ("icontains", "a"),
                        ("icontains", "b"),
                        ("icontains", "c"),
                        ("icontains", "d"),
                    ],
                    [True, False, True, False],
                ),
                rubric_result("cand", "t1", "research_rubrics", [("r", "X", 1)], [0.9]),
            ]
        )
    )
    det = summarize_deterministic(diff_cells(base, cand, 0.05))
    assert det["new_passes"] == 1  # a: fail -> pass
    assert det["new_fails"] == 1  # b: pass -> fail
    assert det["total_passes"] == 2  # a, c pass in candidate
    assert det["total_fails"] == 2  # b, d fail in candidate


def test_diff_test_keys():
    base = _cells(
        make_eval_json(
            [
                rubric_result("prod", "t1", "research_rubrics", [("a", "X", 1)], [0.5]),
                rubric_result("prod", "tg", "research_rubrics", [("g", "X", 1)], [0.5]),
            ]
        )
    )
    cand = _cells(
        make_eval_json(
            [
                rubric_result("cand", "t1", "research_rubrics", [("a", "X", 1)], [0.5]),
                rubric_result("cand", "tn", "research_rubrics", [("n", "X", 1)], [0.5]),
            ]
        )
    )
    only_base, only_cand = diff_test_keys(base, cand)
    assert only_base == ["tg"]
    assert only_cand == ["tn"]


# --- render_html ---------------------------------------------------------


def test_render_html_contains_summary_and_markers():
    base = _cells(
        make_eval_json(
            [
                rubric_result(
                    "prod",
                    "t1",
                    "research_rubrics",
                    [("a", "X", 1), ("b", "X", 1)],
                    [0.5, 0.9],
                ),
            ]
        )
    )
    cand = _cells(
        make_eval_json(
            [
                rubric_result(
                    "cand",
                    "t1",
                    "research_rubrics",
                    [("a", "X", 1), ("b", "X", 1)],
                    [0.9, 0.4],
                ),
            ]
        )
    )
    diffs = diff_cells(base, cand, 0.05)
    out = render_html(diffs, ([], []), 0.05)
    assert "<html" in out.lower()
    assert "Rubric Tests:" in out
    assert "1 improved" in out
    assert "1 regressed" in out
    assert "research_rubrics" in out
    assert "status-improved" in out
    assert "status-regressed" in out


def test_render_html_per_suite_and_aggregate_summaries():
    base = _cells(
        make_eval_json(
            [
                rubric_result("prod", "t1", "research_rubrics", [("a", "X", 1)], [0.9]),
                deterministic_result(
                    "prod",
                    "f1",
                    "simple_facts",
                    [("icontains", "x"), ("icontains", "y")],
                    [True, True],
                ),
            ]
        )
    )
    cand = _cells(
        make_eval_json(
            [
                rubric_result("cand", "t1", "research_rubrics", [("a", "X", 1)], [0.4]),
                deterministic_result(
                    "cand",
                    "f1",
                    "simple_facts",
                    [("icontains", "x"), ("icontains", "y")],
                    [True, False],
                ),
            ]
        )
    )
    diffs = diff_cells(base, cand, 0.05)
    out = render_html(diffs, ([], []), 0.05)
    # Both suites grouped and labelled.
    assert "research_rubrics" in out and "simple_facts" in out
    # Per-suite deterministic summary present.
    assert "Deterministic Tests:" in out
    assert "1 new fails" in out
    assert "total passes" in out
    # Aggregate at the very top includes both kinds.
    agg = out.split("research_rubrics")[0]
    assert "Rubric Tests:" in agg
    assert "Deterministic Tests:" in agg


def test_render_html_deterministic_shows_pass_fail_and_blank_delta():
    base = _cells(
        make_eval_json(
            [
                deterministic_result(
                    "prod", "f1", "simple_facts", [("icontains", "x")], [True]
                ),
            ]
        )
    )
    cand = _cells(
        make_eval_json(
            [
                deterministic_result(
                    "cand", "f1", "simple_facts", [("icontains", "x")], [False]
                ),
            ]
        )
    )
    diffs = diff_cells(base, cand, 0.05)
    out = render_html(diffs, ([], []), 0.05)
    assert ">pass<" in out  # baseline verdict
    assert ">fail<" in out  # candidate verdict
    assert "status-regressed" in out


def test_render_html_shows_drift_banner():
    base = _cells(
        make_eval_json(
            [
                rubric_result("prod", "t1", "research_rubrics", [("a", "X", 1)], [0.5]),
            ]
        )
    )
    cand = _cells(
        make_eval_json(
            [
                rubric_result("cand", "t1", "research_rubrics", [("a", "X", 1)], [0.5]),
            ]
        )
    )
    diffs = diff_cells(base, cand, 0.05)
    out = render_html(diffs, (["t_missing"], []), 0.05)
    assert "config drift" in out.lower()
    assert "t_missing" in out


def test_render_html_escapes_markup():
    base = _cells(
        make_eval_json(
            [
                rubric_result(
                    "prod",
                    "t1",
                    "research_rubrics",
                    [("<script>x</script>", "X", 1)],
                    [0.5],
                ),
            ]
        )
    )
    cand = _cells(
        make_eval_json(
            [
                rubric_result(
                    "cand",
                    "t1",
                    "research_rubrics",
                    [("<script>x</script>", "X", 1)],
                    [0.9],
                ),
            ]
        )
    )
    diffs = diff_cells(base, cand, 0.05)
    out = render_html(diffs, ([], []), 0.05)
    assert "<script>x</script>" not in out
    assert "&lt;script&gt;" in out


# --- main / CLI ----------------------------------------------------------


def _write(tmp_path, name, ev):
    p = tmp_path / name
    p.write_text(json.dumps(ev), encoding="utf-8")
    return p


def test_main_writes_report_and_returns_zero(tmp_path, capsys):
    base = _write(
        tmp_path,
        "base.json",
        make_eval_json(
            [
                rubric_result("prod", "t1", "research_rubrics", [("a", "X", 1)], [0.5]),
            ]
        ),
    )
    cand = _write(
        tmp_path,
        "cand.json",
        make_eval_json(
            [
                rubric_result("cand", "t1", "research_rubrics", [("a", "X", 1)], [0.9]),
            ]
        ),
    )
    out = tmp_path / "report.html"
    rc = main([str(base), str(cand), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    assert "1 improved" in out.read_text(encoding="utf-8")
    assert "1 improved" in capsys.readouterr().out


def test_main_compares_all_suites_by_default(tmp_path):
    # No --suite given: both rubric and deterministic suites appear.
    base = _write(
        tmp_path,
        "base.json",
        make_eval_json(
            [
                rubric_result("prod", "t1", "research_rubrics", [("a", "X", 1)], [0.5]),
                deterministic_result(
                    "prod", "f1", "simple_facts", [("icontains", "x")], [True]
                ),
            ]
        ),
    )
    cand = _write(
        tmp_path,
        "cand.json",
        make_eval_json(
            [
                rubric_result("cand", "t1", "research_rubrics", [("a", "X", 1)], [0.9]),
                deterministic_result(
                    "cand", "f1", "simple_facts", [("icontains", "x")], [True]
                ),
            ]
        ),
    )
    out = tmp_path / "report.html"
    main([str(base), str(cand), "--out", str(out)])
    text = out.read_text(encoding="utf-8")
    assert "research_rubrics" in text
    assert "simple_facts" in text


def test_main_respects_suite_override(tmp_path):
    base = _write(
        tmp_path,
        "base.json",
        make_eval_json(
            [
                rubric_result("prod", "t1", "research_rubrics", [("a", "X", 1)], [0.5]),
                rubric_result(
                    "prod", "h1", "agentharm_refusal", [("nr", "non_refusal", 1)], [1.0]
                ),
            ]
        ),
    )
    cand = _write(
        tmp_path,
        "cand.json",
        make_eval_json(
            [
                rubric_result("cand", "t1", "research_rubrics", [("a", "X", 1)], [0.9]),
                rubric_result(
                    "cand", "h1", "agentharm_refusal", [("nr", "non_refusal", 1)], [0.0]
                ),
            ]
        ),
    )
    out = tmp_path / "report.html"
    main([str(base), str(cand), "--out", str(out), "--suite", "research_rubrics"])
    text = out.read_text(encoding="utf-8")
    assert "research_rubrics" in text
    assert "agentharm_refusal" not in text


# --- alternating per-test row highlight ----------------------------------


def _row_parities_by_test(out, test_descs):
    result = {t: set() for t in test_descs}
    for cls, body in re.findall(r'<tr class="([^"]*)">(.*?)</tr>', out, re.DOTALL):
        parity = "a" if "test-a" in cls else "b" if "test-b" in cls else None
        for t in test_descs:
            if f">{t}<" in body:
                result[t].add(parity)
    return result


def test_render_html_rows_of_same_test_share_one_parity():
    base = _cells(
        make_eval_json(
            [
                rubric_result(
                    "prod",
                    "t_a",
                    "research_rubrics",
                    [("a", "A", 1), ("b", "B", 1)],
                    [0.5, 0.5],
                ),
                rubric_result(
                    "prod",
                    "t_b",
                    "research_rubrics",
                    [("c", "C", 1), ("d", "D", 1)],
                    [0.5, 0.5],
                ),
            ]
        )
    )
    cand = _cells(
        make_eval_json(
            [
                rubric_result(
                    "cand",
                    "t_a",
                    "research_rubrics",
                    [("a", "A", 1), ("b", "B", 1)],
                    [0.6, 0.6],
                ),
                rubric_result(
                    "cand",
                    "t_b",
                    "research_rubrics",
                    [("c", "C", 1), ("d", "D", 1)],
                    [0.6, 0.6],
                ),
            ]
        )
    )
    out = render_html(diff_cells(base, cand, 0.05), ([], []), 0.05)
    parities = _row_parities_by_test(out, ["t_a", "t_b"])
    assert len(parities["t_a"]) == 1
    assert len(parities["t_b"]) == 1
    assert parities["t_a"] != parities["t_b"]


def test_render_html_defines_neutral_row_colors():
    out = render_html([], ([], []), 0.05)
    assert ".test-a" in out and ".test-b" in out


# --- promptfoo UI links --------------------------------------------------


def test_extract_cells_search_prefers_metadata_id():
    ev = make_eval_json(
        [
            rubric_result(
                "prod",
                "researchrubrics[X] abc123",
                "research_rubrics",
                [("a", "A", 1)],
                [0.5],
                metadata_extra={"sample_id": "abc123"},
            ),
        ]
    )
    cell = next(iter(extract_cells(ev).values()))
    assert cell.search == "abc123"


def test_extract_cells_search_falls_back_to_description():
    ev = make_eval_json(
        [
            rubric_result(
                "prod",
                "agentharm[x] 9-2 foo",
                "agentharm_refusal",
                [("nr", "non_refusal", 1)],
                [1.0],
            ),
        ]
    )
    cell = next(iter(extract_cells(ev).values()))
    assert cell.search == "agentharm[x] 9-2 foo"


def test_read_eval_id_prefers_top_level():
    assert read_eval_id(make_eval_json([], eval_id="eval-xyz")) == "eval-xyz"


def test_read_eval_id_falls_back_to_baseline_meta():
    assert read_eval_id({"_baseline_meta": {"eval_id": "eval-frozen"}}) == "eval-frozen"


def test_render_html_links_scores_when_eval_ids_given():
    base = _cells(
        make_eval_json(
            [
                rubric_result(
                    "prod",
                    "researchrubrics[X] abc123",
                    "research_rubrics",
                    [("a", "A", 1)],
                    [0.5],
                    metadata_extra={"sample_id": "abc123"},
                ),
            ]
        )
    )
    cand = _cells(
        make_eval_json(
            [
                rubric_result(
                    "cand",
                    "researchrubrics[X] abc123",
                    "research_rubrics",
                    [("a", "A", 1)],
                    [0.9],
                    metadata_extra={"sample_id": "abc123"},
                ),
            ]
        )
    )
    diffs = diff_cells(base, cand, 0.05)
    out = render_html(
        diffs,
        ([], []),
        0.05,
        baseline_eval_id="eval-base",
        candidate_eval_id="eval-cand",
    )
    assert 'href="http://localhost:3000/eval/eval-base?search=abc123"' in out
    assert 'href="http://localhost:3000/eval/eval-cand?search=abc123"' in out
    assert ">0.50</a>" in out


def test_render_html_custom_base_url():
    base = _cells(
        make_eval_json(
            [
                rubric_result(
                    "prod",
                    "t",
                    "research_rubrics",
                    [("a", "A", 1)],
                    [0.5],
                    metadata_extra={"sample_id": "s1"},
                ),
            ]
        )
    )
    cand = _cells(
        make_eval_json(
            [
                rubric_result(
                    "cand",
                    "t",
                    "research_rubrics",
                    [("a", "A", 1)],
                    [0.9],
                    metadata_extra={"sample_id": "s1"},
                ),
            ]
        )
    )
    diffs = diff_cells(base, cand, 0.05)
    out = render_html(
        diffs,
        ([], []),
        0.05,
        baseline_eval_id="e1",
        candidate_eval_id="e2",
        ui_base_url="http://host:9999",
    )
    assert 'href="http://host:9999/eval/e1?search=s1"' in out


def test_render_html_no_links_without_eval_ids():
    base = _cells(
        make_eval_json(
            [
                rubric_result("prod", "t", "research_rubrics", [("a", "A", 1)], [0.5]),
            ]
        )
    )
    cand = _cells(
        make_eval_json(
            [
                rubric_result("cand", "t", "research_rubrics", [("a", "A", 1)], [0.9]),
            ]
        )
    )
    diffs = diff_cells(base, cand, 0.05)
    out = render_html(diffs, ([], []), 0.05)
    assert "<a href" not in out


def test_render_html_no_link_for_missing_side():
    base = _cells(make_eval_json([]))
    cand = _cells(
        make_eval_json(
            [
                rubric_result(
                    "cand",
                    "t",
                    "research_rubrics",
                    [("a", "A", 1)],
                    [0.9],
                    metadata_extra={"sample_id": "s1"},
                ),
            ]
        )
    )
    diffs = diff_cells(base, cand, 0.05)
    out = render_html(
        diffs, ([], []), 0.05, baseline_eval_id="e1", candidate_eval_id="e2"
    )
    assert 'href="http://localhost:3000/eval/e2?search=s1"' in out
    assert "/eval/e1?" not in out


# --- errored-test markers ------------------------------------------------


def _provider_error_entry(desc, sample_id, label="cand"):
    return {
        "provider": {"label": label},
        "prompt": {"label": "user_only"},
        "failureReason": 2,
        "error": "API error: 504 Gateway Timeout",
        "testCase": {
            "description": desc,
            "metadata": {"suite": "research_rubrics", "sample_id": sample_id},
            "assert": [{"type": "llm-rubric", "value": "a", "metric": "A"}],
        },
        "gradingResult": {"componentResults": []},
    }


def test_errored_tests_detects_provider_and_grader_errors():
    grader_err = {
        "provider": {"label": "p"},
        "prompt": {"label": "user_only"},
        "failureReason": 1,
        "error": "Aggregate score 0.00 < 0.5 threshold",
        "testCase": {
            "description": "grader_err",
            "metadata": {"suite": "research_rubrics"},
            "assert": [{"type": "llm-rubric", "value": "a", "metric": "A"}],
        },
        "gradingResult": {
            "componentResults": [
                {"score": 0, "reason": "Bedrock API invoke model error: AccessDenied"}
            ]
        },
    }
    ev = make_eval_json(
        [
            _provider_error_entry("prov_err", "s0", label="p"),
            grader_err,
            rubric_result("p", "ok", "research_rubrics", [("a", "A", 1)], [0.9]),
        ]
    )
    assert errored_tests(ev) == {"prov_err", "grader_err"}


def test_errored_tests_clean_zero_is_not_error():
    refusal = {
        "provider": {"label": "p"},
        "prompt": {"label": "user_only"},
        "failureReason": 1,
        "error": "Aggregate score 0.00 < 0.5 threshold",
        "testCase": {
            "description": "refused",
            "metadata": {"suite": "agentharm_refusal"},
            "assert": [{"type": "llm-rubric", "value": "nr", "metric": "nr"}],
        },
        "gradingResult": {
            "componentResults": [
                {"score": 0, "reason": "The output is a refusal to assist."}
            ]
        },
    }
    assert errored_tests(make_eval_json([refusal])) == set()


def test_render_html_missing_errored_cell_shows_error_link():
    base = _cells(
        make_eval_json(
            [
                rubric_result(
                    "prod",
                    "t_err",
                    "research_rubrics",
                    [("a", "A", 1)],
                    [0.5],
                    metadata_extra={"sample_id": "sid-err"},
                ),
            ]
        )
    )
    cand_json = make_eval_json([_provider_error_entry("t_err", "sid-err")])
    cand = extract_cells(cand_json)
    diffs = diff_cells(base, cand, 0.05)
    out = render_html(
        diffs,
        (["t_err"], []),
        0.05,
        baseline_eval_id="eval-base",
        candidate_eval_id="eval-cand",
        candidate_errored=errored_tests(cand_json),
    )
    assert ">ERROR</a>" in out
    assert 'href="http://localhost:3000/eval/eval-cand?search=sid-err"' in out


def test_render_html_missing_not_errored_stays_dash():
    base = _cells(
        make_eval_json(
            [
                rubric_result(
                    "prod", "gone", "research_rubrics", [("g", "G", 1)], [0.5]
                ),
            ]
        )
    )
    cand = _cells(make_eval_json([]))
    diffs = diff_cells(base, cand, 0.05)
    out = render_html(
        diffs,
        (["gone"], []),
        0.05,
        baseline_eval_id="eb",
        candidate_eval_id="ec",
        candidate_errored=set(),
    )
    assert "ERROR" not in out
    assert "—" in out


def test_render_html_new_errored_cell_links_to_baseline():
    base_json = make_eval_json([_provider_error_entry("t_new", "sidn", label="prod")])
    base = extract_cells(base_json)
    cand = _cells(
        make_eval_json(
            [
                rubric_result(
                    "cand",
                    "t_new",
                    "research_rubrics",
                    [("a", "A", 1)],
                    [0.9],
                    metadata_extra={"sample_id": "sidn"},
                ),
            ]
        )
    )
    diffs = diff_cells(base, cand, 0.05)
    out = render_html(
        diffs,
        ([], ["t_new"]),
        0.05,
        baseline_eval_id="eb",
        candidate_eval_id="ec",
        baseline_errored=errored_tests(base_json),
    )
    assert 'href="http://localhost:3000/eval/eb?search=sidn"' in out
    assert ">ERROR</a>" in out


# --- copy-as-curl --------------------------------------------------------

_PROVIDER_YAML = """\
id: openai:chat:Qwen/Qwen3-Next-80B-A3B-Thinking-FP8
label: gw_prod
config:
  apiBaseUrl: http://127.0.0.1:8082/v1
  apiKey: dummy        # placeholder comment
  temperature: 0.7
  max_tokens: 100000
"""


def test_parse_provider_yaml_strips_comments_and_quotes():
    cfg = parse_provider_yaml(_PROVIDER_YAML)
    assert cfg["label"] == "gw_prod"
    assert cfg["id"] == "openai:chat:Qwen/Qwen3-Next-80B-A3B-Thinking-FP8"
    assert cfg["apiBaseUrl"] == "http://127.0.0.1:8082/v1"
    assert cfg["apiKey"] == "dummy"
    assert cfg["temperature"] == "0.7"
    assert cfg["max_tokens"] == "100000"


def test_extract_request_info_includes_model_and_messages():
    ev = {
        "results": {
            "results": [
                {
                    "provider": {"id": "openai:chat:Qwen/Q", "label": "gw_prod"},
                    "prompt": {"raw": '[{"role":"user","content":"hi"}]'},
                    "testCase": {
                        "description": "t1",
                        "metadata": {"suite": "research_rubrics"},
                    },
                }
            ]
        }
    }
    info = extract_request_info(ev)
    assert info["t1"]["model"] == "Qwen/Q"
    assert info["t1"]["provider_label"] == "gw_prod"
    assert info["t1"]["messages_raw"] == '[{"role":"user","content":"hi"}]'


def test_build_curl_has_endpoint_model_messages_and_shell_escapes():
    ep = {
        "url": "http://127.0.0.1:8082/v1",
        "api_key": "dummy",
        "temperature": "0.7",
        "max_tokens": "100000",
    }
    messages_raw = '[{"role":"user","content":"it' + "'" + 's ok"}]'
    curl = build_curl(ep, "Qwen/Q", messages_raw)
    assert curl.startswith("curl http://127.0.0.1:8082/v1/chat/completions ")
    assert "Authorization: Bearer dummy" in curl
    assert '"model": "Qwen/Q"' in curl
    assert '"temperature": 0.7' in curl
    assert '"max_tokens": 100000' in curl
    assert "'\\''" in curl


def test_build_curl_empty_when_no_url():
    assert build_curl({"url": None}, "m", "[]") == ""


def test_resolve_endpoints_from_yaml_files(tmp_path):
    (tmp_path / "providers").mkdir()
    (tmp_path / "providers" / "p.yaml").write_text(_PROVIDER_YAML, encoding="utf-8")
    ev = {
        "config": {"providers": ["file://providers/p.yaml"]},
        "results": {"results": [{"provider": {"label": "gw_prod"}}]},
    }
    eps = resolve_endpoints(ev, tmp_path)
    assert eps["gw_prod"]["url"] == "http://127.0.0.1:8082/v1"
    assert eps["gw_prod"]["api_key"] == "dummy"


def test_resolve_endpoints_override_url_without_yaml(tmp_path):
    ev = {
        "config": {"providers": []},
        "results": {"results": [{"provider": {"label": "gw_x"}}]},
    }
    eps = resolve_endpoints(ev, tmp_path, override_url="http://host:9/v1")
    assert eps["gw_x"]["url"] == "http://host:9/v1"


def test_render_html_copy_button_on_score_and_error_cells():
    base = _cells(
        make_eval_json(
            [
                rubric_result(
                    "prod",
                    "t_ok",
                    "research_rubrics",
                    [("a", "A", 1)],
                    [0.5],
                    metadata_extra={"sample_id": "s1"},
                ),
            ]
        )
    )
    cand_json = make_eval_json([_provider_error_entry("t_ok", "s1")])
    cand = extract_cells(cand_json)
    diffs = diff_cells(base, cand, 0.05)
    out = render_html(
        diffs,
        (["t_ok"], []),
        0.05,
        candidate_errored=errored_tests(cand_json),
        baseline_curls={"t_ok": "curl http://b/v1/chat/completions"},
        candidate_curls={"t_ok": "curl http://c/v1/chat/completions"},
    )
    assert "copy-btn" in out
    assert "navigator.clipboard.writeText" in out
    assert 'data-curl="curl http://b/v1/chat/completions"' in out
    assert 'data-curl="curl http://c/v1/chat/completions"' in out


def test_render_html_no_copy_button_for_bare_missing():
    base = _cells(
        make_eval_json(
            [
                rubric_result(
                    "prod", "gone", "research_rubrics", [("g", "G", 1)], [0.5]
                ),
            ]
        )
    )
    cand = _cells(make_eval_json([]))
    diffs = diff_cells(base, cand, 0.05)
    out = render_html(diffs, (["gone"], []), 0.05, candidate_curls={})
    assert '<button type="button" class="copy-btn"' not in out


# --- eval-name header ----------------------------------------------------


def test_render_html_shows_eval_names_in_header():
    base = _cells(
        make_eval_json(
            [
                rubric_result("prod", "t", "research_rubrics", [("a", "A", 1)], [0.5]),
            ]
        )
    )
    cand = _cells(
        make_eval_json(
            [
                rubric_result("cand", "t", "research_rubrics", [("a", "A", 1)], [0.9]),
            ]
        )
    )
    diffs = diff_cells(base, cand, 0.05)
    out = render_html(
        diffs,
        ([], []),
        0.05,
        baseline_eval_id="eval-base-123",
        candidate_eval_id="eval-cand-456",
    )
    assert "Baseline" in out and "Candidate" in out
    assert "eval-base-123" in out and "eval-cand-456" in out
    assert 'href="http://localhost:3000/eval/eval-base-123"' in out
    assert 'href="http://localhost:3000/eval/eval-cand-456"' in out


def test_render_html_eval_header_handles_missing_id():
    out = render_html([], ([], []), 0.05)
    assert "Baseline" in out and "Candidate" in out
    assert "(unknown)" in out


# --- select-best head-to-head -------------------------------------------


def _select_best(provider, desc, suite, won):
    """One result with a single select-best assertion (won => component passes)."""
    return deterministic_result(provider, desc, suite, [("select-best", "crit")], [won])


def test_extract_cells_tags_select_best_as_best_kind():
    ev = make_eval_json([_select_best("prod", "t1", "ab", True)])
    cell = extract_cells(ev)[CellKey("t1", "user_only", "crit")]
    assert cell.kind == "best"
    assert cell.passed is True


def test_best_winner_is_the_passing_side():
    base = _cells(make_eval_json([_select_best("prod", "t1", "ab", True)]))
    cand = _cells(make_eval_json([_select_best("cand", "t1", "ab", False)]))
    (diff,) = diff_cells(base, cand, 0.05)
    assert diff.kind == "best"
    assert best_winner(diff) == "prod"

    base = _cells(make_eval_json([_select_best("prod", "t1", "ab", False)]))
    cand = _cells(make_eval_json([_select_best("cand", "t1", "ab", True)]))
    (diff,) = diff_cells(base, cand, 0.05)
    assert best_winner(diff) == "candidate"


def test_best_winner_undecided_when_a_side_missing():
    base = _cells(make_eval_json([_select_best("prod", "t1", "ab", True)]))
    (diff,) = diff_cells(base, {}, 0.05)  # candidate side absent
    assert best_winner(diff) is None


def test_select_best_excluded_from_deterministic_summary():
    base = _cells(make_eval_json([_select_best("prod", "t1", "ab", True)]))
    cand = _cells(make_eval_json([_select_best("cand", "t1", "ab", False)]))
    diffs = diff_cells(base, cand, 0.05)
    det = summarize_deterministic(diffs)
    assert det == {"new_passes": 0, "new_fails": 0, "total_passes": 0, "total_fails": 0}


def test_summarize_best_counts_winners():
    base = _cells(
        make_eval_json(
            [
                _select_best("prod", "t1", "ab", True),
                _select_best("prod", "t2", "ab", False),
            ]
        )
    )
    cand = _cells(
        make_eval_json(
            [
                _select_best("cand", "t1", "ab", False),
                _select_best("cand", "t2", "ab", True),
            ]
        )
    )
    counts = summarize_best(diff_cells(base, cand, 0.05))
    assert counts == {"prod": 1, "candidate": 1, "undecided": 0}


def test_render_html_best_column_shows_winner():
    base = _cells(make_eval_json([_select_best("prod", "t1", "ab", False)]))
    cand = _cells(make_eval_json([_select_best("cand", "t1", "ab", True)]))
    out = render_html(diff_cells(base, cand, 0.05), ([], []), 0.05)
    assert "<th>best</th>" in out
    assert "Best (head-to-head):" in out
    assert "candidate" in out
