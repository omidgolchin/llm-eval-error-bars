"""Tests for the breadth-arm harness (src/errorbars/codeeval.py).

The scorer-validation test runs EvalPlus's own checks on the first
few pinned MBPP+ problems: canonical solutions must pass all base
and plus tests, broken variants must fail. It is skipped when the
pinned ``evalplus`` package is not installed.
"""

import pytest

from errorbars import codeeval

evalplus = pytest.importorskip("evalplus")


def test_prompt_template_is_the_frozen_one():
    template = codeeval.prompt_template()
    assert template == (
        "Please provide a self-contained Python script that solves the "
        "following problem in a markdown code block:\n```python\n"
        "{problem}\n```\n")


def test_build_prompt_strips_problem():
    msg = codeeval.build_prompt('  """\nWrite a function.\n"""\n')
    assert msg.startswith("Please provide a self-contained Python script")
    assert '"""\nWrite a function.\n"""' in msg
    assert msg.endswith("```\n")


def test_pinned_datasets_have_frozen_sizes():
    assert len(codeeval.load_mbppplus_raw()) == 378
    assert len(codeeval.load_humanevalplus_raw()) == 164


def test_scorer_validation_first_problems():
    report = codeeval.validate_scorer(dataset="mbpp", n_problems=3,
                                      isolated=True)
    assert report["ok"], report
    for entry in report["problems"].values():
        assert entry["canonical"]["pass"]
        assert all(not r["pass"] for r in entry["broken"].values())
