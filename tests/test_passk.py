"""Tests for the breadth-arm pass@k analysis (src/errorbars/passk.py)."""

import numpy as np
import pytest

from errorbars import passk


# ---------------------------------------------------------------------------
# Chen et al. unbiased estimator
# ---------------------------------------------------------------------------

def test_passk_known_values():
    assert passk.passk_unbiased(0, 20, 4) == 0.0
    assert passk.passk_unbiased(20, 20, 4) == 1.0
    assert passk.passk_unbiased(7, 20, 1) == pytest.approx(7 / 20)
    # 1 - C(15, 4)/C(20, 4) for c = 5, n = 20, k = 4
    from math import comb
    assert passk.passk_unbiased(5, 20, 4) == pytest.approx(
        1 - comb(15, 4) / comb(20, 4))
    # fewer failures than k: certain to hit a success
    assert passk.passk_unbiased(18, 20, 4) == 1.0
    with pytest.raises(ValueError):
        passk.passk_unbiased(5, 4, 5)


def test_problem_values_match_scalar():
    counts = np.array([0, 3, 11, 20])
    vals = passk.problem_values(counts, 20, 4)
    for c, v in zip(counts, vals):
        assert v == pytest.approx(passk.passk_unbiased(int(c), 20, 4))


# ---------------------------------------------------------------------------
# W_m: exact sum (Milletari & Kundu Eq. 30), Monte Carlo as the check
# ---------------------------------------------------------------------------

def test_w_m_exact_reduces_at_k1():
    for p in (0.05, 0.3, 0.5, 0.9):
        for m in (5, 20):
            assert passk.w_m_exact(1, m, p) == pytest.approx(p * (1 - p) / m)


def test_w_m_exact_matches_monte_carlo():
    rng = np.random.default_rng(20261123)
    for k, m, p in ((4, 20, 0.4), (4, 5, 0.6), (2, 10, 0.25), (8, 20, 0.7)):
        exact = passk.w_m_exact(k, m, p)
        mc = passk.w_m_mc(k, m, p, draws=200_000, rng=rng)
        # relative tolerance, with an absolute floor: where h_k is
        # near-degenerate the variance is ~1e-7 and MC noise dominates
        assert mc == pytest.approx(exact, rel=0.05, abs=1e-6)


# ---------------------------------------------------------------------------
# Correction factor: the frozen unit test
# ---------------------------------------------------------------------------

def test_correction_factor_reduces_at_K_equals_n():
    rng = np.random.default_rng(1)
    counts = rng.binomial(20, 0.45, size=378)
    for N in (25, 50, 100, 200):
        comp = passk.correction_factor(counts, 20, 4, N, 20)
        assert comp["factor"] == pytest.approx(np.sqrt(1 - N / 378))
        assert comp["Var_B"] == 0.0
        assert comp["W_K"] == comp["W_n"]


def test_correction_factor_components_for_K_below_n():
    rng = np.random.default_rng(2)
    counts = rng.binomial(20, 0.5, size=120)
    comp = passk.correction_factor(counts, 20, 4, 50, 10,
                                   rng=np.random.default_rng(3))
    assert comp["Var_B"] > 0
    assert comp["V_sup"] > 0
    assert comp["factor"] == pytest.approx(
        np.sqrt((comp["Var_A"] + comp["Var_B"]) / comp["V_sup"]))


# ---------------------------------------------------------------------------
# Decision rule and arm outcome
# ---------------------------------------------------------------------------

def test_decide_boundaries():
    # corrected coverage 0.95 at B=2000: MC upper well above 0.94
    assert passk.decide(0.95, 2000)["verdict"] == "HOLDS"
    # coverage 0.90: MC upper below 0.94
    assert passk.decide(0.90, 2000)["verdict"] == "UNDER-COVERS"
    # coverage 0.99: MC lower above 0.97 -> CONSERVATIVE takes precedence
    assert passk.decide(0.99, 2000)["verdict"] == "CONSERVATIVE"


def test_arm_outcome_wording():
    assert passk.arm_outcome({"a": "HOLDS", "b": "HOLDS"}) == \
        "the guidance carries over to pass@k"
    assert passk.arm_outcome({"a": "HOLDS", "b": "CONSERVATIVE"}) == \
        "the guidance carries over to pass@k"
    out = passk.arm_outcome({"model-x": "UNDER-COVERS", "b": "HOLDS"})
    assert out == "carries over, with a named exception (model-x)"
    assert passk.arm_outcome({"a": "UNDER-COVERS", "b": "UNDER-COVERS"}) == \
        "does not carry over"


def test_inclusion_rule():
    # pass@1 = 0.5, pass@4 high but below the 0.98 ceiling
    counts = np.full(100, 10)
    assert passk.included_in_decision(counts, 20)
    # near-ceiling model: pass@4 above 0.98 -> descriptive only
    counts = np.full(100, 19)
    assert not passk.included_in_decision(counts, 20)
    # below the floor: pass@1 < 0.02
    counts = np.zeros(100, dtype=int)
    counts[0] = 20
    assert not passk.included_in_decision(counts, 20)


# ---------------------------------------------------------------------------
# Coverage study smoke test on a synthetic pool
# ---------------------------------------------------------------------------

def test_coverage_study_smoke():
    rng = np.random.default_rng(4)
    counts = rng.binomial(20, rng.beta(6, 6, size=60))
    rows = passk.coverage_study(
        counts, 20, [dict(passk.PRIMARY_CELL), {"k": 1, "N": 25, "K": 5}],
        B=40, seed=5, bootstrap_draws=50, n_subsets=50)
    primary, naive_cell = rows
    assert primary["label"] == "primary"
    assert primary["correction_factor"] == pytest.approx(np.sqrt(1 - 50 / 60))
    assert 0.0 <= primary["cluster_t_coverage_corrected"] <= 1.0
    assert "naive_wilson_coverage" in naive_cell
    assert "naive_wilson_coverage" not in primary
