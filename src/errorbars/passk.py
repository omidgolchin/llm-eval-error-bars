"""Breadth-arm analysis: interval coverage for mean pass@k on code runs.

Implements the design frozen 2026-11-23 (see ``FREEZE_BREADTH.md``).
The estimand is the super-population mean of the per-problem pass@k
value: for problem ``i`` with success probability ``p_i`` over the
model's sampling distribution, ``pass@k`` is ``1 - (1 - p_i)**k``,
estimated from ``n`` samples per problem by the unbiased estimator of
Chen et al. (2021). Intervals are built by clustering over problems.

Primary cell: ``k = 4``, ``N = 50``, ``K = n = 20``. With ``K = n``
each subsample uses every sample a problem has, and the finite-pool
correction is exact: the correction factor is ``sqrt(1 - N/P)`` with
``P`` the pool size after exclusions.

Exploratory grid: ``k in {1, 2, 4, 8}``, ``N in {25, 50, 100, 200}``,
``K in {5, 10, 20}`` with ``k <= K``. For ``K < n`` the correction
factor is ``sqrt((Var_A + Var_B) / V_sup)`` with

- ``S2``    variance across problems of ``v_i(k, n)``;
- ``Var_A = (1 - N/P) * S2 / N``;
- ``Var_B`` mean over problems of the variance of ``v_i(k, K)`` across
  random K-of-n subsets of that problem's samples (Monte Carlo, 500
  subsets per problem; the subset success count is hypergeometric, so
  the subsets are drawn through that distribution);
- ``W_m``   mean over problems of ``Var[v_i(k, m)]`` when the success
  count is ``Binomial(m, p_hat_i)``. ``W_m`` is computed with the
  exact sum of Milletari & Kundu (2026, Eq. 30) -- see
  :func:`w_m_exact`. A Monte Carlo estimate (:func:`w_m_mc`) is kept
  as the implementation check. ``W_m`` is a plug-in at ``p_hat_i`` and
  is labelled approximate wherever it is used;
- ``V_sup = (S2 - W_n + W_K) / N``.

At ``K = n``: ``Var_B = 0`` and ``W_K = W_n``, so the factor reduces
to ``sqrt(1 - N/P)``. That reduction is a unit test
(``tests/test_passk.py``).

Coverage is scored against the pool truth (the mean of ``v_i(k, n)``
over all ``P`` problems). Both the raw coverage and the corrected
coverage (interval half-width scaled by the correction factor) are
reported, together with the analytic prediction: the raw coverage a
perfectly calibrated interval would show against the finite-pool
truth, ``2 * F_t(t_crit / factor) - 1`` with ``t_crit`` the 0.975
quantile of Student's t with ``N - 1`` degrees of freedom.

The decision at the primary cell is read on the cluster-t interval.
Per included model, with a Clopper-Pearson Monte Carlo interval on
the corrected coverage (``B = 2000`` replicates):

- ``HOLDS``:       the MC upper bound is at least 0.94;
- ``UNDER-COVERS``: the MC upper bound is below 0.94;
- ``CONSERVATIVE``: the MC lower bound is above 0.97 (reported, not
  a failure; it takes precedence over HOLDS).

A model is included in the decision only if its pool pass@1 is at
least 0.02 and its pool pass@4 is at most 0.98; otherwise it is
reported descriptively. The arm outcome wording is pre-written:

- no included model under-covers: "the guidance carries over to
  pass@k";
- exactly one: "carries over, with a named exception";
- two or more: "does not carry over".

The naive comparator is computed at ``k = 1`` only: a Wilson interval
on all ``N * K`` binary outcomes of the subsample, exploratory. The
primary cell has no naive comparator.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "passk_unbiased",
    "problem_values",
    "w_m_exact",
    "w_m_mc",
    "correction_factor",
    "clopper_pearson",
    "decide",
    "arm_outcome",
    "coverage_study",
    "pool_passk",
    "included_in_decision",
    "PRIMARY_CELL",
    "GRID_K",
    "GRID_N",
    "GRID_K_SUB",
]

PRIMARY_CELL = {"k": 4, "N": 50, "K": 20}
GRID_K = (1, 2, 4, 8)
GRID_N = (25, 50, 100, 200)
GRID_K_SUB = (5, 10, 20)

NOMINAL = 0.95
HOLDS_UPPER = 0.94
CONSERVATIVE_LOWER = 0.97


# ---------------------------------------------------------------------------
# pass@k estimator (Chen et al. 2021)
# ---------------------------------------------------------------------------

def passk_unbiased(c: int, n: int, k: int) -> float:
    """Unbiased pass@k estimate from ``c`` successes in ``n`` samples."""
    if k > n:
        raise ValueError(f"k={k} exceeds n={n}")
    if c < 0 or c > n:
        raise ValueError(f"c={c} outside [0, {n}]")
    if n - c < k:
        return 1.0
    # 1 - prod_{i=0..k-1} (n - c - i) / (n - i)
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod


def _passk_from_counts(counts: np.ndarray, n: int, k: int) -> np.ndarray:
    """Vectorised pass@k for an array of success counts (each out of n)."""
    counts = np.asarray(counts, dtype=float)
    out = np.ones_like(counts)
    # problems where n - c >= k need the product form
    need = (n - counts) >= k
    if np.any(need):
        c = counts[need]
        prod = np.ones_like(c)
        for i in range(k):
            prod *= (n - c - i) / (n - i)
        out[need] = 1.0 - prod
    out[counts <= 0] = 0.0 if k > 0 else out[counts <= 0]
    # c == 0 must give exactly 0 (product form already does: prod = 1)
    return out


def problem_values(counts: np.ndarray, n: int, k: int) -> np.ndarray:
    """Per-problem pass@k values ``v_i(k, n)`` from full-sample counts."""
    return _passk_from_counts(np.asarray(counts), n, k)


def pool_passk(counts: np.ndarray, n: int, k: int) -> float:
    """Pool pass@k: the mean of ``v_i(k, n)`` over all problems."""
    return float(np.mean(problem_values(counts, n, k)))


def included_in_decision(counts: np.ndarray, n: int) -> bool:
    """Frozen inclusion rule: pool pass@1 >= 0.02 and pass@4 <= 0.98."""
    return pool_passk(counts, n, 1) >= 0.02 and pool_passk(counts, n, 4) <= 0.98


# ---------------------------------------------------------------------------
# W_m: variance of v_i(k, m) under Binomial(m, p) counts
# ---------------------------------------------------------------------------

def w_m_exact(k: int, m: int, p: float) -> float:
    """Exact ``Var[h_k(C)]`` for ``C ~ Binomial(m, p)``.

    Milletari & Kundu (2026), Section 3.3, Eq. (30), with ``N := m``
    and ``rho := p``::

        Var(h_k(C) | p) = sum_c h_k(c)^2 * C(m, c) p^c (1-p)^(m-c)
                          - (1 - (1-p)^k)^2

    where ``h_k(c)`` is the unbiased pass@k estimator. The second term
    is exact: ``E[h_k(C)] = 1 - (1-p)^k``. At ``k = 1`` this reduces
    to ``p (1 - p) / m``. This is a plug-in quantity when ``p`` is the
    observed ``p_hat_i``.
    """
    if m < k:
        raise ValueError(f"m={m} < k={k}: pass@k undefined")
    c = np.arange(0, m + 1)
    h = _passk_from_counts(c, m, k)
    pmf = stats.binom.pmf(c, m, p)
    return float(np.sum(h**2 * pmf) - (1.0 - (1.0 - p) ** k) ** 2)


def w_m_mc(k: int, m: int, p: float, draws: int = 2000,
           rng: np.random.Generator | None = None) -> float:
    """Monte Carlo estimate of :func:`w_m_exact` (implementation check).

    The frozen design specified ``W_m`` by Monte Carlo with 2000
    draws; the exact sum is the definition and this estimate is kept
    as the check. Note the 2000-draw estimate is itself noisy
    (roughly +/-6-9% at mid-range ``p``).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    c = rng.binomial(m, p, size=draws)
    h = _passk_from_counts(c, m, k)
    return float(np.var(h, ddof=1))


# ---------------------------------------------------------------------------
# Finite-pool correction factor
# ---------------------------------------------------------------------------

def _subset_values(counts: np.ndarray, n: int, k: int, K: int,
                   rng: np.random.Generator,
                   n_subsets: int = 500) -> np.ndarray:
    """``v_i(k, K)`` for random K-of-n subsets: shape (P, n_subsets).

    The success count in a random K-subset of a problem with ``c``
    successes in ``n`` samples is Hypergeometric(n, c, K).
    """
    c = np.asarray(counts)
    draws = rng.hypergeometric(c[:, None], (n - c)[:, None], K,
                               size=(c.size, n_subsets))
    return _passk_from_counts(draws, K, k)


def correction_factor(counts: np.ndarray, n: int, k: int, N: int,
                      K: int, rng: np.random.Generator | None = None,
                      n_subsets: int = 500) -> dict:
    """The finite-pool correction factor and its components.

    Returns a dict with ``factor``, ``S2``, ``Var_A``, ``Var_B``,
    ``W_K``, ``W_n`` and ``V_sup``. At ``K = n`` the factor is exactly
    ``sqrt(1 - N/P)``.
    """
    counts = np.asarray(counts)
    P = counts.size
    if K > n:
        raise ValueError(f"K={K} exceeds n={n}")
    v_full = problem_values(counts, n, k)
    S2 = float(np.var(v_full, ddof=1))
    p_hat = counts / n
    W_n = float(np.mean([w_m_exact(k, n, p) for p in p_hat]))
    W_K = float(np.mean([w_m_exact(k, K, p) for p in p_hat]))
    V_sup = (S2 - W_n + W_K) / N
    Var_A = (1.0 - N / P) * S2 / N
    if K == n:
        Var_B = 0.0
        factor = float(np.sqrt(1.0 - N / P))
    else:
        if rng is None:
            rng = np.random.default_rng(0)
        sub = _subset_values(counts, n, k, K, rng, n_subsets=n_subsets)
        Var_B = float(np.mean(np.var(sub, axis=1, ddof=1)))
        factor = float(np.sqrt((Var_A + Var_B) / V_sup))
    return {"factor": factor, "S2": S2, "Var_A": Var_A, "Var_B": Var_B,
            "W_K": W_K, "W_n": W_n, "V_sup": V_sup}


# ---------------------------------------------------------------------------
# Monte Carlo intervals and the frozen decision rule
# ---------------------------------------------------------------------------

def clopper_pearson(successes: int, B: int, level: float = 0.95) -> tuple:
    """Clopper-Pearson interval for a coverage proportion."""
    alpha = 1.0 - level
    lo = 0.0 if successes == 0 else float(
        stats.beta.ppf(alpha / 2, successes, B - successes + 1))
    hi = 1.0 if successes == B else float(
        stats.beta.ppf(1 - alpha / 2, successes + 1, B - successes))
    return lo, hi


def decide(corrected_coverage: float, B: int, level: float = 0.95) -> dict:
    """Apply the frozen decision rule to a corrected coverage."""
    successes = int(round(corrected_coverage * B))
    lo, hi = clopper_pearson(successes, B, level)
    if lo > CONSERVATIVE_LOWER:
        verdict = "CONSERVATIVE"
    elif hi < HOLDS_UPPER:
        verdict = "UNDER-COVERS"
    else:
        verdict = "HOLDS"
    return {"verdict": verdict, "coverage": corrected_coverage,
            "mc_lower": lo, "mc_upper": hi, "B": B}


def arm_outcome(verdicts: dict) -> str:
    """Pre-written arm outcome wording from per-model verdicts.

    ``verdicts`` maps model name -> verdict string, for the models
    included in the decision only.
    """
    under = [m for m, v in verdicts.items() if v == "UNDER-COVERS"]
    if not under:
        return "the guidance carries over to pass@k"
    if len(under) == 1:
        return f"carries over, with a named exception ({under[0]})"
    return "does not carry over"


# ---------------------------------------------------------------------------
# Coverage study
# ---------------------------------------------------------------------------

def _cluster_t_interval(values: np.ndarray, level: float = 0.95):
    """Cluster-t interval over problems for one subsample."""
    N = values.size
    est = float(np.mean(values))
    if N < 2:
        return est, est, est, 0.0
    se = float(np.std(values, ddof=1) / np.sqrt(N))
    crit = float(stats.t.ppf(1 - (1 - level) / 2, N - 1))
    return est, est - crit * se, est + crit * se, se


def _bootstrap_interval(values: np.ndarray, rng: np.random.Generator,
                        draws: int, level: float = 0.95):
    """Percentile bootstrap over problems for one subsample."""
    N = values.size
    idx = rng.integers(0, N, size=(draws, N))
    means = values[idx].mean(axis=1)
    alpha = 1 - level
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _wilson(successes: int, total: int, level: float = 0.95):
    """Wilson interval on a pooled proportion (naive comparator)."""
    if total == 0:
        return 0.0, 0.0, 1.0
    p = successes / total
    z = float(stats.norm.ppf(1 - (1 - level) / 2))
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    return p, float(centre - half), float(centre + half)


def coverage_study(counts: np.ndarray, n: int, cells: list,
                   B: int = 2000, seed: int = 0, level: float = 0.95,
                   bootstrap_draws: int = 1000,
                   n_subsets: int = 500) -> list:
    """Run the coverage study for one model.

    ``counts``: integer array of shape (P,) with the number of
    successes out of ``n`` samples for each problem in the pool.
    ``cells``: list of dicts with keys ``k``, ``N``, ``K``. Every cell
    is labelled exploratory except the frozen primary cell.

    Returns one dict per cell with raw and corrected coverage for the
    cluster-t interval, the bootstrap interval reported beside it, the
    analytic prediction, the correction factor and its components,
    and (at ``k = 1`` only) the naive Wilson comparator.
    """
    counts = np.asarray(counts, dtype=int)
    P = counts.size
    rng = np.random.default_rng(seed)
    rows = []
    for cell in cells:
        k, N, K = cell["k"], cell["N"], cell["K"]
        if k > K or K > n or N > P:
            raise ValueError(f"invalid cell {cell} for P={P}, n={n}")
        truth = pool_passk(counts, n, k)
        comp = correction_factor(counts, n, k, N, K, rng=rng,
                                 n_subsets=n_subsets)
        factor = comp["factor"]
        crit = float(stats.t.ppf(1 - (1 - level) / 2, N - 1))
        analytic = float(2 * stats.t.cdf(crit / factor, N - 1) - 1)

        hits_t = hits_t_corr = hits_b = hits_b_corr = 0
        widths_t = []
        naive_hits = 0
        for _ in range(B):
            draw = rng.choice(P, size=N, replace=False)
            c_draw = counts[draw]
            if K == n:
                vals = problem_values(c_draw, n, k)
                sub_counts = c_draw
            else:
                sub_counts = rng.hypergeometric(c_draw, n - c_draw, K)
                vals = _passk_from_counts(sub_counts, K, k)
            est, lo, hi, _se = _cluster_t_interval(vals, level)
            half = (hi - lo) / 2
            hits_t += lo <= truth <= hi
            hits_t_corr += (est - half * factor) <= truth <= (est + half * factor)
            blo, bhi = _bootstrap_interval(vals, rng, bootstrap_draws, level)
            bhalf = (bhi - blo) / 2
            best = float(np.mean(vals))
            hits_b += blo <= truth <= bhi
            hits_b_corr += (best - bhalf * factor) <= truth <= (best + bhalf * factor)
            widths_t.append(hi - lo)
            if k == 1:
                # naive comparator: Wilson on all N*K binary outcomes
                _p, nlo, nhi = _wilson(int(sub_counts.sum()), N * K, level)
                naive_hits += nlo <= truth <= nhi
        row = {
            "k": k, "N": N, "K": K,
            "label": "primary" if cell == PRIMARY_CELL else "exploratory",
            "truth": truth,
            "B": B,
            "correction_factor": factor,
            **{f"comp_{key}": val for key, val in comp.items() if key != "factor"},
            "analytic_prediction": analytic,
            "cluster_t_coverage_raw": hits_t / B,
            "cluster_t_coverage_corrected": hits_t_corr / B,
            "bootstrap_coverage_raw": hits_b / B,
            "bootstrap_coverage_corrected": hits_b_corr / B,
            "cluster_t_median_width": float(np.median(widths_t)),
        }
        if k == 1:
            row["naive_wilson_coverage"] = naive_hits / B
        rows.append(row)
    return rows
