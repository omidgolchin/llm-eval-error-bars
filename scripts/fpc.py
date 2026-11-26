# Finite-pool correction for the frozen coverage study.
# Post-freeze exploratory analysis (Claude's review, 2026-11-23; approved
# by Omid). See REPORT_P01_FPC.md for the write-up.
#
# Step 1 reruns the frozen subsampling protocol replicate by replicate
# (same generator and seeds as src/errorbars/analyze.py) and scores every
# interval twice: the frozen rule (pool truth inside the interval) and
# the corrected rule |estimate - truth| <= half-width * sqrt(1 - N/400).
# The frozen numbers must reproduce results/coverage_<model>.csv exactly;
# the script checks every cell and stops if one does not.
#
# Step 2 is the split-pool check: subsample from one half of the 400
# questions, take the truth from the other half's full pool.
#
# Usage:
#   .venv/bin/python scripts/fpc.py step1 --model qwen
#   .venv/bin/python scripts/fpc.py combine
#   .venv/bin/python scripts/fpc.py step2
#   .venv/bin/python scripts/fpc.py step3 --model qwen
#   .venv/bin/python scripts/fpc.py combine3

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from errorbars import analyze
from errorbars.analyze import (FAST_METHODS, LEVEL, SHORT, _iter_designs,
                               _mc_interval, bootstrap_fast)
from errorbars.intervals import intervals

DESIGNS_N = [25, 50, 100, 200]
DESIGNS_K = [1, 2, 3, 5]
B = 2000
POOL_N = 400

MODELS = sorted(SHORT)  # frozen order: seed = 1000 + index in this list
SLUG_TO_MODEL = {v: k for k, v in SHORT.items()}

STEP2_METHODS = ("naive", "question_means", "cluster", "bootstrap")
SPLIT_SEED = 20261123   # declared split seed for step 2
STEP2_SEED = 3100       # interval/subsampling stream for step 2 (+ model idx)


def load_mat(model):
    tidy = pd.read_parquet(ROOT / "outputs" / "correctness.parquet")
    return analyze.model_matrix(tidy, model)


def one_interval(method, sub, seed):
    # exactly the calls coverage_study makes, plus the point estimate
    if method == "bootstrap":
        lo, hi, _ = bootstrap_fast(sub, level=LEVEL, seed=seed)
        return float(sub.mean()), lo, hi
    r = intervals(sub, level=LEVEL, methods=(method,), seed=seed)[method]
    return r["estimate"], r["lower"], r["upper"]


# ---------------------------------------------------------------------------
# step 1: corrected coverage on the frozen replicates
# ---------------------------------------------------------------------------

def step1(model):
    slug = SHORT[model]
    seed = 1000 + MODELS.index(model)
    mat = load_mat(model)
    reference = float(np.nanmean(mat))
    frozen = pd.read_csv(ROOT / "results" / f"coverage_{slug}.csv")
    records = []
    for d in _iter_designs(mat, DESIGNS_N, DESIGNS_K, B, seed):
        n, k = d["N"], d["K"]
        shrink = np.sqrt(1 - n / POOL_N)
        for m in FAST_METHODS:
            hits_frozen, hits_corr, widths = 0, 0, []
            for b, sub in enumerate(d["subs"]):
                est, lo, hi = one_interval(m, sub, d["seeds"][m][b])
                hits_frozen += int(lo <= reference <= hi)
                hits_corr += int(abs(est - reference) <= (hi - lo) / 2 * shrink)
                widths.append(hi - lo)
            cov_f, flo, fhi = _mc_interval(hits_frozen, B)
            cov_c, clo, chi = _mc_interval(hits_corr, B)
            # the frozen numbers must come out identical; stop if not
            frow = frozen[(frozen.method == m) & (frozen.N == n)
                          & (frozen.K == k)].iloc[0]
            if cov_f != frow.coverage or abs(np.median(widths)
                                             - frow.median_width) > 1e-9:
                raise SystemExit(
                    f"REPRODUCTION FAILED {slug} {m} N={n} K={k}: got "
                    f"coverage {cov_f}, width {np.median(widths)}; frozen "
                    f"has {frow.coverage}, {frow.median_width}")
            records.append({
                "model": slug, "method": m, "N": n, "K": k, "B": B,
                "coverage": cov_f, "mc_lower": flo, "mc_upper": fhi,
                "corrected_coverage": cov_c, "corrected_mc_lower": clo,
                "corrected_mc_upper": chi,
                "median_width": float(np.median(widths)),
                "reference": reference})
            print(f"{slug} {m} N={n} K={k}: frozen {cov_f}, "
                  f"corrected {cov_c}", flush=True)
    out = ROOT / "results" / f"fpc_coverage_{slug}.csv"
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"wrote {out}", flush=True)


def combine():
    parts = [pd.read_csv(ROOT / "results" / f"fpc_coverage_{s}.csv")
             for s in ("qwen", "glimmer", "inkling")]
    out = ROOT / "results" / "fpc_coverage.csv"
    pd.concat(parts, ignore_index=True).to_csv(out, index=False)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# step 2: split-pool check
# ---------------------------------------------------------------------------

def step2():
    tidy = pd.read_parquet(ROOT / "outputs" / "correctness.parquet")
    # one declared split of the 400 questions into two halves of 200;
    # matrix rows are in question_id order, the same for every model
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(POOL_N)
    draw_half, truth_half = sorted(perm[:200]), sorted(perm[200:])
    qids = (tidy[tidy.model == MODELS[0]].drop_duplicates("question_id")
            .sort_values("question_id")["question_id"].tolist())
    draw_set = set(int(i) for i in draw_half)
    split = pd.DataFrame({
        "row": list(range(POOL_N)),
        "question_id": qids,
        "half": ["draw" if i in draw_set else "truth"
                 for i in range(POOL_N)]})
    split.to_csv(ROOT / "results" / "fpc_splitpool_split.csv", index=False)

    records = []
    for mi, model in enumerate(MODELS):
        mat = analyze.model_matrix(tidy, model)
        truth_mat = mat[truth_half]
        truth = float(truth_mat.mean())
        # the truth is itself a 200-question mean; its sampling SE
        truth_se = float(truth_mat.mean(axis=1).std(ddof=1) / np.sqrt(200))
        draw_mat = mat[draw_half]
        rng = np.random.default_rng(STEP2_SEED + mi)
        for n in (25, 50):
            subs = []
            for _ in range(B):
                qi = rng.choice(200, size=n, replace=False)
                sub = np.empty((n, 5))
                for r, q in enumerate(qi):
                    si = rng.choice(5, size=5, replace=False)
                    sub[r] = draw_mat[q, si]
                subs.append(sub)
            seeds = {m: [int(rng.integers(1 << 31)) for _ in range(B)]
                     for m in STEP2_METHODS}
            for m in STEP2_METHODS:
                hits, widths = 0, []
                for b, sub in enumerate(subs):
                    est, lo, hi = one_interval(m, sub, seeds[m][b])
                    hits += int(lo <= truth <= hi)
                    widths.append(hi - lo)
                cov, clo, chi = _mc_interval(hits, B)
                records.append({
                    "model": SHORT[model], "method": m, "N": n, "K": 5,
                    "B": B, "coverage": cov, "mc_lower": clo,
                    "mc_upper": chi,
                    "median_width": float(np.median(widths)),
                    "truth": truth, "truth_se": truth_se})
                print(f"split {SHORT[model]} {m} N={n}: {cov}", flush=True)
    out = ROOT / "results" / "fpc_splitpool.csv"
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# step 3: exact-factor re-score of the step-1 replicates
# ---------------------------------------------------------------------------
# The step-1 factor sqrt(1 - N/400) is exact for K = 5 and approximate for
# K < 5: it does not model that at K < 5 the drawn seeds are themselves
# shared with the pool. The exact factor, from the variance decomposition
# of (estimate - pool truth) against the estimator's super-population
# variance, is
#
#   factor^2 = [ (1 - N/P) * S2 + VarB_K ] / (S2 - W_5 + W_K)
#
# where S2 is the variance across questions of the 5-seed question means,
# W_m = mean_i p_i (1 - p_i) / m is the binomial within-question variance
# of a question mean over m seeds (p_i = question i's pool rate), and
# VarB_K is the mean over questions of the variance of a K-of-5 subset
# mean across subsets of the question's seeds (hypergeometric:
# p_i (1 - p_i) / K * (5 - K) / 4; zero at K = 5). At K = 5 the factor
# reduces to sqrt(1 - N/P) exactly; step 3 asserts that, asserts the
# frozen numbers still reproduce, and asserts every K = 5 corrected
# coverage equals the step-1 value. The p_i plug-in makes the K < 5
# factors approximate; they are labelled so in the output.

from itertools import combinations

NPOOL_SEEDS = 5


def exact_factors(mat):
    # mat: 400 x 5 correctness matrix (may contain NaN for failed calls,
    # which the frozen policy grades incorrect before this point only in
    # analyze.model_matrix output; here rows are complete 0/1)
    p = np.nanmean(mat, axis=1)                    # pool rate per question
    s2 = float(np.var(np.nanmean(mat, axis=1), ddof=1))
    w = {m: float(np.mean(p * (1 - p)) / m) for m in (1, 2, 3, 5)}
    varb = {}
    for k in (1, 2, 3, 5):
        if k == NPOOL_SEEDS:
            varb[k] = 0.0
            continue
        # exact enumeration of all K-of-5 subsets, per question
        subs = np.array(list(combinations(range(NPOOL_SEEDS), k)))
        vals = np.nanmean(mat[:, subs], axis=2)    # 400 x n_subsets
        varb[k] = float(np.mean(np.var(vals, axis=1, ddof=0)))
        # closed-form check: hypergeometric variance of a subset mean
        closed = float(np.mean(p * (1 - p)) / k
                       * (NPOOL_SEEDS - k) / (NPOOL_SEEDS - 1))
        if abs(varb[k] - closed) > 1e-9:
            raise SystemExit(f"VarB mismatch K={k}: enumeration "
                             f"{varb[k]} vs closed form {closed}")
    factors = {}
    for n in DESIGNS_N:
        for k in DESIGNS_K:
            num = (1 - n / POOL_N) * s2 + varb[k]
            den = s2 - w[NPOOL_SEEDS] + w[k]
            factors[(n, k)] = float(np.sqrt(num / den))
            if k == NPOOL_SEEDS and abs(factors[(n, k)]
                                        - np.sqrt(1 - n / POOL_N)) > 1e-12:
                raise SystemExit(f"unit test failed: exact factor at "
                                 f"K=5, N={n} is {factors[(n, k)]}, not "
                                 f"sqrt(1 - N/400)")
    return factors


def step3(model):
    slug = SHORT[model]
    seed = 1000 + MODELS.index(model)
    mat = load_mat(model)
    reference = float(np.nanmean(mat))
    factors = exact_factors(mat)
    frozen = pd.read_csv(ROOT / "results" / f"coverage_{slug}.csv")
    step1res = pd.read_csv(ROOT / "results" / f"fpc_coverage_{slug}.csv")
    records = []
    for d in _iter_designs(mat, DESIGNS_N, DESIGNS_K, B, seed):
        n, k = d["N"], d["K"]
        shrink = np.sqrt(1 - n / POOL_N)
        exact = factors[(n, k)]
        for m in FAST_METHODS:
            hits_frozen, hits_s1, hits_ex, widths = 0, 0, 0, []
            for b, sub in enumerate(d["subs"]):
                est, lo, hi = one_interval(m, sub, d["seeds"][m][b])
                hw = (hi - lo) / 2
                hits_frozen += int(lo <= reference <= hi)
                hits_s1 += int(abs(est - reference) <= hw * shrink)
                hits_ex += int(abs(est - reference) <= hw * exact)
                widths.append(hi - lo)
            cov_f, flo, fhi = _mc_interval(hits_frozen, B)
            cov_s1, _, _ = _mc_interval(hits_s1, B)
            cov_ex, elo, ehi = _mc_interval(hits_ex, B)
            frow = frozen[(frozen.method == m) & (frozen.N == n)
                          & (frozen.K == k)].iloc[0]
            if cov_f != frow.coverage or abs(np.median(widths)
                                             - frow.median_width) > 1e-9:
                raise SystemExit(
                    f"REPRODUCTION FAILED {slug} {m} N={n} K={k}: got "
                    f"coverage {cov_f}, width {np.median(widths)}; frozen "
                    f"has {frow.coverage}, {frow.median_width}")
            srow = step1res[(step1res.method == m) & (step1res.N == n)
                            & (step1res.K == k)].iloc[0]
            if cov_s1 != srow.corrected_coverage:
                raise SystemExit(
                    f"STEP-1 MISMATCH {slug} {m} N={n} K={k}: re-scored "
                    f"{cov_s1}, stored {srow.corrected_coverage}")
            if k == NPOOL_SEEDS and cov_ex != cov_s1:
                raise SystemExit(
                    f"K=5 IDENTITY FAILED {slug} {m} N={n}: exact {cov_ex}"
                    f" != step-1 {cov_s1}")
            records.append({
                "model": slug, "method": m, "N": n, "K": k, "B": B,
                "coverage": cov_f, "mc_lower": flo, "mc_upper": fhi,
                "corrected_step1": cov_s1,
                "corrected_exact": cov_ex, "exact_mc_lower": elo,
                "exact_mc_upper": ehi,
                "factor_step1": float(shrink), "factor_exact": exact,
                "median_width": float(np.median(widths)),
                "reference": reference})
            print(f"{slug} {m} N={n} K={k}: frozen {cov_f}, "
                  f"step1 {cov_s1}, exact {cov_ex} "
                  f"(factor {exact:.4f} vs {shrink:.4f})", flush=True)
    out = ROOT / "results" / f"fpc_exact_coverage_{slug}.csv"
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"wrote {out}", flush=True)


def combine3():
    parts = [pd.read_csv(ROOT / "results" / f"fpc_exact_coverage_{s}.csv")
             for s in ("qwen", "glimmer", "inkling")]
    out = ROOT / "results" / "fpc_exact_coverage.csv"
    pd.concat(parts, ignore_index=True).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["step1", "combine", "step2",
                                     "step3", "combine3"])
    ap.add_argument("--model", choices=list(SLUG_TO_MODEL))
    args = ap.parse_args()
    t0 = time.time()
    if args.step == "step1":
        step1(SLUG_TO_MODEL[args.model])
    elif args.step == "combine":
        combine()
    elif args.step == "step3":
        step3(SLUG_TO_MODEL[args.model])
    elif args.step == "combine3":
        combine3()
    else:
        step2()
    print(f"elapsed {time.time() - t0:.0f}s")
