"""I-040: does Ye's (2026) variance-adaptive region change P01's advice?

Runs the arms frozen in I040_FROZEN_RULE.md on the existing rollouts
(outputs/correctness.parquet, 3 models x 400 questions x 5 seeds).
$0 spend, no new model calls.

VA is implemented from the paper (arXiv:2611.21266, sec.4.3 and
Algorithm 1): the region keeps every mixing law G whose tail-count
means mu_k(G) satisfy |T_k - mu_k(G)| <= r(V_k(G)) for k in a fixed
index set, with the Bernstein radius r(V) = sqrt(2 L V / m) + L/(3 m),
L = log(2 |K| / alpha). On a grid of cells with envelopes for P_k, V_k
and the target h, and with the radius replaced by its tangent cuts,
each endpoint is a linear program (scipy/HiGHS).

Stages: gate (validate VA on one of Ye's simulation laws), then
E1 (fixed-benchmark mean), E2 (VA off-label under the coverage_study
protocol), E3 (pass@k, fixed-benchmark, descriptive). The frozen
CONFIRM / REVISE / KILL rule is applied at the end.

Run:  .venv/bin/python scripts/i040.py [gate|e1|e2|e3|all]
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from errorbars.analyze import SHORT, _iter_designs, model_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.05
LEVEL = 0.95
ETA = 1e-2            # envelope tolerance (Ye's default is 1e-3; the
                      # coarser grid is a logged deviation, see report)
B = 1000              # replicates per cell (frozen rule)
BOOT_DRAWS = 1000     # plug-in bootstrap resamples per replicate
MODELS = ["arize-ai/qwen-2-1.5b-instruct",
          "meta-models/Muse-Glimmer-30B",
          "thinkingmachines/Inkling"]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def cp_interval(hits, b):
    """Clopper-Pearson Monte Carlo interval for a coverage proportion."""
    p = hits / b
    lo = 0.0 if hits == 0 else float(stats.beta.ppf(0.025, hits, b - hits + 1))
    hi = 1.0 if hits == b else float(stats.beta.ppf(0.975, hits + 1, b - hits))
    return p, lo, hi


def wilson(succ, n, level=LEVEL):
    """Pooled Wilson interval, same formula as intervals.py's naive."""
    p = succ / n
    z = float(stats.norm.ppf(1 - (1 - level) / 2))
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return centre - half, centre + half


def plugin_bootstrap(values, stat_fn, rng):
    """Percentile bootstrap over problems: resample the per-problem
    values with replacement, recompute the plug-in statistic."""
    m = len(values)
    idx = rng.integers(0, m, size=(BOOT_DRAWS, m))
    draws = stat_fn(values[idx])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# VA region (Ye 2026, sec.4.3 / Algorithm 1)
# ---------------------------------------------------------------------------

def make_K(n):
    """Ye's default index set K_geo, intersected with {1..n}."""
    if n <= 16:
        return list(range(1, n + 1))
    ks = set(range(1, 17)) | {n}
    ks |= {int(round(x)) for x in np.geomspace(16, n, 16)}
    return sorted(k for k in ks if 1 <= k <= n)


def _bennett_u(y):
    """Invert h(u) = (1 + u) log(1 + u) - u = y for u >= 0, via the
    Lambert W closed form u = exp(1 + W((y - 1) / e)) - 1."""
    from scipy.special import lambertw
    y = np.asarray(y, dtype=float)
    u = np.exp(1.0 + np.real(lambertw((y - 1.0) / np.e))) - 1.0
    return np.maximum(u, 0.0)


def bennett_rs(V0, L, m):
    """(radius, slope) at variance points V0 for the Bennett bound:
    r(V) solves V h(r / V) = L / m. The slope is from implicit
    differentiation: r'(V) = u - h(u) / log(1 + u) at u = r / V,
    with h(u) = L / (m V) at the solution."""
    V0 = np.asarray(V0, dtype=float)
    y = L / (m * V0)
    u = _bennett_u(y)
    r = V0 * u
    slope = np.where(u > 1e-6, u - y / np.log1p(u),
                     u / 2 - u**2 / 12)
    return r, slope


def bernstein_rs(V0, L, m):
    """(radius, slope) at variance points V0 for the Bernstein bound
    r(V) = sqrt(2 L V / m) + L / (3 m)."""
    V0 = np.asarray(V0, dtype=float)
    a = np.sqrt(2 * L / m)
    return a * np.sqrt(V0) + L / (3 * m), a / (2 * np.sqrt(V0))


def tail_probs(k_arr, n, p_arr):
    """P_k(p) = Pr(Bin(n, p) >= k), rows = k values, cols = p values."""
    return stats.binom.sf(np.asarray(k_arr)[:, None] - 1, n,
                          np.asarray(p_arr)[None, :])


def build_partition(n, K, h_funcs, eta=ETA):
    """Data-independent partition of [0, 1] with cells {0} and {1},
    bisected until every cell has P_k(b) - P_k(a) <= eta for all k in K
    and h(b) - h(a) <= eta for every (nondecreasing) target h.
    Returns (a, b) cell endpoint arrays."""
    pts = set(np.linspace(0, 1, 201).tolist())
    pts |= set(np.geomspace(1e-7, 1 - 1e-7, 80).tolist())
    pts |= {1.0 - x for x in np.geomspace(1e-7, 1 - 1e-7, 80)}
    # seed the transition zone of every P_k, where the envelopes move
    grid = np.linspace(0, 1, 2001)
    P = tail_probs(K, n, grid)
    for row in P:
        for q in (0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98):
            pts.add(float(grid[np.argmin(np.abs(row - q))]))
    pts = np.array(sorted(pts))
    for _ in range(60):
        P = tail_probs(K, n, pts)
        var = np.diff(P, axis=1).max()
        for h in h_funcs:
            var = max(var, np.diff(h(pts)).max())
        if var <= eta:
            break
        bad = np.zeros(len(pts) - 1, dtype=bool)
        bad |= (np.diff(P, axis=1) > eta).any(axis=0)
        for h in h_funcs:
            bad |= np.diff(h(pts)) > eta
        mid = (pts[:-1] + pts[1:]) / 2
        pts = np.sort(np.concatenate([pts, mid[bad]]))
    # cells: {0}, then [pts_i, pts_{i+1}], then {1}
    a = np.concatenate([[0.0], pts[:-1], [1.0]])
    b = np.concatenate([[0.0], pts[1:], [1.0]])
    return a, b


class VA:
    """Precomputed envelopes and LP rows for one (n, partition)."""

    def __init__(self, n, K, a, b, tangents=None, radius="bernstein"):
        self.n, self.K = n, K
        self.a, self.b = a, b
        self.J = len(a)
        self.radius = radius
        self.certified_ = []
        Pa = tail_probs(K, n, a)
        Pb = tail_probs(K, n, b)
        self.Plo, self.Phi = Pa, Pb
        # Vbar_kj = max of u(x) = x(1-x) over x in [P_k(a), P_k(b)]
        u = lambda x: x * (1 - x)  # noqa: E731
        Vbar = np.maximum(u(Pa), u(Pb))
        mid = (Pa <= 0.5) & (Pb >= 0.5)
        Vbar[mid] = 0.25
        self.Vbar = Vbar
        # Initial tangent points V0 for the radius cuts. Ye's software
        # uses V0 = t^2 down to t = 1e-7; those near-zero tangents have
        # slopes in the thousands and make the LP badly scaled, while
        # their radius overshoot at V = 0 is only a * sqrt(V0) / 2, so
        # this set starts at 1e-4 and the cut rounds add tangents at the
        # realised variances. Any finite tangent set is valid
        # (Theorem 4.3): the cuts only outer-approximate the radius.
        # The precision check passes Ye's full set explicitly.
        if tangents is None:
            tangents = list(np.geomspace(1e-4, 0.25, 8))
        self.tangents = list(tangents)

    def _rs(self, V0, L, m):
        if self.radius == "bennett":
            return bennett_rs(V0, L, m)
        return bernstein_rs(V0, L, m)

    def _r(self, V, L, m):
        """Radius values at variance points V (no slope)."""
        V = np.asarray(V, dtype=float)
        if self.radius == "bennett":
            out = np.zeros_like(V)
            pos = V > 0
            out[pos] = V[pos] * _bennett_u(L / (m * V[pos]))
            return out
        return np.sqrt(2 * L * V / m) + L / (3 * m)

    def _rows(self, m, tangents):
        """LP rows for the given problem count m and tangent set.
        Returns (A_ub, c_hi_part, c_lo_part) where the right-hand sides
        still need the observed tail counts T."""
        L = np.log(2 * len(self.K) / ALPHA)
        rows, rhs_const, sign, kidx = [], [], [], []
        for V0 in tangents:
            r0, s0 = self._rs(np.array([V0]), L, m)
            r0, s0 = float(r0[0]), float(s0[0])
            scale = 1.0 / (1.0 + s0 * 0.25)   # row scaling, same LP
            rows.append(-(self.Phi + s0 * self.Vbar) * scale)
            rhs_const.append(np.full(len(self.K), (r0 - s0 * V0) * scale))
            sign.append(np.full(len(self.K), -scale))
            kidx.append(np.arange(len(self.K)))
            rows.append((self.Plo - s0 * self.Vbar) * scale)
            rhs_const.append(np.full(len(self.K), (r0 - s0 * V0) * scale))
            sign.append(np.full(len(self.K), scale))
            kidx.append(np.arange(len(self.K)))
        return (np.vstack(rows), np.concatenate(rhs_const),
                np.concatenate(sign), np.concatenate(kidx), L)

    def _endpoint(self, A, rhs, obj):
        """One LP endpoint via the dual problem (the primal has a cell
        per column but only a few hundred rows, so the dual is far
        smaller for the solver). Returns (value, w) or (None, None).
        The primal solution w is recovered from the dual problem's
        constraint marginals, for placing the next round of cuts."""
        R = A.shape[0]
        A_dual = np.hstack([-A.T, np.ones((self.J, 1))])
        d_obj = np.concatenate([rhs, [-1.0]])
        bounds = [(0.0, None)] * R + [(None, None)]
        res = linprog(d_obj, A_ub=A_dual, b_ub=obj, bounds=bounds,
                      method="highs")
        if res.status != 0:
            return None, None
        w = -np.asarray(res.ineqlin.marginals, dtype=float)
        if w.min() < -1e-6 or abs(w.sum() - 1) > 1e-4:
            w = -w
        if w.min() < -1e-6 or abs(w.sum() - 1) > 1e-4:
            w = None
        else:
            w = np.clip(w, 0, None)
        return -float(res.fun), w

    def intervals_multi(self, counts, targets, max_rounds=2,
                        certify=False):
        """VA intervals for several targets from the same counts.
        targets: list of (h_lo, h_hi) per-cell envelopes. The tangent
        set is shared across targets, so cuts found for one target
        tighten the rest. Returns a list of (lo, hi), (nan, nan) if
        the region is empty.

        With certify=True (Ye's dual certification, Algorithm 1
        step 5), cut rounds continue until the LP optimum satisfies
        the exact nonlinear constraints, so the endpoint is certified
        against the true region rather than padded; the per-target
        outcome is left in self.certified_. Without it, endpoints get
        the small outward pad used in the main run."""
        counts = np.asarray(counts, dtype=int)
        m = len(counts)
        T = np.array([np.mean(counts >= k) for k in self.K])
        # sure constraints as two extra LP rows: w_{0} <= share of zero
        # counts, w_{1} <= share of full counts (sec.3.5 of the paper)
        sure = np.zeros((2, self.J))
        sure[0, 0] = 1.0
        sure[1, -1] = 1.0
        sure_rhs = np.array([np.mean(counts == 0), np.mean(counts == self.n)])
        tangents = list(self.tangents)
        out = []
        self.certified_ = []
        for h_lo, h_hi in targets:
            pair = []
            ok_pair = True
            for oi, obj in enumerate((h_lo, -h_hi)):  # min, then max
                val = None
                ok = False
                for _ in range(max_rounds):
                    A, rhs_c, sign, kidx, L = self._rows(m, tangents)
                    A = np.vstack([A, sure])
                    rhs = np.concatenate([rhs_c + sign * T[kidx], sure_rhs])
                    val, w = self._endpoint(A, rhs, obj)
                    if val is None:
                        self.certified_ = [False] * len(targets)
                        return [(np.nan, np.nan)] * len(targets)
                    if w is None:
                        break
                    # exact (nonlinear) constraints; add cuts if broken
                    Vk = self.Vbar @ w
                    r = self._r(Vk, L, m)
                    viol = (T - self.Phi @ w > r + 1e-7) | \
                           (self.Plo @ w - T > r + 1e-7)
                    if not viol.any():
                        ok = True
                        break
                    new = [float(v) for v in Vk[viol] if v > 1e-12]
                    if not new:
                        break
                    tangents = sorted(set(tangents) | set(new))
                pair.append(val if oi == 0 else -val)
                ok_pair = ok_pair and ok
            self.certified_.append(ok_pair)
            if certify and ok_pair:
                out.append((max(0.0, pair[0]), min(1.0, pair[1])))
            else:
                # small outward pad in place of Ye's dual certification
                out.append((max(0.0, pair[0] - 1e-6),
                            min(1.0, pair[1] + 1e-6)))
        return out

    def interval(self, counts, h_lo, h_hi, max_rounds=2, certify=False):
        """VA interval for one target; see intervals_multi."""
        return self.intervals_multi(counts, [(h_lo, h_hi)],
                                    max_rounds=max_rounds,
                                    certify=certify)[0]


def h_envelopes(va, h):
    """Per-cell [inf, sup] of a nondecreasing h on the partition."""
    return h(va.a), h(va.b)


# ---------------------------------------------------------------------------
# checkpointing: every replicate's endpoints are appended to a progress
# CSV as soon as they exist, so an interrupted run resumes where it
# stopped instead of starting over
# ---------------------------------------------------------------------------

def load_progress(path):
    if Path(path).exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def append_row(path, row):
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not Path(path).exists(), index=False)


# ---------------------------------------------------------------------------
# validation gate: Ye's law F1, m = 250, n = 32, theta_delta functional
# ---------------------------------------------------------------------------

def run_gate():
    n, m, reps = 32, 250, 250
    d = 0.01          # gate cell: Ye's law F1, theta_delta, delta = 0.01
    rng = np.random.default_rng(20261123)
    # F1 = 0.3 delta_0 + 0.7 Beta(0.3, 1.5); one F_m drawn and held fixed
    p = np.where(rng.random(m) < 0.3, 0.0, rng.beta(0.3, 1.5, size=m))
    h = lambda x: np.minimum(x / d, 1.0)  # noqa: E731
    truth = float(h(p).mean())
    prog_path = ROOT / "results" / "i040_gate_progress.csv"
    prog = load_progress(prog_path)
    done = set(prog["rep"]) if len(prog) else set()
    K = make_K(n)
    va = VA(n, K, *build_partition(n, K, [h]))
    print(f"gate: partition J = {va.J} cells, |K| = {len(K)}; "
          f"{len(done)} replicates already done", flush=True)
    env = h_envelopes(va, h)
    crng = np.random.default_rng(20261124)
    for rep in range(reps):
        c = crng.binomial(n, p)
        if rep in done:
            crng.integers(0, m, size=(BOOT_DRAWS, m))  # keep stream aligned
            continue
        lo, hi = va.interval(c, *env)
        blo, bhi = plugin_bootstrap(c / n, lambda v: np.minimum(
            v / d, 1.0).mean(axis=1), crng)
        append_row(prog_path, {"rep": rep, "va_lo": lo, "va_hi": hi,
                               "boot_lo": blo, "boot_hi": bhi})
        if (rep + 1) % 50 == 0:
            print(f"gate: {rep + 1}/{reps} replicates", flush=True)
    prog = load_progress(prog_path)
    rows = []
    for meth in ("va", "boot"):
        hits = int(((prog[f"{meth}_lo"] <= truth)
                    & (truth <= prog[f"{meth}_hi"])).sum())
        cov, lo_, hi_ = cp_interval(hits, len(prog))
        rows.append({"law": "F1", "m": m, "n": n, "delta": d,
                     "method": meth, "B": len(prog), "coverage": cov,
                     "mc_lower": lo_, "mc_upper": hi_,
                     "median_width": float((prog[f"{meth}_hi"]
                                             - prog[f"{meth}_lo"]).median()),
                     "truth": truth})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "i040_gate.csv", index=False)
    print(out.to_string(index=False), flush=True)
    va_cov = out[out.method == "va"]["coverage"].iloc[0]
    boot_cov = out[out.method == "boot"]["coverage"].iloc[0]
    passed = len(prog) >= 200 and va_cov >= 0.94 \
        and boot_cov < va_cov - 0.03
    print(f"GATE: va coverage {va_cov:.3f}, bootstrap {boot_cov:.3f} -> "
          f"{'PASS' if passed else 'FAIL'}", flush=True)
    return passed


# ---------------------------------------------------------------------------
# E1 (fixed-benchmark mean) and E3 (pass@k, fixed-benchmark)
# ---------------------------------------------------------------------------

def run_e1_e3(mats):
    # I040_SKIP_E3=1 drops the descriptive pass@k targets (E3 carries
    # no decision weight) to protect the decision-bearing E1 cells
    # against the execution time-box; logged as a deviation.
    import os
    skip_e3 = os.environ.get("I040_SKIP_E3") == "1"
    from math import comb
    prog_path = ROOT / "results" / "i040_e1_progress.csv"
    prog = load_progress(prog_path)
    done = set()
    if len(prog):
        done = {(r.slug, int(r.n), int(r.rep))
                for r in prog.itertuples()}
    for n in (5, 2):
        for mi, model in enumerate(MODELS):
            mat = mats[model]
            phat = mat.mean(axis=1)          # per-question success rates
            K = make_K(n)
            targets = [lambda x: x] + \
                [(lambda k: lambda x: 1 - (1 - x) ** k)(k) for k in (2, 4)]
            part_targets = targets[:1] if skip_e3 else targets
            va = VA(n, K, *build_partition(n, K, part_targets))
            env_mean = h_envelopes(va, lambda x: x)
            env_pass = {k: h_envelopes(va, targets[1 + j])
                        for j, k in enumerate((2, 4))}
            rng = np.random.default_rng(20261125 + 100 * mi + n)
            for rep in range(B):
                c = rng.binomial(n, phat)
                if (SHORT[model], n, rep) in done:
                    # keep the rng stream aligned: one bootstrap draw
                    # for the mean, two more for pass@k when n == 5
                    rng.integers(0, 400, size=(BOOT_DRAWS, 400))
                    if n == 5 and not skip_e3:
                        rng.integers(0, 400, size=(BOOT_DRAWS, 400))
                        rng.integers(0, 400, size=(BOOT_DRAWS, 400))
                    continue
                va_targets = [env_mean]
                if n == 5 and not skip_e3:
                    va_targets += [env_pass[2], env_pass[4]]
                va_out = va.intervals_multi(c, va_targets)
                row = {"slug": SHORT[model], "n": n, "rep": rep,
                       "va_lo": va_out[0][0], "va_hi": va_out[0][1]}
                blo, bhi = plugin_bootstrap(c / n,
                                            lambda v: v.mean(axis=1), rng)
                row["boot_lo"], row["boot_hi"] = blo, bhi
                row["wilson_lo"], row["wilson_hi"] = wilson(int(c.sum()),
                                                           400 * n)
                x = c / n
                half = 1.959964 * x.std(ddof=1) / np.sqrt(400)
                row["normal_lo"] = x.mean() - half
                row["normal_hi"] = x.mean() + half
                if n == 5 and not skip_e3:
                    for j, k in enumerate((2, 4)):
                        row[f"va{k}_lo"], row[f"va{k}_hi"] = va_out[1 + j]
                        # plug-in bootstrap, Chen et al. unbiased pass@k
                        g = np.array([1.0 - (comb(n - ci, k) / comb(n, k)
                                             if n - ci >= k else 0.0)
                                      for ci in c])
                        blo, bhi = plugin_bootstrap(
                            g, lambda v: v.mean(axis=1), rng)
                        row[f"boot{k}_lo"], row[f"boot{k}_hi"] = blo, bhi
                append_row(prog_path, row)
            print(f"E1/E3 replicates done: {SHORT[model]} n={n}",
                  flush=True)
    aggregate_e1_e3(mats)


def aggregate_e1_e3(mats):
    prog = load_progress(ROOT / "results" / "i040_e1_progress.csv")
    rows1, rows3 = [], []
    for model in MODELS:
        mat = mats[model]
        phat = mat.mean(axis=1)
        theta = float(phat.mean())
        pass_truth = {k: float((1 - (1 - phat) ** k).mean()) for k in (2, 4)}
        for n in (5, 2):
            sub = prog[(prog.slug == SHORT[model]) & (prog.n == n)]
            b = len(sub)
            for meth in ("va", "boot", "wilson", "normal"):
                hits = int(((sub[f"{meth}_lo"] <= theta)
                            & (theta <= sub[f"{meth}_hi"])).sum())
                cov, lo_, hi_ = cp_interval(hits, b)
                rows1.append({"estimand": "E1", "model": model,
                              "slug": SHORT[model], "n": n, "method": meth,
                              "B": b, "coverage": cov, "mc_lower": lo_,
                              "mc_upper": hi_,
                              "median_width": float((sub[f"{meth}_hi"]
                                                     - sub[f"{meth}_lo"]
                                                     ).median()),
                              "reference": theta})
            if n == 5 and "va2_lo" in prog.columns:
                for k in (2, 4):
                    for meth in ("va", "boot"):
                        col = f"{meth}{k}"
                        hits = int(((sub[f"{col}_lo"] <= pass_truth[k])
                                    & (pass_truth[k] <= sub[f"{col}_hi"]))
                                   .sum())
                        cov, lo_, hi_ = cp_interval(hits, b)
                        rows3.append({"estimand": "E3", "model": model,
                                      "slug": SHORT[model], "k": k,
                                      "method": meth, "B": b,
                                      "coverage": cov, "mc_lower": lo_,
                                      "mc_upper": hi_,
                                      "median_width": float(
                                          (sub[f"{col}_hi"]
                                           - sub[f"{col}_lo"]).median()),
                                      "reference": pass_truth[k]})
    pd.DataFrame(rows1).to_csv(ROOT / "results" / "i040_e1.csv", index=False)
    pd.DataFrame(rows3).to_csv(ROOT / "results" / "i040_e3.csv", index=False)
    print("E1/E3 aggregated", flush=True)


# ---------------------------------------------------------------------------
# E2: VA off-label under the existing coverage_study protocol
# ---------------------------------------------------------------------------

def run_e2(mats):
    n = 5
    K = make_K(n)
    va = VA(n, K, *build_partition(n, K, [lambda x: x]))
    env = h_envelopes(va, lambda x: x)
    prog_path = ROOT / "results" / "i040_e2_progress.csv"
    prog = load_progress(prog_path)
    done = set()
    if len(prog):
        done = {(r.slug, int(r.N), int(r.idx)) for r in prog.itertuples()}
    # primary cell (50, 5) for every model first, then the secondary
    # (100, 5): the decision rule reads the primary cell, so it is
    # completed first under the execution time-box
    for Nt in (50, 100):
        for mi, model in enumerate(MODELS):
            mat = mats[model]
            # regenerate the study's own subsample stream (seed
            # 1000 + index, designs in config order, B = 2000);
            # VA uses the first 1000 subsamples of the target design
            for d in _iter_designs(mat, [25, 50, 100, 200], [1, 2, 3, 5],
                                   2000, 1000 + mi):
                if (d["N"], d["K"]) != (Nt, 5):
                    continue
                for idx, sub in enumerate(d["subs"][:B]):
                    if (SHORT[model], Nt, idx) in done:
                        continue
                    c = sub.sum(axis=1).astype(int)
                    lo, hi = va.interval(c, *env)
                    append_row(prog_path, {"slug": SHORT[model], "N": Nt,
                                           "idx": idx, "va_lo": lo,
                                           "va_hi": hi})
                print(f"E2 replicates done: {SHORT[model]} N={Nt}",
                      flush=True)
                break
    prog = load_progress(prog_path)
    rows = []
    for model in MODELS:
        reference = float(mats[model].mean())
        for N in (50, 100):
            sub = prog[(prog.slug == SHORT[model]) & (prog.N == N)]
            b = len(sub)
            hits = int(((sub.va_lo <= reference)
                        & (reference <= sub.va_hi)).sum())
            cov, lo_, hi_ = cp_interval(hits, b)
            rows.append({"estimand": "E2", "model": model,
                         "slug": SHORT[model], "N": N, "K": 5,
                         "method": "va_offlabel", "B": b, "coverage": cov,
                         "mc_lower": lo_, "mc_upper": hi_,
                         "median_width": float((sub.va_hi - sub.va_lo)
                                               .median()),
                         "reference": reference})
    pd.DataFrame(rows).to_csv(ROOT / "results" / "i040_e2.csv", index=False)
    print("E2 aggregated", flush=True)


# ---------------------------------------------------------------------------
# precision check: the E2 cells above ran with runtime shortcuts
# (eta = 1e-2, tangents from V0 = 1e-4, Bernstein radius only, 1e-6
# endpoint padding). This stage re-runs VA at Ye's defaults --
# eta = 1e-3, the full tangent set V0 = t^2 down to t = 1e-7, the
# Bennett radius, dual certification instead of padding -- on the
# first subsamples of the E2 primary cells, to measure how much the
# shortcuts widened VA where the KILL clause is decided on width.
# ---------------------------------------------------------------------------

PRECISION_MODELS = ("glimmer", "inkling")


def default_va(n, K):
    """VA at Ye's defaults on the mean target."""
    tangents = list(np.geomspace(1e-7, 0.5, 30) ** 2)
    va = VA(n, K, *build_partition(n, K, [lambda x: x], eta=1e-3),
            tangents=tangents, radius="bennett")
    return va, h_envelopes(va, lambda x: x)


def e2_counts(mats, model, reps):
    """The study's own (50, 5) subsample stream, first `reps` draws,
    as per-question success counts -- the identical draws the E2 arm
    used, regenerated the identical way."""
    mi = MODELS.index(model)
    mat = mats[model]
    for d in _iter_designs(mat, [25, 50, 100, 200], [1, 2, 3, 5],
                           2000, 1000 + mi):
        if (d["N"], d["K"]) != (50, 5):
            continue
        return [sub.sum(axis=1).astype(int) for sub in d["subs"][:reps]]
    raise RuntimeError("design (50, 5) not found")


def run_precision(reps=100):
    n = 5
    K = make_K(n)
    tidy = pd.read_parquet(ROOT / "outputs" / "correctness.parquet")
    mats = {m: model_matrix(tidy, m) for m in MODELS}
    # the shortcut solver, built exactly as in run_e2
    va_cut = VA(n, K, *build_partition(n, K, [lambda x: x]))
    env_cut = h_envelopes(va_cut, lambda x: x)
    va_def, env_def = default_va(n, K)
    print(f"precision: shortcut partition J = {va_cut.J}, "
          f"default partition J = {va_def.J}", flush=True)
    e2prog = load_progress(ROOT / "results" / "i040_e2_progress.csv")
    prog_path = ROOT / "results" / "i040_precision_progress.csv"
    prog = load_progress(prog_path)
    done = set()
    if len(prog):
        done = {(r.slug, int(r.idx)) for r in prog.itertuples()}
    for model in MODELS:
        slug = SHORT[model]
        if slug not in PRECISION_MODELS:
            continue
        counts = e2_counts(mats, model, reps)
        rec = e2prog[(e2prog.slug == slug) & (e2prog.N == 50)]
        rec = rec.set_index("idx")
        for idx, c in enumerate(counts):
            if (slug, idx) in done:
                continue
            # integrity: the shortcut solver on these counts must
            # reproduce the E2 arm's recorded endpoints, proving the
            # draws are the identical ones in the identical order
            lo_c, hi_c = va_cut.interval(c, *env_cut)
            if abs(lo_c - rec.loc[idx, "va_lo"]) > 1e-7 or \
                    abs(hi_c - rec.loc[idx, "va_hi"]) > 1e-7:
                raise RuntimeError(
                    f"shortcut endpoints do not reproduce for {slug} "
                    f"idx {idx}; subsample stream mismatch, stopping")
            t0 = time.time()
            lo, hi = va_def.interval(c, *env_def, max_rounds=60,
                                     certify=True)
            append_row(prog_path,
                       {"slug": slug, "idx": idx, "def_lo": lo,
                        "def_hi": hi,
                        "certified": int(va_def.certified_[0]),
                        "seconds": round(time.time() - t0, 2)})
            if (idx + 1) % 10 == 0:
                print(f"precision {slug}: {idx + 1}/{reps}", flush=True)
        print(f"precision replicates done: {slug}", flush=True)
    aggregate_precision()


def aggregate_precision():
    prog = load_progress(ROOT / "results" / "i040_precision_progress.csv")
    e2prog = load_progress(ROOT / "results" / "i040_e2_progress.csv")
    rows = []
    for slug in PRECISION_MODELS:
        sub = prog[prog.slug == slug].sort_values("idx")
        rec = e2prog[(e2prog.slug == slug) & (e2prog.N == 50)
                     & (e2prog.idx.isin(sub.idx))]
        cut_med = float((rec.va_hi - rec.va_lo).median())
        def_med = float((sub.def_hi - sub.def_lo).median())
        rows.append({"slug": slug, "replicates": len(sub),
                     "shortcut_median_width": cut_med,
                     "default_median_width": def_med,
                     "pct_narrower": 1.0 - def_med / cut_med,
                     "certified_all": bool(sub.certified.all())})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "i040_precision.csv", index=False)
    print(out.to_string(index=False), flush=True)
    g = out[out.slug == "glimmer"].iloc[0]
    i = out[out.slug == "inkling"].iloc[0]
    if g.pct_narrower < 0.09 or i.pct_narrower < 0.18:
        print("PRECISION BRANCH: at Ye's defaults VA cannot trigger "
              "KILL (Glimmer needs >= 9% narrower, Inkling >= 18%); "
              "CONFIRM stands as robust", flush=True)
    else:
        print("PRECISION BRANCH: both thresholds crossed; run "
              "precision-full before applying the frozen rule",
              flush=True)
    return out


def run_precision_full():
    """Branch 2 only: re-run the two decisive E2 primary cells in
    full (B = 1000, same stream) at Ye's defaults, then apply the
    frozen KILL test to the new numbers (Qwen's shortcut cell stands
    as-is). Logged as a deviation wherever the numbers are reported;
    both versions are kept."""
    n = 5
    K = make_K(n)
    tidy = pd.read_parquet(ROOT / "outputs" / "correctness.parquet")
    mats = {m: model_matrix(tidy, m) for m in MODELS}
    va_def, env_def = default_va(n, K)
    prog_path = ROOT / "results" / "i040_precision_full_progress.csv"
    prog = load_progress(prog_path)
    done = set()
    if len(prog):
        done = {(r.slug, int(r.idx)) for r in prog.itertuples()}
    for model in MODELS:
        slug = SHORT[model]
        if slug not in PRECISION_MODELS:
            continue
        counts = e2_counts(mats, model, B)
        for idx, c in enumerate(counts):
            if (slug, idx) in done:
                continue
            lo, hi = va_def.interval(c, *env_def, max_rounds=60,
                                     certify=True)
            append_row(prog_path,
                       {"slug": slug, "idx": idx, "def_lo": lo,
                        "def_hi": hi,
                        "certified": int(va_def.certified_[0])})
            if (idx + 1) % 100 == 0:
                print(f"precision-full {slug}: {idx + 1}/{B}",
                      flush=True)
        print(f"precision-full replicates done: {slug}", flush=True)
    prog = load_progress(prog_path)
    rows = []
    for model in MODELS:
        slug = SHORT[model]
        if slug not in PRECISION_MODELS:
            continue
        sub = prog[prog.slug == slug]
        reference = float(mats[model].mean())
        hits = int(((sub.def_lo <= reference)
                    & (reference <= sub.def_hi)).sum())
        cov, lo_, hi_ = cp_interval(hits, len(sub))
        rows.append({"estimand": "E2", "model": model, "slug": slug,
                     "N": 50, "K": 5, "method": "va_offlabel_defaults",
                     "B": len(sub), "coverage": cov, "mc_lower": lo_,
                     "mc_upper": hi_,
                     "median_width": float((sub.def_hi - sub.def_lo)
                                           .median()),
                     "reference": reference})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "i040_e2_defaults.csv", index=False)
    print(out.to_string(index=False), flush=True)
    # frozen KILL test on the new numbers for the two re-run models;
    # Qwen's shortcut cell (width condition already failed by a wide
    # margin) stands as-is
    kill = []
    for slug in ("qwen", "glimmer", "inkling"):
        if slug in PRECISION_MODELS:
            va = out[out.slug == slug].iloc[0]
        else:
            e2 = pd.read_csv(ROOT / "results" / "i040_e2.csv")
            va = e2[(e2.slug == slug) & (e2.N == 50)].iloc[0]
        old = pd.read_csv(ROOT / "results" / f"coverage_{slug}.csv")
        boot = old[(old.N == 50) & (old.K == 5)
                   & (old.method == "bootstrap")].iloc[0]
        tol = 2.0 * np.sqrt(va.coverage * (1 - va.coverage) / va.B
                            + boot.coverage * (1 - boot.coverage)
                            / boot.B)
        if va.coverage >= boot.coverage - tol and \
                va.median_width <= 0.90 * boot.median_width:
            kill.append(slug)
    print(f"KILL at Ye's defaults: {kill} "
          f"({'KILL TRIGGERS' if len(kill) >= 2 else 'no KILL'})",
          flush=True)
    return kill


# ---------------------------------------------------------------------------
# the frozen decision rule, applied mechanically
# ---------------------------------------------------------------------------

def decide():
    e1 = pd.read_csv(ROOT / "results" / "i040_e1.csv")
    e2 = pd.read_csv(ROOT / "results" / "i040_e2.csv")
    kill, revise = [], []
    for slug in ("qwen", "inkling", "glimmer"):
        # KILL check, E2 primary cell (50, 5): VA vs the study bootstrap
        va = e2[(e2.slug == slug) & (e2.N == 50)].iloc[0]
        old = pd.read_csv(ROOT / "results" / f"coverage_{slug}.csv")
        boot = old[(old.N == 50) & (old.K == 5)
                   & (old.method == "bootstrap")].iloc[0]
        # "within Monte Carlo error" (frozen rule, KILL clause) read as:
        # the coverages differ by less than 2 MC standard errors of
        # their difference, SE = sqrt(c1(1-c1)/B1 + c2(1-c2)/B2)
        tol = 2.0 * np.sqrt(va.coverage * (1 - va.coverage) / va.B
                            + boot.coverage * (1 - boot.coverage) / boot.B)
        if va.coverage >= boot.coverage - tol and \
                va.median_width <= 0.90 * boot.median_width:
            kill.append(slug)
        # REVISE check, E1 primary cell (n = 5)
        e1m = e1[(e1.slug == slug) & (e1.n == 5)]
        va1 = e1m[e1m.method == "va"].iloc[0]
        va_ok = va1.mc_lower <= 0.95 <= va1.mc_upper or va1.mc_lower > 0.95
        ok_width = True
        for _, comp in e1m[e1m.method != "va"].iterrows():
            comp_ok = comp.mc_lower <= 0.95 <= comp.mc_upper \
                or comp.mc_lower > 0.95
            if comp_ok and not va1.median_width <= 0.90 * comp.median_width:
                ok_width = False
        if va_ok and ok_width:
            revise.append(slug)
    print(f"KILL models: {kill}; REVISE models: {revise}", flush=True)
    if len(kill) >= 2 and len(revise) >= 2:
        verdict = "inconclusive (triggers point both ways)"
    elif len(kill) >= 2:
        verdict = "KILL"
    elif len(revise) >= 2:
        verdict = "REVISE"
    elif len(kill) == 1 or len(revise) == 1:
        verdict = "inconclusive (trigger in exactly one model)"
    else:
        verdict = "CONFIRM"
    print(f"DECISION: {verdict}", flush=True)
    return verdict


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if stage in ("gate", "all"):
        if not run_gate():
            print("STOP: implementation unverified (frozen rule).")
            return
    if stage in ("e1", "e2", "all"):
        tidy = pd.read_parquet(ROOT / "outputs" / "correctness.parquet")
        mats = {m: model_matrix(tidy, m) for m in MODELS}
        if stage in ("e1", "all"):
            run_e1_e3(mats)
        if stage in ("e2", "all"):
            run_e2(mats)
    if stage == "precision":
        run_precision()
    if stage == "precision-full":
        run_precision_full()
    if stage in ("all", "decide"):
        decide()
    print(f"elapsed: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
