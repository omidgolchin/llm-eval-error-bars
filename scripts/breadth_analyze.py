"""Breadth-arm analysis: coverage study, decision, replication check.

Reads the per-problem success counts produced by
``scripts/breadth_run.py score`` and runs the frozen analysis
(``src/errorbars/passk.py``): the primary cell first, then the
exploratory grid, all labelled. Writes CSVs under
``breadth/results/`` and prints the per-model verdicts and the
pre-written arm outcome.

The replication check compares the fresh pool's corrected cluster
coverage at N = 50, K = 5 (computed by the same machinery on the
fresh 400 x 5 correctness matrices) against the frozen corrected
values in ``results/fpc_coverage_*.csv``: a model replicates if the
difference is less than 2 SE of the difference; the Inkling
exception replicates if its fresh corrected MC upper bound is below
0.95. The generalised wording gate is at least 2 of 3 models
replicating.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from errorbars import passk  # noqa: E402

BREADTH = ROOT / "breadth"
MODELS = [
    "arize-ai/qwen-2-1.5b-instruct",
    "thinkingmachines/Inkling",
    "meta-models/Muse-Glimmer-30B",
]
SHORT = {"arize-ai/qwen-2-1.5b-instruct": "qwen",
         "thinkingmachines/Inkling": "inkling",
         "meta-models/Muse-Glimmer-30B": "glimmer"}


def counts_from_scored(counts_path: Path, model: str, n: int) -> np.ndarray:
    counts = json.loads(counts_path.read_text())
    vals = []
    for key, v in counts.items():
        m, _task = key.split("|", 1)
        if m == model:
            assert v["samples"] == n, (key, v)
            vals.append(v["successes"])
    return np.array(sorted(vals), dtype=int)


def run_pool_analysis(counts_path: Path, n: int, B: int, seed: int) -> dict:
    out_dir = BREADTH / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    verdicts = {}
    summary = {}
    for model in MODELS:
        counts = counts_from_scored(counts_path, model, n)
        cells = [dict(passk.PRIMARY_CELL)] + [
            {"k": k, "N": N, "K": K}
            for k in passk.GRID_K for N in passk.GRID_N
            for K in passk.GRID_K_SUB if k <= K
            and {"k": k, "N": N, "K": K} != passk.PRIMARY_CELL]
        rows = passk.coverage_study(counts, n, cells, B=B, seed=seed)
        path = out_dir / f"passk_coverage_{SHORT[model]}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        primary = rows[0]
        inc = passk.included_in_decision(counts, n)
        dec = passk.decide(primary["cluster_t_coverage_corrected"], B)
        summary[model] = {
            "pool_pass1": passk.pool_passk(counts, n, 1),
            "pool_pass4": passk.pool_passk(counts, n, 4),
            "included_in_decision": inc,
            "primary": primary,
            "decision": dec,
        }
        if inc:
            verdicts[model] = dec["verdict"]
    summary["arm_outcome"] = passk.arm_outcome(verdicts)
    (out_dir / "passk_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def run_replication_check(scored_path: Path, B: int, seed: int) -> dict:
    rows = [json.loads(line) for line in open(scored_path) if line.strip()]
    out = {}
    n_replicating = 0
    for model in MODELS:
        mat: dict = {}
        for r in rows:
            if r["model"] == model:
                mat.setdefault(r["task_id"], {})[r["sample"]] = int(r["correct"])
        questions = sorted(mat)
        counts = np.array([sum(mat[q].values()) for q in questions], dtype=int)
        res = passk.coverage_study(counts, 5, [{"k": 1, "N": 50, "K": 5}],
                                   B=B, seed=seed)[0]
        fresh_cov = res["cluster_t_coverage_corrected"]
        fresh_lo, fresh_hi = passk.clopper_pearson(
            int(round(fresh_cov * B)), B)
        fpc = list(csv.DictReader(
            open(ROOT / "results" / f"fpc_coverage_{SHORT[model]}.csv")))
        frozen = [r for r in fpc if r["method"] == "cluster"
                  and int(r["N"]) == 50 and int(r["K"]) == 5][0]
        frozen_cov = float(frozen["corrected_coverage"])
        se_diff = float(np.sqrt(fresh_cov * (1 - fresh_cov) / B
                                + frozen_cov * (1 - frozen_cov) / B))
        replicates = abs(fresh_cov - frozen_cov) < 2 * se_diff
        entry = {"fresh_corrected_coverage": fresh_cov,
                 "fresh_mc": [fresh_lo, fresh_hi],
                 "frozen_corrected_coverage": frozen_cov,
                 "se_of_difference": se_diff,
                 "replicates": bool(replicates)}
        if SHORT[model] == "inkling":
            entry["exception_replicates"] = bool(fresh_hi < 0.95)
        n_replicating += int(replicates)
        out[model] = entry
    out["generalised_wording_gate"] = n_replicating >= 2
    (BREADTH / "results" / "replication_check.json").write_text(
        json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--counts", default="breadth/outputs/pool_scored.counts.json")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--replication-scored",
                    default="breadth/outputs/replication_scored.jsonl")
    ap.add_argument("--skip-replication", action="store_true")
    args = ap.parse_args()
    summary = run_pool_analysis(ROOT / args.counts, args.n_samples,
                                args.B, args.seed)
    for model in MODELS:
        s = summary[model]
        print(f"{model}: pass@1={s['pool_pass1']:.3f} "
              f"pass@4={s['pool_pass4']:.3f} included={s['included_in_decision']} "
              f"verdict={s['decision']['verdict']}")
    print("arm outcome:", summary["arm_outcome"])
    if not args.skip_replication and Path(args.replication_scored).exists():
        rep = run_replication_check(ROOT / args.replication_scored,
                                    args.B, args.seed)
        print("replication:", json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
