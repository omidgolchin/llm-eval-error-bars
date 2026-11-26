"""Breadth-arm runner: pilot, pool and replication arms.

Frozen design: see ``FREEZE_BREADTH.md``. Raw generations are always
written to disk before any scoring happens. Generation is resumable
by run_id and guarded by the arm budget: cumulative spend is tracked
in ``breadth/outputs/spend.jsonl`` and generation stops before any
call that would cross the $36 automatic stop (hard cap $45).

Subcommands:

- ``validate``     scorer validation (canonical pass, broken fail)
                   for MBPP+ and HumanEval+; writes
                   ``breadth/validation_report.json``.
- ``pilot``        HumanEval+, first 20 problems x 20 samples x 3
                   models; calibrates tokens, cost and truncation.
                   Pilot data is never scored into the study (it is
                   scored here only to estimate pass@1 for the trim
                   rule and the projection).
- ``pool``         MBPP+, all 378 problems x n samples x 3 models.
- ``score``        score a raw-generations file (extraction +
                   EvalPlus checks), writing per-sample pass flags
                   and per-problem success counts.
- ``replication``  fresh GSM8K pool: 400 items drawn with seed
                   20261123 from the 919 test items outside the
                   frozen 400, x 5 seeds x 3 models at the frozen
                   GSM8K settings; graded with the frozen grader.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from errorbars import codeeval  # noqa: E402
from errorbars.run import get_client  # noqa: E402

BREADTH = ROOT / "breadth"
OUT = BREADTH / "outputs"
MODELS = [
    "arize-ai/qwen-2-1.5b-instruct",
    "thinkingmachines/Inkling",
    "meta-models/Muse-Glimmer-30B",
]
BUDGET_CAP_USD = 45.0
BUDGET_STOP_USD = 36.0
FRESH_POOL_SEED = 20261123
FROZEN_POOL_SEED = 20261112


# ---------------------------------------------------------------------------
# spend ledger
# ---------------------------------------------------------------------------

def _spend_path() -> Path:
    return OUT / "spend.jsonl"


def spent_so_far() -> float:
    p = _spend_path()
    if not p.exists():
        return 0.0
    return sum(json.loads(line)["cost_usd"] for line in open(p) if line.strip())


def _record_spend(entry: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(_spend_path(), "a") as f:
        f.write(json.dumps(entry) + "\n")


def _prices() -> dict:
    snap = json.loads((BREADTH / "models_snapshot_2026-11-23.json").read_text())
    return {m: (snap[m]["pricing"]["input"], snap[m]["pricing"]["output"])
            for m in MODELS}


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def _existing_run_ids(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(line)["run_id"] for line in open(path) if line.strip()}


def generate(jobs: list, out_path: Path, temperature: float, top_p: float,
             max_tokens: int, arm: str) -> dict:
    """Run generation jobs, saving every raw generation before scoring.

    Each job: {model, task_id, sample, prompt}. Stops before crossing
    the automatic budget stop.
    """
    client = get_client()
    prices = _prices()
    done = _existing_run_ids(out_path)
    spent = spent_so_far()
    stats = {"calls": 0, "cost_usd": 0.0, "stopped_on_budget": False}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        for job in jobs:
            run_id = f"{job['model']}|{job['task_id']}|{job['sample']}"
            if run_id in done:
                continue
            if spent >= BUDGET_STOP_USD:
                stats["stopped_on_budget"] = True
                break
            row = {"run_id": run_id, "arm": arm, "model": job["model"],
                   "task_id": job["task_id"], "sample": job["sample"],
                   "text": None, "finish_reason": None,
                   "prompt_tokens": None, "completion_tokens": None,
                   "error": None}
            try:
                resp = client.chat.completions.create(
                    model=job["model"],
                    messages=[{"role": "user", "content": job["prompt"]}],
                    temperature=temperature, top_p=top_p,
                    max_tokens=max_tokens, seed=job["sample"])
                choice = resp.choices[0]
                row["text"] = choice.message.content
                row["finish_reason"] = choice.finish_reason
                row["prompt_tokens"] = resp.usage.prompt_tokens
                row["completion_tokens"] = resp.usage.completion_tokens
            except Exception as e:  # recorded, counted like the frozen policy
                row["error"] = str(e)[:300]
            f.write(json.dumps(row) + "\n")
            f.flush()
            pin, pout = prices[job["model"]]
            cost = ((row["prompt_tokens"] or 0) * pin
                    + (row["completion_tokens"] or 0) * pout) / 1e6
            spent += cost
            stats["calls"] += 1
            stats["cost_usd"] += cost
            _record_spend({"arm": arm, "run_id": run_id, "cost_usd": cost,
                           "ts": time.time()})
    return stats


def code_jobs(records: list, models: list, n_samples: int,
              prompt_of) -> list:
    jobs = []
    for rec in records:
        prompt = prompt_of(rec)
        for model in models:
            for s in range(n_samples):
                jobs.append({"model": model, "task_id": rec["task_id"],
                             "sample": s, "prompt": prompt})
    return jobs


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score_raw(raw_path: Path, dataset: str, out_path: Path) -> dict:
    """Extract and score every raw generation; write counts per problem."""
    problems = codeeval._evalplus_problems(dataset)
    expected_all = codeeval._groundtruth(dataset, problems)
    rows = [json.loads(line) for line in open(raw_path) if line.strip()]
    counts: dict = {}
    scored = []
    for row in rows:
        problem = problems[row["task_id"]]
        code = None
        if row.get("text"):
            code = codeeval.extract_solution(row["text"],
                                             problem["entry_point"])
        if code is None:
            res = {"pass": False, "base_status": "extract_fail",
                   "plus_status": "extract_fail"}
        else:
            res = codeeval.score_solution_isolated(
                dataset, problem, expected_all[row["task_id"]], code)
        scored.append({**row, "text": None, "extracted": code is not None,
                       **res})
        key = (row["model"], row["task_id"])
        c = counts.setdefault(key, {"successes": 0, "samples": 0})
        c["samples"] += 1
        c["successes"] += int(res["pass"])
    with open(out_path, "w") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")
    counts_out = {f"{m}|{t}": v for (m, t), v in sorted(counts.items())}
    counts_path = out_path.with_suffix(".counts.json")
    counts_path.write_text(json.dumps(counts_out, indent=1))
    return {"scored": len(scored), "counts_path": str(counts_path)}


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

def cmd_validate(_args) -> None:
    report = {}
    for dataset in ("mbpp", "humaneval"):
        report[dataset] = codeeval.validate_scorer(dataset=dataset)
    (BREADTH / "validation_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({d: {"ok": r["ok"], "docker": r["docker_available"],
                          "problems": len(r["problems"])}
                      for d, r in report.items()}, indent=1))


def cmd_pilot(_args) -> None:
    records = codeeval.load_humanevalplus_raw()[:20]
    jobs = code_jobs(records, MODELS, 20,
                     lambda r: codeeval.build_prompt(r["prompt"]))
    stats = generate(jobs, OUT / "pilot_raw.jsonl", temperature=0.8,
                     top_p=1.0, max_tokens=1024, arm="pilot")
    print(json.dumps(stats, indent=1))
    print("score with: breadth_run.py score --raw breadth/outputs/pilot_raw.jsonl "
          "--dataset humaneval --out breadth/outputs/pilot_scored.jsonl")


def cmd_pool(args) -> None:
    records = codeeval.load_mbppplus_raw()
    jobs = code_jobs(records, MODELS, args.n_samples,
                     lambda r: codeeval.build_prompt(r["prompt"]))
    stats = generate(jobs, OUT / "pool_raw.jsonl", temperature=0.8,
                     top_p=1.0, max_tokens=args.max_tokens, arm="pool")
    print(json.dumps(stats, indent=1))


def cmd_replication(_args) -> None:
    from errorbars.data import load_gsm8k
    from errorbars.grade import grade

    df = load_gsm8k("test")
    frozen_idx = np.sort(np.random.default_rng(FROZEN_POOL_SEED).choice(
        len(df), size=400, replace=False))
    remaining = np.array([i for i in range(len(df)) if i not in set(frozen_idx)])
    assert len(remaining) == 919, len(remaining)
    fresh_idx = np.sort(np.random.default_rng(FRESH_POOL_SEED).choice(
        remaining, size=400, replace=False))
    fresh = df.iloc[fresh_idx].reset_index(drop=True)
    (BREADTH / "fresh_pool_20261123.json").write_text(json.dumps(
        {"seed": FRESH_POOL_SEED,
         "question_ids": list(fresh["question_id"])}, indent=1))

    template = (ROOT / "prompts" / "gsm8k_prompt.txt").read_text()
    jobs = []
    gold = {}
    for _, rec in fresh.iterrows():
        gold[rec["question_id"]] = rec["gold"]
        for model in MODELS:
            for s in range(5):
                jobs.append({"model": model, "task_id": rec["question_id"],
                             "sample": s,
                             "prompt": template.replace("{question}",
                                                        rec["question"])})
    raw_path = OUT / "replication_raw.jsonl"
    stats = generate(jobs, raw_path, temperature=0.7, top_p=1.0,
                     max_tokens=512, arm="replication")
    # grade with the frozen grader; one row per (model, question, seed)
    rows = [json.loads(line) for line in open(raw_path) if line.strip()]
    with open(OUT / "replication_scored.jsonl", "w") as f:
        for row in rows:
            correct = False
            if row.get("text"):
                correct = grade(row["text"], gold[row["task_id"]])[0] == 1
            f.write(json.dumps({**row, "text": None, "correct": correct}) + "\n")
    print(json.dumps(stats, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    sub.add_parser("pilot")
    p_pool = sub.add_parser("pool")
    p_pool.add_argument("--n-samples", type=int, default=20)
    p_pool.add_argument("--max-tokens", type=int, default=1024)
    p_score = sub.add_parser("score")
    p_score.add_argument("--raw", required=True)
    p_score.add_argument("--dataset", choices=["mbpp", "humaneval"],
                         required=True)
    p_score.add_argument("--out", required=True)
    sub.add_parser("replication")
    args = ap.parse_args()
    if args.cmd == "validate":
        cmd_validate(args)
    elif args.cmd == "pilot":
        cmd_pilot(args)
    elif args.cmd == "pool":
        cmd_pool(args)
    elif args.cmd == "score":
        print(json.dumps(score_raw(Path(args.raw), args.dataset,
                                   Path(args.out)), indent=1))
    elif args.cmd == "replication":
        cmd_replication(args)


if __name__ == "__main__":
    main()
