"""Breadth-arm harness: prompt, extraction and sandboxed scoring.

Scoring follows EvalPlus v0.3.1 exactly (pinned in
``FREEZE_BREADTH.md``): the model's raw output is saved before any
processing; the EvalPlus sanitizer extracts the solution; the
sanitized code is checked with EvalPlus's own ``check_correctness``
against the ground truth computed from the canonical solution, over
both the base and the plus inputs. A sample passes only if it passes
all base and all plus tests. An extraction failure counts as a fail.

Execution isolation: EvalPlus executes each check in a separate
process with its reliability guard (memory capped through
``EVALPLUS_MAX_MEMORY_BYTES``; the freeze sets 2 GB) and per-test
time limits of ``max(min_time_limit, gt_time_limit_factor * t)`` with
the EvalPlus defaults ``min_time_limit = 1`` second and
``gt_time_limit_factor = 4``. Where Docker is available the whole
scoring step can be run inside ``docker run --network none
--memory 2g`` (see ``scripts/breadth_run.py``); this environment has
no Docker, so scoring runs one process per sample through
:func:`score_solution` and the gap is recorded in
``FREEZE_BREADTH.md`` as an item to close before the pool run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BREADTH_DIR = REPO_ROOT / "breadth"
PROMPT_TEMPLATE_PATH = BREADTH_DIR / "mbppplus_prompt.txt"
MBPPPLUS_PATH = BREADTH_DIR / "data" / "MbppPlus-v0.2.0.jsonl"
HUMANEVALPLUS_PATH = BREADTH_DIR / "data" / "HumanEvalPlus-v0.1.10.jsonl"

MIN_TIME_LIMIT = 1.0
GT_TIME_LIMIT_FACTOR = 4.0
MAX_MEMORY_BYTES = 2 * 1024 * 1024 * 1024

# Pin EvalPlus's loaders to the committed dataset files. These are
# read by evalplus at import time, so they are set here, before any
# evalplus import can happen.
os.environ.setdefault("MBPP_OVERRIDE_PATH", str(MBPPPLUS_PATH))
os.environ.setdefault("HUMANEVAL_OVERRIDE_PATH", str(HUMANEVALPLUS_PATH))

PASS = "pass"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def prompt_template() -> str:
    """The frozen prompt template (EvalPlus v0.3.1 chat construction)."""
    return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")


def build_prompt(task_prompt: str) -> str:
    """Build the single user message for one problem.

    Mirrors EvalPlus v0.3.1's OpenAI-backend construction:
    ``instruction_prefix + "\\n```python\\n{prompt.strip()}\\n```"``
    where the instruction prefix is "Please provide a self-contained
    Python script that solves the following problem in a markdown
    code block:".
    """
    return prompt_template().replace("{problem}", task_prompt.strip())


# ---------------------------------------------------------------------------
# Dataset loading (pinned files; EvalPlus loaders for scoring inputs)
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_mbppplus_raw() -> list:
    """The pinned MBPP+ records, exactly as distributed."""
    return load_jsonl(MBPPPLUS_PATH)


def load_humanevalplus_raw() -> list:
    """The pinned HumanEval+ records, exactly as distributed."""
    return load_jsonl(HUMANEVALPLUS_PATH)


def _evalplus_problems(dataset: str) -> dict:
    """Problems through EvalPlus's own loaders (input deserialisation
    included), reading the pinned files via the override paths."""
    if dataset == "mbpp":
        os.environ["MBPP_OVERRIDE_PATH"] = str(MBPPPLUS_PATH)
        from evalplus.data import get_mbpp_plus
        return get_mbpp_plus()
    if dataset == "humaneval":
        os.environ["HUMANEVAL_OVERRIDE_PATH"] = str(HUMANEVALPLUS_PATH)
        from evalplus.data import get_human_eval_plus
        return get_human_eval_plus()
    raise ValueError(f"unknown dataset {dataset!r}")


# ---------------------------------------------------------------------------
# Extraction (EvalPlus sanitizer)
# ---------------------------------------------------------------------------

def extract_solution(raw_text: str, entry_point: str):
    """Sanitize a raw generation. Returns None on extraction failure.

    A failure here counts as a failed sample in the study; it is
    never repaired by hand.
    """
    from evalplus.sanitize import sanitize
    try:
        code = sanitize(raw_text, entrypoint=entry_point)
    except Exception:
        return None
    if not code or not code.strip():
        return None
    return code


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def docker_available() -> bool:
    """True if a Docker daemon answers (the frozen isolation route)."""
    if shutil.which("docker") is None:
        return False
    try:
        out = subprocess.run(["docker", "info"], capture_output=True,
                             timeout=15)
        return out.returncode == 0
    except Exception:
        return False


def _groundtruth(dataset: str, problems: dict) -> dict:
    from evalplus.evaluate import get_groundtruth
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    if dataset == "mbpp":
        from evalplus.data import get_mbpp_plus_hash
        return get_groundtruth(problems, get_mbpp_plus_hash(),
                               MBPP_OUTPUT_NOT_NONE_TASKS)
    from evalplus.data import get_human_eval_plus_hash
    return get_groundtruth(problems, get_human_eval_plus_hash(), [])


def score_solution(dataset: str, problem: dict, expected: dict,
                   code: str) -> dict:
    """Score one sanitized solution with EvalPlus's check_correctness.

    Pass = the base check and the plus check both report "pass"
    (all base and all plus tests pass).
    """
    os.environ.setdefault("EVALPLUS_MAX_MEMORY_BYTES",
                          str(MAX_MEMORY_BYTES))
    from evalplus.evaluate import check_correctness
    ret = check_correctness(
        dataset, 0, problem, code, expected,
        base_only=False, fast_check=False, identifier=None,
        min_time_limit=MIN_TIME_LIMIT,
        gt_time_limit_factor=GT_TIME_LIMIT_FACTOR)
    base_status = ret["base"][0]
    plus_status = ret["plus"][0]
    return {"pass": base_status == PASS and plus_status == PASS,
            "base_status": base_status, "plus_status": plus_status}


def score_solution_isolated(dataset: str, problem: dict, expected: dict,
                            code: str) -> dict:
    """Score in a fresh subprocess (one process per sample).

    Used when Docker is unavailable: the child imports this module,
    scores the single sample, and prints the result as JSON. The
    frozen 2 GB limit is enforced by EvalPlus's own reliability
    guard on the process that executes the sample
    (``EVALPLUS_MAX_MEMORY_BYTES``); the child itself gets a
    generous 16 GB address-space backstop, because an address-space
    rlimit counts virtual memory and the imports plus the
    ground-truth cache alone exceed 2 GB of it.

    Only the dataset, task_id and code cross the process boundary:
    the child rebuilds the problem and the ground truth itself,
    because a JSON round-trip would turn the tuples in the inputs
    and expected outputs into lists and change the semantics.
    """
    payload = json.dumps({"dataset": dataset,
                          "task_id": problem["task_id"], "code": code})
    child = (
        "import json, resource, sys\n"
        "resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))\n"
        "sys.path.insert(0, 'src')\n"
        "from errorbars import codeeval\n"
        "p = json.loads(sys.stdin.read())\n"
        "problems = codeeval._evalplus_problems(p['dataset'])\n"
        "problem = problems[p['task_id']]\n"
        "expected = codeeval._groundtruth(p['dataset'], problems)[p['task_id']]\n"
        "print(json.dumps(codeeval.score_solution(p['dataset'], problem, "
        "expected, p['code'])))\n"
    )
    out = subprocess.run([sys.executable, "-c", child], input=payload,
                         capture_output=True, text=True,
                         cwd=str(REPO_ROOT), timeout=600)
    if out.returncode != 0:
        return {"pass": False, "base_status": "error",
                "plus_status": "error",
                "stderr": out.stderr[-500:]}
    return json.loads(out.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# Scorer validation: canonical solutions pass, broken variants fail
# ---------------------------------------------------------------------------

def canonical_code(dataset: str, problem: dict) -> str:
    """The full canonical solution as EvalPlus executes it.

    For MBPP+ the canonical solution is self-contained. For
    HumanEval+ it is a function body and needs the prompt (signature
    and docstring) in front of it, as in EvalPlus's ground truth.
    """
    if dataset == "humaneval":
        return problem["prompt"] + problem["canonical_solution"]
    return problem["canonical_solution"]


def broken_variants(dataset: str, problem: dict) -> dict:
    """Deliberately broken variants of a problem's canonical solution."""
    canon = canonical_code(dataset, problem)
    entry = problem["entry_point"]
    return {
        # the entry point is gone: extraction/execution cannot find it
        "renamed_entry": canon.replace(entry, entry + "_broken", 1),
        # truncated mid-body: not a complete definition
        "truncated": canon[: max(1, len(canon) // 2)],
        # not Python at all
        "garbage": "def broken(:\n  this is not python",
    }


def validate_scorer(dataset: str = "mbpp", n_problems: int = 12,
                    isolated: bool = True) -> dict:
    """Run the scorer validation on the first ``n_problems`` problems.

    Every canonical solution must pass all base and plus tests; every
    broken variant must fail. Returns a report dict; ``ok`` is True
    only if every expectation holds.
    """
    problems = _evalplus_problems(dataset)
    expected_all = _groundtruth(dataset, problems)
    score = score_solution_isolated if isolated else score_solution
    task_ids = list(problems)[:n_problems]
    report = {"dataset": dataset, "docker_available": docker_available(),
              "problems": {}, "ok": True}
    for task_id in task_ids:
        problem = problems[task_id]
        expected = expected_all[task_id]
        entry = {"canonical": score(dataset, problem, expected,
                                    canonical_code(dataset, problem)),
                 "broken": {}}
        if not entry["canonical"]["pass"]:
            report["ok"] = False
        for name, code in broken_variants(dataset, problem).items():
            res = score(dataset, problem, expected, code)
            entry["broken"][name] = res
            if res["pass"]:
                report["ok"] = False
        report["problems"][task_id] = entry
    return report
