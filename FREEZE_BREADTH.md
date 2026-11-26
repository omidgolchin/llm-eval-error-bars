# Freeze note: P01 breadth arm (pass@k on code generation + fresh-pool replication)

Frozen 2026-11-23, before the first paid call of the arm. This note,
the PLAN.md addendum (§15) and the code named below are committed and
pushed publicly before any paid call, and the OSF registration of
this freeze is submitted before the first paid call.

## Questions

1. Does the interval guidance of the frozen study carry over to
   pass@k on code generation?
2. Does the coverage audit replicate on a fresh GSM8K pool?

## Dataset pin: MBPP+ (EvalPlus)

- Framework release: **EvalPlus v0.3.1** (GitHub release, published
  2024-10-20; tag object `e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2`,
  https://github.com/evalplus/evalplus/releases/tag/v0.3.1).
- Data version: **MbppPlus v0.2.0** — the version EvalPlus v0.3.1
  loads by default (`MBPP_PLUS_VERSION = "v0.2.0"` in
  `evalplus/data/mbpp.py`); release commit
  `64fc4195b858a17cdfdb3324f0baf37939144e14` in
  `evalplus/mbppplus_release`. **378 problems** (`Mbpp/2` …
  `Mbpp/809`), confirmed in the pinned file.
- Pinned artefact: `breadth/data/MbppPlus-v0.2.0.jsonl`, the
  decompressed release asset `MbppPlus.jsonl.gz` from
  https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz
  - SHA-256 (decompressed .jsonl bytes):
    `b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad`
  - MD5 (EvalPlus's own convention, `get_mbpp_plus_hash`):
    `ee43ecabebf20deef4bb776a405ac5b1`
  - SHA-256 of the .gz as downloaded:
    `af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63`
  - Hash method: SHA-256 over the exact bytes of the decompressed
    file committed at the path above; recomputing it on that file
    must reproduce the value.
- Pass = all base **and** all plus tests pass (EvalPlus
  `check_correctness` with `base_only=False`).
- Pilot dataset: **HumanEval+ v0.1.10** (the EvalPlus v0.3.1
  default), 164 problems, pinned at
  `breadth/data/HumanEvalPlus-v0.1.10.jsonl`, SHA-256
  `42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f`,
  MD5 `fe585eb4df8c88d844eeb463ea4d0302`, release commit
  `68cd26d53a0dec69f85eafe1f82a2a74155a2bd6` in
  `evalplus/humanevalplus_release`. Pilot data is never scored into
  the study.

### Licence

- The EvalPlus framework, including the MBPP+ curation and the added
  plus tests as distributed in the EvalPlus repository, is under the
  **Apache License 2.0** (LICENSE file in evalplus/evalplus at
  v0.3.1; the `evalplus/eval/` subpackage additionally complies with
  the MIT License for its HumanEval-derived parts).
- The underlying MBPP problems are **CC-BY-4.0**: the official
  dataset card (Hugging Face `google-research-datasets/mbpp`,
  Google Research) states "Licensing Information: CC-BY-4.0"
  (Austin et al. 2021, arXiv:2108.07732).
- MBPP+ is therefore used with attribution to both: EvalPlus
  (Liu et al. 2023) for the framework and added tests, and
  Austin et al. 2021 for the underlying problems. The pinned hash
  above fixes the exact artefact used, independent of licence
  interpretation.

## Models and version check

The same three models as the frozen study. On 2026-11-23 the
Together model list (free endpoint, 274 models) was checked: **all
three endpoints still exist under the identical identifiers**, so no
model is reported as a different version. Snapshot:
`breadth/models_snapshot_2026-11-23.json`.

| Together ID | Display name | Weights link shown | Context | Price in/out per 1M |
|---|---|---|---|---|
| `arize-ai/qwen-2-1.5b-instruct` | Arize AI Qwen 2 1.5B Instruct | huggingface.co/togethercomputer/arize-ai-qwen-2-1.5b-instruct | 32,768 | $0.10 / $0.10 |
| `thinkingmachines/Inkling` | Inkling FP4 | huggingface.co/thinkingmachines/Inkling-NVFP4 | 524,288 | $1.00 / $4.05 |
| `meta-models/Muse-Glimmer-30B` | Muse Glimmer 30B | huggingface.co/togethercomputer/onyx_final_hf-fp8-mlp | 131,072 | $0.35 / $1.50 |

## Prompt (verbatim)

One user message per sample, built exactly as EvalPlus v0.3.1's
OpenAI backend builds it for chat models
(`instruction_prefix + "\n```python\n{prompt.strip()}\n```"`). The
template is committed at `breadth/mbppplus_prompt.txt`:

```
Please provide a self-contained Python script that solves the following problem in a markdown code block:
```python
{problem}
```
```

with `{problem}` replaced by the problem's `prompt` field, stripped.
SHA-256 of the template file: `82aa04f763b20f69a03e6dfbbbc0c33925008819f5236e9a6fa9ba0f53726c05`.

## Sampling

- Pool and pilot: n = 20 samples per problem, temperature 0.8,
  top_p 1.0, max_tokens 1024 (2048 for a model whose pilot
  truncation rate exceeds 5%, decided at the pilot, before the pool
  run, and recorded).
- Code extraction: the EvalPlus sanitizer (v0.3.1). An extraction
  failure counts as a fail.
- Raw generations are saved before scoring
  (`scripts/breadth_run.py` writes the raw file first; scoring is a
  separate step).

## Primary cell and decision rules

- Primary cell: **k = 4, N = 50, K = n = 20**. With K = n the
  finite-pool correction is exact at `sqrt(1 - N/P)`, P the pool
  size after exclusions.
- Method: cluster-t interval over problems, `v_i` from all K samples
  via the unbiased pass@k estimator (Chen et al. 2021,
  arXiv:2107.03374); the problem bootstrap is reported beside it.
  Raw (uncorrected) coverage is always reported next to the
  corrected coverage, with the analytic-prediction column (the raw
  coverage a perfectly calibrated interval would show against the
  finite-pool truth).
- Inclusion rule: a model enters the coverage decision only if its
  pool pass@1 is at least 0.02 **and** its pool pass@4 is at most
  0.98. A model outside these bounds is reported descriptively.
- Decision per included model, on corrected coverage with a
  Clopper-Pearson MC interval (B = 2000):
  - **HOLDS**: MC upper bound at least 0.94;
  - **UNDER-COVERS**: MC upper bound below 0.94;
  - **CONSERVATIVE**: MC lower bound above 0.97 (reported, not a
    failure; takes precedence over HOLDS).
- Arm outcome (pre-written):
  - every included model HOLDS → "the guidance carries over to
    pass@k";
  - exactly one model UNDER-COVERS → "carries over, with a named
    exception";
  - two or more → "does not carry over".

## Exploratory grid and the K < n correction

Grid: k in {1, 2, 4, 8}, N in {25, 50, 100, 200}, K in {5, 10, 20},
k <= K. Every output labels these cells exploratory. For K < 20 the
correction factor is `sqrt((Var_A + Var_B) / V_sup)` with:

- `S2` = variance across problems of `v_i(k, 20)`;
- `Var_A = (1 - N/P) * S2 / N`;
- `Var_B` = mean over problems of the variance of `v_i(k, K)` across
  random K-of-20 subsets of that problem's samples (Monte Carlo, 500
  subsets per problem; drawn through the hypergeometric distribution
  of the subset success count);
- `W_m` = mean over problems of `Var[v_i(k, m)]` under
  Binomial(m, p_hat_i) resampling. **Definition (adopted at freeze):
  the exact sum** of Milletarì & Kundu (2026), §3.3 Eq. (30):
  `Var(h_k(C)|p) = sum_c h_k(c)^2 * C(m,c) p^c (1-p)^(m-c)
  - (1 - (1-p)^k)^2`, verified against 400,000-draw Monte Carlo
  across 28 cells (agreement within MC noise; reduces exactly to
  `p(1-p)/m` at k = 1). The Monte Carlo estimate (2000 draws, as
  originally specified) is kept in the code as the implementation
  check; it is itself noisy (±6–9% at mid-range p). `W_m` remains a
  plug-in at `p_hat_i`, and K < 20 corrections that use it are
  labelled approximate;
- `V_sup = (S2 - W_20 + W_K) / N`.

At K = 20 the factor reduces to `sqrt(1 - N/P)`; that reduction is a
unit test (`tests/test_passk.py`).

Naive comparator: **k = 1 only** — Wilson on all N x K binary
outcomes, exploratory. The primary cell has no naive comparator.
(The pooled pass@k plug-in construction is not used: it is biased
upward by Jensen's inequality.)

## Pilot (HumanEval+)

20 problems (the first 20 task_ids of the pinned file) x 20 samples
x 3 models. It calibrates tokens, cost, truncation and the harness
without touching the MBPP+ pool. Pilot data is never scored into the
study; it is scored only to estimate pass@1 for the trim rule and
the projection.

## Replication arm (fresh GSM8K pool)

- 400 GSM8K test items not among the frozen 400, drawn with seed
  **20261123** from the remaining 919 (the frozen pool is the
  seed-20261112 draw of PLAN.md §3; the fresh draw is
  `numpy.random.default_rng(20261123).choice(remaining, size=400,
  replace=False)`, indices sorted; the drawn question IDs are
  recorded in `breadth/fresh_pool_20261123.json` when the arm runs).
  The seed matches the OSF registration draft.
- x 5 seeds x 3 models, the frozen prompt and settings (temperature
  0.7, top_p 1.0, max_tokens 512), graded by the frozen grader.
- The frozen pipeline runs with the exact finite-pool correction
  inside the fresh pool; read at N = 50, K = 5.
- Replicates, per model: the fresh corrected cluster coverage
  differs from the frozen corrected value by less than 2 SE of the
  difference. The Inkling exception replicates if its fresh
  corrected MC upper bound is below 0.95.
- Gate for generalised public wording: the replication holds in at
  least 2 of 3 models with unchanged versions.
- Estimated cost about $5. This arm is never trimmed.

## Budget and trims

- Cap **$45** for the whole arm (pilot, pass@k pool, replication);
  generation stops automatically at **$36** (tracked per call in
  `breadth/outputs/spend.jsonl`).
- If the pilot's projection exceeds the cap, trim in this order:
  1. n = 20 → 16, with the primary K staying equal to n;
  2. leave out of the pool run any model whose pilot pass@1 is
     below 0.02;
  3. n = 16 → 12.
- Never trimmed: the replication arm.

## Harness

- Scoring: EvalPlus v0.3.1 `check_correctness` (base + plus),
  ground truth from the canonical solutions, per-test time limits
  `max(min_time_limit, gt_time_limit_factor * t)` with the EvalPlus
  defaults min_time_limit 1 s and gt_time_limit_factor 4, recorded
  here as frozen.
- Isolation (frozen): EvalPlus execution inside Docker with
  `--network none`, 2 GB memory, one process per sample.
  **Gap recorded at freeze:** Docker is not available in the
  environment where this freeze was prepared (`docker` not
  installed). There, scoring runs one process per sample in a fresh
  subprocess; the 2 GB limit is enforced by EvalPlus's own
  reliability guard on the process that executes the sample
  (`EVALPLUS_MAX_MEMORY_BYTES` = 2 GB), with a generous
  address-space backstop on the wrapper process. **Closing the Docker gap —
  or re-recording the equivalent isolation — is an item to close
  before the pool run.**
- Scorer validation (committed with this freeze): canonical
  solutions pass all base and plus tests and deliberately broken
  variants (renamed entry point, truncated body, non-Python garbage)
  fail, on the first 12 problems of each dataset;
  `breadth/validation_report.json`, produced by
  `scripts/breadth_run.py validate`, re-runnable any time.

## Hashes (SHA-256, filled at freeze commit)

- prompt template `breadth/mbppplus_prompt.txt`: `82aa04f763b20f69a03e6dfbbbc0c33925008819f5236e9a6fa9ba0f53726c05`
- analysis script `src/errorbars/passk.py`: `329d4bd3ad42ca7076c9fea1bd0a1b11ac2d7a7bd91268def126a48f380ec7f3`
- harness `src/errorbars/codeeval.py`: `10a3f6b58224abe25fc5ce2988fdfc4913af167643a304659fad087b501aa01b`
- runner `scripts/breadth_run.py`: `17f8db92c759b056835974d51007313ea40f0ad441df5649cca6028292a32d2e`
- analysis CLI `scripts/breadth_analyze.py`: `9b4b8b07ffa625d1371c20cd6d2ab9932144c3f8f2f4ff233aa2f4589d30d4ca`

Recompute with `sha256sum` on the committed files; the values in the
committed version of this note are the frozen ones.
