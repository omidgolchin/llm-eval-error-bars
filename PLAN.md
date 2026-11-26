# Plan: Error bars for repeated-run LLM evaluations

Status: Stage 2 plan, written and frozen before the main run. It stops at
Checkpoint 2 for approval; no main-run call is made until this plan is
approved. `FREEZE_MANIFEST.txt` records SHA-256 hashes of this plan, the
config, the prompt, and the code as frozen.

Date frozen: 2026-11-12.

## 1. Research question

When an LLM is evaluated on N questions with K repeated runs (seeds) per
question, which interval methods actually cover the true accuracy at their
nominal 95% level, and how should a fixed budget of calls be split between
more questions and more seeds?

### Honest framing

Clustering uncertainty by question is **not** this project's idea. It is the
recommendation of Miller (2024), "Adding Error Bars to Evals: A Statistical
Approach to Language Model Evaluations" (arXiv:2411.00640), and this project
follows it. The contributions here are:

1. an empirical coverage check of several interval methods on real model
   runs (not only on simulated data);
2. a hierarchical Bayesian interval compared against the frequentist
   methods on the same runs;
3. practical budget guidance: how to split a fixed number of calls between
   questions and seeds;
4. a reusable function (`errorbars.intervals`) and an interactive
   playground built on the real runs.

No claim is made anywhere in the project that clustering by question
originated here.

## 2. Estimands

- **Primary:** the population accuracy over questions like these — the mean
  accuracy over the population of GSM8K-like questions, averaged over the
  model's sampling randomness (temperature sampling across seeds). All main
  interval methods target this estimand, each in its own way (see §8 for
  each method's stated target).
- **Secondary:** the paired accuracy difference between two models run on
  the same questions with the same seeds.
- **Reported separately, different estimand:** the seed-only interval
  targets accuracy on *these* N questions only, treating the questions as
  fixed. It says nothing about new questions and is always labelled as a
  different estimand, never mixed into the primary comparison.

## 3. Benchmark and data

- Dataset: GSM8K, Hugging Face `openai/gsm8k`, config `main`, **test**
  split (1,319 questions), loaded by `src/errorbars/data.py`.
- Gold answer: the number after `####` in the reference solution, parsed
  and normalized by the same code that grades model output
  (`src/errorbars/grade.py:parse_gold`). Question IDs are
  `gsm8k-test-<index>`.
- **Reference pool:** 400 questions sampled once from the test split,
  without replacement, with fixed seed **20261112**
  (`sample_questions`: `numpy.random.default_rng(20261112).choice(1319,
  size=400, replace=False)`, indices sorted). This pool is the same for all
  three models and for every analysis design.
- **Reference truth** for a model: the mean of `correct` over all
  400 × 5 = 2,000 cells of that model's main run. Coverage is measured
  against this reference value; it is a finite-pool reference, not the
  unknowable population value (see Limitations, §12).

## 4. Models

Chosen by pilot accuracy in the 45–85% band (Stage 1 pilots:
`results/pilot_summary.md`, `results/pilot2_summary.md`; pilot design
40 questions × 2 seeds, identical prompt and sampling settings). Approved
at revised Checkpoint 1 on 2026-11-12. All three are served via Together
AI serverless (`https://api.together.xyz/v1`).

| Model (Together ID) | Pilot accuracy | Pilot parse-fail | Pilot cost / call | Projected cost, 2,000 main-run calls |
|---|---|---|---|---|
| `arize-ai/qwen-2-1.5b-instruct` | 52.5% (42/80) | 0.0% (0/80) | $0.000033 | $0.07 |
| `thinkingmachines/Inkling` | 78.8% (63/80) | 7.5% (6/80) | $0.001303 | $2.61 |
| `meta-models/Muse-Glimmer-30B` | 71.3% (57/80) | 26.2% (21/80) | $0.000666 | $1.33 |

- The three models are from three different families, as required.
- **Projected total main-run spend ≈ $4.01** for 6,000 calls.
- **Stop rule:** if the pilot-based projection had exceeded $10
  (`config.yaml: main.cost_stop_usd`), the main run would not start and the
  overrun would be reported instead. $4.01 is under the threshold.
- Pilot spend already incurred (reported for completeness, not part of the
  main run): $0.0504 (pilot 1) + $0.71 (pilot 2) ≈ $0.76.
- Costs are computed from Together catalog prices at pilot time combined
  with observed mean prompt/completion tokens; the final summary reports
  actual spend on the same basis.

## 5. Run design (Stage 3, after Checkpoint 2 approval)

- **Grid:** 400 pool questions × 5 seeds × 3 models = **6,000 calls**.
  Seeds are the integers **[0, 1, 2, 3, 4]** (`config.yaml: main.seeds`);
  each (model, question, seed) cell is one call, and the cell's seed is
  passed to the API as that call's `seed` parameter, so the 5 replicates
  of a question use 5 distinct seeds.
- **Call order:** all 6,000 jobs are shuffled once before running
  (`random.Random(20261112).shuffle` in `run_grid`), so model, question,
  and seed are interleaved rather than run in blocks.
- **Sampling settings** (from `config.yaml`, identical to the pilots):
  temperature 0.7, top_p 1.0, max_tokens 512.
- **Prompt:** the single fixed prompt in `prompts/gsm8k_prompt.txt`
  (zero-shot chain-of-thought ending with "Give the final answer after
  '####'."), with `{question}` replaced by the question text. No per-model
  or per-question prompt variation.
- **Command:**

  ```
  .venv/bin/python -m errorbars.run \
    --models arize-ai/qwen-2-1.5b-instruct thinkingmachines/Inkling \
             meta-models/Muse-Glimmer-30B \
    --limit-questions-from-pool --n-questions 400 \
    --seeds 0 1 2 3 4 --question-seed 20261112 \
    --out outputs/raw_runs.jsonl
  ```

  (`--limit-questions-from-pool` samples the frozen 400-question pool from
  the test split with `pool_seed` 20261112; `--n-questions 400` then keeps
  all 400.)
- **Recorded per call** in `outputs/raw_runs.jsonl`: `run_id`
  (`model|question_id|seed`), requested model, question_id, seed,
  `model_returned` (the model name returned by the API), raw response
  `text`, `finish_reason`, `prompt_tokens`, `completion_tokens`, and an
  `error` field (null on success). Runs are resumable by run_id: cells
  already present in the file are skipped. Dry-run placeholder rows are
  never written to this file, and the runner refuses a real run if
  placeholders are found in it.
- **API key handling:** the key is read from the `TOGETHER_API_KEY`
  environment variable (or a git-ignored `.env`); where a credential helper is configured
  (CREDENTIAL_HELPER_DIR), it supplies a surrogate key instead.
  It is never printed, logged, or committed.
- **Grading** (`src/errorbars/grade.py`, exactly as implemented): take the
  first number after the last `####` in the response; if there is no
  number after the marker (or no marker), fall back to the last number in
  the text. Normalization strips commas, dollar signs, percent signs, and
  whitespace, and drops a trailing `.0` (via numeric normalization, so
  `42.0` becomes `42`). A response with no number at all is a **parse
  failure** and grades incorrect. Correctness is numeric equality with the
  gold answer. Raw text is always stored, and the parse-failure share per
  model is reported in the analysis.

## 6. Failure policy

- Each call gets up to **3 attempts** with exponential backoff
  (2 s, then 4 s) in `run.py`. (On an authentication rejection the runner
  rebuilds its client before the next attempt.)
- A call that fails all 3 attempts is recorded with
  `finish_reason: "error"`, empty text, and the error message. It grades
  as **incorrect** (empty text has no parseable answer) and is counted as a
  failed call.
- Failed-call counts are reported per model in `results/summary.md`, and
  are not silently dropped or re-run beyond the 3 attempts.

## 7. Coverage study design (Stage 4)

For each model separately, using its 400 × 5 correctness matrix:

- **Reference truth:** the model's full-pool mean accuracy (§3).
- **Designs:** N ∈ {25, 50, 100, 200} questions × K ∈ {1, 2, 3, 5} seeds
  (`config.yaml: analysis.designs_N`, `designs_K`).
- **Subsamples:** B = **2,000** per design. Each subsample draws N
  questions **without replacement** from the 400, then, per drawn
  question, K of its 5 seeds **without replacement**
  (`analyze.coverage_study`).
- For every subsample, every interval method (§8) is computed; coverage is
  the share of subsamples whose interval contains the reference truth.
- **Reporting per method and design:** coverage with a **Monte Carlo 95%
  interval** (Wilson interval for the coverage proportion over the B
  subsamples, `analyze._mc_interval`), the **median interval width**, and
  **N/400** (`pool_fraction`). The default design for headline tables is
  N = 50, K = 5.
- **Finite-pool caveat:** subsamples are drawn from a finite pool of 400,
  so at large N (especially N = 200, pool fraction 0.5) the subsampling
  distribution is narrower than sampling fresh questions from the
  population would be; coverage results at large N are therefore
  conservative in a specific, stated way, and N/400 is always reported
  alongside.
- **Bayesian method exception:** because MCMC is slow, `bayes` is run on
  only the **first 200 of the 2,000 subsamples** per design
  (`config.yaml: analysis.bayes_subsamples`); its Monte Carlo interval is
  computed with B = 200. This is stated openly wherever its coverage is
  reported.
- **Paired-difference study:** for each pair of models (and pre-specified
  in particular for the headline pair), designs N ∈ {25, 50, 100, 200} ×
  K ∈ {1, 5} (`analysis.paired_designs_K`), same subsampling scheme,
  reference = the full-pool paired difference of the two reference
  accuracies' underlying cells (mean of per-cell differences over the
  matched 400 × 5 grid); coverage and median width reported for the three
  paired methods in §8 with Monte Carlo intervals as above.
- Random seeds: the coverage study is seeded per model (seed = 1000 +
  model index in `analyze.main`); each interval computation inside a
  subsample draws a fresh random seed from the study RNG (relevant to the
  bootstrap and Bayes methods).

## 8. Interval methods — exact definitions as implemented

All methods live in `src/errorbars/intervals.py` and operate on an
(N, K) matrix of 0/1 correctness values (NaN = missing cell). Notation:
`x_qk` is the correctness of question q under seed k, `p̂` is the pooled
mean over observed cells, the nominal level is 0.95 throughout, `z` is the
0.975 standard-normal quantile, and `t` is the 0.975 quantile of Student's
t with the stated degrees of freedom. Methods (1)–(5) plus (6) are
computed for every subsample in the coverage study; the reusable
`intervals()` function defaults to methods (1)–(5), with `seed_only`
available on request.

1. **naive — Wilson interval on the pooled proportion.**
   All N·K answers are treated as independent Bernoulli trials. With
   n = number of observed cells and p̂ the pooled proportion, the Wilson
   centre is `(p̂ + z²/(2n)) / (1 + z²/n)` and the half-width is
   `(z / (1 + z²/n)) · sqrt(p̂(1−p̂)/n + z²/(4n²))`.
   The reported `se` is the Wald standard error `sqrt(p̂(1−p̂)/n)`.
   Target (stated in its note): population accuracy, but expected to be
   too narrow when questions differ in difficulty.

2. **question_means — per-question means + t-interval.**
   Each question's mean `m_q` over its observed seeds is computed; the
   estimate is `mean(m_q)` over the N questions; the standard error is
   `sd(m_q, ddof=1) / sqrt(N)`; the interval is
   `estimate ± t(df = N−1) · se`. Targets the primary estimand.

3. **cluster — cluster-robust by question.**
   Estimate: pooled mean `p̂`. Per-question score sums
   `u_q = Σ_k (x_qk − p̂)`; variance of the estimate is
   `(N / (N−1)) · Σ_q u_q² / (total observed cells)²` — i.e. the
   cluster-robust sandwich variance with the small-sample correction
   **N/(N−1)**. The interval is `p̂ ± crit · se` where the critical value
   is **Student's t with df = N−1** (the code uses `_z_or_t(level, N−1)`;
   it is explicitly not the normal value). Targets the primary estimand.

4. **bootstrap — two-stage bootstrap, percentile.**
   2,000 draws (`analysis.bootstrap_draws`; `intervals(bootstrap_draws)`
   default 2000). Each draw: resample N questions **with replacement**;
   for each drawn question, resample its observed seed values **with
   replacement** (as many values as that question has observed); the
   statistic is the pooled mean of the resampled cells. The interval is
   the 2.5%–97.5% percentiles of the 2,000 statistics; the estimate is
   the pooled mean of the original matrix; the reported `se` is the
   standard deviation (ddof=1) of the bootstrap statistics. Targets the
   primary estimand.

5. **bayes — hierarchical Bayesian logistic with question random effect.**
   For each question, `y_q` correct out of `k_q` observed seeds is
   modelled as `y_q ~ Binomial(k_q, logit⁻¹(μ + u_q))` with
   `u_q ~ Normal(0, σ)`. Priors: **μ ~ Normal(0, 1.5)**,
   **σ ~ HalfNormal(1.5)**. Sampled with NumPyro NUTS (500 warmup, 1,000
   posterior draws, code defaults in `_bayes`). The target is the
   population accuracy **E[logit⁻¹(μ + u)]** for `u ~ Normal(0, σ)`,
   computed per posterior draw by 32-node Gauss–Hermite quadrature over
   the standard normal. The interval is the equal-tailed 95% posterior
   interval of that quantity; the estimate is its posterior mean. Run on
   only 200 of the 2,000 coverage subsamples per design (§7). Imported
   lazily; requires the optional `numpyro`/`jax` dependencies.

6. **seed_only — different estimand.**
   Questions are treated as fixed. With `p_q` the per-question mean over
   `k_q` observed seeds, the estimate is `mean(p_q)` and the variance is
   `Σ_q p_q(1−p_q)/k_q / N²` (pure within-question binomial variance);
   the interval uses the **normal** critical value `z`. This interval is
   for accuracy on these questions only.

**Paired methods** (`paired_intervals`, matched on question_id and seed;
only cells observed for both models are used):

- **naive (paired_naive):** two-sample interval treating the models as
  independent: estimate = `mean(A) − mean(B)` over each model's observed
  cells; `se = sqrt(var(A)/n_A + var(B)/n_B)` with sample variances
  (ddof=1); normal critical value.
- **paired_cluster:** per-cell differences `d_qk = a_qk − b_qk` on matched
  cells; estimate = pooled mean of `d`; cluster-robust standard error by
  question exactly as in method (3) applied to the difference matrix
  (including the N/(N−1) correction, N = number of matched questions);
  **t critical value with df = N−1**.
- **paired_bootstrap:** the two-stage bootstrap of method (4) applied to
  the difference matrix `d` (resample questions, then seeds within
  question, statistic = pooled mean difference), percentile interval,
  2,000 draws.

## 9. Decision rule

A method is said to **under-cover** at a design if and only if both hold:

- its measured coverage is **below 0.93**, and
- its Monte Carlo 95% interval **excludes 0.95** (lies entirely below
  0.95).

Any weaker pattern (e.g. coverage 0.94 with an interval containing 0.95)
is reported as-is and not labelled under-coverage.

## 10. Variance components and budget analysis

- **Variance components** (`analyze.variance_components`, one-way ANOVA
  on question means, balanced 400 × 5 matrix): with question means `m_q`
  and grand mean `g`,
  `MS_between = K · Σ_q (m_q − g)² / (N−1)`,
  `MS_within = Σ_qk (x_qk − m_q)² / (N(K−1))`,
  **σ²_w = MS_within** (within-question / seed variance) and
  **σ²_q = max((MS_between − MS_within) / K, 0)** (between-question
  variance). Computed per model on the full pool matrix.
- **Predicted variance** of an accuracy estimate at design (N, K):
  **σ²_q/N + σ²_w/(N·K)**; predicted SE is its square root, and predicted
  interval width is proportional to it.
- **Budget analysis:** fix the total budget at **N·K = 250 calls** and
  compare the splits **(250, 1), (125, 2), (83, 3), (50, 5)**
  (`config.yaml: analysis.budget_splits`; note 83 × 3 = 249 ≈ 250).
  For each split, report the predicted width (from the variance
  components) and the **empirical** median width, obtained from dedicated
  subsamples at exactly those (N, K) splits with B = 2,000, for at least
  the naive and cluster methods. The budget frontier figure (§11) plots
  width against the split so the questions-vs-seeds trade-off is read off
  directly.
- **Predicted vs empirical SE:** for every coverage design (N, K), the
  empirical SE of the estimate across the 2,000 subsamples is recorded
  and plotted against the predicted SE from the variance components.

## 11. Outputs and figures

- `outputs/raw_runs.jsonl` — one row per call (§5).
- `outputs/correctness.parquet` — exactly the columns
  `model, question_id, seed, correct`, one row per cell (6,000 rows),
  built by `analyze.build_correctness` from the raw runs and the GSM8K
  gold answers. Failed calls appear as incorrect; parse failures are
  incorrect with no parsed answer (tracked alongside in the analysis
  table, not in the parquet).
- `results/figures/` — 200 dpi PNGs with a colour-blind-safe palette:
  1. coverage dot plot per method and design, with Monte Carlo intervals
     as error bars and a dashed line at 0.95;
  2. median interval width by method across designs;
  3. predicted vs empirical standard error;
  4. budget frontier: interval width across the N·K = 250 splits.
- `results/summary.md` — must contain: per-model coverage and median
  width at the default design (N = 50, K = 5) for every method; paired
  results; variance components per model; the budget result (best split
  and the widths behind it); failed-call counts per model; parse-failure
  rates per model; every amendment from §13's log, repeated in full; and
  the **total model spend** (main run, plus pilot spend to date, stated
  separately).

## 12. Limitations

- **One benchmark:** GSM8K only, under this one prompt. Results may not
  transfer to other benchmarks, prompts, or sampling settings.
- **Three models:** one model set, chosen for mid-range pilot accuracy on
  GSM8K; coverage behaviour at accuracy near 0 or 1 is out of scope by
  design.
- **Finite reference pool:** "truth" is the mean over 400 questions × 5
  seeds, not the population value; the coverage study inherits the
  finite-pool conservatism described in §7.
- **Output behaviour is part of one model's difficulty:** Muse-Glimmer-30B
  had a 26.2% parse-failure rate in the pilot (21/80 responses with no
  parseable answer, mostly empty or truncated text at the 512-token cap).
  Part of its measured inaccuracy therefore reflects output behaviour
  rather than question difficulty. Parse failures are reported separately
  per model (§11) so this component is visible, and the same caveat
  applies when interpreting its coverage results.
- The Bayesian coverage numbers rest on 200 subsamples per design, so
  their Monte Carlo intervals are wider than the other methods'.

## 13. Implementation status and spec-vs-code notes

Checked against the code at freeze time; nothing here is papered over.

1. `intervals.py` implements all six single-model methods and the three
   paired methods exactly as defined in §8, including the cluster
   correction N/(N−1) with a t critical value (df = N−1), the Bayes
   priors, and the quadrature target. The reusable function matches this
   plan.
2. `analyze.py`'s coverage study, Monte Carlo intervals, and variance
   components exist and match §7 and §10. Its `main()` is **partial**: it
   writes the default-design coverage table, variance components, and one
   coverage figure per model, but it does **not** yet implement the
   paired-difference study, the budget analysis (including the dedicated
   (250, 1) and (83, 3) subsample designs, which are outside the
   §7 N-grid), the empirical-SE recording, or the full figure set of
   §11, and its current figure palette is matplotlib `tab10`, not a
   colour-blind-safe palette. Extending `analyze.py` to this plan is
   Stage 4 work, done after the main run and reviewed at Checkpoint 3;
   any such extension changes code after the freeze and will be recorded
   in the amendment log if it changes any definition in §§7–10 (pure
   additions that implement this plan as written are not amendments).
3. `variance_components` requires a balanced matrix without NaNs;
   `analyze.main` currently passes the matrix through
   `nan_to_num(..., nan=0.0)`. Under the failure policy a failed call is
   a recorded row that grades incorrect (0), so the main-run matrix
   should contain no NaNs; if a cell were missing entirely, this would
   silently treat it as incorrect. Stage 4 must assert the matrix is
   complete (400 × 5 per model, no NaNs) before applying ANOVA, and
   report any missing cells instead of filling them.
4. `freeze.py`'s docstring says to run it "after PLAN.md is approved";
   the staged spec for this project freezes the plan as part of Stage 2,
   before Checkpoint 2, so that what is approved is a fixed, hashed
   document. The freeze is therefore run at Stage 2; if approval at
   Checkpoint 2 changes anything, the plan is amended (§14) and
   re-frozen, with both events recorded.
5. `grade.py` normalization also strips percent signs, beyond the
   commas / dollar signs / trailing `.0` listed in the original spec;
   §5 documents the behaviour as implemented.

## 14. Amendment log

Empty at freeze time. Any later change to this plan is appended here with
the date, the change, the reason, and whether outputs had been seen when
the change was made ("Outputs seen? Yes/No"). Amendments are also repeated
in `results/summary.md`.

| Date | Change | Reason | Outputs seen? |
|---|---|---|---|
| — | — | — | — |
| 2026-11-23 | No change to the plan. Post-freeze exploratory analysis recorded below (finite-pool correction), reported in `REPORT_P01_FPC.md`. | Claude's review of 2026-11-23 (finite-pool gate), approved by Omid. | Yes |

**Post-freeze analysis, 2026-11-23 (not an amendment).** The frozen
coverage study scores intervals against the mean of the same 400×5 pool
the subsamples are drawn from, so frozen coverage overstates
super-population coverage by an amount that grows with N. The analysis
in `REPORT_P01_FPC.md` (code `scripts/fpc.py`) rescores the frozen
replicates with a finite-pool-corrected criterion and adds a split-pool
check; the frozen numbers in `results/coverage_*.csv` are unchanged and
are reported alongside the corrected ones, never replaced. Findings and
the proposed README headline sentence are in the report; the README
itself is not changed by this entry.

2026-11-23: post-freeze check against variance-adaptive regions (scripts/i040.py; I040_FROZEN_RULE.md); decision CONFIRM.

## 15. Breadth arm (addendum, frozen 2026-11-23)

A separate arm, frozen in `FREEZE_BREADTH.md` before its first paid
call; nothing in §§1–14 changes. It asks two things: whether the
interval guidance carries over to pass@k on code generation, and
whether the coverage audit replicates on a fresh GSM8K pool.

- **Code arm:** MBPP+ (EvalPlus v0.3.1, data v0.2.0, 378 problems,
  pinned and hashed in the freeze note), the same three models
  (endpoints re-verified unchanged on 2026-11-23), n = 20 samples at
  temperature 0.8, EvalPlus's instruct prompt quoted verbatim in the
  freeze note, extraction by the EvalPlus sanitizer (a failure
  counts as a fail), pass = all base and plus tests pass.
- **Pilot:** HumanEval+ (v0.1.10), 20 problems × 20 samples × 3
  models, for calibration only; pilot data is never scored into the
  study.
- **Primary cell:** k = 4, N = 50, K = n = 20, so the finite-pool
  correction is exact at √(1 − N/P). Decision read on the cluster-t
  interval over problems (problem bootstrap beside it), per included
  model, with the pre-written HOLDS / UNDER-COVERS / CONSERVATIVE
  rules and arm-outcome wording of the freeze note. Inclusion:
  pool pass@1 ≥ 0.02 and pool pass@4 ≤ 0.98.
- **Exploratory grid:** k ∈ {1, 2, 4, 8}, N ∈ {25, 50, 100, 200},
  K ∈ {5, 10, 20}, k ≤ K, labelled exploratory; the K < n
  correction factor and its components are defined in the freeze
  note, with W_m defined by the exact sum of Milletarì & Kundu
  (2026) Eq. (30) and Monte Carlo kept as the check. Naive
  comparator at k = 1 only (Wilson on the pooled binaries),
  exploratory; no naive comparator at the primary cell.
- **Replication arm:** 400 fresh GSM8K test items drawn with seed
  20261123 from the 919 outside the frozen pool, × 5 seeds × 3
  models at the frozen settings, the frozen pipeline with the exact
  correction inside the fresh pool, read at N = 50, K = 5; the 2-SE
  replication rule and the ≥ 2-of-3 gate for generalised wording
  are in the freeze note. Never trimmed.
- **Budget:** $45 cap for the arm, automatic stop at $36; trim
  order n 20 → 16 (K stays equal to n), then drop pool models with
  pilot pass@1 < 0.02, then n 16 → 12.
- **Pre-registration:** this addendum and the freeze note are
  committed and pushed publicly before the first paid call; the OSF
  registration of the freeze is submitted before the first paid
  call.
