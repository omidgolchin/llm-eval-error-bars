# I-040 — Frozen decision rule (written BEFORE any I-040 run)

Frozen: 2026-11-23, on Omid's instruction to proceed, under the project
council's conditions (record: council-2026-11-23.md). This rule may not be
edited after any I-040 result exists. Any departure from it is logged as a
deviation in the report, with the reason, before results are interpreted.

## What is being stress-tested

P01's recommendation: for repeated-run LLM evaluations, report question-level
intervals (cluster-robust t on question means / cluster bootstrap) for the
mean accuracy of a *new-question* (super-population) target; never pool all
N×K answers into one Wilson interval. I-040 asks whether Ye's (2026)
variance-adaptive (VA) regions — built for the *fixed-benchmark* estimand —
change that recommendation when both are run on P01's real rollouts
(3 models × 400 GSM8K questions × 5 seeds, frozen pool, $0, no new calls).

## Estimands — kept separate, never conflated

- **E1 — fixed-benchmark mean accuracy.** θ = mean of the per-question
  success probabilities p_i over the fixed 400 questions. Reference law F̂:
  p̂_i = c_i/5 from the full pool. Replicates: for each question draw
  c_i* ~ Bin(n, p̂_i), n = 5 (primary), n = 2 (secondary). Coverage is
  measured against θ(F̂).
- **E2 — super-population mean accuracy.** P01's original estimand and
  protocol exactly (`coverage_study`: subsample N questions without
  replacement, K of the 5 runs; truth = full-pool mean). Primary cell
  (N=50, K=5); secondary (N=100, K=5). Existing methods' numbers are reused
  from `results/coverage_*.csv` (B=2000), not recomputed. VA is computed on
  the same subsamples and is labelled **off-label** here (Ye makes no
  super-population claim for VA without a concentration layer).
- **E3 — pass@k, fixed-benchmark only, descriptive.** θ_k = mean_i
  [1 − (1 − p_i)^k] under F̂, k ∈ {2, 4}, n = 5. No decision weight.

## Methods

E1: VA region for the mean functional (Ye 2026 LP construction); plug-in
percentile bootstrap over questions (resample questions' counts, percentile
interval for the mean of p̂); pooled Wilson over all 400n answers; normal
interval on question means treating p̂_i as exact. E2: VA (off-label) added
to the existing five methods under the identical protocol. E3: VA and
plug-in bootstrap.

VA implementation: Ye's released code if it can be used as-is; otherwise an
implementation from the paper (arXiv:2611.21266, main text). Either way it
must pass the validation gate below before any real-rollout number is
produced.

## Validation gate (before real rollouts)

Reproduce Ye's qualitative simulation behaviour on one of his laws
(F1–F5, m=250, n=32, θ_δ functional), ≥200 replicates:
(i) VA empirical coverage ≥ 0.94 at nominal 0.95, and
(ii) the plug-in percentile bootstrap's coverage is materially below VA's
(the direction of Ye's 41/135 finding).
If the gate fails: STOP. Report "implementation unverified". No real-rollout
results are produced or reported.

## Replicates and Monte Carlo error

B = 1000 replicates per cell (E1, E3, and the VA additions in E2). If VA's
runtime forces B = 500 for VA only, that is a logged deviation, not a rule
change. Every coverage number is reported with its Clopper–Pearson Monte
Carlo interval (the repo's existing convention).

## Decision rule — applied mechanically

Let coverage and median width be per model, per cell, at nominal 95%.

- **KILL** (P01's recommendation is withdrawn pending rewrite): on the E2
  primary cell, VA dominates the cluster bootstrap — VA coverage ≥ bootstrap
  coverage (within Monte Carlo error) AND VA median width ≤ 0.90 × bootstrap
  median width — in at least 2 of the 3 models.
- **REVISE** (recommendation stands for the super-population target but gains
  a written fixed-benchmark caveat): on the E1 primary cell, VA's coverage
  Monte Carlo interval contains or exceeds 0.95 AND VA's median width is
  ≤ 0.90 × the median width of every E1 comparator whose coverage also
  contains or exceeds 0.95, in at least 2 of the 3 models.
- **CONFIRM** (recommendation stands unchanged; I-040 reported as a
  robustness pass): neither KILL nor REVISE triggers.
- Anything else — a trigger in exactly 1 model, or triggers pointing both
  ways — is **inconclusive**: reported as such, no guidance change, no
  post-hoc threshold adjustment.

E3 (pass@k) carries no decision weight in any case.

## Time-box and spend

Execution is capped at 4 hours wall-clock from launch, with the validation
gate as a checkpoint. If the box expires mid-arm, stop and report what
exists, labelled partial. Spend: $0. Existing rollouts only. Nothing is
pushed or published; results land as `results/i040_*.csv` and
`REPORT_I040.md` in this repo, committed locally.
