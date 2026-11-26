# Finite-pool correction, exact-factor re-score

Post-freeze exploratory analysis, 2026-11-23. Companion to
REPORT_P01_FPC.md. The frozen numbers in `results/coverage_*.csv` and
the step-1 corrected numbers in `results/fpc_coverage.csv` are
unchanged; this re-score is reported alongside them, never instead of
them. Code: `scripts/fpc.py` step 3 (`step3` / `combine3`). Data:
`results/fpc_exact_coverage.csv`. Cost: $0, existing rollouts only.

## Why this exists

Step 1 of REPORT_P01_FPC scored a replicate as covered when
|estimate − pool truth| ≤ half-width × √(1 − N/400). That factor is
exact for K = 5, where the subsample uses every seed the pool has, and
approximate for K < 5: it does not model that at K < 5 the drawn seeds
are themselves shared with the pool, so it shrinks the half-width too
much. This re-score replaces the flat factor with the exact one from
the variance decomposition of (estimate − pool truth) against the
estimator's super-population variance:

  factor² = [ (1 − N/P) · S² + Var_B(K) ] / (S² − W_5 + W_K)

- S² = variance across the 400 questions of the 5-seed question means;
- W_m = mean over questions of p_i (1 − p_i) / m, the binomial
  within-question variance of a question mean over m seeds
  (p_i = question i's pool rate);
- Var_B(K) = mean over questions of the variance of a K-of-5 subset
  mean across subsets of the question's seeds (hypergeometric:
  p_i (1 − p_i) / K · (5 − K) / 4; zero at K = 5). Computed by exact
  enumeration of all subsets and checked against the closed form;
  the two agree to under 1e-9.

At K = 5 the factor reduces to √(1 − N/400) exactly. The run asserts
this, asserts the frozen coverage and median width still reproduce in
every cell, asserts the step-1 corrected coverage reproduces in every
cell, and asserts every K = 5 exact-factor coverage equals the step-1
value. The p_i plug-in makes the K < 5 factors approximate; they are
labelled so wherever they are used.

## Result

- **K = 5: identical.** All 60 cells (3 models × 4 N × 5 methods)
  match step 1 exactly. The corrected cluster coverage at K = 5
  stands: Qwen 0.949 / 0.947 / 0.948 / 0.946, Glimmer 0.948 / 0.945 /
  0.939 / 0.956, Inkling 0.896 / 0.931 / 0.938 / 0.939 at
  N = 25 / 50 / 100 / 200 — and with it every reading REPORT_P01_FPC
  drew from the K = 5 cells, including Inkling under-covering at
  every N and the naive interval's corrected 0.60–0.77.
- **K < 5: step 1 over-shrank, as expected.** Every exact-factor
  corrected coverage is at or above its step-1 value (no cell moves
  down). Mean change: +0.014 at K = 1, +0.012 at K = 2, +0.006 at
  K = 3, 0 at K = 5. The largest single move is Qwen at N = 200,
  K = 1 (cluster and question-means 0.881 → 0.967): at K = 1 the
  exact factor is 0.899–1.032 depending on N and model, against the
  flat 0.707–0.968, because a one-seed estimate shares almost none
  of its seed noise with the five-seed pool truth.
- **No reading changes.** Under the exact factor the cluster
  interval's corrected coverage is 0.947–0.968 (Qwen) and
  0.937–0.960 (Glimmer) across all designs — about nominal
  throughout. Inkling remains the weakest model at every K, and the
  seed-only interval remains far below nominal everywhere (at most
  0.639 after correction). The naive interval at K = 1 moves up
  (Qwen 0.930–0.967), consistent with pooling being legitimate when
  each question contributes one run; its K = 5 under-coverage is
  untouched.

## Reproduce

```
.venv/bin/python scripts/fpc.py step3 --model qwen     # also glimmer, inkling
.venv/bin/python scripts/fpc.py combine3               # -> results/fpc_exact_coverage.csv
```

Step 3 refuses to write output if any frozen cell, any step-1
corrected cell, or the K = 5 identity fails to reproduce.
