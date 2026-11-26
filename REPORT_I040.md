# I-040 — Variance-adaptive regions vs P01's recommendation

Design and decision rule: `I040_FROZEN_RULE.md` (frozen 2026-11-23 before
any I-040 result existed; not edited since). Execution: `scripts/i040.py`.
Spend: $0. Existing rollouts only (`outputs/correctness.parquet`,
3 models × 400 GSM8K questions × 5 seeds). Nothing pushed or published.
Wall-clock used: ≈ 2 h 35 min of the 4 h box.

**Outcome in one line: VA is not competitive at K ≤ 5 for a single mean.
Decision per the frozen rule: CONFIRM — P01's guidance is unchanged.**

## Decision-rule readings (stated before any results)

Two phrases in the frozen rule needed a numeric reading. Both were fixed
here, before the results tables below were assembled, and applied
mechanically.

1. **"Within Monte Carlo error" (KILL clause).** VA coverage counts as
   ≥ bootstrap coverage "within Monte Carlo error" when the two
   coverages differ by less than **2 Monte Carlo standard errors of
   their difference**, with

   SE_diff = sqrt( c_VA (1 − c_VA) / B_VA + c_boot (1 − c_boot) / B_boot ),

   B_VA = 1000 (I-040 replicates), B_boot = 2000 (the original study's
   replicates, reused from `results/coverage_*.csv`). The tolerance
   2 × SE_diff is computed per model from that model's two coverages;
   the values used are: Qwen **0.008**, Glimmer **0.015**, Inkling
   **0.012**.

2. **"Materially below" (validation gate, criterion ii).** Read as:
   the plug-in bootstrap's coverage is at least 0.03 (3 coverage points)
   below VA's coverage in the same gate cell.

## Deviations and implementation choices (logged before interpretation)

The frozen rule allows departures only as logged deviations. The VA
implementation follows Ye (2026, arXiv:2611.21266) §4.3 / Algorithm 1 —
tail counts T_k, Bernstein radius r(V) = sqrt(2LV/m) + L/(3m) with
L = log(2|K|/α), K = K_geo, cell envelopes for P_k, V_k and h, tangent
cuts for the radius, the two sure constraints — with these choices,
each forced by LP runtime at B = 1000 replicates and each in the
conservative direction (wider VA intervals, never narrower):

- **Own implementation, not Ye's released code.** VA ran on our own
  implementation of Ye's Algorithm 1, written from the paper; Ye's
  released `valoc` code was not run. That is a deviation from the
  frozen rule's first preference ("Ye's released code if it can be
  used as-is"). Its consequence for how the gate below may be read:
  the gate checks that this implementation reproduces Ye's
  *qualitative behaviour* — VA valid but wide, the plug-in bootstrap
  failing on the same law — not his published widths. No width in
  this report is a reproduction of a number from Ye's paper or
  software.

- **Envelope tolerance η = 1e-2** instead of Ye's default 1e-3 for the
  partition refinement (P_k and h variation per cell). Coarser cells
  widen the envelopes and hence the VA intervals.
- **Tangent set** starts at V0 = 1e-4 (Ye's software goes to V0 ≈ 1e-14;
  those tangents have slopes in the thousands and made the LP solver
  orders of magnitude slower). Up to 2 cutting-plane rounds add tangents
  at the realised V̄_k(w). Any finite tangent set gives a valid outer
  approximation (Ye's Theorem 4.3), so this costs width, not validity.
- **Bernstein radius only**; Ye's optional Bennett cuts are not imposed
  (the paper presents Bernstein as the primary radius; Bennett would
  only narrow the intervals).
- **Endpoints padded outward by 1e-6** in place of Ye's dual
  certification step (Algorithm 1, step 5).
- The LP is solved **in dual form** (same optimum; checked equal to the
  primal solve on test instances) because the primal's thousands of
  near-identical cell columns made the solver unusably slow.
- **Validation gate replicates:** planned 250, aggregated at **B = 210**
  (the rule requires ≥ 200). A first gate attempt was lost at 200/250
  replicates when a service restart wiped the un-checkpointed run; the
  reported run checkpoints every replicate to
  `results/i040_gate_progress.csv` and was stopped once past 200 to
  protect the arms' share of the time-box. Gate cell: law **F1**,
  δ = **0.01** (the rule fixes the law family, m = 250, n = 32 and the
  θ_δ functional; the δ choice is stated here).
- **E3 (pass@k) was not run.** Under the time-box, the pass@k VA
  targets tripled the per-replicate LP cost of E1 while carrying no
  decision weight in the frozen rule (E3 is descriptive only), so they
  were dropped (`I040_SKIP_E3=1`) to protect the decision-bearing
  cells. `results/i040_e3.csv` is header-only. Consequence for exact
  reproducibility: E1's replicate streams consume the RNG with one
  bootstrap draw per replicate; rerun E1 with the same flag to
  reproduce `results/i040_e1.csv` bit-for-bit.
- E2 execution order was changed mid-run (primary (50, 5) cells for all
  models before the secondary (100, 5) cells) for the same reason;
  the subsample stream, seeds and per-cell B are unchanged. VA is
  computed on the **first 1000** of the study's own 2000 subsamples
  per cell, as the rule specifies; the existing methods' numbers are
  reused from `results/coverage_*.csv` at B = 2000, not recomputed.
- Monte Carlo intervals on coverage are **Clopper–Pearson**, per the
  frozen rule (the repo's earlier study used Wilson intervals in
  `analyze._mc_interval`; I-040 follows the rule).

## Validation gate (before any real-rollout number)

Ye's law F1 = 0.3δ₀ + 0.7·Beta(0.3, 1.5), one F_m drawn and held fixed
(θ_0.01(F_m) = 0.5205), m = 250 problems, n = 32 draws per problem,
B = 210 replicates (`results/i040_gate.csv`):

| method | coverage | MC interval | median width |
|---|---|---|---|
| VA | 1.000 | [0.983, 1.000] | 0.607 |
| plug-in percentile bootstrap | 0.000 | [0.000, 0.017] | 0.120 |

Criterion (i): VA coverage 1.000 ≥ 0.94 ✓. Criterion (ii): bootstrap
coverage 0.000, a full 1.000 below VA's — materially below ✓.
**Gate: PASS.** The implementation reproduces Ye's published
qualitative behaviour: VA is valid but wide; the plug-in bootstrap is
narrow and, for this functional, worthless. (The bootstrap's failure
here is a bias failure, not noise: at δ = 0.01 the plug-in functional
of the redrawn counts concentrates far from θ_δ(F_m).)

## Results

### E1 — fixed-benchmark mean accuracy

Truth: the pool's mean per-question success rate (Qwen 0.4945,
Glimmer 0.6480, Inkling 0.8615). Replicates redraw counts
c_i ~ Bin(n, p̂_i) from the reference law. B = 1000 per cell
(`results/i040_e1.csv`). Coverage [Clopper–Pearson MC interval],
median width.

**n = 5 (primary):**

| model | VA | bootstrap | pooled Wilson | normal |
|---|---|---|---|---|
| Qwen | 1.000 [0.996, 1.000], 0.100 | 1.000, 0.077 | 0.996, 0.044 | 1.000, 0.078 |
| Glimmer | 1.000 [0.996, 1.000], 0.063 | 1.000, 0.087 | 1.000, 0.042 | 1.000, 0.087 |
| Inkling | 1.000 [0.996, 1.000], 0.047 | 1.000, 0.063 | 1.000, 0.030 | 1.000, 0.063 |

**n = 2 (secondary):** every method covers 0.998–1.000 in every model.
VA widths 0.113 / 0.077 / 0.059 (Qwen / Glimmer / Inkling) against
pooled Wilson 0.069 / 0.066 / 0.048 — the same pattern as n = 5: VA
narrower than the bootstrap and the normal interval for Glimmer and
Inkling, widest for Qwen, and wider than pooled Wilson everywhere.
Full numbers in the CSV.

On this estimand everything covers: at m = 400 the redraw design is
benign, and the comparison is decided by width alone. VA pays a
1.2–2.3× width premium over the pooled Wilson interval it would have
to beat.

Two qualifications keep that reading honest. First, every comparator
in the set targets a different estimand: the question-level
bootstrap, the normal interval and the pooled Wilson interval are
all built for sampling questions (or answers) afresh, not for a
fixed benchmark. On the fixed benchmark every method over-covers
(0.996–1.000) — the pooled Wilson interval included, because its
variance carries the between-question spread, which a fixed
benchmark does not have. No method in the set is a true
fixed-benchmark interval (one built on a within-question standard
error), so E1 shows VA losing on width to a method that is itself
conservative here. Second, VA is not the widest interval in the set:
at n = 5 it is narrower than both the question-level bootstrap and
the normal interval for Glimmer (0.063 vs 0.087) and Inkling (0.047
vs 0.063); it is widest only for Qwen. What it never beats, in any
model, is the pooled Wilson interval.

### E2 — super-population mean (P01's original protocol; VA off-label)

VA run on the study's own subsample stream, first 1000 of 2000
subsamples (`results/i040_e2.csv`); cluster bootstrap and cluster-t
reused from `results/coverage_*.csv` at B = 2000.

One caution on every coverage number in this section: they are
measured against the frozen pool's mean, so they sit on the
pool-truth scale and share the finite-pool inflation quantified in
`REPORT_P01_FPC.md` — subsample and truth draw on the same 400
questions, which flatters coverage by an amount that grows with N.
The numbers stay descriptive; the decision below reads the width
condition and the coverage ordering, not the levels.

**N = 50, K = 5 (primary):**

| model | VA (off-label) | cluster bootstrap | cluster-t |
|---|---|---|---|
| Qwen | 1.000 [0.996, 1.000], 0.321 | 0.970, 0.216 | 0.962, 0.208 |
| Glimmer | 0.962 [0.948, 0.973], 0.242 | 0.962, 0.244 | 0.957, 0.249 |
| Inkling | 0.985 [0.975, 0.992], 0.192 | 0.951, 0.176 | 0.942, 0.181 |

**N = 100, K = 5 (secondary):** VA coverage 0.999 / 0.950 / 0.970
(Qwen / Glimmer / Inkling), widths 0.224 / 0.156 / 0.124, against
cluster bootstrap 0.986 / 0.968 / 0.969 at widths 0.154 / 0.172 /
0.126.

Descriptively, VA used off-label covers the super-population target
about as well as the cluster methods — but it is **wider than the
cluster bootstrap in all three primary cells** (and in two of three
secondary cells), so it dominates nothing.

### E3 — pass@k, fixed-benchmark

Not run (see deviations). No decision weight in the frozen rule.

## Decision (applied mechanically from the frozen rule)

**KILL test** (E2 primary, per model: VA coverage ≥ bootstrap coverage
within MC error AND VA median width ≤ 0.90 × bootstrap's):

| model | coverage condition | width condition |
|---|---|---|
| Qwen | 1.000 ≥ 0.970 − 0.008 ✓ | 0.321 ≤ 0.194 ✗ |
| Glimmer | 0.962 ≥ 0.962 − 0.015 ✓ | 0.242 ≤ 0.220 ✗ |
| Inkling | 0.985 ≥ 0.951 − 0.012 ✓ | 0.192 ≤ 0.158 ✗ |

KILL triggers in **0 of 3** models.

**REVISE test** (E1 primary, per model: VA's coverage MC interval
contains or exceeds 0.95 AND VA ≥ 10% narrower than every comparator
whose coverage also contains/exceeds 0.95): VA's coverage condition
holds in all 3 models (interval [0.996, 1.000]), but the width
condition fails in all 3 — the pooled Wilson interval qualifies on
coverage everywhere and is 33–56% *narrower* than VA. REVISE triggers
in **0 of 3** models.

Neither clause triggers → **CONFIRM**. P01's recommendation stands as
written: report question-level cluster-t / cluster-bootstrap intervals
for the super-population mean; for a model's accuracy beyond the
benchmark's questions (the super-population mean), never pool into
one Wilson interval. The scoping is part of the guidance, not a
footnote to it: E1 shows that for the fixed questions alone, the
pooled interval over-covers rather than under-covers — the warning
against pooling is a claim about the super-population target, and
only there.

The decision does not depend on either numeric reading fixed at the
top of this report. The "materially below" threshold concerns only
the gate, which passed by a margin of 1.000 against 0.000; the KILL
clause's Monte Carlo tolerance moves only its coverage condition,
which holds under any plausible reading. The clause fails on width
in every model, and width carries no tolerance — change either
reading and CONFIRM is unchanged.

Framing, stated plainly: this is **not** a stress test the
recommendation survived against the odds. It records that **VA is not
competitive at K ≤ 5 for a single mean** — it buys its validity with a
width premium (1.2–2.3× on E1; wider than the cluster bootstrap on
every E2 primary cell) that no clause of the frozen rule rewards. That
is consistent with Ye's own results, where VA's width premium persists
at far larger n; a KILL was close to impossible by construction, and a
REVISE would have required VA to out-narrow a pooled interval at the
one sample size where pooling is cheapest. The one genuinely new
observation is descriptive only: used off-label on E2, VA's coverage
of the super-population mean is respectable (0.95–1.00, on the
pool-truth scale — see the caution in the E2 section) — at a width
the cluster bootstrap beats everywhere, so it changes no guidance.

## Precision check (post-decision): did the runtime shortcuts decide KILL?

The KILL clause failed on width alone, and every runtime shortcut
listed in the deviations (η = 1e-2, the truncated tangent set, the
Bernstein radius without Bennett cuts, the 1e-6 endpoint padding in
place of dual certification) widens VA. So the shortcuts push toward
CONFIRM, and by how much had to be measured before CONFIRM could be
reported as robust. The margins: on the E2 primary cells VA would
have to be at least 40% narrower to trigger KILL for Qwen (out of
reach), at least 9% for Glimmer (0.242 → ≤ 0.220) and at least 18%
for Inkling (0.192 → ≤ 0.158); KILL needs two of three models, so it
turns on Glimmer and Inkling together.

**Method.** VA was re-run at Ye's defaults — η = 1e-3 partition
(3,979 cells, against 511), the full tangent set V0 = t² down to
t = 1e-7, the Bennett radius in place of Bernstein-only, and dual
certification (cut rounds until the LP optimum satisfies the exact
nonlinear constraints) in place of endpoint padding — on the first
100 subsamples of the two E2 primary cells, the identical draws in
the identical order the E2 arm used. Identity of the draws was
verified, not assumed: the shortcut solver on the regenerated counts
reproduces the E2 arm's recorded endpoints exactly for every
replicate checked. Per-replicate results:
`results/i040_precision_progress.csv`; summary:
`results/i040_precision.csv`.

| model | replicates | shortcut median width | Ye-defaults median width | narrower |
|---|---|---|---|---|
| Glimmer | 100 | 0.241 | 0.228 | 5.4% |
| Inkling | 100 | 0.189 | 0.176 | 7.1% |

Certification: all 100 Glimmer replicates certified; 97 of 100
Inkling replicates certified — the remaining 3 did not certify
within 60 cut rounds and kept the 1e-6 outward pad, which widens
those intervals, so the measured narrowing is if anything
understated.

**Branch taken.** Glimmer is 5.4% narrower at Ye's defaults, below
the 9% its KILL cell would need; Inkling is 7.1%, below its 18%.
Either one alone settles it: at Ye's defaults KILL cannot trigger,
so the pre-specified full re-run of the two cells was not run. Even
if the same percentages held across all 1,000 replicates, the
full-cell medians would be ≈ 0.229 (Glimmer) and ≈ 0.178 (Inkling),
still above the KILL width thresholds of 0.220 and 0.158. **CONFIRM
stands as robust: the runtime shortcuts did not decide KILL.** The
widths in the E2 table above remain the shortcut versions; both
versions of the decisive comparison are the ones in this section.

## Completeness

- Gate: complete at B = 210 (rule: ≥ 200), PASS.
- E1: complete — 3 models × n ∈ {5, 2} × 4 methods at B = 1000.
- E2: complete — VA at 3 models × N ∈ {50, 100} at B = 1000;
  reference methods reused at B = 2000.
- E3: **not run** (deviation, above). The report is otherwise complete;
  the decision does not depend on E3.
- Precision check: complete — VA at Ye's defaults on the first 100
  subsamples of the Glimmer and Inkling E2 primary cells (Glimmer
  5.4% narrower, Inkling 7.1%); the branch that would have required
  a full re-run at defaults was not taken, and CONFIRM stands as
  robust.
- Per-replicate endpoints are kept in `results/i040_*_progress.csv`.
- Repo tests: 40/40 pass after the run.
