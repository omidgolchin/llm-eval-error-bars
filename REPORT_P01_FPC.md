# Finite-pool correction for the P01 coverage study

Post-freeze exploratory analysis, 2026-11-23. Directed by Claude's review
of the P01 council package and approved by Omid. Recorded in PLAN.md §14
as a post-freeze analysis, not an amendment. The frozen numbers in
`results/coverage_*.csv` are unchanged and are never replaced by the
corrected ones; both are reported side by side. Code: `scripts/fpc.py`.
Data: `results/fpc_coverage.csv`, `results/fpc_splitpool.csv`,
`results/fpc_splitpool_split.csv`. Cost: $0, existing rollouts only.

## Why this exists

The frozen coverage study measures coverage against the mean of the same
400×5 pool the subsamples are drawn from (N of the 400 questions, without
replacement). Subsample and truth share questions, so a correctly
calibrated interval covers the pool truth more often than its nominal
rate — increasingly so as N grows. The frozen coverage therefore
overstates super-population coverage, by an amount that grows with N.
This analysis asks what the frozen replicates say once that sharing is
accounted for.

## Method

**Step 1 — corrected scoring of the frozen replicates.** The frozen
subsampling protocol (`src/errorbars/analyze.py`, `_iter_designs`, same
seeds) was rerun replicate by replicate, and every interval was scored
twice: the frozen rule (pool truth inside the interval) and the corrected
rule — a replicate counts as covered when
|estimate − pool truth| ≤ half-width × √(1 − N/400). The correction is
exact in its logic for K = 5 (all five pool seeds observed, so the only
sharing is the question overlap) and approximate for K < 5 (the drawn
seeds are also shared with the pool, which the factor does not model).
Before any corrected number was accepted, the frozen coverage and median
width had to reproduce `results/coverage_*.csv` exactly: they did, for
all 240 fast-method cells (5 methods × 16 designs × 3 models). The
Bayesian method could not be rescored — see Deviations.

**Step 2 — split-pool check.** The 400 questions were split into two
halves of 200 with one fixed random split (seed 20261123, declared before
running; the split itself is in `results/fpc_splitpool_split.csv`).
Subsamples (N ∈ {25, 50}, K = 5, B = 2000, the fast methods except
seed_only) were drawn from one half, and the truth was the other half's
full 200×5 mean. Unlike step 1, the truth here is not shared with the
subsample at all — but it is itself a 200-question mean, so it carries
its own sampling error (SE ≈ 0.021–0.030 depending on model), and the
realised difference between the two half-means enters the result
directly. It is a robustness check on step 1, not a replacement for it.

## 1. Step 1 results

Corrected coverage of the cluster interval at K = 5 (nominal 0.95;
Monte Carlo 95% intervals in brackets):

| model | N=25 | N=50 | N=100 | N=200 |
|---|---|---|---|---|
| Qwen | 0.949 [.938, .958] | 0.947 [.936, .956] | 0.948 [.937, .957] | 0.946 [.935, .955] |
| Glimmer | 0.948 [.937, .957] | 0.945 [.934, .954] | 0.939 [.928, .949] | 0.956 [.946, .964] |
| Inkling | 0.896 [.882, .909] | 0.931 [.919, .941] | 0.938 [.926, .947] | 0.939 [.927, .948] |

The bootstrap tracks the cluster interval closely: Qwen 0.954–0.961,
Glimmer 0.934–0.959 (slightly below nominal at N = 25), Inkling
0.883–0.942.

Reading:

- **Qwen and Glimmer.** Once the finite pool is accounted for, the
  cluster interval covers at about the nominal level at every N. The
  frozen table's rise toward 0.99 at N = 200 was the pool, not the
  method. Glimmer at N = 100 (0.939) is marginally low.
- **Inkling.** The cluster and bootstrap intervals under-cover at every
  N studied, worst at small N (0.896 at N = 25, 0.931 at N = 50). Every
  Monte Carlo interval excludes 0.95. The small-N under-coverage the
  README reports for Inkling is real, and against a super-population it
  is larger than the frozen table shows.
- **Naive pooled Wilson.** Its under-coverage is also understated by the
  frozen table, and increasingly with N. Corrected coverage at K = 5 is
  roughly flat in N — Qwen 0.75–0.77, Glimmer 0.64–0.68, Inkling
  0.60–0.65 — where the frozen table suggested it recovered to 0.80–0.90
  at N = 200. The headline direction (naive under-covers, badly) is
  safe; its size at large N was understated.

Full step-1 tables (frozen → corrected, every method, every design;
Monte Carlo intervals for every cell are in `results/fpc_coverage.csv`):

### Qwen-2-1.5B-Instruct (reference accuracy 0.4945)
K = 1:
| method | N=25 | N=50 | N=100 | N=200 |
|---|---|---|---|---|
| naive | 0.960 → 0.887 | 0.968 → 0.929 | 0.966 → 0.913 | 0.978 → 0.880 |
| question_means | 0.960 → 0.960 | 0.968 → 0.948 | 0.966 → 0.933 | 0.978 → 0.880 |
| cluster | 0.960 → 0.960 | 0.968 → 0.948 | 0.966 → 0.933 | 0.978 → 0.880 |
| bootstrap | 0.954 → 0.893 | 0.966 → 0.939 | 0.962 → 0.928 | 0.978 → 0.884 |
| seed_only | 0.000 → 0.000 | 0.000 → 0.000 | 0.000 → 0.000 | 0.000 → 0.000 |

K = 2:
| method | N=25 | N=50 | N=100 | N=200 |
|---|---|---|---|---|
| naive | 0.908 → 0.883 | 0.915 → 0.881 | 0.931 → 0.875 | 0.965 → 0.839 |
| question_means | 0.954 → 0.947 | 0.958 → 0.940 | 0.973 → 0.948 | 0.987 → 0.923 |
| cluster | 0.954 → 0.947 | 0.958 → 0.940 | 0.973 → 0.948 | 0.987 → 0.923 |
| bootstrap | 0.968 → 0.948 | 0.975 → 0.956 | 0.982 → 0.962 | 0.993 → 0.945 |
| seed_only | 0.628 → 0.619 | 0.623 → 0.584 | 0.661 → 0.599 | 0.738 → 0.582 |

K = 3:
| method | N=25 | N=50 | N=100 | N=200 |
|---|---|---|---|---|
| naive | 0.876 → 0.822 | 0.875 → 0.845 | 0.888 → 0.829 | 0.948 → 0.842 |
| question_means | 0.959 → 0.952 | 0.963 → 0.951 | 0.970 → 0.933 | 0.994 → 0.940 |
| cluster | 0.959 → 0.952 | 0.963 → 0.951 | 0.970 → 0.933 | 0.994 → 0.940 |
| bootstrap | 0.969 → 0.960 | 0.975 → 0.964 | 0.980 → 0.960 | 0.996 → 0.961 |
| seed_only | 0.644 → 0.633 | 0.653 → 0.624 | 0.682 → 0.618 | 0.777 → 0.590 |

K = 5:
| method | N=25 | N=50 | N=100 | N=200 |
|---|---|---|---|---|
| naive | 0.783 → 0.762 | 0.796 → 0.750 | 0.829 → 0.768 | 0.903 → 0.764 |
| question_means | 0.955 → 0.949 | 0.962 → 0.947 | 0.976 → 0.948 | 0.993 → 0.946 |
| cluster | 0.955 → 0.949 | 0.962 → 0.947 | 0.976 → 0.948 | 0.993 → 0.946 |
| bootstrap | 0.962 → 0.954 | 0.970 → 0.955 | 0.986 → 0.961 | 0.996 → 0.961 |
| seed_only | 0.604 → 0.588 | 0.608 → 0.581 | 0.658 → 0.586 | 0.746 → 0.581 |

### Muse-Glimmer-30B (reference accuracy 0.6480)
K = 1:
| method | N=25 | N=50 | N=100 | N=200 |
|---|---|---|---|---|
| naive | 0.940 → 0.894 | 0.971 → 0.918 | 0.974 → 0.943 | 0.991 → 0.920 |
| question_means | 0.956 → 0.940 | 0.956 → 0.956 | 0.974 → 0.943 | 0.993 → 0.934 |
| cluster | 0.956 → 0.940 | 0.956 → 0.956 | 0.974 → 0.943 | 0.993 → 0.934 |
| bootstrap | 0.940 → 0.928 | 0.956 → 0.949 | 0.971 → 0.942 | 0.989 → 0.926 |
| seed_only | 0.000 → 0.000 | 0.000 → 0.000 | 0.000 → 0.000 | 0.000 → 0.000 |

K = 2:
| method | N=25 | N=50 | N=100 | N=200 |
|---|---|---|---|---|
| naive | 0.885 → 0.806 | 0.882 → 0.838 | 0.913 → 0.867 | 0.953 → 0.862 |
| question_means | 0.950 → 0.950 | 0.963 → 0.949 | 0.980 → 0.956 | 0.993 → 0.944 |
| cluster | 0.950 → 0.950 | 0.963 → 0.949 | 0.980 → 0.956 | 0.993 → 0.944 |
| bootstrap | 0.953 → 0.929 | 0.961 → 0.948 | 0.982 → 0.957 | 0.995 → 0.944 |
| seed_only | 0.319 → 0.319 | 0.359 → 0.347 | 0.406 → 0.367 | 0.471 → 0.345 |

K = 3:
| method | N=25 | N=50 | N=100 | N=200 |
|---|---|---|---|---|
| naive | 0.776 → 0.744 | 0.810 → 0.768 | 0.841 → 0.767 | 0.907 → 0.749 |
| question_means | 0.944 → 0.937 | 0.958 → 0.947 | 0.979 → 0.949 | 0.996 → 0.947 |
| cluster | 0.944 → 0.937 | 0.958 → 0.947 | 0.979 → 0.949 | 0.996 → 0.947 |
| bootstrap | 0.944 → 0.925 | 0.961 → 0.945 | 0.983 → 0.949 | 0.997 → 0.952 |
| seed_only | 0.357 → 0.352 | 0.360 → 0.332 | 0.404 → 0.346 | 0.480 → 0.347 |

K = 5:
| method | N=25 | N=50 | N=100 | N=200 |
|---|---|---|---|---|
| naive | 0.664 → 0.638 | 0.689 → 0.670 | 0.712 → 0.649 | 0.829 → 0.682 |
| question_means | 0.952 → 0.948 | 0.957 → 0.945 | 0.964 → 0.939 | 0.993 → 0.956 |
| cluster | 0.952 → 0.948 | 0.957 → 0.945 | 0.964 → 0.939 | 0.993 → 0.956 |
| bootstrap | 0.954 → 0.933 | 0.963 → 0.944 | 0.969 → 0.941 | 0.995 → 0.959 |
| seed_only | 0.295 → 0.287 | 0.334 → 0.315 | 0.349 → 0.299 | 0.433 → 0.314 |

### Inkling (reference accuracy 0.8615)
K = 1:
| method | N=25 | N=50 | N=100 | N=200 |
|---|---|---|---|---|
| naive | 0.929 → 0.864 | 0.949 → 0.927 | 0.972 → 0.955 | 0.989 → 0.943 |
| question_means | 0.874 → 0.874 | 0.936 → 0.927 | 0.965 → 0.955 | 0.989 → 0.943 |
| cluster | 0.874 → 0.874 | 0.936 → 0.927 | 0.965 → 0.955 | 0.989 → 0.943 |
| bootstrap | 0.865 → 0.865 | 0.960 → 0.928 | 0.974 → 0.915 | 0.987 → 0.938 |
| seed_only | 0.000 → 0.000 | 0.000 → 0.000 | 0.000 → 0.000 | 0.000 → 0.000 |

K = 2:
| method | N=25 | N=50 | N=100 | N=200 |
|---|---|---|---|---|
| naive | 0.840 → 0.817 | 0.857 → 0.837 | 0.902 → 0.821 | 0.948 → 0.840 |
| question_means | 0.914 → 0.912 | 0.949 → 0.927 | 0.964 → 0.936 | 0.987 → 0.941 |
| cluster | 0.914 → 0.912 | 0.949 → 0.927 | 0.964 → 0.936 | 0.987 → 0.941 |
| bootstrap | 0.911 → 0.858 | 0.958 → 0.925 | 0.971 → 0.936 | 0.990 → 0.943 |
| seed_only | 0.255 → 0.255 | 0.364 → 0.314 | 0.371 → 0.312 | 0.468 → 0.334 |

K = 3:
| method | N=25 | N=50 | N=100 | N=200 |
|---|---|---|---|---|
| naive | 0.804 → 0.789 | 0.808 → 0.744 | 0.825 → 0.738 | 0.906 → 0.757 |
| question_means | 0.930 → 0.899 | 0.930 → 0.921 | 0.961 → 0.930 | 0.989 → 0.946 |
| cluster | 0.930 → 0.899 | 0.930 → 0.921 | 0.961 → 0.930 | 0.989 → 0.946 |
| bootstrap | 0.930 → 0.896 | 0.948 → 0.917 | 0.968 → 0.930 | 0.989 → 0.949 |
| seed_only | 0.298 → 0.298 | 0.332 → 0.315 | 0.365 → 0.308 | 0.470 → 0.341 |

K = 5:
| method | N=25 | N=50 | N=100 | N=200 |
|---|---|---|---|---|
| naive | 0.661 → 0.600 | 0.704 → 0.645 | 0.706 → 0.625 | 0.799 → 0.636 |
| question_means | 0.906 → 0.896 | 0.943 → 0.930 | 0.962 → 0.938 | 0.992 → 0.939 |
| cluster | 0.906 → 0.896 | 0.943 → 0.930 | 0.962 → 0.938 | 0.992 → 0.939 |
| bootstrap | 0.912 → 0.883 | 0.951 → 0.924 | 0.969 → 0.935 | 0.992 → 0.942 |
| seed_only | 0.262 → 0.259 | 0.288 → 0.272 | 0.328 → 0.292 | 0.395 → 0.282 |

## 2. Split-pool results

Coverage against the other half's mean (K = 5, B = 2000; Monte Carlo
95% intervals in brackets):

| model | method | N=25 | N=50 |
|---|---|---|---|
| Qwen | naive | 0.773 [.754, .791] | 0.804 [.786, .821] |
| Qwen | question_means / cluster | 0.956 [.946, .964] | 0.964 [.955, .971] |
| Qwen | bootstrap | 0.963 [.953, .970] | 0.971 [.963, .978] |
| Glimmer | naive | 0.339 [.319, .360] | 0.135 [.121, .151] |
| Glimmer | question_means / cluster | 0.823 [.805, .839] | 0.600 [.578, .621] |
| Glimmer | bootstrap | 0.776 [.757, .794] | 0.569 [.547, .591] |
| Inkling | naive | 0.622 [.601, .643] | 0.531 [.509, .552] |
| Inkling | question_means / cluster | 0.964 [.955, .971] | 0.962 [.952, .969] |
| Inkling | bootstrap | 0.952 [.941, .960] | 0.948 [.937, .956] |

The truth-half means were 0.501 (Qwen), 0.705 (Glimmer) and 0.879
(Inkling); the draw-half means were 0.488, 0.591 and 0.844.

Reading:

- **Qwen and Inkling** behave as step 1 predicts: with a truth that
  shares nothing with the subsample, the cluster interval covers at
  about the nominal level (0.956–0.964), and Inkling's step-1
  under-coverage does not reappear against this particular truth —
  consistent with step 1 being the sharper instrument (it averages over
  2,000 overlaps of the full pool rather than hinging on one split).
- **Glimmer is dominated by the split itself.** The one declared split
  put substantially easier questions in the truth half: the half-means
  differ by 0.114, about twice the sampling standard error of an
  N = 50 subsample mean (≈0.06) and close to the median interval
  half-width (0.127). An interval then covers the truth only when the
  subsample happens to land high, so the low numbers (0.82 / 0.60)
  measure the realised half difference, not miscalibration. This is the known weakness of a single split,
  stated here rather than repaired by re-rolling the split: the seed
  was declared in advance and stands. For Glimmer the split-pool check
  is therefore uninformative about calibration, in either direction.
- Naive under-covers in every split-pool cell, as everywhere else.

## 3. Proposed README headline sentence

Adjusted from Claude's draft to what steps 1–2 show (the one change:
Inkling under-covers at every N studied, not only at N ≤ 50):

> Cluster-robust and bootstrap intervals cover at about the nominal
> level for two of three models once the finite pool is accounted for;
> for the third (Inkling) they under-cover at every sample size studied,
> worst at N ≤ 50.

(Not yet applied to README.md; the README change waits for Claude's
review, per his instruction.)

## 4. A note on N = 200

Wherever the frozen results cite coverage at N = 200 (cluster and
bootstrap ≈ 0.99), those values are the finite pool at work — at
N = 200 a subsample is half the pool, so the truth it is scored against
is half-built from the subsample's own questions. They are not evidence
that the methods are over-cautious. On the corrected scale the same
cells sit at 0.94–0.96.

## 5. Deviations from the brief

1. **Bayes was not recomputed.** `results/bayes_cache/*.csv` hold only
   per-design aggregates (coverage and median width over 200
   subsamples), not per-replicate interval bounds, so the corrected
   criterion cannot be applied to stored output. Regenerating the
   bounds means rerunning the frozen MCMC arm — 200 fits × 16 designs
   × 3 models = 9,600 fits. Measured cost on this machine: ≈20 s per
   fit at N = 100 after a ≈60 s first fit (JAX compile), so the arm
   would take on the order of two days — far outside the 3-hour
   time-box alongside steps 1–2. No fresh replicates were substituted;
   the Bayes rows of the frozen table stand without a corrected
   counterpart.
2. **Step 2 uses new, declared seeds** (split seed 20261123; subsampling
   streams 3100 + model index). Step 2 has no frozen counterpart, so
   there were no frozen replicates to reuse; the seeds are recorded
   here and in `scripts/fpc.py` so the check is reproducible.
3. **Step 2's first launch crashed before producing any output** (a
   scripting error in building the split table). It was fixed and
   rerun with the design unchanged; the declared split seed was not
   altered.

## Reproduce

```
.venv/bin/python scripts/fpc.py step1 --model qwen     # also glimmer, inkling
.venv/bin/python scripts/fpc.py combine                # -> results/fpc_coverage.csv
.venv/bin/python scripts/fpc.py step2                  # -> results/fpc_splitpool.csv
```

Step 1 refuses to write output if any fast-method cell fails to
reproduce the frozen coverage and median width.
