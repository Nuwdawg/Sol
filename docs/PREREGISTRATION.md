# Pre-registration: what the next GPU run must show

Written before any pod is created, at the user's requirement: *"the agent should not
spend your next GPU run until it has written down the hypothesis, expected
measurable gain, and the exact result that would justify abandoning or continuing
it."*

Everything below is fixed in advance. Results are compared against these numbers,
not against numbers chosen afterwards.

## Baseline, frozen

| quantity | value |
|---|---|
| validation PQ | 0.4558638687292126 |
| TP / FP / FN | 860 / 397 / 465 |
| SQ / RQ | 0.6843 / 0.6662 |
| proposal recall at IoU 0.5 (pool, no rules) | **1016 / 1325 = 76.7%** |
| public offset | 0.067 - 0.078 below validation |
| public score to beat | 0.39 |

Validation 0.472 prints 0.40. Validation ~0.53 prints 0.46.

## The primary metric is not PQ

PQ conflates the three stages. Every arm below reports, in this order:

1. **proposal recall at IoU 0.5** - the share of truths with *any* candidate above
   0.5, before confidence, erosion, area floor or painting;
2. **the exp129 bucket histogram** - >=0.5 / 0.3-0.5 / 0.1-0.3 / <0.1;
3. **instance-level matched IoU at native 2048 resolution**;
4. PQ last, as the summary.

Rule from exp094, which cost a whole experiment line: **IoU is measured at native
resolution, never at crop resolution against a native baseline.** Nearest-neighbour
resampling to 256 inflated mean IoU by +0.012 with no model involved.

## H1 - boundary refinement

**Hypothesis.** A model that re-segments an already-localized region at higher
effective resolution converts truths whose best proposal sits at IoU 0.3-0.5 into
matches. Filaments have a 2.2 px median half-width at 2048, so a crop upsampled 3x
gives roughly 7 px, which is where a boundary becomes representable at all.

**Population, corrected 2026-09-16 21:34 UTC.** **165** validation truths (12.5%),
not the 194 first claimed. exp129 bucketed truths by their best *raw* proposal, but
the pipeline works on masks eroded 1 px, and exp131/exp132 showed erosion is itself
a boundary correction that lifts proposal recall from 76.7% to 79.4%. Truths whose
raw proposal is 0.3-0.5 but clear 0.5 after erosion were never boundary failures.

**Oracle ceiling.** +0.0507 PQ (0.4559 -> 0.5066), all 165 converted at IoU 0.65.

**Expected realistic gain.** Converting 40% of the 165 gives PQ ~0.4769, i.e.
**+0.021**, which prints 0.40-0.41.

**This is the decisive point: H1 succeeding does not reach 0.46.** Its entire oracle
is +0.051, and 0.46 publicly needs validation near 0.53. H1 is therefore demoted to
a follow-on, not the next GPU spend.

**Continue if:** on the 194, measured at native resolution, the refiner raises mean
IoU by >= +0.05 over the unrefined proposal AND converts >= 30% across the 0.5
threshold.

**Abandon if:** native-resolution mean IoU gain < +0.05, or crossings up minus
crossings down <= 0. exp092 scored +0.001 by this standard and would have been
killed on the first criterion.

## H2 - emission rule

**Hypothesis.** 156 truths have a proposal at IoU >= 0.5 that the emission rule
discards. exp130 attributes each to the confidence floor, the 1 px erosion, the
300 px area floor, or paint-order. If erosion or the area floor dominates, this is a
rule change worth ~+0.05 with no training at all.

**Oracle ceiling.** +0.0498 PQ.

**Continue if:** any rule setting raises PQ by >= +0.010 on validation without
lowering proposal recall.

**ABANDONED 2026-09-16 21:14 UTC, on the criterion below.** exp130 attributed the
203 unmatched good proposals:

| cause | count | share |
|---|---|---|
| **below the 0.35 confidence floor** | **168** | **82.8%** |
| dropped by the 300 px area floor | 13 | 6.4% |
| destroyed by the 1 px erosion | 12 | 5.9% |
| pixels lost to a higher-scoring mask | 10 | 4.9% |

A size-conditional rule addresses at most the 25 killed by erosion and the area
floor, not 156. My thin-filament suspicion was specific, testable and **false**: the
median truth area of these misses is 932 px, comfortably above the 300 px floor.

"Below the confidence floor" means the detector found these filaments and scored
them under 0.35. Admitting them is the sub-threshold verifier problem of exp118-127,
where the oracle ceiling was +0.021 and every deployable rule returned about +0.002.
H2 is therefore not a free +0.05; it is the same wall reached from the other side.

Count note: exp129 reported 156 truths whose best **raw** proposal clears 0.5, while
exp130 counts 203 unmatched in the final emission, which also includes truths lost
to another annotation pass's painting. The two are different measurements and 156
must not be quoted for 203.

*Original criterion, retained for the record:* abandon if the best setting gains
< +0.010 **and** a size-conditional rule also gains < +0.010.

The distinction matters and I nearly wrote a bad kill criterion here. exp062 swept
**global** rules - one confidence, one erosion, one area floor applied to every
mask. If the 156 losses concentrate in small or thin instances, then a conditional
rule (no erosion below some area, a lower area floor for high-elongation candidates,
or erosion scaled to the local half-width) is a hypothesis exp062 never tested. A
global sweep finding its own optimum does not falsify a conditional rule, so the
conditional form must be tested before this arm is abandoned.

Filaments have a 2.2 px median half-width. A 1 px erosion removes roughly half of a
thin body's width, so the suspicion that our own rule destroys the thinnest true
positives is specific and testable, not a fishing expedition.

*This arm needs no GPU and is tested first.*

## H3 - detector retrain, ablated  [PROMOTED to the next GPU spend]

**Why this and not the refiner.** The corrected decomposition leaves two large
buckets, and they are the same kind of problem:

| bucket | count | oracle gain | nature |
|---|---|---|---|
| lost to the 0.35 confidence floor | 172 | +0.054 | the detector found these and ranked them under 0.35 - a **calibration** failure |
| boundary failure at 0.3-0.5 eroded | 165 | +0.051 | the detector's mask is wrong - a **mask quality** failure |
| both | 337 (25% of all truths) | **+0.0986 -> PQ 0.5545** | both are detector quality |

A refiner addresses the second bucket only, and partially. A retrain is the only
lever that touches both, because a better-trained detector both scores real
filaments higher and draws them better. Validation 0.5545 is the 0.46 region
publicly; +0.021 from a refiner is not.

The honest caveat: I have **no evidence yet** that preprocessing or dedup improves
either bucket. That is exactly what the thresholds below test, and arm A exists so
that a null result is interpretable rather than ambiguous.

The user's requirement: evaluate each change separately, never stacked.

| arm | change | purpose |
|---|---|---|
| A | raw images, all annotator passes | reproduce the current detector |
| B | + preprocessing | limb-flattened, histogram-matched, off-disk zeroed |
| C | + label dedup | one annotator pass per training photograph |

**Measured on this release:** 974 training image_ids cover 601 unique photographs
(1.62x duplication); 215 of 249 multi-pass photographs have annotators disagreeing
on instance count, one labelled 8, 14 and 21 times.

**Continue B if:** proposal recall at IoU 0.5 rises by >= 2 points over arm A
(76.7% -> 78.7%).

**Continue C if:** proposal recall rises by >= 2 points over arm B, or the
0.3-0.5 bucket shrinks by >= 20 truths.

**Abandon the whole retrain if:** arm A fails to reproduce the current detector's
proposal recall within 2 points, because then the harness is measuring something
other than the change under test.

**Cost.** A100 80GB PCIe, community, **$1.19/h**. Arm A is the control and must run;
three arms at roughly 2 h each is ~$7, against the $9.95 precedent from exp097.

## What would falsify the whole direction

If arms A, B and C all land within 2 points of 76.7% proposal recall, and H1's
refiner fails its native-resolution criterion, then the remaining error is neither
localization nor boundary under these architectures, and the next step is a
different architecture (attention, properly trained and evaluated) rather than more
variants of this one.

Mask2Former is **not** currently evidence either way: exp046/exp053 built the
infrastructure but ran on CPU and never produced a working model, and the one
submission from that line scored 0.20. That is an untested direction, not a failed
one.
