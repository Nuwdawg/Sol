# Competition host guidance: prediction quality and scientific reasoning

**User instruction, received 2026-09-13:** keep this somewhere important and let
it inform decisions. Source is the discussion excerpt pasted by the user; no
discussion URL was supplied or independently verified.

The excerpt identifies Azim Ahmadzadeh as the competition host, replying to
Rahul S.'s claim that the challenge is about scoring conventions and a loose,
multiply annotated consensus. The host calls the observation partially correct
but rejects that characterization of the task.

## What the host says

- Annotation quality varies. Filaments are fluid physical objects observed from
  about 150 million km away; uncertain boundaries are part of the real problem.
- A single scalar cannot adequately describe true model performance.
- According to the host's description of the rubric, the judging committee will
  inspect individual predictions and the logic of the algorithm.
- Participants should focus on prediction quality rather than score alone.

Key excerpt supplied by the user:

> Our judging committee will look at individual predictions, as well as the logic of the algorithm.

> So, again, we encourage all participants to focus on the quality of their predictions instead of their score.

The pasted reply also says "Anything beyond 3.5 PQ is of great interest to us."
That is preserved verbatim. Its scale or possible typo is unverified; do not
silently reinterpret it as 0.35 or use it to change the user's >0.46 goal.

## How this changes our decisions

1. Keep both objectives visible: the user's verified public score >0.46 and
   scientifically credible, useful filament predictions. Meeting a score alone
   does not establish a better segmentation algorithm.
2. Before promoting a model or postprocessing change, compare representative
   native-resolution overlays with the existing best. Include difficult thin
   filaments and crowded regions, and inspect instances the change adds/removes.
   Record misses, false positives, fragmentation/merging and boundary distortion.
   These are visual judgments, not extra ground-truth labels for the test set.
3. Explain why an intervention should improve predictions. Thresholds, mask NMS,
   TTA and ensembling remain legitimate experiments; do not justify them solely
   by a public-score bump or knowingly trade away coherent structures without
   documenting the cost.
4. Preserve per-image examples, checkpoint/configuration provenance and measured
   tradeoffs so a reviewer can assess algorithmic reasoning and limitations.
5. Treat annotation disagreement as uncertainty to measure, not an established
   performance ceiling. The host's reply neither proves a ceiling nor replaces
   the official metric with a consensus mask. Continue canonical photograph-level
   splits and individual-annotation-record PQ; never merge annotators for scoring.
6. Reject access to public-source test annotations as a route to a higher score.
   Compare predictions with the supplied observations and legitimate validation
   annotations, without inventing or acquiring test labels.

## Immediate application

The exp062 D4 probe improved public PQ from0.38 to0.39, but that alone does not
prove its added/removed masks are physically better. Review those changes. The
exp063 NMS, weight-blend, soft-D4 and Mask2 continuation probes remain authorized;
review their saved predictions before treating any score as a promotion decision.
Do not terminate a live experiment solely because this adds a review step.
