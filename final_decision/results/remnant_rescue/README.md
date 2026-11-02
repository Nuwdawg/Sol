# exp169 + guard, minus remnants, plus Swin-B rescues: a learning-free +0.0020 (2026-11-02)

**File:** `submission_exp169_remnant_rescue.csv` (1,149 instances, 179 photographs, SHA256 `F9423B74...`).
Code: `experiments/exp_200_test_truth/src/combo.py` (validation) and `combo_test.py` (test).
`reproduce/run.py --stages ... remnant_rescue` rebuilds it.

It is exp169's rule and the deployed guard, with two inference-time changes. Neither is trained on
anything:
1. **Remnant filter.** exp169 paints its instances in descending score, so an instance keeps only
   the pixels higher-ranked ones left. An instance that keeps less than 70% of its own vote mask is
   dropped. Until now only instances under 100 px were dropped.
2. **Swin-B rescue.** A Mask2Former instance scoring in [0.60, 0.68) is also kept when Swin-B's best
   full-view partner (IoU > 0.5) scores 0.95 or more. Without YOLO support exp169's score cannot
   reach 0.68 (11 view scores over 19 slots), so this band holds Mask2Former instances that YOLO did
   not confirm.

## Validation, per record (exp168's two folds, seed 2026)

| rule | PQ | half A | half B |
|---|---|---|---|
| exp169 + guard (submitted, public 0.41) | 0.48893 | 0.49323 | 0.48411 |
| + remnant filter | 0.48990 | 0.49471 | 0.48452 |
| + Swin-B rescue | 0.48998 | 0.49470 | 0.48471 |
| **+ both (this file)** | **0.49095** | **0.49618** | **0.48511** |

- The two gains add up, and the file gains on both halves.
- Any remnant cutoff from 0.7 to 0.9 gives the identical result (0.5-0.6: +0.0008 instead of +0.0010).
- The rescue threshold was picked after seeing the results, so its +0.0010 is optimistic. Under
  exp168's fold-choice protocol it did not survive.

**Expected public:** tuned rules have reached the board at about 0.845 x validation, so
0.49095 x 0.845 = 0.415, right on the line between 0.41 and 0.42. exp169's own public score is
somewhere in [0.405, 0.415). The file prints 0.42 only if exp169 sits in the top fifth or so of
that range: roughly a 30-40% chance, not a sure gain.

## Checks

1. Parity: with neither change, the same test path rebuilds the submitted
   `submission_exp169_guarded.csv` byte for byte (SHA256 `EC00408E...`).
2. Review at native resolution against exp169 + guard (`review_vs_exp169_guarded.jpg`, all
   changes): 1,144 instances identical (median IoU 1.0).
   - **5 added** by the rescue (412-4,192 px): dark filaments on the disk.
   - **6 removed** by the remnant filter (162-3,155 px): mostly the tail of a filament. Another
     instance holds the filament's main body, and the removed one was a lower-ranked, longer version.
3. Trade-off, recorded: the removed tails are filament material. Adding them to the main instance
   instead of dropping them scores 0.48905 against 0.48990. On validation, annotators mostly do not
   draw those faint ends, so the main body alone matches them better.

## Also tried on 2026-11-02, per record, no file (all scored against exp169's 0.48895)

| idea | result | file |
|---|---|---|
| truth rasterised differently on test? | no: every annotation's `area` equals pycocotools' pixel count (300 of 300) | |
| fill holes / keep largest component / redraw as polygons | 0.48888 / 0.48702 / 0.469-0.474 | `shape_post_val.json` |
| paint small instances first | 0.48950; can punch holes in large filaments, not used | `paint_order_val.json` |
| SAM-style stability score (logit +1 vs -1 mask) | best 0.48895 (no weight) | `stability_val.json` |
| decoder-layer agreement instead of the final class probability | 0.4889 / 0.4882 / 0.4855 | `remnants_val.json` |
| add uncovered Swin-B detections that a YOLO model confirms | best +0.0012 in-sample, cross-validated -0.006 | `orphans_val.json` |
| merge fragments another model sees as one filament | -0.003 to -0.006 | `merges_val.json` |
| more rescues: 2-3 of exp184a's three models agree, or Swin-B >= 0.97 in [0.50, 0.60) | no change (0.49095 in every case) | |
