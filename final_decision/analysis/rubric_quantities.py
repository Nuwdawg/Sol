"""The rubric's quantitative items for the final picks, on validation, per annotation record.

The host's self-evaluation notebook shows, besides PQ:
  - the distribution of IoU over every overlapping ground-truth / prediction pair (IoU > 0),
  - the same for Dice (2 IoU / (1 + IoU) for a pair),
  - one-to-many and many-to-one counts: for each ground-truth filament, how many predictions
    overlap it, and for each prediction, how many ground-truth filaments it overlaps (IoU > 0).
This computes them, with the notebook's definitions, for three files rebuilt on the 106 validation
photographs (180 records): exp139 single view (second pick), exp169 + guard, and the remnant +
rescue rule (first pick).

    python final_decision/analysis/rubric_quantities.py ANNOTATIONS VAL_CACHE SHIFT_CACHE INGREDIENTS OUT.json SWINB...
"""
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.exp_033_inference.src.sweep import records_for  # noqa: E402
from experiments.exp_149_evidence_fusion.src import evidence as E  # noqa: E402
from experiments.exp_157_shift_views.src.views import merged_cache  # noqa: E402
from experiments.exp_168_joint_tuning.src.joint import scores  # noqa: E402
from experiments.exp_199_swinb_votes.src.guard import guard  # noqa: E402
from experiments.exp_199_swinb_votes.src.votes import merge_shards  # noqa: E402
from experiments.exp_200_test_truth.src.hypotheses import as_bytes, match, total  # noqa: E402
from experiments.exp_200_test_truth.src.paint_order import ORDERS  # noqa: E402
from experiments.exp_200_test_truth.src.remnants import paint_fraction  # noqa: E402
from experiments.exp_200_test_truth.src.swinb_gate import support  # noqa: E402


def quantities(finals: dict, passes: dict) -> dict:
    ious, gt_deg, pred_deg, rows = [], Counter(), Counter(), []
    for name, records in passes.items():
        preds = finals[name]
        for truth in records:
            rows.append(match(preds, truth))
            if not preds or not truth:
                gt_deg.update([0] * len(truth))
                pred_deg.update([0] * len(preds))
                continue
            m = np.asarray(mu.iou(preds, truth, [0] * len(truth))).reshape(len(preds), len(truth))
            ious.extend(m[m > 0].tolist())
            gt_deg.update((m > 0).sum(axis=0).tolist())
            pred_deg.update((m > 0).sum(axis=1).tolist())
    ious = np.asarray(ious)
    dice = 2 * ious / (1 + ious)
    pq = total(rows)
    q = lambda a: {str(p): round(float(np.quantile(a, p)), 4) for p in (0.1, 0.25, 0.5, 0.75, 0.9)}
    gt_total, pred_total = sum(gt_deg.values()), sum(pred_deg.values())
    return dict(pq=pq["pq"], sq=pq["sq"], rq=pq["rq"], tp=pq["tp"], fp=pq["fp"], fn=pq["fn"],
                overlapping_pairs=int(ious.size), mean_iou=float(ious.mean()), mean_dice=float(dice.mean()),
                iou_quantiles=q(ious), dice_quantiles=q(dice),
                share_pairs_iou_above_half=float((ious > 0.5).mean()),
                gt_overlapped_by=dict(sorted(gt_deg.items())), pred_overlaps=dict(sorted(pred_deg.items())),
                one_to_many=int(sum(v for k, v in gt_deg.items() if k >= 2)),
                many_to_one=int(sum(v for k, v in pred_deg.items() if k >= 2)),
                gt_total=gt_total, pred_total=pred_total)


def main() -> None:
    annotations, cache_path, shift_path, ingredients_path, out = map(Path, sys.argv[1:6])
    swinb_paths = [Path(p) for p in sys.argv[6:]]
    records, _ = records_for(annotations)
    names = sorted(records)
    passes = {n: [[as_bytes(t) for t in truth] for _, truth in sorted(records[n])] for n in names}
    prep = pickle.loads(ingredients_path.read_bytes())
    cache = merged_cache(cache_path, shift_path)
    swinb, _ = merge_shards(swinb_paths, names)
    files = {"exp139_single": {}, "exp169_guard": {}, "remnant_rescue": {}}
    for n in names:
        base = [as_bytes(r) for _, r in E.paint([(c["score"], E.Crop(c["rle"])) for c in cache[n]["full"]
                                                 if c["score"] >= 0.85], 100)]
        s = scores(prep[n], 4.0, 0.3)
        s_b = support(cache[n]["full"], [c for c in swinb[n]["full"] if c["p"] >= 0.5])
        plain = [as_bytes(r) for _, r in E.paint([(float(v), c) for v, c in zip(s, prep[n]["crops"]) if v >= 0.68], 100)]
        kept = [(float(v), c) for v, b, c in zip(s, s_b, prep[n]["crops"]) if v >= 0.68 or (0.60 <= v < 0.68 and b >= 0.95)]
        combo = [as_bytes(r) for r in paint_fraction(kept, ORDERS["score_desc"], 0.7)]
        files["exp139_single"][n] = base
        files["exp169_guard"][n] = plain + guard(base, plain)
        files["remnant_rescue"][n] = combo + guard(base, combo)
    report = {}
    for tag, finals in files.items():
        report[tag] = quantities(finals, passes)
        r = report[tag]
        print(tag, json.dumps(dict(pq=round(r["pq"], 5), mean_iou=round(r["mean_iou"], 4), mean_dice=round(r["mean_dice"], 4),
                                   pairs=r["overlapping_pairs"], one_to_many=r["one_to_many"], many_to_one=r["many_to_one"],
                                   gt_total=r["gt_total"], pred_total=r["pred_total"])), flush=True)
    out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
