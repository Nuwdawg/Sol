"""Rebuild exp169 on validation (parity-gated) and audit what its YOLO veto removes.

exp169 scores each Mask2Former candidate as s^0.3 * mean^0.7, mean = (sum of 11 view scores + 4 * sum
of 2 YOLO supports) / 19. With no YOLO support, mean <= 11/19 and the score cannot reach the 0.68
threshold even at s = 1, so any candidate neither YOLO pool matches at IoU > 0.5 is removed.

This script (CPU, existing caches only):
  1. replays exp169's validation rule with the exp169 kernel's own code and requires
     PQ 0.48894859995182627 with TP/FP/FN 892/304/433;
  2. finds exp139-deployed validation instances that exp169 drops, and scores each against every
     annotation record of its photograph: TP for how many records, by area and exp139 score;
  3. evaluates a 'protect' variant: exp169's output plus exp139 instances dropped by the veto whose
     area and exp139 score exceed a cut, painted after exp169's instances (no overlap).
argv: payload_dir annotations val_cache shift_val_cache yolov8l_val_pool yolo11m_val_pool out_json
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pycocotools.mask as mu

payload, annotations, val_cache, val_shift, v8l, y11m, out_path = sys.argv[1:8]
sys.path.insert(0, payload)
from experiments.exp_033_inference.src.sweep import records_for, pq_rle  # noqa: E402
from experiments.exp_149_evidence_fusion.src import evidence as E  # noqa: E402
from experiments.exp_157_shift_views.src.views import merged_cache  # noqa: E402
from experiments.exp_159_yolo_votes.src.votes import load_pool  # noqa: E402
from experiments.exp_168_joint_tuning.src.joint import prepared, scores  # noqa: E402
from shared.utils import aggregate_pq, paint_panoptic  # noqa: E402

records, _ = records_for(annotations)
cache = merged_cache(Path(val_cache), Path(val_shift))
pools = {"yolo11m": load_pool(Path(y11m)), "yolov8l": load_pool(Path(v8l))}
tags = sorted(pools)
names = sorted(records)

pool = {}
for n in names:
    p = prepared(cache[n], [pools[t].get(n, []) for t in tags], ("d4", "shift"))
    pool[n] = list(zip(map(float, scores(p, 4.0, 0.3)), p["crops"]))
rows, per = E.score_rule(pool, records, (0.68,), min_area=100)
print("REPLAY", json.dumps(rows[0]))
assert abs(rows[0]["pq"] - 0.48894859995182627) < 1e-9 and (rows[0]["tp"], rows[0]["fp"], rows[0]["fn"]) == (892, 304, 433)

def emitted(entries, conf, min_area=100):
    """exp169's own painter (evidence.paint) on its crops; returns (score, dense mask)."""
    kept = E.paint([(s, c) for s, c in entries if s >= conf], min_area)
    return [(s, mu.decode(r)) for s, r in kept]

def enc(m):
    r = mu.encode(np.asfortranarray(m.astype(np.uint8)))
    return r

def ious(a, b):
    if not a or not b:
        return np.zeros((len(a), len(b)))
    return np.asarray(mu.iou(a, b, [0] * len(b))).reshape(len(a), len(b))

audit = []
protect_rows = {}
CUTS = [(None, None), (3000, 0.85), (3000, 0.90), (2000, 0.85), (2000, 0.90), (1000, 0.90), (0, 0.95)]
for cut in CUTS:
    protect_rows[cut] = []
for n in names:
    new = emitted(pool[n], 0.68)
    new_m = [m for _, m in new]
    new_r = [enc(m) for m in new_m]
    base_c = [(c["score"], mu.decode(E.as_rle(c["rle"]))) for c in cache[n]["full"] if c["score"] >= 0.85]
    base = [(s, m) for s, m, _ in paint_panoptic(base_c, min_area=100)]
    base_r = [enc(m) for _, m in base]
    m = ious(base_r, new_r)
    dropped = [i for i in range(len(base)) if (m[i].max() if m.shape[1] else 0.0) < 0.5]
    truths = [t for _, t in records[n]]
    for i in dropped:
        hits = [float(ious([base_r[i]], t)[0].max()) if t else 0.0 for t in truths]
        audit.append(dict(photo=n, area=int(base[i][1].sum()), score=float(base[i][0]),
                          records=len(truths), tp_records=int(sum(h > 0.5 for h in hits)),
                          overlap_new=float(m[i].max()) if m.shape[1] else 0.0))
    # protection variants: add dropped exp139 instances above the cut, painted after exp169's
    for cut in CUTS:
        extra = []
        if cut[0] is not None:
            claimed = np.zeros((2048, 2048), np.uint8)
            for mm in new_m:
                claimed |= mm.astype(np.uint8)
            for i in dropped:
                s, mm = base[i]
                if mm.sum() >= cut[0] and s >= cut[1]:
                    rem = mm.astype(np.uint8) & (1 - claimed)
                    if rem.sum() >= 100:
                        claimed |= rem
                        extra.append(enc(rem))
        pred = new_r + extra
        for t in truths:
            protect_rows[cut].append(pq_rle(pred, t))

def summarise(sel):
    if not sel:
        return {}
    tp_any = sum(1 for a in sel if a["tp_records"] > 0)
    rec = sum(a["records"] for a in sel); tp_rec = sum(a["tp_records"] for a in sel)
    return dict(n=len(sel), matched_any_record=tp_any, record_pairs=rec, matched_record_pairs=tp_rec,
                share_matched_pairs=round(tp_rec / rec, 3), median_score=round(float(np.median([a["score"] for a in sel])), 3))

report = {"replay": rows[0], "dropped_total": summarise(audit),
          "dropped_area>=3000": summarise([a for a in audit if a["area"] >= 3000]),
          "dropped_area_1000_3000": summarise([a for a in audit if 1000 <= a["area"] < 3000]),
          "dropped_area<1000": summarise([a for a in audit if a["area"] < 1000]),
          "dropped_score>=0.95": summarise([a for a in audit if a["score"] >= 0.95]),
          "dropped_reshaped(overlap>0)": summarise([a for a in audit if a["overlap_new"] > 0]),
          "dropped_deleted(no overlap)": summarise([a for a in audit if a["overlap_new"] == 0]),
          "protect": {}}
for cut, rws in protect_rows.items():
    a = aggregate_pq(rws)
    report["protect"][str(cut)] = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in a.items()}
for k, v in report.items():
    print(k, json.dumps(v))
json.dump(dict(report, audit=audit), open(out_path, "w"), indent=1)
