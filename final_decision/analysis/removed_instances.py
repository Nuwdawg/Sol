"""Which exp139 test instances does exp169 drop, and how confident was exp139 about them?

Inputs: exp139 deployed test CSV (parity_deployed.csv), exp169 test CSV (submission_joint.csv),
exp151a epoch-40 test candidate cache (full view, with scores). Test images and labels are not used.
For each exp139 instance: best IoU with any exp169 instance of the same photograph; it is
'kept' at IoU >= 0.5, 'reshaped' if some exp169 instance overlaps it below 0.5, 'deleted' if none
overlaps. Its exp139 score is the score of the full-view candidate it was painted from.
"""
import csv
import json
import sys
from collections import Counter, defaultdict

import numpy as np
import pycocotools.mask as mu

def load(path):
    by = defaultdict(list)
    for row in csv.DictReader(open(path)):
        stem = row["filament_id"].rsplit("_", 1)[0]
        by[stem].append({"size": [2048, 2048], "counts": row["segmentation_rle"].encode()})
    return by

base, new = load(sys.argv[1]), load(sys.argv[2])
cache = json.load(open(sys.argv[3]))
cache = cache.get("photographs", cache)

def ious(a, b):
    if not a or not b:
        return np.zeros((len(a), len(b)))
    return np.asarray(mu.iou(a, b, [0] * len(b))).reshape(len(a), len(b))

rows = []
for stem, insts in base.items():
    key = stem + ".jpeg" if stem + ".jpeg" in cache else stem
    entry = cache[key]
    full = entry["full"] if isinstance(entry, dict) else entry
    cands = [c for c in full if c["score"] >= 0.85]
    crles = [c["rle"] for c in cands]
    for c in crles:
        if isinstance(c.get("counts"), str):
            c["counts"] = c["counts"].encode()
    m_new = ious(insts, new.get(stem, []))
    m_c = ious(insts, crles)
    for i, r in enumerate(insts):
        best_new = float(m_new[i].max()) if m_new.shape[1] else 0.0
        status = "kept" if best_new >= 0.5 else ("reshaped" if best_new > 0 else "deleted")
        # the painted mask is a subset of its candidate: take the candidate covering it best
        if m_c.shape[1]:
            inter = [float(mu.area(mu.merge([r, c], intersect=True))) for c in crles]
            j = int(np.argmax(inter))
            score = cands[j]["score"]
        else:
            score = float("nan")
        rows.append({"stem": stem, "status": status, "score": score, "area": int(mu.area(r)), "best_iou_new": best_new})

c = Counter(r["status"] for r in rows)
print("BASE INSTANCES", len(rows), dict(c))
for status in ("deleted", "reshaped"):
    sel = [r for r in rows if r["status"] == status]
    s = np.array([r["score"] for r in sel]); a = np.array([r["area"] for r in sel])
    print(status, "n", len(sel), "score>=0.95:", int((s >= 0.95).sum()), "score>=0.97:", int((s >= 0.97).sum()),
          "median score", round(float(np.median(s)), 3) if len(s) else None,
          "area>=3000:", int((a >= 3000).sum()), "median area", int(np.median(a)) if len(a) else None)
kept = np.array([r["score"] for r in rows if r["status"] == "kept"])
print("kept median score", round(float(np.median(kept)), 3))
top = sorted([r for r in rows if r["status"] == "deleted"], key=lambda r: -r["area"])[:15]
for r in top:
    print("  deleted", r["stem"], "area", r["area"], "exp139 score", round(r["score"], 3))
json.dump(rows, open(sys.argv[4], "w"), indent=0)
