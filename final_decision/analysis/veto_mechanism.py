"""Why does exp169's YOLO vote veto an instance? Whole-instance miss, or fragments? (CPU, existing pools)

exp159's support for a Mask2Former candidate M is the score of the YOLO instance with the best IoU,
if that IoU > 0.5, else 0. A filament that YOLO returns as two or more pieces has no single piece
with IoU > 0.5, so it gets no support even though YOLO saw it. For every exp139 instance that exp169
drops (validation: from veto_check.json; test: from removed_instances.json), per YOLO pool:
  best_iou   best single-instance IoU with M
  frag_cover |M ∩ union(Y_j)| / |M| over YOLO instances Y_j with >= 50% of their own area inside M
             (pieces of this filament, not neighbours that merely touch it)
argv: annotations val_cache v8l_val y11m_val veto_check.json removed_instances.json base_csv
      e040_test_cache v8l_test exp002_test_views_0 exp002_test_views_1 out_json payload_dir
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycocotools.mask as mu

(annotations, val_cache, v8l_val, y11m_val, veto_path, removed_path, base_csv,
 test_cache, v8l_test, views0, views1, out_path, payload) = sys.argv[1:14]
sys.path.insert(0, payload)
from experiments.exp_149_evidence_fusion.src import evidence as E  # noqa: E402
from experiments.exp_159_yolo_votes.src.votes import load_pool  # noqa: E402
from experiments.exp_162_final_submission.src.final import fused_views  # noqa: E402
from shared.utils import paint_panoptic  # noqa: E402

def measures(m_rle, pool_entries):
    if not pool_entries:
        return 0.0, 0.0
    ys = [E.as_rle(p["rle"]) for p in pool_entries]
    iou = np.asarray(mu.iou([m_rle], ys, [0] * len(ys))).ravel()
    area_m = mu.area(m_rle)
    pieces = []
    for y, r in zip(ys, iou):
        if r <= 0:
            continue
        inter = mu.area(mu.merge([m_rle, y], intersect=True))
        if inter >= 0.5 * mu.area(y):
            pieces.append(y)
    cover = 0.0
    if pieces:
        u = mu.merge(pieces, intersect=False)
        cover = float(mu.area(mu.merge([m_rle, u], intersect=True)) / area_m)
    return float(iou.max()), cover

def classify(b8, c8, b11, c11):
    if max(b8, b11) > 0.5:
        return "single-piece support exists"
    if max(c8, c11) >= 0.5:
        return "seen by YOLO in pieces (coverage >= 0.5)"
    if max(c8, c11) > 0:
        return "partly seen (coverage < 0.5)"
    return "not seen by YOLO"

out = {}
# ---------- validation ----------
pools = {"v8l": load_pool(Path(v8l_val)), "y11m": load_pool(Path(y11m_val))}
cache = json.load(open(val_cache))["photographs"]
audit = json.load(open(veto_path))["audit"]
rows = []
for item in audit:
    name = item["photo"]
    cands = [(c["score"], mu.decode(E.as_rle(c["rle"]))) for c in cache[name]["full"] if c["score"] >= 0.85]
    target = None
    for s, m, a in paint_panoptic(cands, min_area=100):
        if a == item["area"] and abs(s - item["score"]) < 1e-9:
            target = mu.encode(np.asfortranarray(m))
    if target is None:
        continue
    b8, c8 = measures(target, pools["v8l"].get(name, []))
    b11, c11 = measures(target, pools["y11m"].get(name, []))
    rows.append(dict(item, v8l_iou=b8, v8l_cover=c8, y11m_iou=b11, y11m_cover=c11, why=classify(b8, c8, b11, c11)))
tab = defaultdict(lambda: [0, 0, 0])
for r in rows:
    tab[r["why"]][0] += 1
    tab[r["why"]][1] += r["tp_records"]
    tab[r["why"]][2] += r["records"]
out["validation"] = {k: dict(instances=v[0], matched_record_pairs=v[1], record_pairs=v[2]) for k, v in tab.items()}
print("VALIDATION dropped by exp169 (instances, matched pairs / pairs):")
for k, v in out["validation"].items():
    print("   ", k, v)

# ---------- test ----------
tpools = {"v8l": load_pool(Path(v8l_test)), "y11m": fused_views([Path(views0), Path(views1)])}
base = defaultdict(list)
for row in csv.DictReader(open(base_csv)):
    base[row["filament_id"].rsplit("_", 1)[0]].append({"size": [2048, 2048], "counts": row["segmentation_rle"].encode()})
removed = json.load(open(removed_path))
trows = []
for r in removed:
    if r["status"] == "kept":
        continue
    target = next((b for b in base[r["stem"]] if int(mu.area(b)) == r["area"]), None)
    if target is None:
        continue
    name = r["stem"] + ".jpeg"
    b8, c8 = measures(target, tpools["v8l"].get(name, []))
    b11, c11 = measures(target, tpools["y11m"].get(name, []))
    trows.append(dict(r, why=classify(b8, c8, b11, c11), large=r["area"] >= 3000))
ttab = defaultdict(lambda: [0, 0])
for r in trows:
    ttab[r["why"]][0] += 1
    ttab[r["why"]][1] += int(r["large"])
out["test"] = {k: dict(instances=v[0], large_ge_3000px=v[1]) for k, v in ttab.items()}
print("TEST dropped by exp169 (instances, of which >= 3000 px):")
for k, v in out["test"].items():
    print("   ", k, v)
json.dump(dict(out, validation_rows=rows, test_rows=trows), open(out_path, "w"), indent=1)
