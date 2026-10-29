"""How well does a committee of two human annotators predict a third? (labels only, CPU)

On validation photographs with three annotation records, hold each annotator out in turn
and build a prediction from the other two:
  single       one other annotator's masks (annotator-vs-annotator)
  agreed       instances both others drew (matched at IoU > 0.5), first annotator's mask
  agreed_and   same instances, pixel-wise AND of the two masks
  agreed_or    same instances, pixel-wise OR of the two masks
  union        every instance either drew (first annotator's mask where both drew it)
Scored with the repo's exp033 matching (IoU > 0.5) and pooled like the competition.
Compares with exp139 epoch 40 on exactly the same records (deployed rule, from the exp147 cache).
argv: annotations, exp147 val cache
"""
import json
import sys

import numpy as np
import pycocotools.mask as mu

sys.path.insert(0, "/home/user/Sol")
from experiments.exp_033_inference.src.sweep import evaluate, records_for, pq_rle  # noqa: E402
from shared.utils import aggregate_pq  # noqa: E402

records, _ = records_for(sys.argv[1])
raw = json.load(open(sys.argv[2]))["photographs"]
cache = {n: [{"score": e["score"], "rle": e["rle"]} for e in r["full"]] for n, r in raw.items()}
_, per_record = evaluate(cache, records, conf=0.85, grow=0, min_area=100)

def ious(a, b):
    if not a or not b:
        return np.zeros((len(a), len(b)))
    return np.asarray(mu.iou(a, b, [0] * len(b))).reshape(len(a), len(b))

def disjoint(rles):
    """Paint in order so the prediction is pixel-disjoint, as a submission must be."""
    claimed = None
    out = []
    for r in rles:
        m = mu.decode(r).astype(np.uint8)
        if claimed is None:
            claimed = np.zeros_like(m)
        m = m & (1 - claimed)
        if m.sum() < 1:
            continue
        claimed |= m
        out.append(mu.encode(np.asfortranarray(m)))
    return out

rows = {k: [] for k in ("single", "agreed", "agreed_and", "agreed_or", "union", "model")}
for name, passes in records.items():
    if len(passes) != 3:
        continue
    for c in range(3):
        held_id, held = passes[c]
        a, b = [passes[i][1] for i in range(3) if i != c]
        m = ious(a, b)
        pairs = sorted(np.argwhere(m > 0.5).tolist(), key=lambda p: -m[p[0], p[1]])
        used_a, used_b, agreed, agreed_and, agreed_or = set(), set(), [], [], []
        for i, j in pairs:
            if i in used_a or j in used_b:
                continue
            used_a.add(i); used_b.add(j)
            agreed.append(a[i])
            agreed_and.append(mu.merge([a[i], b[j]], intersect=True))
            agreed_or.append(mu.merge([a[i], b[j]], intersect=False))
        union = list(a) + [b[j] for j in range(len(b)) if j not in used_b]
        rows["single"].append(pq_rle(list(a), held))
        rows["agreed"].append(pq_rle(disjoint(agreed), held))
        rows["agreed_and"].append(pq_rle(disjoint(agreed_and), held))
        rows["agreed_or"].append(pq_rle(disjoint(agreed_or), held))
        rows["union"].append(pq_rle(disjoint(union), held))
        rows["model"].append(per_record[held_id])
out = {}
for k, v in rows.items():
    a = aggregate_pq(v)
    out[k] = {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in a.items()} | {"records": len(v)}
    print(k, out[k])
json.dump(out, open(sys.argv[3], "w"), indent=1)
