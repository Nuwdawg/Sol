"""Phase-2 diagnosis of exp139 epoch 40 on the canonical validation split (CPU only).

Inputs (all read-only):
  argv[1] training annotations JSON (sha256 5da9e92b...)
  argv[2] exp147 validation candidate cache (sol-exp147-candidate-cache-t4/val_cache.json)
  argv[3] output JSON path

Steps
  1. Parity: rebuild the deployed rule (full view, score >= 0.85, no erosion, min area 100,
     paint by score) with the repo's own exp033 evaluator and require PQ 0.4705249990.
  2. Break the pooled PQ down by photograph multiplicity (1/2/3 annotation records), site,
     annotator group and year.
  3. Classify every FN and FP: best IoU against the scored record, and whether the same
     object was drawn by a *different* annotator of the same photograph (disagreement)
     or by nobody (model error).
  4. Host-notebook rubric quantities: IoU and Dice over all overlapping pairs (IoU > 0), and
     the one-to-many / many-to-one degree counts.
  5. Photograph bootstrap of the pooled PQ (noise floor for 106 and for ~90 photographs).
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np
import pycocotools.mask as mu

sys.path.insert(0, "/home/user/Sol")
from experiments.exp_033_inference.src.sweep import evaluate, records_for, pq_rle  # noqa: E402
from shared.utils import aggregate_pq  # noqa: E402

annotations, cache_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
EXPECTED = 0.470524998967835

records, split = records_for(annotations)          # {photo: [(image_id, [gt rle])]}
raw = json.load(open(cache_path))["photographs"]
cache = {name: [{"score": e["score"], "rle": e["rle"]} for e in rec["full"]] for name, rec in raw.items()}

summary, per_record = evaluate(cache, records, conf=0.85, grow=0, min_area=100)
print("PARITY", json.dumps(summary))
assert abs(summary["pq"] - EXPECTED) < 1e-9, "does not reproduce exp139's 0.4705249990"

# Rebuild the deployed predictions per photograph (same code path as evaluate()).
from shared.utils import paint_panoptic  # noqa: E402
from experiments.exp_033_inference.src.sweep import encode  # noqa: E402
deployed = {}
for name in records:
    cands = [(e["score"], mu.decode(e["rle"])) for e in cache[name] if e["score"] >= 0.85]
    deployed[name] = [(s, encode(m)) for s, m, _ in paint_panoptic(cands, min_area=100)]

coco = json.load(open(annotations))
image_meta = {r["id"]: r for r in coco["images"]}

def group_of(image_id):
    return image_id.split("-")[0]

# ---------- 2. breakdowns ----------
def pooled(rows):
    a = aggregate_pq(rows)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in a.items()} | {"records": len(rows)}

by = defaultdict(lambda: defaultdict(list))
for name, passes in records.items():
    mult = len(passes)
    for image_id, _ in passes:
        r = per_record[image_id]
        by["multiplicity"][str(mult)].append(r)
        by["site"][name.rsplit(".", 1)[0][-2:]].append(r)
        by["group"][group_of(image_id)[:4]].append(r)
        by["year"][name[:4]].append(r)
breakdowns = {k: {g: pooled(v) for g, v in sorted(d.items())} for k, d in by.items()}
for k, d in breakdowns.items():
    print("BREAKDOWN", k)
    for g, v in d.items():
        print("   ", g, v)

# ---------- 3. error taxonomy ----------
def ious(a, b):
    if not a or not b:
        return np.zeros((len(a), len(b)))
    return np.asarray(mu.iou(a, b, [0] * len(b))).reshape(len(a), len(b))

fn_classes, fp_classes = Counter(), Counter()
fn_by_mult, fp_by_mult = defaultdict(Counter), defaultdict(Counter)
cand_rles = {name: [e["rle"] for e in cache[name]] for name in records}
cand_scores = {name: np.array([e["score"] for e in cache[name]]) for name in records}
fn_candidate = Counter()
for name, passes in records.items():
    preds = [r for _, r in deployed[name]]
    mult = len(passes)
    for image_id, truth in passes:
        others = [g for oid, t in passes if oid != image_id for g in t]
        m = ious(preds, truth)                     # pred x gt
        # one-to-one matches at IoU > 0.5 (unique by construction)
        tp_p = set(np.argwhere(m > 0.5)[:, 0].tolist()) if m.size else set()
        tp_g = set(np.argwhere(m > 0.5)[:, 1].tolist()) if m.size else set()
        o_g = ious(truth, others)                  # this gt x other annotators' gt
        o_p = ious(preds, others)                  # pred x other annotators' gt
        c = ious(cand_rles[name], truth)           # every full-view candidate x gt
        for j in range(len(truth)):
            if j in tp_g:
                continue
            best = m[:, j].max() if len(preds) else 0.0
            band = "0.3-0.5" if best >= 0.3 else "0.1-0.3" if best >= 0.1 else "0-0.1" if best > 0 else "none"
            if mult == 1:
                who = "single-annotator photo"
            else:
                who = "drawn by another annotator" if (o_g.shape[1] and o_g[j].max() > 0.5) else "only this annotator"
            fn_classes[(band, who)] += 1
            fn_by_mult[str(mult)][band] += 1
            # did any candidate (any score) cover it at IoU > 0.5?
            hit = np.where(c[:, j] > 0.5)[0] if c.size else []
            fn_candidate["candidate>0.5 exists" if len(hit) else "no candidate>0.5"] += 1
            if len(hit):
                s = cand_scores[name][hit].max()
                fn_candidate["best covering score " + ("<0.3" if s < 0.3 else "0.3-0.6" if s < 0.6 else "0.6-0.85" if s < 0.85 else ">=0.85 (lost in painting/area)")] += 1
        for i in range(len(preds)):
            if i in tp_p:
                continue
            best = m[i].max() if len(truth) else 0.0
            band = "0.3-0.5" if best >= 0.3 else "0.1-0.3" if best >= 0.1 else "0-0.1" if best > 0 else "none"
            if mult == 1:
                who = "single-annotator photo"
            else:
                who = "matches another annotator" if (o_p.shape[1] and o_p[i].max() > 0.5) else "matches no annotator"
            fp_classes[(band, who)] += 1
            fp_by_mult[str(mult)][band] += 1

print("FN classes", sum(fn_classes.values()))
for k, v in sorted(fn_classes.items(), key=lambda x: -x[1]):
    print("   ", k, v)
print("FN candidate coverage", dict(fn_candidate))
print("FP classes", sum(fp_classes.values()))
for k, v in sorted(fp_classes.items(), key=lambda x: -x[1]):
    print("   ", k, v)

# ---------- 4. host rubric quantities ----------
iou_all, dice_all, gt_deg, pred_deg = [], [], Counter(), Counter()
for name, passes in records.items():
    preds = [r for _, r in deployed[name]]
    for image_id, truth in passes:
        m = ious(preds, truth)
        if m.size:
            inter_iou = m[m > 0]
            iou_all.extend(inter_iou.tolist())
            dice_all.extend((2 * inter_iou / (1 + inter_iou)).tolist())
            gt_deg.update((m > 0).sum(axis=0).tolist())
            pred_deg.update((m > 0).sum(axis=1).tolist())
        else:
            gt_deg.update([0] * len(truth)); pred_deg.update([0] * len(preds))
host = {"mean_iou_overlapping_pairs": float(np.mean(iou_all)), "mean_dice_overlapping_pairs": float(np.mean(dice_all)),
        "pairs": len(iou_all), "gt_degree": dict(sorted(gt_deg.items())), "pred_degree": dict(sorted(pred_deg.items()))}
print("HOST", json.dumps(host))

# ---------- annotator vs annotator on multi-annotated validation photographs ----------
aa = []
for name, passes in records.items():
    if len(passes) < 2:
        continue
    for a in range(len(passes)):
        for b in range(len(passes)):
            if a != b:
                aa.append(pq_rle(passes[a][1], passes[b][1]))
print("ANNOTATOR_VS_ANNOTATOR", pooled(aa))
model_multi = [per_record[i] for name, passes in records.items() if len(passes) >= 2 for i, _ in passes]
print("MODEL_ON_MULTI", pooled(model_multi))

# ---------- 5. photograph bootstrap ----------
rng = np.random.default_rng(0)
names = sorted(records)
rows_by_photo = {n: [per_record[i] for i, _ in records[n]] for n in names}
def boot(k, reps=4000):
    vals = []
    for _ in range(reps):
        pick = rng.choice(names, size=k, replace=True)
        vals.append(aggregate_pq([r for n in pick for r in rows_by_photo[n]])["pq"])
    return {"sd": float(np.std(vals)), "p2.5": float(np.percentile(vals, 2.5)), "p97.5": float(np.percentile(vals, 97.5))}
b106, b90 = boot(106), boot(90)
print("BOOT106", b106, "BOOT90", b90)

json.dump({"parity": summary, "breakdowns": breakdowns,
           "fn_classes": {" | ".join(k): v for k, v in fn_classes.items()},
           "fn_candidate": dict(fn_candidate),
           "fp_classes": {" | ".join(k): v for k, v in fp_classes.items()},
           "fn_by_multiplicity": {k: dict(v) for k, v in fn_by_mult.items()},
           "fp_by_multiplicity": {k: dict(v) for k, v in fp_by_mult.items()},
           "host_rubric": host, "annotator_vs_annotator": pooled(aa), "model_on_multi": pooled(model_multi),
           "bootstrap": {"106": b106, "90": b90}}, open(out_path, "w"), indent=1)
print("WROTE", out_path)
