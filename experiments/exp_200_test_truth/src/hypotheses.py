"""exp200: which construction of the test truth explains the leaderboard?

Validation PQ runs 0.06-0.08 above the public score for every model family, and nothing tried so
far explained it (final_decision/decision_brief.pdf, section 2e). Validation scores each annotator's
version of a photograph as its own record. The submission format has one image per photograph
(`<stem>_<k>`), so the hidden test truth must combine a photograph's annotators somehow, and the
organisers have not said how (forum 729487, 753043). Their notebook's metric counts every truth /
prediction pair above IoU 0.5, so overlapping truth is legal input to it.

This scores the validation predictions of submissions whose public scores are known under several
constructions of the truth, and prints them next to the public scores:

  record        each annotator's version separately, counts pooled (what validation does now)
  pooled        all annotators' filaments in one image, overlapping (the notebook's metric as is)
  union_graph   filaments that overlap across annotators merged into their union
  union_px      the union of all annotators' pixels, split into connected components
  majority_px   pixels drawn by more than half of the annotators, as components
  inter_px      pixels drawn by every annotator, as components
  first / last  one annotator only (records ordered by image id)

    python -m experiments.exp_200_test_truth.src.hypotheses --annotations ... --cache ... --shift ...
        --yolo yolov8l=... --yolo yolo11m=... --yolov8l-pool ... [--output report.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_149_evidence_fusion.src import evidence as E
from experiments.exp_157_shift_views.src.views import merged_cache
from experiments.exp_159_yolo_votes.src.votes import load_pool
from experiments.exp_168_joint_tuning.src.joint import prepared, scores

MIN_COMPONENT = 100          # a consensus component smaller than this is not kept as a filament
PUBLIC = {"exp139": 0.41, "exp169": 0.41, "exp097": 0.39}
RECORDED = {"exp139": 0.4705249990, "exp169": 0.48894859995182627, "exp097": 0.466806}


def as_bytes(rle: dict) -> dict:
    counts = rle["counts"]
    return dict(size=[int(v) for v in rle["size"]], counts=counts.encode("ascii") if isinstance(counts, str) else counts)


def match(predictions: list[dict], truth: list[dict]) -> dict:
    """The organisers' get_pq_score for one image: every pair above IoU 0.5 is a true positive."""
    if not truth:
        return dict(tp=0, fp=len(predictions), fn=0, iou_sum=0.0)
    if not predictions:
        return dict(tp=0, fp=0, fn=len(truth), iou_sum=0.0)
    iou = np.asarray(mu.iou(predictions, truth, [0] * len(truth))).reshape(len(predictions), len(truth))
    hit = iou > 0.5
    return dict(tp=int(hit.sum()), iou_sum=float(iou[hit].sum()),
                fp=int((hit.sum(1) == 0).sum()), fn=int((hit.sum(0) == 0).sum()))


def total(rows: list[dict]) -> dict:
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    iou_sum = sum(r["iou_sum"] for r in rows)
    sq = iou_sum / tp if tp else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if tp + fp + fn else 0.0
    return dict(pq=sq * rq, sq=sq, rq=rq, tp=tp, fp=fp, fn=fn)


def components(binary: np.ndarray) -> list[dict]:
    count, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    out = []
    for label in range(1, count):
        part = labels == label
        if part.sum() >= MIN_COMPONENT:
            out.append(mu.encode(np.asfortranarray(part.astype(np.uint8))))
    return out


def union_graph(passes: list[list[dict]]) -> list[dict]:
    """Merge filaments of different annotators that share any pixel; keep the rest as drawn."""
    items = [(a, rle) for a, rles in enumerate(passes) for rle in rles]
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    rles = [rle for _, rle in items]
    if len(rles) > 1:
        iou = np.asarray(mu.iou(rles, rles, [0] * len(rles))).reshape(len(rles), len(rles))
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i][0] != items[j][0] and iou[i, j] > 0:
                    parent[find(i)] = find(j)
    groups = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(rles[i])
    return [g[0] if len(g) == 1 else mu.merge(g, intersect=False) for g in groups.values()]


def pixel_vote(passes: list[list[dict]], need) -> list[dict]:
    count = None
    for rles in passes:
        drawn = (mu.decode(mu.merge(rles, intersect=False)) if len(rles) > 1 else mu.decode(rles[0])) if rles \
            else np.zeros((2048, 2048), np.uint8)
        count = drawn.astype(np.uint8) if count is None else count + drawn
    return components(count >= need(len(passes)))


def truths(passes: list[list[dict]]) -> dict:
    """Every construction of one photograph's truth, as {name: list of truth images (lists of RLEs)}."""
    k = len(passes)
    return {
        "record": passes,
        "pooled": [[rle for rles in passes for rle in rles]],
        "union_graph": [union_graph(passes)],
        "union_px": [pixel_vote(passes, lambda n: 1)],
        "majority_px": [pixel_vote(passes, lambda n: n // 2 + 1)],
        "inter_px": [pixel_vote(passes, lambda n: n)],
        "first": [passes[0]],
        "last": [passes[-1]],
        "_annotators": k,
    }


def as_text(rle: dict) -> dict:
    counts = rle["counts"]
    return dict(size=[int(v) for v in rle["size"]], counts=counts.decode("ascii") if isinstance(counts, bytes) else counts)


def predictions(args) -> dict:
    """Validation predictions of the submissions whose public scores are known (cached in --predictions)."""
    records, _ = records_for(args.annotations)
    if args.predictions and args.predictions.exists():
        return records, json.loads(args.predictions.read_text())
    cache = merged_cache(args.cache, args.shift)
    pools = [load_pool(Path(p)) for _, p in (s.split("=", 1) for s in sorted(args.yolo))]
    v8 = load_pool(args.yolov8l_pool)
    out = {"exp139": {}, "exp169": {}, "exp097": {}}
    for name in sorted(records):
        p = prepared(cache[name], [pl.get(name, []) for pl in pools], ("d4", "shift"))
        entries = list(zip(map(float, scores(p, 4.0, 0.3)), p["crops"]))
        out["exp169"][name] = [rle for _, rle in E.paint([(s, c) for s, c in entries if s >= 0.68], 100)]
        out["exp139"][name] = [rle for _, rle in E.paint([(c["score"], E.Crop(c["rle"])) for c in cache[name]["full"]
                                                          if c["score"] >= 0.85], 100)]
        out["exp097"][name] = [rle for _, rle in E.paint([(e["score"], E.Crop(e["rle"])) for e in v8.get(name, [])
                                                          if e["score"] >= 0.35], 300)]
    out = {model: {name: [as_text(r) for r in rles] for name, rles in by.items()} for model, by in out.items()}
    if args.predictions:
        args.predictions.write_text(json.dumps(out))
    return records, out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--shift", type=Path, required=True)
    parser.add_argument("--yolo", action="append", default=[])
    parser.add_argument("--yolov8l-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None, help="cache of the validation predictions")
    args = parser.parse_args()

    records, preds = predictions(args)
    built = {name: truths([[as_bytes(r) for r in rles] for _, rles in sorted(records[name])]) for name in records}
    names = [k for k in next(iter(built.values())) if not k.startswith("_")]
    table = {}
    for model, by_photo in preds.items():
        rows = {construction: [] for construction in names}
        for name, truth in built.items():
            mine = [as_bytes(r) for r in by_photo[name]]
            for construction in names:
                rows[construction].extend(match(mine, image) for image in truth[construction])
        table[model] = {construction: total(r) for construction, r in rows.items()}
        if abs(table[model]["record"]["pq"] - RECORDED[model]) > 2e-6:
            raise SystemExit(f"{model}: record PQ {table[model]['record']['pq']} is not the recorded {RECORDED[model]}")
    print("construction".ljust(14), *(f"{m:>9}" for m in table), " public:", PUBLIC)
    for construction in names:
        print(construction.ljust(14), *(f"{table[m][construction]['pq']:9.4f}" for m in table))
    annotators = {}
    for truth in built.values():
        annotators[truth["_annotators"]] = annotators.get(truth["_annotators"], 0) + 1
    report = dict(public=PUBLIC, annotators_per_photo=annotators, table=table)
    if args.output:
        args.output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
