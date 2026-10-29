"""exp159: do the two YOLO models add independent votes to exp154's rule?

exp154's rule scores each Mask2Former candidate by sqrt(score x mean score over its eight dihedral
views) and votes its mask over those views. The dihedral views share one network's errors. Two YOLO
detectors trained separately (exp097 YOLOv8l-seg at 2048, exp002 YOLO11m-seg at 1280, both with
dihedral fusion) err differently (exp102 found cross-architecture pooling the first ensemble that
helped YOLO). Here each YOLO pool contributes one more vote to the mean score: the score of its
candidate with the best IoU > 0.5, 0 if none, with weight w. Masks stay Mask2Former's dihedral vote,
because the YOLO masks follow another boundary convention (exp104: they are best eroded by 1 px).

    python -m experiments.exp_159_yolo_votes.src.votes --cache exp147_val_cache.json
        --yolo yolov8l=d4_pool_stage2_best.json --yolo yolo11m=exp002_val_pool.json --annotations ... --output ...
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.cached_tta import vote_rle
from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_149_evidence_fusion.src import evidence as E
from experiments.exp_157_shift_views.src.views import CONFS, as_text, partners


def load_pool(path: Path) -> dict:
    """exp146's loader: the repository's two pool layouts as {file_name: [{score, rle}]}."""
    raw = json.loads(Path(path).read_text())
    pool = {}
    for key, entries in raw.items():
        name = key if key.endswith((".jpeg", ".jpg", ".png")) else f"{key}.jpeg"
        items = []
        for entry in entries:
            if "rle" in entry:
                items.append(dict(score=float(entry["score"]), rle=entry["rle"]))
            else:                                        # exp005 layout: {size, counts, features}
                items.append(dict(score=float(entry["features"]["score"]),
                                  rle=dict(size=entry["size"], counts=entry["counts"])))
        pool[name] = items
    return pool


def yolo_support(entry: dict, pool_entries: list[dict]) -> np.ndarray:
    rles = [E.as_rle(c["rle"]) for c in entry["full"]]
    if not rles or not pool_entries:
        return np.zeros(len(rles))
    others = [E.as_rle(p["rle"]) for p in pool_entries]
    ious = np.asarray(mu.iou(rles, others, [0] * len(others))).reshape(len(rles), len(others))
    best = ious.argmax(axis=1)
    scores = np.array([p["score"] for p in pool_entries])
    return np.where(ious.max(axis=1) > 0.5, scores[best], 0.0)


def pool_for(entry: dict, yolo: list[np.ndarray], weight: float,
             sources: tuple[str, ...] = ("d4",)) -> list[tuple[float, "E.Crop"]]:
    groups, total, views = partners(entry, sources)
    s = np.array([c["score"] for c in entry["full"]], np.float64)
    total = total + weight * sum(yolo) if yolo else total
    views = views + weight * len(yolo)
    scores = np.sqrt(s * total / views)
    out = []
    for i, group in enumerate(groups):
        mask = entry["full"][i]["rle"] if len(group) == 1 else as_text(vote_rle(group, max(1, math.ceil(0.5 * len(group)))))
        out.append((float(scores[i]), E.Crop(mask)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--yolo", action="append", default=[], help="tag=pool.json, repeatable")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = E.load_cache(args.cache)
    records, _ = records_for(args.annotations)
    pools = {tag: load_pool(Path(path)) for tag, path in (spec.split("=", 1) for spec in args.yolo)}
    support = {n: {tag: yolo_support(cache[n], pool.get(n, [])) for tag, pool in pools.items()} for n in records}
    base_rows, base_per = E.score_rule({n: [(c["score"], E.Crop(c["rle"])) for c in cache[n]["full"]] for n in records},
                                       records, (0.85,), min_area=100)
    baseline = base_per[0.85]
    report = dict(deployed=base_rows[0], variants={})
    variants = [("d4_vote_geo", [], 0.0)]
    for tag in pools:
        variants += [(f"+{tag}_w1", [tag], 1.0), (f"+{tag}_w2", [tag], 2.0)]
    if len(pools) > 1:
        variants += [("+all_w1", list(pools), 1.0), ("+all_w2", list(pools), 2.0)]
    for name, tags, weight in variants:
        pool = {n: pool_for(cache[n], [support[n][t] for t in tags], weight) for n in records}
        rows, per = E.score_rule(pool, records, CONFS, min_area=100)
        best = max(rows, key=lambda r: r["pq"])
        report["variants"][name] = dict(best=best, bootstrap=E.bootstrap_difference(baseline, per[best["conf"]], records))
        print("VARIANT", name, json.dumps(report["variants"][name]), flush=True)
    (args.output / "exp159_report.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
