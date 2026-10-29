"""exp154: does a vote over the dihedral views give masks closer to the annotations?

exp153 found that 176 of exp139's 444 validation misses have a confident candidate whose mask
overlaps the truth at IoU 0.1-0.5: the filament is found, its extent disagrees. The seven
non-identity dihedral views are independent-looking estimates of the same mask. Here each
full-view candidate keeps its score (or exp149's geometric score) and only its mask changes:
it becomes the k-of-n pixel vote over the candidate and its partner in every view (the view
candidate with the best IoU > 0.5), for vote shares 0.25, 0.5 and 0.75. The deployed emission
then runs unchanged. A mask change moves SQ and the matches near IoU 0.5; it cannot add
instances.

    python -m experiments.exp_154_view_masks.src.view_masks --cache val_cache.json
        --annotations ... --output /kaggle/working
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
from experiments.exp_152_rule_and_emit.src.emit import geometric_scores

CONFS = tuple(np.round(np.arange(0.70, 0.96, 0.01), 2))


def voted_masks(entry: dict, share: float) -> list[dict]:
    full = entry["full"]
    rles = [E.as_rle(c["rle"]) for c in full]
    partners = [[r] for r in rles]
    for view in range(1, 8):
        members = [E.as_rle(c["rle"]) for c in entry.get("d4", []) if c["view"] == view]
        if not members or not rles:
            continue
        ious = np.asarray(mu.iou(rles, members, [0] * len(members))).reshape(len(rles), len(members))
        for i in range(len(rles)):
            j = int(ious[i].argmax())
            if ious[i, j] > 0.5:
                partners[i].append(members[j])
    out = []
    for group in partners:
        needed = max(1, math.ceil(share * len(group)))
        out.append(vote_rle(group, needed) if len(group) > 1 else dict(size=group[0]["size"],
                                                                        counts=group[0]["counts"].decode("ascii")
                                                                        if isinstance(group[0]["counts"], bytes)
                                                                        else group[0]["counts"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = E.load_cache(args.cache)
    records, _ = records_for(args.annotations)
    report = {}
    baseline_records = None
    geometric = {name: geometric_scores(cache[name]) for name in records}
    variants = [("full_masks", None)] + [(f"vote_{share}", share) for share in (0.25, 0.5, 0.75)]
    for tag, share in variants:
        masks = {name: ([c["rle"] for c in cache[name]["full"]] if share is None else voted_masks(cache[name], share))
                 for name in records}
        for scoring in ("score", "geometric"):
            pool = {}
            for name in records:
                scores = ([c["score"] for c in cache[name]["full"]] if scoring == "score" else geometric[name])
                pool[name] = [(float(s), E.Crop(m)) for s, m in zip(scores, masks[name])]
            rows, per_record = E.score_rule(pool, records, CONFS, min_area=100)
            best = max(rows, key=lambda r: r["pq"])
            key = f"{tag}+{scoring}"
            if key == "full_masks+score":
                baseline_records = per_record[0.85]
            diff = E.bootstrap_difference(baseline_records, per_record[best["conf"]], records)
            report[key] = dict(best=best, bootstrap=diff)
            print("VARIANT", key, json.dumps(dict(best=best, bootstrap=diff)), flush=True)
    (args.output / "exp154_report.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
