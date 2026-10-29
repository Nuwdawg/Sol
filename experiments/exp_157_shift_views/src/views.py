"""exp157: extra views as votes on each deployed candidate - dihedral turns, and half-stride shifts.

exp154 showed that the majority vote over the eight dihedral views gives masks closer to the
annotations (SQ +0.0035) and, with the geometric score, +0.0082 PQ. The dihedral views keep the
stride-4 mask grid's phase, so they cannot vote on where a boundary falls inside a 4-pixel cell.
Translations by 2 px (half the stride) can. This module generalises exp154's rule to any set of
views:

- partners: for each full-view candidate, the best-IoU (> 0.5) candidate of every extra view;
- mask: the k-of-n pixel vote over the candidate and its partners, k = ceil(share x n);
- score: sqrt(score x mean score over all views), a view without a partner counting 0.

`sources` names the cache keys whose views vote: "d4" has views 1-7, "shift" views 1-3.
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

VIEWS = {"d4": range(1, 8), "shift": range(1, 4)}
CONFS = tuple(np.round(np.arange(0.70, 0.96, 0.01), 2))


def partners(entry: dict, sources: tuple[str, ...]) -> tuple[list[list[dict]], np.ndarray, int]:
    """Masks voting for each full candidate, the summed partner scores, and the number of views."""
    full = entry["full"]
    rles = [E.as_rle(c["rle"]) for c in full]
    groups = [[r] for r in rles]
    total = np.array([c["score"] for c in full], np.float64)
    views = 1
    for source in sources:
        for view in VIEWS[source]:
            views += 1
            members = [c for c in entry.get(source, []) if c["view"] == view]
            if not members or not rles:
                continue
            member_rles = [E.as_rle(c["rle"]) for c in members]
            ious = np.asarray(mu.iou(rles, member_rles, [0] * len(member_rles))).reshape(len(rles), len(members))
            for i in range(len(rles)):
                j = int(ious[i].argmax())
                if ious[i, j] > 0.5:
                    groups[i].append(member_rles[j])
                    total[i] += members[j]["score"]
    return groups, total, views


def as_text(rle: dict) -> dict:
    counts = rle["counts"]
    return dict(size=list(rle["size"]), counts=counts.decode("ascii") if isinstance(counts, bytes) else counts)


def rule_pool(entry: dict, sources: tuple[str, ...], share: float | None, geometric: bool) -> list[tuple[float, "E.Crop"]]:
    groups, total, views = partners(entry, sources)
    scores = np.array([c["score"] for c in entry["full"]], np.float64)
    if geometric:
        scores = np.sqrt(scores * total / views)
    pool = []
    for i, group in enumerate(groups):
        if share is None or len(group) == 1:
            mask = entry["full"][i]["rle"]
        else:
            mask = vote_rle(group, max(1, math.ceil(share * len(group))))
        pool.append((float(scores[i]), E.Crop(as_text(mask) if isinstance(mask, dict) else mask)))
    return pool


def merged_cache(base: Path, extra: Path | None) -> dict:
    """exp147's cache (full, d4) with another run's views (e.g. shift) added per photograph."""
    cache = E.load_cache(base)
    if extra is not None:
        more = E.load_cache(extra)
        for name, entry in more.items():
            for key, value in entry.items():
                if key not in ("full", "split"):
                    cache[name][key] = value
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True, help="exp147 validation cache (full, d4)")
    parser.add_argument("--extra", type=Path, default=None, help="cache holding the shift views")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = merged_cache(args.cache, args.extra)
    records, _ = records_for(args.annotations)
    report, baseline = {}, None
    variants = [("deployed", (), None, False), ("d4_vote_geo", ("d4",), 0.5, True)]
    if all("shift" in cache[n] for n in records):
        variants += [("shift_vote", ("shift",), 0.5, False), ("shift_vote_geo", ("shift",), 0.5, True),
                     ("d4shift_vote_geo", ("d4", "shift"), 0.5, True), ("d4shift_vote75_geo", ("d4", "shift"), 0.75, True)]
    for tag, sources, share, geometric in variants:
        pool = {n: rule_pool(cache[n], sources, share, geometric) for n in records}
        rows, per = E.score_rule(pool, records, CONFS, min_area=100)
        best = max(rows, key=lambda r: r["pq"])
        if tag == "deployed":
            baseline = per[0.85]
        report[tag] = dict(best=best, bootstrap=E.bootstrap_difference(baseline, per[best["conf"]], records))
        print("VARIANT", tag, json.dumps(report[tag]), flush=True)
    (args.output / "exp157_report.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
