"""exp152: choose the emission rule on held-out photographs, then write test submissions.

Two rules are compared on a validation cache of a model that did not train on validation
(the exp150 control, or exp139 epoch 40):

- `single`: the deployed rule, one view, confidence threshold on the model's score;
- `geometric`: the score replaced by sqrt(score x mean score over the eight dihedral views),
  where a view contributes the score of its candidate with the best IoU > 0.5 and 0 otherwise.
  exp149 measured +0.0043 [-0.0026, +0.0116] for it on exp139 epoch 40.

For each rule the confidence threshold is swept (0.50-0.95 by 0.01) and the area floor is
100, exp139's. The rule and threshold are then applied, unchanged, to the test cache of the
model chosen for submission, and each CSV is written with exp033's emission, checked for
empty or overlapping instances, and summarised with its row count and SHA256. A review
sheet compares the chosen emission with truth on validation crops.

    python -m experiments.exp_152_rule_and_emit.src.emit --val-cache ... --test-cache ...
        --annotations ... --images <train_images> --output /kaggle/working
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from experiments.exp_149_evidence_fusion.src import evidence as E

CONFS = tuple(np.round(np.arange(0.50, 0.96, 0.01), 2))
PARITY_ROWS = 1192
PARITY_SHA256 = "CCA9C06EB7FC70E6FBFF11609554A490F6F0A0BD34A48E88F648F434FC986BFF"


def geometric_scores(entry: dict) -> np.ndarray:
    full = entry["full"]
    scores = np.array([c["score"] for c in full], np.float64)
    if not full:
        return scores
    rles = [E.as_rle(c["rle"]) for c in full]
    total = scores.copy()
    for view in range(1, 8):
        members = [c for c in entry.get("d4", []) if c["view"] == view]
        iou, partner = E.best_overlap(rles, [E.as_rle(c["rle"]) for c in members],
                                      np.array([c["score"] for c in members]))
        total += np.where(iou > 0.5, partner, 0.0)
    return np.sqrt(scores * total / 8.0)


RULES = ("single", "geometric", "vote", "geometric_vote")


def pool_for(cache: dict, rule: str) -> dict:
    """`vote` rules replace each full-view mask by the majority vote over it and its partner in each
    dihedral view (exp154); `geometric` rules rescore with sqrt(score x eight-view mean score) (exp149)."""
    from experiments.exp_154_view_masks.src.view_masks import voted_masks
    pool = {}
    for name, entry in cache.items():
        scores = geometric_scores(entry) if rule.startswith("geometric") else np.array([c["score"] for c in entry["full"]])
        masks = voted_masks(entry, 0.5) if rule.endswith("vote") else [c["rle"] for c in entry["full"]]
        pool[name] = [(float(s), E.Crop(m)) for s, m in zip(scores, masks)]
    return pool


def write_csv(pool: dict, conf: float, min_area: int, destination: Path) -> dict:
    written, photos = 0, 0
    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filament_id", "segmentation_rle"])
        for name in sorted(pool):
            stem = Path(name).stem
            painted = E.paint([(s, c) for s, c in pool[name] if s >= conf], min_area)
            claimed = np.zeros((E.SIZE, E.SIZE), np.uint8)
            kept = 0
            for index, (_, rle) in enumerate(painted, start=1):
                mask = mu.decode(rle)
                if not mask.sum():
                    raise SystemExit(f"empty instance for {stem}")
                if (claimed & mask).any():
                    raise SystemExit(f"overlapping instances for {stem}")
                claimed |= mask
                counts = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]
                writer.writerow([f"{stem}_{index}", counts])
                kept += 1
            written += kept
            photos += kept > 0
    return dict(path=destination.name, rows=written, photographs_with_rows=photos, conf=conf, min_area=min_area,
                sha256=hashlib.sha256(destination.read_bytes()).hexdigest().upper())


def compare_csv(ours: Path, theirs: Path) -> dict:
    """Instance-level agreement of two submission files: every instance matched at IoU >= 0.99?"""
    def read(path):
        rows = {}
        with Path(path).open() as handle:
            for row in csv.DictReader(handle):
                rows.setdefault(row["filament_id"].rsplit("_", 1)[0], []).append(
                    dict(size=[E.SIZE, E.SIZE], counts=row["segmentation_rle"].encode("ascii")))
        return rows
    a, b = read(ours), read(theirs)
    worst, unmatched = 1.0, 0
    for stem in set(a) | set(b):
        x, y = a.get(stem, []), b.get(stem, [])
        if len(x) != len(y):
            unmatched += abs(len(x) - len(y))
        if x and y:
            ious = np.asarray(mu.iou(x, y, [0] * len(y))).reshape(len(x), len(y))
            worst = min(worst, float(ious.max(axis=1).min()), float(ious.max(axis=0).min()))
    return dict(min_iou=worst, count_differences=unmatched)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val-cache", type=Path, required=True, help="cache of a model that did not see validation")
    parser.add_argument("--test-cache", action="append", default=[], help="tag=path, repeatable")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--parity-tag", default="e040", help="test cache that must reproduce exp139's CSV")
    parser.add_argument("--parity-csv", type=Path, default=None, help="exp139's submitted CSV")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from experiments.exp_033_inference.src.sweep import records_for
    records, _ = records_for(args.annotations)
    val = E.load_cache(args.val_cache)
    report = dict(val_cache=str(args.val_cache), rules={})
    per_record = {}
    for rule in RULES:
        rows, records_at = E.score_rule(pool_for(val, rule), records, CONFS, min_area=args.min_area)
        best = max(rows, key=lambda r: r["pq"])
        at85 = next(r for r in rows if abs(r["conf"] - 0.85) < 1e-9)
        per_record[rule] = records_at[best["conf"]]
        report["rules"][rule] = dict(best=best, at_085=at85, curve=[(r["conf"], round(r["pq"], 5)) for r in rows])
        print("RULE", rule, json.dumps(dict(best=best, at_085=at85)), flush=True)
    for rule in RULES[1:]:
        report[f"{rule}_vs_single"] = E.bootstrap_difference(per_record["single"], per_record[rule], records)
        print("BOOTSTRAP", rule, json.dumps(report[f"{rule}_vs_single"]), flush=True)
    for spec in args.test_cache:
        tag, path = spec.split("=", 1)
        test = E.load_cache(Path(path))
        if len(test) != 180:
            raise SystemExit(f"{tag}: expected 180 test photographs, found {len(test)}")
        report[tag] = {}
        if tag == args.parity_tag:
            # The deployed exp139 rule on this cache must rebuild the submitted file byte for byte
            # (CLAUDE.md section 5): 1,192 rows, SHA256 from docs/ACTIVE_EXPERIMENT.md.
            facts = write_csv(pool_for(test, "single"), 0.85, args.min_area, args.output / f"parity_{tag}_single_085.csv")
            facts["identical_bytes"] = facts["sha256"] == PARITY_SHA256
            if args.parity_csv is not None:
                facts.update(compare_csv(args.output / f"parity_{tag}_single_085.csv", args.parity_csv))
            report[tag]["parity"] = facts
            print("PARITY", json.dumps(facts), flush=True)
            if facts["rows"] != PARITY_ROWS or (not facts["identical_bytes"] and facts.get("min_iou", 0) < 0.99):
                raise SystemExit("the test path does not reproduce exp139's submitted CSV")
        for rule in RULES:
            if rule != "single" and not all("d4" in entry for entry in test.values()):
                continue
            conf = report["rules"][rule]["best"]["conf"]
            facts = write_csv(pool_for(test, rule), conf, args.min_area, args.output / f"submission_{tag}_{rule}.csv")
            report[tag][rule] = facts
            print("CSV", tag, rule, json.dumps(facts), flush=True)
    (args.output / "exp152_report.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
