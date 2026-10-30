"""exp200q: does pooled truth predict exp_028's recall probe (public 0.32) and its baseline (0.36)?

exp_028 is the one past submission built to emit more than the validation optimum: exp_010's detector
at confidence 0.30, plus the candidates beneath it that exp_026's crop verifier (trained on "some
annotator drew this") passes at 0.50. Per record it lost 0.041 on validation and about 0.04 in public.
Under pooled one-to-one truth a promoted candidate that matches any annotator's filament is a true
positive, so if the leaderboard scored that way the probe should have lost much less there.

Rebuilt from the original weights (dataset sol-exp010-session1, exp_026's checkpoint) with
predict_verified.py's exact detector call and rule, on validation and on test. The test files must
reproduce the submitted CSV before the validation numbers stand for it (CLAUDE.md section 5).

    python -m experiments.exp_200_test_truth.src.recall_probe cache --weights best.pt --verifier best.pt
        --images val_images --output val.jsonl
    python -m experiments.exp_200_test_truth.src.recall_probe score --annotations ... --val val.jsonl
        --test test.jsonl --reference exp028.csv --output recall_probe.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mu

from experiments.exp_025_verifier.src.harvest import crop_of
from experiments.exp_025_verifier.src.model import Verifier
from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_200_test_truth.src.hypotheses import as_bytes, match, total
from experiments.exp_200_test_truth.src.matching import one_to_one
from shared.utils import decode_rle, encode_rle, paint_panoptic

FULL = 2048
FLOOR, IOU, MAX_DET = 0.05, 0.60, 100           # predict_verified.py's detector call
BASE_CONF, MIN_AREA = 0.30, 250                 # its emission rule; the gate is 0.50 for exp_028
PUBLIC = {None: 0.36, 0.50: 0.32}               # exp_010 session 1 and exp_028, from the submissions list


def cache(args) -> None:
    """Detector and verifier on every photograph of a folder, one JSON line each, resumable."""
    from ultralytics import YOLO
    detector = YOLO(str(args.weights))
    checkpoint = torch.load(args.verifier, map_location="cpu")
    verifier = Verifier(checkpoint.get("width", 24))
    verifier.load_state_dict(checkpoint["state"])
    verifier.eval()
    done = {json.loads(line)["name"] for line in args.output.open()} if args.output.exists() else set()
    paths = [p for p in sorted(args.images.glob("*.jpeg")) if p.name not in done]
    with args.output.open("a") as fh:
        for path in paths:
            result = detector.predict(str(path), imgsz=FULL, conf=FLOOR, iou=IOU, max_det=MAX_DET,
                                      retina_masks=True, verbose=False)[0]
            entries = []
            if result.masks is not None and len(result.masks.data):
                raw = result.masks.data.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                photograph = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                for index in range(raw.shape[0]):
                    mask = raw[index].astype(np.uint8)
                    if mask.shape != (FULL, FULL):
                        mask = (cv2.resize(mask.astype(np.float32), (FULL, FULL),
                                           interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
                    p = 0.0                        # predict_verified.py leaves an uncroppable mask at 0
                    cropped = crop_of(photograph, mask)
                    if cropped is not None:
                        patch, patch_mask = cropped
                        x = np.stack([patch.astype(np.float32) / 255.0, patch_mask.astype(np.float32)])[None]
                        with torch.no_grad():
                            p = float(1.0 / (1.0 + np.exp(-verifier(torch.from_numpy(x)).numpy()[0])))
                    entries.append(dict(conf=float(confs[index]), p=p, counts=encode_rle(mask)))
            fh.write(json.dumps(dict(name=path.name, candidates=entries)) + "\n")
            fh.flush()


def emit(pool: dict, gate: float | None) -> dict:
    """predict_verified.py's rule: confidence >= 0.30 paints first; verifier passes paint at 0.29."""
    out = {}
    for name, entries in pool.items():
        candidates = []
        for e in entries:
            if e["conf"] >= BASE_CONF:
                candidates.append((e["conf"], decode_rle(e["counts"])))
            elif gate is not None and e["p"] >= gate:
                candidates.append((BASE_CONF - 0.01, decode_rle(e["counts"])))
        out[name] = [m for _, m, _ in paint_panoptic(candidates, min_area=MIN_AREA)] if candidates else []
    return out


def load(path: Path) -> dict:
    return {row["name"]: row["candidates"] for row in map(json.loads, path.open())}


def score(args) -> None:
    records, _ = records_for(args.annotations)
    names = sorted(records)
    passes = {n: [[as_bytes(t) for t in truth] for _, truth in sorted(records[n])] for n in names}
    val, test = load(args.val), load(args.test)
    if sorted(val) != names or len(test) != 180:
        raise SystemExit("the caches do not cover validation and test")
    reference = pd.read_csv(args.reference)
    reference["stem"] = reference["filament_id"].str.rsplit("_", n=1).str[0]
    rows = []
    for gate in (None, 0.50, 0.60, 0.70):
        painted = emit(test, gate)
        if gate == 0.50:                         # parity with the submitted exp_028 file
            ours = {Path(n).stem: [encode_rle(m) for m in ms] for n, ms in painted.items()}
            theirs = reference.groupby("stem")["segmentation_rle"].apply(list).to_dict()
            same = sum(len(set(ours.get(s, [])) & set(c)) for s, c in theirs.items())
            print(json.dumps(dict(parity=dict(rows_ours=sum(map(len, ours.values())), rows_submitted=len(reference),
                                              identical_masks=same))), flush=True)
        preds = {n: [mu.encode(np.asfortranarray(m)) for m in ms] for n, ms in emit(val, gate).items()}
        record = total([match(preds[n], t) for n in names for t in passes[n]])
        pooled = total([one_to_one(preds[n], [x for t in passes[n] for x in t]) for n in names])
        row = dict(gate=gate, test_instances=sum(map(len, painted.values())), val_instances=sum(map(len, preds.values())),
                   record=record, pooled=pooled, public=PUBLIC.get(gate))
        rows.append(row)
        print(json.dumps(dict(gate=gate, test=row["test_instances"], val=row["val_instances"],
                              record=round(record["pq"], 4), record_counts=[record[k] for k in ("tp", "fp", "fn")],
                              pooled=round(pooled["pq"], 4), pooled_counts=[pooled[k] for k in ("tp", "fp", "fn")],
                              public=row["public"])), flush=True)
    args.output.write_text(json.dumps(rows, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--weights", type=Path, required=True)
    c.add_argument("--verifier", type=Path, required=True)
    c.add_argument("--images", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--annotations", type=Path, required=True)
    s.add_argument("--val", type=Path, required=True)
    s.add_argument("--test", type=Path, required=True)
    s.add_argument("--reference", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (cache if args.command == "cache" else score)(args)


if __name__ == "__main__":
    main()
