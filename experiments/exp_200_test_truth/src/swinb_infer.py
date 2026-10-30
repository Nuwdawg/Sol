"""exp200n: Swin-B (exp190b epoch 32) full-view candidates for the union rule, lean and resumable.

exp147's `cache.py` keeps every candidate down to class probability 0.02 with three masks each,
which costs most of a CPU's time at 2048 px. The union rule (union.py) uses only Swin-B candidates
scoring at least 0.75, and a candidate's score (class probability x mean mask probability) cannot
exceed its class probability. This calls the same `run_view` with floor 0.5 and no extras, so every
candidate it writes is identical to cache.py's (checked on a validation photograph: --parity).
One JSON line per photograph; a rerun skips photographs already written.

    python -m experiments.exp_200_test_truth.src.swinb_infer --weights swinb_exp190b_e032.pt
        --images test_images --output swinb_test_full.jsonl [--shard 0 --shards 2] [--parity CACHE]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mu

from experiments.exp_139_mask2former.src.data import NATIVE, read_gray
from experiments.exp_147_candidate_cache.src.cache import encode, load_model, run_view

FLOOR = 0.5
SWINB_SHA256 = "ac9583775f51fbc3ed29f10a75ce19b1cde26b7525cc74041b0a7f51e16498e6"


def candidates(model, path: Path) -> list[dict]:
    gray = read_gray(path)
    out = []
    for e in run_view(model, gray, torch.device("cpu"), FLOOR, NATIVE):
        out.append(dict(query=e["query"], score=e["score"], p=e["p"], m=e["m"], area=e["area"], rle=encode(e["mask"])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--parity", type=Path, default=None,
                        help="an exp147-layout cache of the same photographs: check, write nothing")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    model, facts = load_model(args.weights, torch.device("cpu"))
    if facts["sha256"] != SWINB_SHA256:
        raise SystemExit(f"not the Swin-B checkpoint of the validation caches: {facts['sha256']}")
    names = sorted(p.name for p in args.images.glob("*.jpeg"))[args.shard::args.shards]
    if args.parity:
        stored = json.loads(args.parity.read_text())["photographs"]
        for n in [n for n in names if n in stored]:
            ours = candidates(model, args.images / n)
            theirs = [c for c in stored[n]["full"] if c["p"] >= FLOOR]
            a = [dict(size=c["rle"]["size"], counts=c["rle"]["counts"].encode()) for c in ours]
            b = [dict(size=c["rle"]["size"], counts=c["rle"]["counts"].encode()) for c in theirs]
            iou = np.asarray(mu.iou(a, b, [0] * len(b))).reshape(len(a), len(b)) if a and b else np.zeros((len(a), len(b)))
            ok = len(a) == len(b) and (len(a) == 0 or iou.max(1).min() == 1.0)
            print("PARITY", json.dumps(dict(photo=n, ours=len(a), stored=len(b), identical=bool(ok))), flush=True)
            if not ok:
                raise SystemExit("the lean pass does not reproduce cache.py's candidates")
        return
    done = set()
    if args.output.exists():
        done = {json.loads(line)["name"] for line in args.output.read_text().splitlines() if line.strip()}
    started = time.time()
    with args.output.open("a") as handle:
        for k, n in enumerate(names, start=1):
            if n in done:
                continue
            tick = time.time()
            handle.write(json.dumps(dict(name=n, full=candidates(model, args.images / n))) + "\n")
            handle.flush()
            print("PHOTO", json.dumps(dict(photo=n, done=k, of=len(names), seconds=round(time.time() - tick, 1),
                                           elapsed_hours=round((time.time() - started) / 3600, 2))), flush=True)
    print("DONE", json.dumps(dict(model=facts, photographs=len(names))), flush=True)


if __name__ == "__main__":
    main()
