"""exp141: two free gains for exp139 - average the saved weights, and re-tune its own view fusion.

Stage A, weight average. Epoch 40 and epoch 50 sit on the same cosine path with a tiny
learning rate between them, so their average is a valid model ("model soup"). Mask2Former
has no batch-norm statistics, so nothing needs re-estimating. Each of epoch 40, epoch 50
and their average is scored with the deployment path: float32, single view, exp033 sweep.

Stage B, view fusion. exp139's eight-view (D4) score used exp119's rule (link 0.5, pixel
vote 0.5, score x (0.5 + 0.5 x support/8)) and LOST 0.003 against a single view. That rule
was tuned for YOLO pools. Here the eight views are cached raw, once, and the fusion is
swept coarse-then-refine (CLAUDE.md section 9) over:
  - views: all eight, or the four that need no 90-degree turn;
  - link IoU, the overlap at which two masks join a cluster;
  - pixel vote, the share of a cluster's masks a pixel needs;
  - support scaling: exp119's, none, or a square root.
Emission (confidence, erosion, minimum area) is swept inside each candidate rule, so the
comparison against the single-view 0.47052 is on one scoring path.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mu

from experiments.exp_033_inference.src.cached_tta import vote_rle
from experiments.exp_033_inference.src.sweep import evaluate_thresholds, records_for
from experiments.exp_139_mask2former.src.data import read_gray
from experiments.exp_139_mask2former.src.evaluate import cache_validation, event, load_checkpoint
from experiments.exp_139_mask2former.src.model import D4, encode, predict
from experiments.exp_139_mask2former.src.submit import sha256

CONFS = tuple(round(0.05 * i, 2) for i in range(1, 20))
EMISSION = [(grow, area) for grow in (0, -1) for area in (100, 300)]
FLIPS = [(0, False), (0, True), (2, False), (2, True)]      # identity, mirror, 180 turn, mirror+180
SINGLE_VIEW_REFERENCE = 0.470524998967835                    # exp139 epoch 40, float32


def sweep(cache: dict, records: dict) -> list[dict]:
    rows = []
    for grow, area in EMISSION:
        rows.extend(evaluate_thresholds(cache, records, CONFS, grow=grow, min_area=area))
    return sorted(rows, key=lambda row: -row["pq"])


def average_weights(paths: list[Path]) -> dict:
    states = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    averaged = {}
    for key, value in states[0]["model"].items():
        stack = torch.stack([state["model"][key].float() for state in states])
        averaged[key] = stack.mean(0).to(value.dtype)
    return dict(model=averaged, config=states[0]["config"], epoch=f"soup{'+'.join(str(s['epoch']) for s in states)}")


def raw_views(model, records: dict, images: Path, device: torch.device, floor: float = 0.02) -> dict:
    """Every dihedral view's candidates, back in the native frame, tagged with the view."""
    cache = {}
    for position, name in enumerate(sorted(records), start=1):
        gray = read_gray(images / name)
        entries = []
        for index, (turns, mirror) in enumerate(D4):
            view = np.ascontiguousarray(np.rot90(np.fliplr(gray) if mirror else gray, turns))
            for score, binary in predict(model, view, device, bf16=False, floor=floor):
                restored = np.rot90(binary, -turns)
                if mirror:
                    restored = np.fliplr(restored)
                entries.append(dict(view=index, score=score, rle=encode(np.ascontiguousarray(restored))))
        cache[name] = entries
        if position % 25 == 0:
            print(f"VIEWS {position}/{len(records)}", flush=True)
    return cache


def fuse_views(entries: list[dict], link: float, vote: float, denominator: int, support: str) -> list[dict]:
    """`cached_tta.fuse` with the support scaling and denominator made explicit."""
    if not entries:
        return []
    rles = [entry["rle"] for entry in entries]
    scores = [float(entry["score"]) for entry in entries]
    order = sorted(range(len(entries)), key=lambda index: -scores[index])
    ious = mu.iou(rles, rles, [0] * len(rles))
    used, output = set(), []
    for seed in order:
        if seed in used:
            continue
        members = [seed]
        used.add(seed)
        for other in order:
            if other not in used and ious[seed, other] >= link:
                members.append(other)
                used.add(other)
        binary = vote_rle([rles[index] for index in members], max(1, round(vote * len(members))))
        if mu.area(binary) < 40:
            continue
        mean = float(np.mean([scores[index] for index in members]))
        share = len(members) / denominator
        if support == "exp119":
            scaled = mean * (0.5 + 0.5 * share)
        elif support == "none":
            scaled = mean
        elif support == "sqrt":
            scaled = mean * min(1.0, share) ** 0.5
        else:
            raise ValueError(support)
        output.append(dict(score=scaled, rle=binary))
    return output


def fused_cache(views: dict, records: dict, subset: list[int], link: float, vote: float, support: str) -> dict:
    allowed = set(subset)
    return {name: fuse_views([entry for entry in views[name] if entry["view"] in allowed],
                             link, vote, len(allowed), support)
            for name in records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--msda", choices=("pytorch", "hub"), default="hub")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    log = args.output / "events.log"
    device = torch.device("cuda")
    manifest = json.loads((args.weights / "manifest.json").read_text())
    checkpoints = {}
    for epoch in (40, 50):
        path = args.weights / f"model_e{epoch:03d}.pt"
        if sha256(path) != manifest[path.name]["sha256"]:
            raise SystemExit(f"{path.name} differs from the pod's manifest")
        checkpoints[epoch] = path
    if args.msda != "pytorch":
        from experiments.exp_139_mask2former.src.fast_ops import install_deformable_kernel, verify_deformable_kernel
        install_deformable_kernel(args.msda)
        rows = verify_deformable_kernel(device)
        event("KERNEL", log, passed=all(row["passed"] for row in rows))
        if not all(row["passed"] for row in rows):
            install_deformable_kernel("pytorch")
            event("KERNEL_FALLBACK", log, source="pytorch")
    records, split = records_for(args.annotations)
    report = dict(split=split.summary(), reference=dict(exp139_single_view=SINGLE_VIEW_REFERENCE))

    # ------------------------------------------------------------- stage A ---
    soup_path = args.cache_dir / "model_soup.pt"
    torch.save(average_weights([checkpoints[40], checkpoints[50]]), soup_path)
    singles = {}
    for label, path in (("e040", checkpoints[40]), ("e050", checkpoints[50]), ("soup", soup_path)):
        model, _ = load_checkpoint(path, device)
        cache, seconds = cache_validation(model, records, args.train_images, device, bf16=False, floor=0.02)
        (args.cache_dir / f"{label}_single.json").write_text(json.dumps(cache))
        best = sweep(cache, records)[0]
        singles[label] = best
        event("SINGLE", log, model=label, best=best, seconds=round(seconds, 1))
        del model, cache
        torch.cuda.empty_cache()
    report["single_view"] = singles
    base = max(singles, key=lambda key: singles[key]["pq"])
    report["base"] = base

    # ------------------------------------------------------------- stage B ---
    path = soup_path if base == "soup" else checkpoints[int(base[1:])]
    model, _ = load_checkpoint(path, device)
    started = time.time()
    views = raw_views(model, records, args.train_images, device)
    (args.cache_dir / f"{base}_views.json").write_text(json.dumps(views))
    event("VIEWS_DONE", log, model=base, seconds=round(time.time() - started, 1),
          candidates=sum(len(v) for v in views.values()))
    del model
    torch.cuda.empty_cache()

    coarse = []
    for subset, name in ((list(range(8)), "eight"), ([D4.index(item) for item in FLIPS], "flips")):
        for link in (0.5, 0.7):
            for vote in (0.5, 0.75):
                for support in ("exp119", "none"):
                    cache = fused_cache(views, records, subset, link, vote, support)
                    best = sweep(cache, records)[0]
                    row = dict(views=name, link=link, vote=vote, support=support, best=best)
                    coarse.append(row)
                    event("FUSION", log, stage="coarse", **row)
    coarse.sort(key=lambda row: -row["best"]["pq"])
    report["coarse"] = coarse

    top = coarse[0]
    refined = []
    for link in sorted({round(top["link"] + delta, 2) for delta in (-0.1, 0.0, 0.1) if 0.3 <= top["link"] + delta <= 0.9}):
        for vote in sorted({round(top["vote"] + delta, 3) for delta in (-0.125, 0.0, 0.125)
                            if 0.25 <= top["vote"] + delta <= 0.875}):
            for support in ("exp119", "none", "sqrt"):
                if (link, vote, support) == (top["link"], top["vote"], top["support"]):
                    continue
                subset = list(range(8)) if top["views"] == "eight" else [D4.index(item) for item in FLIPS]
                cache = fused_cache(views, records, subset, link, vote, support)
                best = sweep(cache, records)[0]
                row = dict(views=top["views"], link=link, vote=vote, support=support, best=best)
                refined.append(row)
                event("FUSION", log, stage="refine", **row)
    refined.append(top)
    refined.sort(key=lambda row: -row["best"]["pq"])
    report["refined"] = refined
    report["conclusion"] = dict(best_single=singles[base], best_fusion=refined[0],
                                fusion_gain=refined[0]["best"]["pq"] - singles[base]["pq"],
                                soup_gain=singles["soup"]["pq"] - singles["e040"]["pq"])
    (args.output / "soup_tta_report.json").write_text(json.dumps(report, indent=2))
    event("CONCLUSION", log, **report["conclusion"])


if __name__ == "__main__":
    main()
