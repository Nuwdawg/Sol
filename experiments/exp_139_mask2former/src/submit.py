"""exp139: validate the trained model on a free Kaggle GPU, then write the test submission.

Nothing here is submitted to the competition; the CSV waits for the user.

1. Checkpoint: the epoch with the highest single-view validation PQ in the pod's sweep
   reports. Its SHA256 must equal the pod's upload manifest.
2. Replay: single-view validation again on this GPU in float32 (the pod used bf16). The
   deployment path must reproduce the validation path (CLAUDE.md section 5), so the best
   PQ must agree with the pod's within `--replay-tolerance`.
3. D4: the same checkpoint with eight dihedral views and exp033's vote fusion, same sweep.
4. Choice: the mode and emission rule (confidence, erosion, minimum area) with the highest
   validation PQ.
5. Test: the chosen mode and rule, emitted exactly as the sweep emits them, written with
   emptiness and overlap checks.
6. Review sheet: native-resolution crops of validation photographs with ground truth
   (green) and the chosen emission (red), for the human check CLAUDE.md requires before
   any submission.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import encode, records_for
from experiments.exp_139_mask2former.src.data import NATIVE, read_gray
from experiments.exp_139_mask2former.src.evaluate import cache_validation, event, load_checkpoint, sweep
from shared.utils import paint_panoptic

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def emit(entries: list[dict], conf: float, grow: int, min_area: int) -> list[dict]:
    """exp033 sweep's emission, operation for operation, so the validated rule deploys unchanged."""
    candidates = []
    for entry in entries:
        if entry["score"] < conf:
            continue
        mask = mu.decode(entry["rle"])
        if grow < 0:
            mask = cv2.erode(mask, KERNEL, iterations=-grow)
        elif grow != 0:
            raise ValueError(f"only unaltered and eroded masks are screened: {grow}")
        candidates.append((entry["score"], mask))
    return [encode(mask) for _, mask, _ in paint_panoptic(candidates, min_area=min_area)]


def choose_checkpoint(weights: Path) -> tuple[Path, dict]:
    manifest = json.loads((weights / "manifest.json").read_text())
    reports = {}
    for path in sorted(weights.glob("e[0-9][0-9][0-9]_sweep.json")):
        reports[int(path.name[1:4])] = json.loads(path.read_text())
    available = {epoch: report for epoch, report in reports.items()
                 if (weights / f"model_e{epoch:03d}.pt").is_file()}
    if not available:
        raise SystemExit(f"no checkpoint with a sweep report in {weights}: reports {sorted(reports)}")
    epoch = max(available, key=lambda e: available[e]["best"]["pq"])
    checkpoint = weights / f"model_e{epoch:03d}.pt"
    if sha256(checkpoint) != manifest[checkpoint.name]["sha256"]:
        raise SystemExit(f"{checkpoint.name} differs from the pod's manifest")
    curve = {e: round(r["best"]["pq"], 5) for e, r in sorted(reports.items())}
    return checkpoint, dict(epoch=epoch, pod_best=available[epoch]["best"], curve=curve)


def review_sheet(records: dict, images: Path, cache: dict, rule: dict, output: Path,
                 photographs: int = 6, tile: int = 512) -> None:
    names = sorted(records)
    chosen = [names[i * len(names) // photographs] for i in range(photographs)]
    tiles = []
    for name in chosen:
        gray = read_gray(images / name)
        truth_rles = records[name][0][1]
        predicted = emit(cache[name], rule["conf"], rule["grow"], rule["min_area"])
        truth = [mu.decode(rle) for rle in truth_rles]
        centres = []
        for mask in truth[:2]:
            ys, xs = np.nonzero(mask)
            centres.append((int(xs.mean()), int(ys.mean())))
        for cx, cy in centres or [(NATIVE // 2, NATIVE // 2)]:
            x0 = int(np.clip(cx - tile // 2, 0, NATIVE - tile))
            y0 = int(np.clip(cy - tile // 2, 0, NATIVE - tile))
            canvas = cv2.cvtColor(gray[y0:y0 + tile, x0:x0 + tile], cv2.COLOR_GRAY2BGR)
            for masks, colour in ((truth, (0, 200, 0)), ([mu.decode(r) for r in predicted], (0, 0, 255))):
                for mask in masks:
                    crop = np.ascontiguousarray(mask[y0:y0 + tile, x0:x0 + tile])
                    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(canvas, contours, -1, colour, 1)
            cv2.putText(canvas, f"{Path(name).stem} x{x0} y{y0}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1, cv2.LINE_AA)
            tiles.append(canvas)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(output), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])   # small: slow link at home


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True, help="the pod's weights dataset folder")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--test-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--msda", choices=("pytorch", "hub"), default="hub")
    parser.add_argument("--replay-tolerance", type=float, default=0.005)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--skip-d4", action="store_true")
    parser.add_argument("--device", default="cuda", help="cpu only for smoke tests")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    log = args.output / "events.log"
    device = torch.device(args.device)

    checkpoint, chosen = choose_checkpoint(args.weights)
    event("CHECKPOINT", log, name=checkpoint.name, **chosen)
    if args.msda != "pytorch":
        from experiments.exp_139_mask2former.src.fast_ops import install_deformable_kernel, verify_deformable_kernel
        kernel = install_deformable_kernel(args.msda)
        rows = verify_deformable_kernel(device)
        event("KERNEL", log, **kernel, passed=all(row["passed"] for row in rows))
        if not all(row["passed"] for row in rows):
            install_deformable_kernel("pytorch")
            event("KERNEL_FALLBACK", log, source="pytorch")
    model, state = load_checkpoint(checkpoint, device)
    records, split = records_for(args.annotations)

    results = {}
    for mode in (["single"] if args.skip_d4 else ["single", "d4"]):
        started = time.time()
        cache, seconds = cache_validation(model, records, args.train_images, device, bf16=False,
                                          d4=mode == "d4", floor=0.02)
        cache_path = args.output / f"validation_{mode}_cache.json"
        cache_path.write_text(json.dumps(cache))
        rows = sweep(cache_path, args.annotations, args.processes)
        results[mode] = dict(best=rows[0], inference_seconds=round(seconds, 1),
                             total_seconds=round(time.time() - started, 1), cache=cache)
        (args.output / f"validation_{mode}_sweep.json").write_text(json.dumps(dict(rows=rows), indent=2))
        event("VALIDATION", log, mode=mode, best=rows[0], inference_seconds=round(seconds, 1))

    replay = results["single"]["best"]["pq"]
    difference = abs(replay - chosen["pod_best"]["pq"])
    event("REPLAY", log, pod=chosen["pod_best"]["pq"], here=replay, difference=difference,
          tolerance=args.replay_tolerance)
    if difference > args.replay_tolerance:
        raise SystemExit(f"single-view replay {replay:.5f} differs from the pod's "
                         f"{chosen['pod_best']['pq']:.5f} by more than {args.replay_tolerance}")

    mode = max(results, key=lambda m: results[m]["best"]["pq"])
    rule = results[mode]["best"]
    event("CHOICE", log, mode=mode, rule=rule, alternatives={m: r["best"]["pq"] for m, r in results.items()})
    review_sheet(records, args.train_images, results[mode]["cache"], rule, args.output / "review_sheet.jpg")

    test = sorted(args.test_images.glob("*.jpeg"))
    if len(test) != 180:
        raise SystemExit(f"expected 180 test photographs, found {len(test)}")
    destination = args.output / "submission.csv"
    written, contributing = 0, set()
    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filament_id", "segmentation_rle"])
        from experiments.exp_139_mask2former.src.model import predict_photograph
        for position, path in enumerate(test, start=1):
            entries = predict_photograph(model, read_gray(path), device, False, d4=mode == "d4", floor=0.02)
            predicted = emit(entries, rule["conf"], rule["grow"], rule["min_area"])
            claimed = np.zeros((NATIVE, NATIVE), np.uint8)
            for index, rle in enumerate(predicted, start=1):
                mask = mu.decode(rle)
                if not mask.sum():
                    raise SystemExit(f"empty instance emitted for {path.stem}")
                if (claimed & mask).any():
                    raise SystemExit(f"overlapping instances emitted for {path.stem}")
                claimed |= mask
                writer.writerow([f"{path.stem}_{index}", rle["counts"]])
                written += 1
                contributing.add(path.stem)
            if position % 20 == 0:
                print(f"PROGRESS {position}/180 photographs, {written} rows", flush=True)
    summary = dict(status="complete", checkpoint=checkpoint.name, epoch=chosen["epoch"], curve=chosen["curve"],
                   mode=mode, rule=rule, validation={m: r["best"] for m, r in results.items()},
                   replay=dict(pod=chosen["pod_best"]["pq"], here=replay, difference=difference),
                   split=split.summary(), rows=written, contributing_images=len(contributing),
                   sha256=hashlib.sha256(destination.read_bytes()).hexdigest().upper())
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    for name in ("validation_single_cache.json", "validation_d4_cache.json"):
        (args.output / name).unlink(missing_ok=True)   # large; the sweeps keep the numbers
    event("SUMMARY", log, **{k: v for k, v in summary.items() if k != "curve"})


if __name__ == "__main__":
    main()
