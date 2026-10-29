"""exp147: one GPU pass that caches everything the CPU experiments after it need.

exp139 epoch 40 (public 0.41) is the best model. Its candidates were never kept: every
earlier kernel scored a sweep and deleted the cache. Rule, fusion and verifier ideas
then each cost a GPU run. This script runs the model once per photograph in several
views and stores every candidate with the evidence that later rules may use.

Modes per photograph (float32, native pixels, the hub deformable-attention kernel):

- `full`: the whole 2048 frame, the deployed view. Floor 0.02 on the class
  probability, exactly `model.predict`. Each candidate keeps the score the deployed
  rule uses (class probability x mean mask probability inside the logit-0 mask), the
  two factors, the per-layer class probabilities of the query, its final decoder
  embedding, and the mask thresholded at logits -1, 0 and +1.
- `d4`: the seven non-identity dihedral views, each candidate mapped back to the
  native frame. Floor 0.05.
- `tiles`: nine native 1024 crops at stride 512, the extent 75% of training batches
  had. Candidates are pasted into the native frame. Floor 0.05.
- `scale`: the whole frame resampled to 2560 (validation only). The mask logits are
  resampled straight to 2048. Floor 0.05.

A parity gate re-scores the `full` candidates with exp139's rule (conf 0.85, no
erosion, area 100) and must reproduce exp145's 0.4705249990 before anything is
written as final.

    python -m experiments.exp_147_candidate_cache.src.cache --weights <exp139 weights dir>
        --annotations ... --train-images ... --test-images ... --output /kaggle/working
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import evaluate_thresholds, records_for
from experiments.exp_139_mask2former.src.data import NATIVE, gpu_pixels, read_gray
from experiments.exp_139_mask2former.src.model import D4, build_model

CHECKPOINT = "model_e040.pt"
PARITY_PQ = 0.4705249990          # exp145: exp139 epoch 40, single view, float32, conf 0.85, area 100
PARITY_TOLERANCE = 5e-4
TILE, STRIDE = 1024, 512
SCALE = 2560
MIN_PIXELS = 40                   # model.predict's floor on mask area


def event(tag: str, log: Path | None = None, **payload) -> None:
    line = f"{tag} {json.dumps(payload, sort_keys=True, default=str)}"
    print(line, flush=True)
    if log is not None:
        with Path(log).open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode(binary: np.ndarray) -> dict:
    rle = mu.encode(np.asfortranarray(binary.astype(np.uint8)))
    return dict(size=[int(v) for v in rle["size"]], counts=rle["counts"].decode("ascii"))


def load_model(weights: Path, device: torch.device, name: str = CHECKPOINT):
    """`weights` is the exp139 weights folder (checked against its manifest) or a checkpoint file."""
    if weights.is_file():
        path, name = weights, weights.name
        digest = sha256(path)
    else:
        manifest = json.loads((weights / "manifest.json").read_text())
        path = weights / name
        digest = sha256(path)
        if digest != manifest[name]["sha256"]:
            raise SystemExit(f"{name} differs from the pod's manifest")
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(config=state["config"])
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval(), dict(checkpoint=name, sha256=digest, epoch=state.get("epoch"))


class Embeddings:
    """The class head's input on its last call: the final decoder layer's query embeddings."""

    def __init__(self, model):
        self.value = None
        model.class_predictor.register_forward_hook(self._keep)

    def _keep(self, module, inputs, output):
        self.value = inputs[0]


@torch.inference_mode()
def run_view(model, gray: np.ndarray, device: torch.device, floor: float, out_size: int,
             rich: bool = False, hook: Embeddings | None = None, chunk: int = 8) -> list[dict]:
    """Candidates of one view, masks at `out_size` in that view's frame.

    With out_size equal to the input extent this is `model.predict` operation for
    operation (same upsampling, logit-0 binarisation, score and area floor).
    """
    pixels = gpu_pixels(torch.from_numpy(np.ascontiguousarray(gray))[None], None, device)
    output = model(pixel_values=pixels, output_auxiliary_logits=rich)
    probabilities = output.class_queries_logits[0].float().softmax(-1)[:, 0]
    logits = output.masks_queries_logits[0]
    keep = torch.nonzero(probabilities >= floor).flatten()
    layers = None
    if rich:
        layers = torch.stack([aux["class_queries_logits"][0].float().softmax(-1)[:, 0]
                              for aux in output.auxiliary_logits] + [probabilities])   # [L, Q]
    embeddings = hook.value[0].float() if (rich and hook is not None and hook.value is not None) else None
    candidates = []
    for start in range(0, len(keep), chunk):
        queries = keep[start:start + chunk]
        upsampled = F.interpolate(logits[queries][None].float(), size=(out_size, out_size),
                                  mode="bilinear", align_corners=False)[0]
        binary = upsampled > 0
        area = binary.flatten(1).sum(1)
        inside = (upsampled.sigmoid() * binary).flatten(1).sum(1) / area.clamp(min=1)
        scores = probabilities[queries] * inside
        low = (upsampled > -1).to(torch.uint8).cpu().numpy() if rich else None
        high = (upsampled > 1).to(torch.uint8).cpu().numpy() if rich else None
        binary = binary.to(torch.uint8).cpu().numpy()
        for j, query in enumerate(queries.tolist()):
            if int(area[j]) < MIN_PIXELS:
                continue
            entry = dict(query=query, score=float(scores[j]), p=float(probabilities[query]),
                         m=float(inside[j]), area=int(area[j]), mask=binary[j])
            if rich:
                entry.update(low=low[j], high=high[j], layers=[round(float(v), 5) for v in layers[:, query]])
                if embeddings is not None:
                    entry["embedding"] = [round(float(v), 4) for v in embeddings[query]]
            candidates.append(entry)
    return candidates


def serialise(entry: dict, **extra) -> dict:
    record = {k: v for k, v in entry.items() if k not in ("mask", "low", "high")}
    record["rle"] = encode(entry["mask"])
    if "low" in entry:
        record["rle_low"] = encode(entry["low"])
        record["rle_high"] = encode(entry["high"])
    record.update(extra)
    return record


def full_mode(model, gray, device, hook) -> list[dict]:
    return [serialise(e) for e in run_view(model, gray, device, 0.02, NATIVE, rich=True, hook=hook)]


def d4_mode(model, gray, device) -> list[dict]:
    entries = []
    for view, (k, mirror) in enumerate(D4):
        if view == 0:
            continue                                   # identity is the `full` mode
        image = np.ascontiguousarray(np.rot90(np.fliplr(gray) if mirror else gray, k))
        for entry in run_view(model, image, device, 0.05, NATIVE):
            restored = np.rot90(entry["mask"], -k)
            if mirror:
                restored = np.fliplr(restored)
            entry["mask"] = np.ascontiguousarray(restored)
            entries.append(serialise(entry, view=view))
    return entries


def tile_mode(model, gray, device) -> list[dict]:
    entries = []
    positions = range(0, NATIVE - TILE + 1, STRIDE)
    for y0 in positions:
        for x0 in positions:
            crop = np.ascontiguousarray(gray[y0:y0 + TILE, x0:x0 + TILE])
            for entry in run_view(model, crop, device, 0.05, TILE):
                small = entry["mask"]
                ys, xs = np.nonzero(small)
                edges = []
                if y0 > 0 and ys.min() == 0:
                    edges.append("top")
                if x0 > 0 and xs.min() == 0:
                    edges.append("left")
                if y0 + TILE < NATIVE and ys.max() == TILE - 1:
                    edges.append("bottom")
                if x0 + TILE < NATIVE and xs.max() == TILE - 1:
                    edges.append("right")
                canvas = np.zeros((NATIVE, NATIVE), np.uint8)
                canvas[y0:y0 + TILE, x0:x0 + TILE] = small
                entry["mask"] = canvas
                entries.append(serialise(entry, x0=x0, y0=y0, edges=edges))
    return entries


def scale_mode(model, gray, device) -> list[dict]:
    large = cv2.resize(gray, (SCALE, SCALE), interpolation=cv2.INTER_LINEAR)
    return [serialise(e) for e in run_view(model, large, device, 0.05, NATIVE)]


def worker(args) -> None:
    """One GPU, one shard of photographs, one JSON file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = args.output / f"{args.tag + '_' if args.tag else ''}events_shard{args.shard}.log"
    if device.type == "cuda" and args.msda == "hub":
        from experiments.exp_139_mask2former.src.fast_ops import (install_deformable_kernel,
                                                                  verify_deformable_kernel)
        kernel = install_deformable_kernel("hub")
        rows = verify_deformable_kernel(device)
        passed = all(row["passed"] for row in rows)
        event("KERNEL", log, shard=args.shard, passed=passed, **kernel)
        if not passed:
            install_deformable_kernel("pytorch")
            event("KERNEL_FALLBACK", log, shard=args.shard)
    model, facts = load_model(args.weights, device)
    hook = Embeddings(model)
    jobs = json.loads(Path(args.jobs).read_text())
    mine = jobs[args.shard::args.shards]
    started = time.time()
    cache, seconds = {}, {}
    for position, job in enumerate(mine, start=1):
        gray = read_gray(Path(job["path"]))
        result = {}
        for mode in job["modes"]:
            tick = time.time()
            if mode == "full":
                result[mode] = full_mode(model, gray, device, hook)
            elif mode == "d4":
                result[mode] = d4_mode(model, gray, device)
            elif mode == "tiles":
                result[mode] = tile_mode(model, gray, device)
            elif mode == "scale":
                result[mode] = scale_mode(model, gray, device)
            else:
                raise SystemExit(f"unknown mode {mode}")
            seconds[mode] = seconds.get(mode, 0.0) + time.time() - tick
        cache[job["name"]] = dict(split=job["split"], **result)
        if position % 10 == 0 or position == len(mine):
            event("PROGRESS", log, shard=args.shard, done=position, of=len(mine),
                  elapsed=round(time.time() - started, 1),
                  seconds={k: round(v, 1) for k, v in seconds.items()})
    target = args.output / f"{args.tag + '_' if args.tag else ''}cache_shard{args.shard}.json"
    target.write_text(json.dumps(dict(model=facts, photographs=cache)))
    event("SHARD_DONE", log, shard=args.shard, photographs=len(cache), path=str(target),
          seconds={k: round(v, 1) for k, v in seconds.items()})


def parity(cache: dict, annotations: Path) -> dict:
    records, _ = records_for(annotations)
    pool = {name: [dict(score=e["score"], rle=e["rle"]) for e in cache[name]["full"]] for name in records}
    return evaluate_thresholds(pool, records, (0.85,), grow=0, min_area=100)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--test-images", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-modes", default="full,d4,tiles,scale")
    parser.add_argument("--test-modes", default="full,d4,tiles")
    parser.add_argument("--msda", choices=("hub", "pytorch"), default="hub")
    parser.add_argument("--limit", type=int, default=0, help="first N photographs of each split (smoke test)")
    parser.add_argument("--gpus", type=int, default=-1, help="-1: all visible")
    parser.add_argument("--no-parity", dest="parity", action="store_false",
                        help="skip the exp139 parity gate (for any other checkpoint)")
    parser.add_argument("--tag", default="", help="prefix of the output cache files")
    parser.add_argument("--no-val", action="store_true", help="test photographs only")
    parser.add_argument("--shard", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--shards", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--jobs", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.shard >= 0:
        worker(args)
        return

    log = args.output / f"{args.tag + '_' if args.tag else ''}events.log"
    records, split = records_for(args.annotations)
    val_names = [] if args.no_val else sorted(records)
    test_names = sorted(p.name for p in args.test_images.glob("*.jpeg")) if args.test_images else []
    if args.limit:
        val_names, test_names = val_names[:args.limit], test_names[:args.limit]
    elif test_names and len(test_names) != 180:
        raise SystemExit(f"expected 180 test photographs, found {len(test_names)}")
    jobs = ([dict(name=n, split="val", path=str(args.train_images / n), modes=args.val_modes.split(","))
             for n in val_names]
            + [dict(name=n, split="test", path=str(args.test_images / n), modes=args.test_modes.split(","))
               for n in test_names])
    # Interleave the two splits so every shard sees both kinds of work.
    jobs.sort(key=lambda job: (len(job["modes"]), job["name"]))
    jobs_path = args.output / f"{args.tag + '_' if args.tag else ''}jobs.json"
    jobs_path.write_text(json.dumps(jobs))
    gpus = torch.cuda.device_count() if args.gpus < 0 else args.gpus
    shards = max(1, gpus)
    event("START", log, split=split.summary(), val=len(val_names), test=len(test_names), gpus=gpus,
          val_modes=args.val_modes, test_modes=args.test_modes)
    processes = []
    for shard in range(shards):
        env = dict(os.environ)
        if gpus > 0 and "CUDA_VISIBLE_DEVICES" not in os.environ:
            env["CUDA_VISIBLE_DEVICES"] = str(shard)
        command = [sys.executable, "-u", "-m", "experiments.exp_147_candidate_cache.src.cache",
                   "--weights", str(args.weights), "--annotations", str(args.annotations),
                   "--train-images", str(args.train_images), "--output", str(args.output),
                   "--msda", args.msda, "--shard", str(shard), "--shards", str(shards), "--jobs", str(jobs_path),
                   "--tag", args.tag]
        processes.append(subprocess.Popen(command, env=env))
    codes = [process.wait() for process in processes]
    if any(codes):
        raise SystemExit(f"a shard failed: exit codes {codes}")

    merged, model_facts = {}, None
    for shard in range(shards):
        part = json.loads((args.output / f"{args.tag + '_' if args.tag else ''}cache_shard{shard}.json").read_text())
        model_facts = part["model"]
        merged.update(part["photographs"])
    missing = [job["name"] for job in jobs if job["name"] not in merged]
    if missing:
        raise SystemExit(f"{len(missing)} photographs missing from the shards")
    for split_name in ("val", "test"):
        subset = {n: v for n, v in merged.items() if v["split"] == split_name}
        if subset:
            prefix = f"{args.tag}_" if args.tag else ""
            (args.output / f"{prefix}{split_name}_cache.json").write_text(json.dumps(dict(model=model_facts,
                                                                                          photographs=subset)))
    for shard in range(shards):
        (args.output / f"{args.tag + '_' if args.tag else ''}cache_shard{shard}.json").unlink()
    counts = {split_name: {mode: sum(len(v.get(mode, [])) for v in merged.values() if v["split"] == split_name)
                           for mode in ("full", "d4", "tiles", "scale")} for split_name in ("val", "test")}
    report = dict(model=model_facts, candidates=counts, photographs=dict(val=len(val_names), test=len(test_names)))
    if not args.limit and args.parity and val_names:
        row = parity({n: v for n, v in merged.items() if v["split"] == "val"}, args.annotations)
        report["parity"] = dict(row, expected=PARITY_PQ, difference=row["pq"] - PARITY_PQ)
        event("PARITY", log, **report["parity"])
        if abs(row["pq"] - PARITY_PQ) > PARITY_TOLERANCE:
            (args.output / f"{args.tag + '_' if args.tag else ''}summary.json").write_text(json.dumps(report, indent=2))
            raise SystemExit(f"parity failed: {row['pq']} against {PARITY_PQ}")
    (args.output / f"{args.tag + '_' if args.tag else ''}summary.json").write_text(json.dumps(report, indent=2))
    event("DONE", log, **{k: v for k, v in report.items() if k != "model"})


if __name__ == "__main__":
    main()
