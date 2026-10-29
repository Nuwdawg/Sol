"""Rebuild the final submission from the weights in this repository, stage by stage.

The final submission is exp169's rule with exp199's large-filament guard:

  1. weights  reassemble the three checkpoints from `weights/` and check their SHA256;
  2. m2f      Mask2Former (exp139 epoch 40) candidates on the 180 test photographs:
              the full frame, the seven other dihedral views and three half-stride shifts
              (`experiments/exp_147_candidate_cache`); GPU, about 40 min on two T4s;
  3. yolo     YOLOv8l (exp097) and YOLO11m (exp002) dihedral pools on validation and test
              (`experiments/exp_161_yolo_test_pool`); each validation pool must agree with the
              stored one before its test pool is used; GPU, about 15 min each;
  4. rule     exp169's vote (`experiments/exp_169_joint_submission`), which first replays its
              validation score from the stored validation predictions and checks that the
              Mask2Former test path rebuilds exp139's submitted file; CPU, about 17 min;
  5. guard    re-add the large Mask2Former filaments the vote deleted
              (`experiments/exp_199_swinb_votes/src/guard.py`); CPU, one minute;
  6. check    compare the result with the submitted file (`final_decision/results/`).

Every stage stops on a failed check; none of them changes a model.

    python reproduce/run.py [--data /path/to/MAGFiLO_1.0_Kaggle_2026] --work /path/to/work [--stages ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "reproduce" / "artefacts"
SUBMITTED = ROOT / "final_decision" / "results" / "submission_exp169_guarded.csv"
STAGES = ("weights", "m2f", "yolo", "rule", "guard", "check")
LIMIT = 0          # --limit N: smoke test of the GPU stages on N photographs (rule/guard/check need all 180)
YOLO = {"yolov8l": ("yolov8l_exp097_stage2_best.pt", ART / "exp098_yolov8l_val_pool.json"),
        "yolo11m": ("yolo11m_exp002_best.pt", ART / "exp002_yolo11m_val_pool.json")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def python(module: str, *arguments: str, env: dict | None = None) -> subprocess.Popen:
    """Run one of the repository's modules with the repository on the import path."""
    full = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", **(env or {}))
    return subprocess.Popen([sys.executable, "-u", "-m", module, *arguments], cwd=ROOT, env=full)


def wait(processes: list[subprocess.Popen], what: str) -> None:
    codes = [p.wait() for p in processes]
    if any(codes):
        raise SystemExit(f"{what} failed: exit codes {codes}")


def find_data() -> Path:
    """The competition folder under /kaggle/input, wherever Kaggle mounted it (the kernels' own lookup)."""
    found = next(Path("/kaggle/input").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"), None)
    if found is None:
        raise SystemExit("--data not given and no competition folder found under /kaggle/input")
    return found.parents[1]


def paths(data: Path) -> dict:
    annotations = data / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    found = dict(annotations=annotations, train=data / "train" / "train_images", test=data / "test" / "test_images")
    for name, path in found.items():
        if not path.exists():
            raise SystemExit(f"{name} not found at {path}")
    if sha256(annotations) != "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1":
        raise SystemExit("the training annotations differ from the competition's file")
    if len(list(found["test"].glob("*.jpeg"))) != 180:
        raise SystemExit("expected 180 test photographs")
    return found


def stage_weights(work: Path, _) -> None:
    sys.path.insert(0, str(ROOT / "weights"))
    from assemble import assemble
    assemble(work / "weights")


def kernel_readback(folder: Path) -> None:
    """Which deformable-attention kernel each Mask2Former shard ran, from its event log.

    exp151a and exp157 ran the hub build at snapshot a4431ff2... (torch28-cxx11-cu128). cache.py asks
    for the latest build and falls back to the PyTorch loop if it fails its check; either may move a
    mask by a pixel, which the later stages measure.
    """
    for log in sorted(folder.glob("e040_events_shard*.log")):
        lines = log.read_text().splitlines()
        hub = [json.loads(line.split(" ", 1)[1]) for line in lines if line.startswith("KERNEL ")]
        fallback = any(line.startswith("KERNEL_FALLBACK") for line in lines)
        source = "pytorch (fallback)" if fallback else "hub" if hub else "pytorch (no GPU)"
        same = bool(hub) and not fallback and "a4431ff2d5767c279f0ff2335a87a6828b77c7a0" in hub[0]["path"]
        print("M2F_KERNEL", json.dumps(dict(log=log.name, source=source, original_build=same)), flush=True)
        if not same:
            print(f"WARNING: {log.name} did not run the deformable-attention kernel of the original caches",
                  flush=True)


def stage_m2f(work: Path, data: dict) -> None:
    for stale in (work / "m2f").glob("e040_events_shard*.log"):   # cache.py appends to its logs
        stale.unlink()
    wait([python("experiments.exp_147_candidate_cache.src.cache",
                 "--weights", str(work / "weights" / "mask2former_exp139_e040.pt"),
                 "--annotations", str(data["annotations"]), "--train-images", str(data["train"]),
                 "--test-images", str(data["test"]), "--output", str(work / "m2f"),
                 "--no-val", "--test-modes", "full,d4,shift", "--tag", "e040",
                 *(["--limit", str(LIMIT)] if LIMIT else []))], "Mask2Former cache")
    cache = json.loads((work / "m2f" / "e040_test_cache.json").read_text())
    if cache["model"]["sha256"] != "7f39b5c8c9d45d62c202d6b5059d9b555027b633c13cad22bcee540afd0b6f87":
        raise SystemExit("the test cache was not made with exp139 epoch 40")
    kernel_readback(work / "m2f")
    print("M2F", json.dumps({mode: sum(len(e.get(mode, [])) for e in cache["photographs"].values())
                             for mode in ("full", "d4", "shift")}), flush=True)


def stage_yolo(work: Path, data: dict) -> None:
    import torch
    from experiments.exp_159_yolo_votes.src.votes import load_pool
    from experiments.exp_161_yolo_test_pool.src.pool import compare
    gpus = max(1, torch.cuda.device_count())
    for tag, (weights, stored) in YOLO.items():
        out = work / f"yolo_{tag}"
        wait([python("experiments.exp_161_yolo_test_pool.src.pool", "--weights", str(work / "weights" / weights),
                     "--annotations", str(data["annotations"]), "--train-images", str(data["train"]),
                     "--test-images", str(data["test"]), "--output", str(out),
                     "--shard", str(shard), "--shards", str(gpus), *(["--limit", str(LIMIT)] if LIMIT else []),
                     env={"CUDA_VISIBLE_DEVICES": str(shard)} if torch.cuda.is_available() else None)
              for shard in range(gpus)], f"{tag} pool")
        for split in ("val", "test"):
            merged = {}
            for shard in range(gpus):
                merged.update(json.loads((out / f"yolov8l_{split}_pool_shard{shard}.json").read_text()))
            (out / f"{split}_pool.json").write_text(json.dumps(merged))
        # The validation pool must agree with the stored one that exp169 was tuned on.
        ours = load_pool(out / "val_pool.json")
        theirs = load_pool(stored)
        if LIMIT:                                   # smoke test: only the photographs it covered
            theirs = {name: theirs[name] for name in ours if name in theirs}
        agreement = compare(ours, theirs)
        total = sum(len(v) for v in theirs.values())
        agreement.update(instances=total, missing_photographs=sorted(set(theirs) - set(ours))[:5])
        print("YOLO_GATE", tag, json.dumps(agreement), flush=True)
        # exp002's stored pool came from an older Ultralytics patch release on CPU; exp033 accepted
        # boundary pixels moving, not candidates appearing or vanishing. Allow at most 1% of them.
        if agreement["missing_photographs"] or agreement["count_differences"] > 0.01 * total:
            raise SystemExit(f"{tag}: the rebuilt validation pool differs from the stored one")


def stage_rule(work: Path, data: dict) -> None:
    wait([python("experiments.exp_169_joint_submission.src.joint_submit",
                 "--report", str(ART / "exp168_report.json"),
                 "--val-cache", str(ART / "exp147_val_cache.json"), "--val-shift", str(ART / "exp157_shift_val_cache.json"),
                 "--test-cache", str(work / "m2f" / "e040_test_cache.json"),
                 "--test-shift", str(work / "m2f" / "e040_test_cache.json"),
                 "--val-yolo", f"yolov8l={YOLO['yolov8l'][1]}", "--val-yolo", f"yolo11m={YOLO['yolo11m'][1]}",
                 "--test-yolo", f"yolov8l={work / 'yolo_yolov8l' / 'test_pool.json'}",
                 "--test-yolo", f"yolo11m={work / 'yolo_yolo11m' / 'test_pool.json'}",
                 "--base-csv", str(ART / "exp139_submission.csv"),
                 "--annotations", str(data["annotations"]), "--output", str(work / "rule"))], "exp169 rule")


def stage_guard(work: Path, _) -> None:
    wait([python("experiments.exp_199_swinb_votes.src.guard", "test",
                 "--base", str(work / "rule" / "parity_deployed.csv"),
                 "--final", str(work / "rule" / "submission_joint.csv"), "--output", str(work / "final"))],
         "guard")


def stage_check(work: Path, _) -> None:
    from experiments.exp_152_rule_and_emit.src.emit import compare_csv
    ours = work / "final" / "submission_guarded.csv"
    exact = sha256(ours) == sha256(SUBMITTED)
    report = dict(ours=sha256(ours).upper(), submitted=sha256(SUBMITTED).upper(), identical=exact)
    if not exact:
        report.update(compare_csv(ours, SUBMITTED))
    print("CHECK", json.dumps(report), flush=True)
    (work / "final" / "check.json").write_text(json.dumps(report, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="the competition folder MAGFiLO_1.0_Kaggle_2026 "
                                                  "(default: found under /kaggle/input)")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--limit", type=int, default=0, help="smoke test: first N photographs in m2f and yolo")
    args = parser.parse_args()
    global LIMIT
    LIMIT = args.limit
    if LIMIT and set(args.stages) & {"rule", "guard", "check"}:
        raise SystemExit("--limit is for the m2f and yolo stages only")
    args.work.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    data = paths(args.data or find_data())
    for name in STAGES:
        if name in args.stages:
            print("STAGE", name, flush=True)
            globals()[f"stage_{name}"](args.work, data)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
