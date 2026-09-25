"""exp099 smoke test: exp097's pod code against Ultralytics 8.4.153 on a Kaggle T4.

Why this exists
---------------
Pod qsatuc7fdharyi passed its preflight and then crashed in `on_train_start` on
`len()` of the `_ImageCache` that Ultralytics 8.4.153 builds for a complete RAM
cache. The callbacks had been tested only against assumptions about that API
(CLAUDE.md sections 6 and 8). This script runs the pod code, from the payload
bytes the pod verifies, against the pinned library, through every path a pod run
can take:

- data: `fetch_data.verify`, `build_yolo` and `densest_subset` on the competition
  files;
- preflight: orchestrator and worker subprocesses, with one full validation batch
  inside the peak-memory measurement;
- run A: warmup, two plateau decays, the converged stop, the second
  `on_fit_epoch_end` that `final_eval` makes, stage snapshots, upload hand-offs
  and the summary;
- run B: an out-of-memory error injected into the first epoch through
  Ultralytics' own retry path, which halves the batch and rebuilds the optimizer,
  then the budget stop.

What differs from the pod, and why
----------------------------------
- 640 px, 240 training and 16 validation images, so a T4 finishes in minutes.
  The callbacks do not depend on image size.
- Plateau settings of patience 1 and min_delta 10, so both decays and the stop
  happen within five epochs.
- `publish` and `verify` are local stand-ins, because a kernel must not hold the
  Kaggle key. The stand-ins still stage every file through the real
  `kaggle_sync.stage`. The real upload path already ran on pod qsatuc7fdharyi.

Each check prints `SMOKE_CHECK {json}`. The last line is `SMOKE_RESULT {json}`.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

MODULE = "experiments.exp_097_runpod_yolo.src"
WORKING = Path("/kaggle/working")
WORK = WORKING / "smoke"
TAGS = {"DATA", "STAGE", "EPOCH", "FINAL_VALIDATION", "UPLOAD", "UPLOAD_ERROR", "ERROR",
        "SUMMARY", "PREFLIGHT", "PREFLIGHT_CANDIDATE"}
FAKE_DATASET = "cosmicwitness/exp099-smoke-never-uploaded"


class Checks:
    def __init__(self) -> None:
        self.results: list[dict] = []

    def check(self, name: str, passed, required: bool = True, **detail) -> bool:
        record = dict(name=name, passed=bool(passed), required=required, **detail)
        self.results.append(record)
        print("SMOKE_CHECK " + json.dumps(record, default=str), flush=True)
        return bool(passed)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def events(path: Path, tag: str) -> list[dict]:
    if not path.is_file():
        return []
    prefix = tag + " "
    return [json.loads(line[len(prefix):]) for line in
            path.read_text(encoding="utf-8").splitlines() if line.startswith(prefix)]


def close(value, target: float) -> bool:
    return value is not None and abs(value - target) <= 1e-9 + 1e-6 * target


def run_logged(command: list[str], log: Path, env: dict) -> int:
    """Run a command with its full output in `log`. Echo event lines, and the tail on failure."""
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   env=env, text=True, errors="replace", cwd=WORKING)
        for line in process.stdout:
            handle.write(line)
            if line.split(" ", 1)[0] in TAGS:
                print(line, end="", flush=True)
        code = process.wait()
    print("COMMAND " + json.dumps(dict(log=str(log.relative_to(WORK)), exit_code=code,
                                       seconds=round(time.time() - started, 1))), flush=True)
    if code != 0:
        text = log.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
        print("\n".join(line for line in text.splitlines()[-80:] if line.strip()), flush=True)
    return code


def smoke_dataset(yolo: Path, target: Path, train_count: int, val_count: int) -> Path:
    """The first images of each split of the pod-identical YOLO build, by sorted name."""
    if target.exists():
        shutil.rmtree(target)
    for split, count in (("train", train_count), ("val", val_count)):
        (target / "images" / split).mkdir(parents=True)
        (target / "labels" / split).mkdir(parents=True)
        for label in sorted((yolo / "labels" / split).glob("*.txt"))[:count]:
            image = yolo / "images" / split / f"{label.stem}.jpeg"
            (target / "images" / split / image.name).symlink_to(image.resolve())
            shutil.copy2(label, target / "labels" / split / label.name)
    (target / "data.yaml").write_text(
        f"path: {target.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n")
    return target / "data.yaml"


def loads_as_models(paths: list[Path], env: dict) -> tuple[bool, str]:
    if not paths:
        return False, "no snapshots"
    script = ("import sys\nfrom ultralytics import YOLO\n"
              "for path in sys.argv[1:]:\n"
              "    model = YOLO(path)\n"
              "    print('LOADED', path, model.task, next(model.model.parameters()).dtype, flush=True)\n")
    result = subprocess.run([sys.executable, "-c", script, *map(str, paths)], env=env,
                            capture_output=True, text=True)
    output = (result.stdout + result.stderr)[-600:]
    return result.returncode == 0 and result.stdout.count("LOADED") == len(paths), output


def train_wrapper(argv: list[str]) -> int:
    """`smoke.py train --uploads PATH [--inject-oom-at N] -- <train.py arguments>`."""
    split = argv.index("--")
    own, rest = argv[:split], argv[split + 1:]
    uploads = Path(own[own.index("--uploads") + 1])
    inject_at = int(own[own.index("--inject-oom-at") + 1]) if "--inject-oom-at" in own else 0

    from experiments.exp_097_runpod_yolo.src import train   # sets the allocator variable before torch
    from experiments.exp_097_runpod_yolo.src.kaggle_sync import stage

    def publish(files, dataset_id, title, message, staging, **options):
        started = time.time()
        manifest = stage(files, staging, dataset_id, title)
        uploads.parent.mkdir(parents=True, exist_ok=True)
        with uploads.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(message=message, files=sorted(manifest))) + "\n")
        return dict(dataset=dataset_id, action="version", attempt=1, files=manifest,
                    seconds=round(time.time() - started, 1),
                    bytes=sum(meta["bytes"] for meta in manifest.values()))

    def verify(dataset_id, manifest, **options):
        return dict(verified=True, status="ready", files=len(manifest))

    train.publish, train.verify = publish, verify
    if inject_at:
        from ultralytics.models.yolo.segment import SegmentationTrainer
        original = SegmentationTrainer.preprocess_batch
        calls = dict(count=0)

        def preprocess_batch(self, batch):
            calls["count"] += 1
            if calls["count"] == inject_at:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB (exp099 injection)")
            return original(self, batch)

        SegmentationTrainer.preprocess_batch = preprocess_batch
    sys.argv = ["train.py", *rest]
    return train.main()


def cleanup() -> None:
    """Keep logs and JSON as kernel output; weights, staging copies and symlink trees go."""
    for path in list(WORKING.rglob("*.pt")):
        path.unlink(missing_ok=True)
    directories = [WORK / "yolo_ds", WORK / "smoke_ds", WORK / "preflight_base",
                   *WORK.rglob("staging"), *(WORK / "preflight").glob("subset_b*")]
    for directory in directories:
        shutil.rmtree(directory, ignore_errors=True)


def main() -> int:
    checks = Checks()
    WORK.mkdir(parents=True, exist_ok=True)
    code_root = Path(__file__).resolve().parent
    env = dict(os.environ, PYTHONPATH=str(code_root), PYTHONUNBUFFERED="1",
               CUDA_VISIBLE_DEVICES="0")
    from experiments.exp_097_runpod_yolo.src import fetch_data

    # Data: the pod's verification and YOLO build, on the attached competition files.
    annotations, train_images, test_images = fetch_data.locate(Path("/kaggle/input"))
    verified = fetch_data.verify(annotations, train_images, test_images, 707, 180)
    yolo = fetch_data.build_yolo(annotations, train_images, WORK / "yolo_ds", 0.5)
    checks.check("data_build_matches_pod",
                 yolo["train_labels"] == 974 and yolo["val_labels"] == 180,
                 verified=verified, yolo=yolo)
    checks.check("data_instances_match_pod",
                 (yolo["train_instances"], yolo["val_instances"]) == (6874, 1325), required=False,
                 instances=(yolo["train_instances"], yolo["val_instances"]))
    base = WORK / "preflight_base"
    fetch_data.densest_subset(WORK / "yolo_ds", base, 16)
    data_yaml = smoke_dataset(WORK / "yolo_ds", WORK / "smoke_ds", 240, 16)

    weights = WORK / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-c", "from ultralytics import YOLO; YOLO('yolov8l-seg.pt')"],
                   cwd=weights, env=env, check=True)
    model = weights / "yolov8l-seg.pt"

    # Preflight: two candidates; each final validation must hold one full validation batch.
    preflight_dir = WORK / "preflight"
    code = run_logged([sys.executable, "-u", "-m", f"{MODULE}.preflight", "--model", str(model),
                       "--base", str(base), "--workdir", str(preflight_dir), "--workers", "4",
                       "--imgsz", "640", "--batches", "1,2", "--warmup-batches", "2",
                       "--measure-batches", "4", "--measure-images", "16",
                       "--train-images", "240", "--price-per-hour", "0.69", "--headroom", "0.92",
                       "--output", str(WORK / "preflight.json"),
                       "--log", str(WORK / "preflight_events.log")],
                      WORK / "preflight.log", env)
    report = read_json(WORK / "preflight.json")
    candidates = report.get("candidates", [])
    checks.check("preflight_completes",
                 code == 0 and (report.get("decision") or {}).get("ok")
                 and [c.get("batch") for c in candidates] == [1, 2]
                 and all(not c.get("oom") and c.get("images_per_second") for c in candidates),
                 exit_code=code, decision=report.get("decision"),
                 candidates=[{key: c.get(key) for key in (
                     "batch", "oom", "images_per_second", "peak_reserved_mib",
                     "gpu_utilization_mean", "error")} for c in candidates])
    validation_images = {batch: len(list(
        (preflight_dir / f"subset_b{batch}" / "images" / "val").glob("*.jpeg")))
        for batch in (1, 2)}
    checks.check("preflight_validation_batch", validation_images == {1: 2, 2: 4},
                 validation_images=validation_images)

    common = ["--data", str(data_yaml), "--model", str(model), "--imgsz", "640", "--batch", "2",
              "--workers", "4", "--epoch-ceiling", "30", "--project", str(WORK / "runs"),
              "--price-per-hour", "0.69", "--upload-dataset", FAKE_DATASET]
    wrapper = [sys.executable, "-u", str(Path(__file__).resolve()), "train"]

    # Run A: warmup, two decays, converged stop, final_eval, snapshots and uploads.
    run_a = WORK / "runA"
    art_a = run_a / "artifacts"
    code_a = run_logged([*wrapper, "--uploads", str(run_a / "uploads.jsonl"), "--", *common,
                         "--warmup-epochs", "1", "--patience", "1", "--min-delta", "10",
                         "--decay-factor", "0.2", "--max-decays", "2", "--smoothing", "1",
                         "--name", "runA", "--artifacts", str(art_a), "--budget-usd", "100",
                         "--pod-start-epoch", str(time.time()),
                         "--staging", str(run_a / "staging")],
                        run_a / "train.log", env)
    log_a = art_a / "events.log"
    start_a = next((e for e in events(log_a, "STAGE") if e.get("name") == "train_start"), {})
    cache_a = start_a.get("cache") or {}
    epochs_a = events(log_a, "EPOCH")
    records_a = read_jsonl(art_a / "epochs.jsonl")
    summary_a = read_json(art_a / "summary.json")
    uploads_a = read_jsonl(run_a / "uploads.jsonl")
    checks.check("A_exit_code_zero", code_a == 0 and not events(log_a, "ERROR"),
                 exit_code=code_a, errors=events(log_a, "ERROR"))
    checks.check("A_ram_cache_read_back",
                 cache_a.get("container") == "_ImageCache" and cache_a.get("mode") == "ram"
                 and cache_a.get("cached") == cache_a.get("samples") == 240, cache=cache_a)
    checks.check("A_pipeline_read_back",
                 start_a.get("optimizer") == "AdamW" and start_a.get("initial_lr") == [0.001]
                 and start_a.get("batch") == 2 and start_a.get("batches_per_epoch") == 120
                 and start_a.get("warmup_iterations") == 120, train_start=start_a)
    checks.check("A_convergence_decisions",
                 [(e.get("epoch"), e.get("action")) for e in epochs_a]
                 == [(1, "continue"), (2, "continue"), (3, "decay"), (4, "decay"), (5, "stop")],
                 decisions=[(e.get("epoch"), e.get("action"), e.get("reason")) for e in epochs_a])
    rates = [r.get("learning_rates") or [] for r in records_a]
    checks.check("A_learning_rates_follow_decays",
                 len(rates) == 5 and all(rates[index] and all(close(v, target) for v in rates[index])
                                         for index, target in ((1, 1e-3), (2, 1e-3), (3, 2e-4),
                                                               (4, 4e-5))),
                 learning_rates=rates)
    checks.check("A_losses_recorded",
                 len(records_a) == 5 and all(
                     len(r.get("losses") or {}) >= 3
                     and all(isinstance(v, float) for v in r["losses"].values()) for r in records_a),
                 losses=[r.get("losses") for r in records_a])
    checks.check("A_mask_metrics_recorded",
                 len(epochs_a) == 5 and all(e.get("mask_map50_95") is not None for e in epochs_a),
                 mask_map50_95=[e.get("mask_map50_95") for e in epochs_a])
    finals = events(log_a, "FINAL_VALIDATION")
    checks.check("A_final_eval_is_not_an_epoch",
                 len(finals) == 1 and len(epochs_a) == 5 and len(records_a) == 5,
                 final_validation=finals)
    checks.check("A_summary",
                 summary_a.get("stop_reason") == "converged" and summary_a.get("epochs_completed") == 5
                 and summary_a.get("stages") == 2 and summary_a.get("batch") == 2
                 and summary_a.get("batch_reductions") == [] and bool(summary_a.get("final_validation")),
                 summary={key: summary_a.get(key) for key in (
                     "stop_reason", "epochs_completed", "stages", "batch", "batch_reductions",
                     "ultralytics_best_fitness")})
    snapshots = sorted(art_a.glob("stage*_best.pt"))
    loaded, output = loads_as_models(snapshots, env)
    checks.check("A_stage_snapshots_load",
                 [p.name for p in snapshots] == ["stage1_best.pt", "stage2_best.pt"] and loaded,
                 snapshots=[p.name for p in snapshots], output=output)
    final_files = set(uploads_a[-1]["files"]) if uploads_a else set()
    required_files = {"best.pt", "last.pt", "stage1_best.pt", "stage2_best.pt", "summary.json",
                      "epochs.jsonl", "results.csv", "args.yaml", "resolved_args.json",
                      "gpu_samples.csv", "events.log"}
    checks.check("A_upload_handoffs",
                 [u["message"] for u in uploads_a]
                 == ["stage 1 ended at epoch 3", "stage 2 ended at epoch 4", "final (complete)"]
                 and required_files <= final_files,
                 messages=[u["message"] for u in uploads_a],
                 missing=sorted(required_files - final_files))
    final_upload = read_json(art_a / "final_upload.json")
    summaries = events(log_a, "SUMMARY")
    checks.check("A_final_upload_and_exit",
                 (final_upload.get("verification") or {}).get("verified") is True
                 and bool(summaries) and summaries[-1].get("exit_code") == 0,
                 verification=final_upload.get("verification"), summary_events=summaries)
    import yaml
    args_yaml = WORK / "runs" / "runA" / "args.yaml"
    resolved = yaml.safe_load(args_yaml.read_text()) if args_yaml.is_file() else {}
    wanted = dict(batch=2, imgsz=640, optimizer="AdamW", lr0=0.001, lrf=1.0, cache="ram",
                  mosaic=0.0, epochs=30, warmup_bias_lr=0.0)
    checks.check("A_args_yaml_read_back", {key: resolved.get(key) for key in wanted} == wanted,
                 args={key: resolved.get(key) for key in wanted})
    samples_path = art_a / "gpu_samples.csv"
    rows = list(csv.DictReader(samples_path.open())) if samples_path.is_file() else []
    utilization = max((float(row["utilization"]) for row in rows if row.get("utilization")),
                      default=0.0)
    checks.check("A_gpu_monitor_samples", bool(rows) and utilization > 0, required=False,
                 rows=len(rows), max_utilization=utilization)

    # Run B: injected first-epoch OOM through Ultralytics' retry path, then the budget stop.
    run_b = WORK / "runB"
    art_b = run_b / "artifacts"
    code_b = run_logged([*wrapper, "--uploads", str(run_b / "uploads.jsonl"),
                         "--inject-oom-at", "3", "--", *common, "--warmup-epochs", "1",
                         "--name", "runB", "--artifacts", str(art_b), "--budget-usd", "1.0",
                         "--pod-start-epoch", str(time.time() - 3600),
                         "--staging", str(run_b / "staging")],
                        run_b / "train.log", env)
    log_b = art_b / "events.log"
    reductions = [e for e in events(log_b, "STAGE") if e.get("name") == "batch_reduced"]
    reduction = reductions[0] if reductions else {}
    epochs_b = events(log_b, "EPOCH")
    summary_b = read_json(art_b / "summary.json")
    train_log_b = run_b / "train.log"
    retry_logged = train_log_b.is_file() and "Reducing to batch=1" in train_log_b.read_text(
        encoding="utf-8", errors="replace")
    checks.check("B_exit_code_zero", code_b == 0 and not events(log_b, "ERROR"),
                 exit_code=code_b, errors=events(log_b, "ERROR"))
    checks.check("B_ultralytics_retry_path_taken", retry_logged)
    checks.check("B_batch_reduction_reported",
                 len(reductions) == 1 and reduction.get("previous_batch") == 2
                 and reduction.get("batch") == 1 and reduction.get("batches_per_epoch") == 240
                 and reduction.get("warmup_iterations") == 240
                 and reduction.get("optimizer") == "AdamW"
                 and (reduction.get("cache") or {}).get("cached") == 240,
                 reductions=reductions)
    checks.check("B_budget_stop",
                 [e.get("stop_reason") for e in epochs_b] == ["budget"]
                 and summary_b.get("stop_reason") == "budget"
                 and summary_b.get("epochs_completed") == 1 and summary_b.get("batch") == 1
                 and len(summary_b.get("batch_reductions") or []) == 1,
                 epochs=[(e.get("epoch"), e.get("action"), e.get("stop_reason")) for e in epochs_b],
                 summary={key: summary_b.get(key) for key in (
                     "stop_reason", "epochs_completed", "batch", "batch_reductions")})
    checks.check("B_final_upload_only",
                 [u["message"] for u in read_jsonl(run_b / "uploads.jsonl")] == ["final (complete)"],
                 messages=[u["message"] for u in read_jsonl(run_b / "uploads.jsonl")])

    cleanup()
    failed = [c["name"] for c in checks.results if c["required"] and not c["passed"]]
    warnings = [c["name"] for c in checks.results if not c["required"] and not c["passed"]]
    print("SMOKE_RESULT " + json.dumps(dict(passed=not failed, failed=failed, warnings=warnings,
                                            checks=len(checks.results))), flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        raise SystemExit(train_wrapper(sys.argv[2:]))
    raise SystemExit(main())
