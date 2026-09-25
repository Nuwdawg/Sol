"""Build the private Kaggle dataset that carries exp128's code to the RunPod pod.

Same contract as `package_exp097_code.py`, which is proven, and deliberately not an
edit to it - exp097 must stay reproducible from its own payload.

Two properties are load-bearing and are inherited unchanged:

- The payload is named `.payload`, not `.tar.gz`. Kaggle auto-extracts recognised
  archives when it processes a dataset (observed 2026-09-15), which would destroy
  the single file whose SHA256 the pod checks before running anything.
- The build is reproducible: fixed tar member metadata and a zeroed gzip header
  timestamp, so the hash changes only when the code changes.

What exp128 adds over exp097's file list:

- `experiments/exp_128_detector_retrain/src/preprocess.py` - limb flattening,
  histogram matching to a frozen reference, off-disk zeroing.
- `.../prepare_dataset.py` - the same YOLO build with one annotator pass per
  training photograph and optional preprocessing.
- `.../evaluate_recall.py` - the pre-registered measurement harness. It reports
  proposal recall at IoU 0.5 first and PQ last, and it reproduced the frozen
  baseline exactly (76.7 / 79.4 / 66.4, buckets 880/172/165/45/63, PQ 0.4558638687)
  before being trusted.
- `experiments/exp_033_inference/src/sweep.py` and `shared/utils.py`, which the
  harness imports and exp097's payload did not carry.

Usage:
    python scripts/package_exp128_code.py
    cd kaggle/exp128_code_dataset
    kaggle datasets create -p .      # first time
    kaggle datasets version -p . -m "..." -r skip -t     # afterwards
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "cosmicwitness/sol-exp128-code"
TITLE = "sol exp128 code"
PAYLOAD = "sol_exp128_code.payload"
FILES = [
    "shared/__init__.py",
    "shared/data_split.py",
    "shared/utils.py",
    "experiments/exp_002_yolo_seg/src/prepare_yolo.py",
    "experiments/exp_033_inference/__init__.py",
    "experiments/exp_033_inference/src/__init__.py",
    "experiments/exp_033_inference/src/sweep.py",
    "experiments/exp_097_runpod_yolo/__init__.py",
    "experiments/exp_097_runpod_yolo/src/__init__.py",
    "experiments/exp_097_runpod_yolo/src/events.py",
    "experiments/exp_097_runpod_yolo/src/convergence.py",
    "experiments/exp_097_runpod_yolo/src/telemetry.py",
    "experiments/exp_097_runpod_yolo/src/gpu_monitor.py",
    "experiments/exp_097_runpod_yolo/src/kaggle_sync.py",
    "experiments/exp_097_runpod_yolo/src/fetch_data.py",
    "experiments/exp_097_runpod_yolo/src/preflight.py",
    "experiments/exp_097_runpod_yolo/src/train.py",
    "experiments/exp_128_detector_retrain/__init__.py",
    "experiments/exp_128_detector_retrain/src/__init__.py",
    "experiments/exp_128_detector_retrain/src/preprocess.py",
    "experiments/exp_128_detector_retrain/src/prepare_dataset.py",
    "experiments/exp_128_detector_retrain/src/evaluate_recall.py",
    "experiments/exp_128_detector_retrain/pod/bootstrap.sh",
]


def build_payload() -> bytes:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for name in FILES:
            data = (ROOT / name).read_bytes()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            tar.addfile(info, io.BytesIO(data))
    compressed = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0) as stream:
        stream.write(archive.getvalue())
    return compressed.getvalue()


def main() -> None:
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    manifest = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
    payload = build_payload()
    if build_payload() != payload:
        raise SystemExit("payload build is not reproducible")
    digest = hashlib.sha256(payload).hexdigest()
    directory = ROOT / "kaggle" / "exp128_code_dataset"
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.tar.gz"):
        stale.unlink()   # Kaggle would auto-extract it into the dataset again
    (directory / PAYLOAD).write_bytes(payload)
    (directory / "manifest.json").write_text(json.dumps(
        dict(payload=PAYLOAD, payload_sha256=digest, files=manifest), indent=2, sort_keys=True))
    (directory / "dataset-metadata.json").write_text(json.dumps(
        dict(title=TITLE, id=DATASET_ID, licenses=[dict(name="CC0-1.0")]), indent=2))
    print(json.dumps(dict(directory=str(directory), payload=PAYLOAD,
                          payload_bytes=len(payload), SOL_CODE_SHA256=digest,
                          files=len(FILES)), indent=2))


if __name__ == "__main__":
    main()
