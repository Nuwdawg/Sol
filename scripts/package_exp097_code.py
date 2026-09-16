"""Build the private Kaggle dataset that carries exp097's code to the RunPod pod.

The pod cannot receive files through the RunPod MCP tools, so its launcher pulls
this dataset with the Kaggle CLI and refuses to run unless the payload's SHA256
equals the value printed here. That value is placed in the pod's environment at
creation, so the code that runs is byte-for-byte the code that was reviewed.

Two details make that guarantee hold:
- The payload is a gzip-compressed tar named `sol_exp097_code.payload`, not
  `.tar.gz`. Kaggle auto-extracts recognised archives when it processes a dataset
  (observed 2026-09-15: the uploaded `.tar.gz` was listed as its extracted files),
  which would destroy the single file whose hash is checked.
- The build is reproducible. Tar member metadata and the gzip header timestamp
  are fixed, so the hash changes only when the code changes.

Usage (run the upload from inside the folder: Kaggle CLI 2.2.4 on Windows breaks on
a `-p` path containing forward slashes):
    python scripts/package_exp097_code.py
    cd kaggle/exp097_code_dataset
    kaggle datasets version -p . -m "..." -r skip -t      # the dataset already exists
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "cosmicwitness/sol-exp097-code"
TITLE = "sol exp097 code"
PAYLOAD = "sol_exp097_code.payload"
FILES = [
    "shared/__init__.py",
    "shared/data_split.py",
    "experiments/exp_002_yolo_seg/src/prepare_yolo.py",
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
    "experiments/exp_097_runpod_yolo/pod/bootstrap.sh",
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
    directory = ROOT / "kaggle" / "exp097_code_dataset"
    directory.mkdir(parents=True, exist_ok=True)
    stale = directory / "sol_exp097_code.tar.gz"
    if stale.exists():
        stale.unlink()   # Kaggle would auto-extract it into the dataset again
    (directory / PAYLOAD).write_bytes(payload)
    (directory / "manifest.json").write_text(json.dumps(
        dict(payload=PAYLOAD, payload_sha256=digest, files=manifest), indent=2, sort_keys=True))
    (directory / "dataset-metadata.json").write_text(json.dumps(
        dict(title=TITLE, id=DATASET_ID, licenses=[dict(name="CC0-1.0")]), indent=2))
    print(json.dumps(dict(directory=str(directory), payload=PAYLOAD, payload_bytes=len(payload),
                          SOL_CODE_SHA256=digest, files=len(FILES)), indent=2))


if __name__ == "__main__":
    main()
