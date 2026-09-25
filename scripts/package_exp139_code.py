"""Build the private Kaggle dataset that carries exp139's code to the RunPod pod.

Same contract as `package_exp128_code.py`, which shipped exp128's pod code:

- The payload is named `.payload`, not `.tar.gz`, because Kaggle auto-extracts
  recognised archives and would destroy the file whose SHA256 the pod checks.
- The build is reproducible: fixed tar metadata and a zeroed gzip timestamp, so the
  hash changes only when the code changes.

Files: the exp139 trainer, model, data and evaluator; exp033's scorer and D4 fusion,
which the evaluator calls unchanged; exp097's data fetch, event and Kaggle upload
helpers, which have run on a pod before.

Usage:
    python scripts/package_exp139_code.py
    cd kaggle/exp139_code_dataset
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
DATASET_ID = "cosmicwitness/sol-exp139-code"
TITLE = "sol exp139 code"
PAYLOAD = "sol_exp139_code.payload"
FILES = [
    "shared/__init__.py",
    "shared/data_split.py",
    "shared/utils.py",
    "experiments/exp_033_inference/__init__.py",
    "experiments/exp_033_inference/src/__init__.py",
    "experiments/exp_033_inference/src/sweep.py",
    "experiments/exp_033_inference/src/cached_tta.py",
    "experiments/exp_097_runpod_yolo/__init__.py",
    "experiments/exp_097_runpod_yolo/src/__init__.py",
    "experiments/exp_097_runpod_yolo/src/events.py",
    "experiments/exp_097_runpod_yolo/src/kaggle_sync.py",
    "experiments/exp_097_runpod_yolo/src/fetch_data.py",
    "experiments/exp_139_mask2former/__init__.py",
    "experiments/exp_139_mask2former/src/__init__.py",
    "experiments/exp_139_mask2former/src/data.py",
    "experiments/exp_139_mask2former/src/model.py",
    "experiments/exp_139_mask2former/src/evaluate.py",
    "experiments/exp_139_mask2former/src/emit_test.py",
    "experiments/exp_139_mask2former/src/fast_ops.py",
    "experiments/exp_139_mask2former/src/submit.py",
    "experiments/exp_139_mask2former/src/train.py",
    "experiments/exp_139_mask2former/pod/bootstrap.sh",
]


def build_payload() -> bytes:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for name in FILES:
            data = (ROOT / name).read_bytes()
            if name.endswith(".sh") and b"\r\n" in data:
                raise SystemExit(f"{name} has CRLF line endings; bash on the pod would fail")
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
    directory = ROOT / "kaggle" / "exp139_code_dataset"
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.tar.gz"):
        stale.unlink()
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
