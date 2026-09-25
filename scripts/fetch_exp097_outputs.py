"""Copy exp097's outputs from the private Kaggle dataset to the D drive, for redundancy.

User permission, 2026-09-15: the run's output data may be downloaded to the D
drive. This script downloads nothing else. The Kaggle dataset remains the source
of truth. Every downloaded file is checked against the SHA256 manifest that the
pod wrote before uploading, and the result is saved beside the files.

Run it only after the pod has reported its final upload.

Usage:
    python scripts/fetch_exp097_outputs.py [--dest D:/Sol/exp097_outputs]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

DATASET = "cosmicwitness/sol-exp097-yolov8l-2048"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=Path("D:/Sol/exp097_outputs"))
    parser.add_argument("--dataset", default=DATASET)
    args = parser.parse_args()

    drive = Path(args.dest.anchor)
    if not drive.exists():
        raise SystemExit(f"destination drive {drive} is not present")
    args.dest.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["kaggle", "datasets", "download", args.dataset, "--unzip", "-p", "."],
                            cwd=args.dest, capture_output=True, text=True,
                            env=dict(os.environ, PYTHONUTF8="1"))
    if result.returncode != 0:
        raise SystemExit(f"download failed: {(result.stdout + result.stderr)[-500:]}")

    manifest = json.loads((args.dest / "manifest.json").read_text())
    checks = {}
    for name, meta in sorted(manifest.items()):
        path = args.dest / name
        present = path.is_file()
        checks[name] = dict(
            present=present,
            size_ok=present and path.stat().st_size == meta["bytes"],
            sha256_ok=present and sha256(path) == meta["sha256"])
    verified = all(all(check.values()) for check in checks.values())
    report = dict(dataset=args.dataset, destination=str(args.dest), verified=verified,
                  files=len(checks),
                  bytes=sum(meta["bytes"] for meta in manifest.values()), checks=checks)
    (args.dest / "download_verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))
    if not verified:
        failed = [name for name, check in checks.items() if not all(check.values())]
        raise SystemExit(f"verification failed for: {failed}")


if __name__ == "__main__":
    main()
