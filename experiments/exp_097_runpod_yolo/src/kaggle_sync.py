"""Publish training artifacts from the pod to a private Kaggle dataset, and verify.

Weights never touch the user's PC. The pod stages a complete, self-describing
snapshot (files plus a SHA256 manifest) and uploads it as a new version of one
private dataset. Kaggle kernels then consume that dataset through
`dataset_sources`, which is how evaluation and submission pick the weights up.

Every upload is a full snapshot rather than a delta, so the latest version is
always sufficient on its own, and a run that dies still leaves its most recent
consistent state behind.

Command syntax is Kaggle CLI 2.2.4, read from `kaggle datasets create -h` and
`kaggle datasets version -h`; the pod pins exactly that version. `create` is
private unless `--public` is passed, and it is never passed here. `-t` keeps
tabular files byte-identical instead of letting Kaggle convert them, so the
manifest hashes stay valid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from experiments.exp_097_runpod_yolo.src.events import emit

METADATA_FILE = "dataset-metadata.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kaggle(arguments: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(["kaggle", *arguments], capture_output=True, text=True,
                          timeout=timeout)


def dataset_status(dataset_id: str) -> str | None:
    """Plain-text status ('ready', 'pending', ...) or None when the dataset does not exist."""
    result = kaggle(["datasets", "status", dataset_id], timeout=120)
    text = result.stdout.strip()
    if result.returncode != 0 or not text:
        return None
    return text.splitlines()[-1].strip().lower()


def stage(files: dict[str, Path], directory: Path, dataset_id: str, title: str) -> dict:
    """Copy present artifacts into a clean staging folder and describe them."""
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    manifest = {}
    for name, source in sorted(files.items()):
        if "/" in name or "\\" in name:
            raise ValueError(f"artifact names must be flat file names: {name}")
        source = Path(source)
        if not source.is_file():
            continue
        target = directory / name
        shutil.copy2(source, target)
        manifest[name] = dict(bytes=target.stat().st_size, sha256=sha256(target))
    if not manifest:
        raise RuntimeError("nothing to upload: none of the artifacts exist")
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (directory / METADATA_FILE).write_text(json.dumps(
        dict(title=title, id=dataset_id, licenses=[dict(name="CC0-1.0")]), indent=2))
    return manifest


def publish(files: dict[str, Path], dataset_id: str, title: str, message: str,
            staging: Path, attempts: int = 4, waits: tuple[int, ...] = (30, 90, 240)) -> dict:
    manifest = stage(files, staging, dataset_id, title)
    started = time.time()
    errors = []
    for attempt in range(1, attempts + 1):
        # Re-check each attempt: a first `create` may still be processing when a
        # retry starts, and a retry must then become a `version`.
        exists = dataset_status(dataset_id) is not None
        command = (["datasets", "version", "-p", str(staging), "-m", message, "-r", "skip",
                    "-t", "-q"] if exists else
                   ["datasets", "create", "-p", str(staging), "-r", "skip", "-t", "-q"])
        result = kaggle(command)
        if result.returncode == 0:
            return dict(dataset=dataset_id, action=command[1], attempt=attempt,
                        files=manifest, seconds=round(time.time() - started, 1),
                        bytes=sum(meta["bytes"] for meta in manifest.values()))
        errors.append((result.stdout + result.stderr).strip()[-400:])
        if attempt < attempts:
            time.sleep(waits[min(attempt - 1, len(waits) - 1)])
    raise RuntimeError(f"kaggle upload failed after {attempts} attempts: {errors[-1]}")


def listed_files(dataset_id: str) -> dict[str, int | None]:
    result = kaggle(["datasets", "files", dataset_id, "--format", "json",
                     "--page-size", "200"], timeout=120)
    if result.returncode != 0:
        return {}
    try:
        rows = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    if isinstance(rows, dict):
        rows = rows.get("datasetFiles") or rows.get("files") or []
    listed = {}
    for row in rows if isinstance(rows, list) else []:
        name = row.get("name") or row.get("ref") or row.get("fileName")
        size = row.get("totalBytes", row.get("total_bytes", row.get("size")))
        try:
            size = int(size)
        except (TypeError, ValueError):
            size = None
        if name:
            listed[Path(str(name)).name] = size
    return listed


def verify(dataset_id: str, manifest: dict, timeout: float = 600, interval: float = 30) -> dict:
    """Poll until every staged file is listed with its size, or report what is missing."""
    deadline = time.time() + timeout
    status, listed, missing, mismatched = None, {}, list(manifest), []
    while True:
        status = dataset_status(dataset_id)
        listed = listed_files(dataset_id)
        missing = [name for name in manifest if name not in listed]
        mismatched = [name for name, meta in manifest.items()
                      if listed.get(name) is not None and listed[name] != meta["bytes"]]
        if status == "ready" and not missing and not mismatched:
            return dict(verified=True, status=status, files=len(manifest))
        if time.time() >= deadline:
            return dict(verified=False, status=status, missing=missing,
                        size_mismatch=mismatched, listed=sorted(listed))
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--file", action="append", default=[],
                        help="name=path, repeatable; missing paths are skipped")
    parser.add_argument("--verify-seconds", type=float, default=300)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()
    files = {}
    for item in args.file:
        name, _, path = item.partition("=")
        files[name] = Path(path)
    upload = publish(files, args.dataset, args.title, args.message, args.staging)
    upload["verification"] = verify(args.dataset, upload["files"], timeout=args.verify_seconds)
    emit("UPLOAD", dict(dataset=args.dataset, action=upload["action"], bytes=upload["bytes"],
                        files=len(upload["files"]), seconds=upload["seconds"],
                        verification=upload["verification"]), args.log)
    (args.staging.parent / f"{args.staging.name}_result.json").write_text(
        json.dumps(upload, indent=2))


if __name__ == "__main__":
    main()
