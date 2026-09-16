"""Deployment-consistency tests for exp097 (CLAUDE.md section 5: deploy what was validated).

These fail when the code that would run on the pod is not the code that was
tested and uploaded: a non-reproducible payload, sources edited after the upload,
or a pod request pointing at a stale hash or carrying a real key.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "kaggle" / "exp097_code_dataset"
POD_DIR = ROOT / "experiments" / "exp_097_runpod_yolo" / "pod"


def packager():
    spec = importlib.util.spec_from_file_location(
        "package_exp097_code", ROOT / "scripts" / "package_exp097_code.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_payload_is_reproducible_and_unpacks_to_the_sources():
    module = packager()
    first, second = module.build_payload(), module.build_payload()
    assert first == second
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as tar:
        assert tar.getnames() == module.FILES
        for name in module.FILES:
            assert tar.extractfile(name).read() == (ROOT / name).read_bytes(), name
        assert tar.getmember("experiments/exp_097_runpod_yolo/pod/bootstrap.sh").mode == 0o755


def test_uploaded_payload_matches_current_sources():
    manifest_path = DATASET_DIR / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("code dataset has not been built")
    manifest = json.loads(manifest_path.read_text())
    payload = packager().build_payload()
    assert hashlib.sha256(payload).hexdigest() == manifest["payload_sha256"], (
        "sources changed after the payload was built: rebuild, re-upload, regenerate the pod request")
    assert (DATASET_DIR / manifest["payload"]).read_bytes() == payload
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


def test_pod_request_points_at_the_payload_and_holds_no_key():
    request_path = POD_DIR / "pod_request.json"
    if not request_path.exists():
        pytest.skip("pod request has not been generated")
    request_text = request_path.read_text()
    body = json.loads(request_text)["body"]
    manifest = json.loads((DATASET_DIR / "manifest.json").read_text())
    assert body["env"]["SOL_CODE_SHA256"] == manifest["payload_sha256"]
    assert body["env"]["KAGGLE_KEY"] == "__INSERT_KAGGLE_KEY_AT_LAUNCH__"
    key_file = Path.home() / ".kaggle" / "kaggle.json"
    if key_file.exists():
        assert json.loads(key_file.read_text())["key"] not in request_text
    launcher = (POD_DIR / "launcher.sh").read_text()
    assert body["args"] == "bash -lc '" + launcher.strip() + "'"
    assert "sol_exp097_code.payload" in launcher and ".tar.gz" not in launcher
    assert body["gpu"]["id"].startswith("NVIDIA ") and body["gpu"]["minCudaVersion"] == "12.8"
    assert body["env"]["PIP_BREAK_SYSTEM_PACKAGES"] == "1"
    assert "--break-system-packages" in launcher
    assert body["mounts"]["persistent"]["path"] == "/workspace"
    # The user's cap is 9.95 in total. A relaunch inherits the spend of earlier
    # attempts, so the pod's own budget is that cap minus what RunPod already billed.
    assert 0 < float(body["env"]["SOL_BUDGET_USD"]) <= 9.95
