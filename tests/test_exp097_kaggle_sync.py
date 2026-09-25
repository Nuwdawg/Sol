"""Unit tests for exp097's Kaggle transfer helpers (no network: the CLI is stubbed)."""
from __future__ import annotations

import json
import subprocess

import pytest

from experiments.exp_097_runpod_yolo.src import kaggle_sync


def test_stage_writes_manifest_metadata_and_skips_missing(tmp_path):
    present = tmp_path / "best.pt"
    present.write_bytes(b"weights")
    manifest = kaggle_sync.stage({"best.pt": present, "absent.pt": tmp_path / "absent.pt"},
                                 tmp_path / "staging", "owner/sol-exp097-test", "sol exp097 test")
    assert list(manifest) == ["best.pt"] and manifest["best.pt"]["bytes"] == 7
    metadata = json.loads((tmp_path / "staging" / "dataset-metadata.json").read_text())
    assert metadata["id"] == "owner/sol-exp097-test"
    assert (tmp_path / "staging" / "manifest.json").exists()


def test_stage_rejects_nested_names_and_empty_uploads(tmp_path):
    source = tmp_path / "x.pt"
    source.write_bytes(b"1")
    with pytest.raises(ValueError):
        kaggle_sync.stage({"weights/x.pt": source}, tmp_path / "s", "o/d", "title here")
    with pytest.raises(RuntimeError):
        kaggle_sync.stage({"x.pt": tmp_path / "missing.pt"}, tmp_path / "s2", "o/d", "title here")


def fake(stdout="", code=0):
    return subprocess.CompletedProcess(args=[], returncode=code, stdout=stdout, stderr="")


def test_publish_creates_when_absent_then_versions_on_retry(tmp_path, monkeypatch):
    source = tmp_path / "last.pt"
    source.write_bytes(b"abc")
    calls = []

    def kaggle(arguments, timeout=3600):
        calls.append(arguments)
        if arguments[:2] == ["datasets", "status"]:
            return fake("ready") if len(calls) > 2 else fake(code=1)
        if arguments[1] == "create":
            return fake(code=1)          # e.g. transient failure after partial creation
        return fake("ok")

    monkeypatch.setattr(kaggle_sync, "kaggle", kaggle)
    monkeypatch.setattr(kaggle_sync.time, "sleep", lambda seconds: None)
    upload = kaggle_sync.publish({"last.pt": source}, "o/d", "title here", "msg", tmp_path / "stg")
    assert upload["action"] == "version" and upload["attempt"] == 2
    assert "--public" not in sum(calls, [])


def test_listed_files_accepts_camel_and_snake_sizes(monkeypatch):
    rows = [dict(name="best.pt", totalBytes=10), dict(name="last.pt", total_bytes="20"),
            dict(ref="owner/d/summary.json", size=None)]
    monkeypatch.setattr(kaggle_sync, "kaggle", lambda a, timeout=0: fake(json.dumps(rows)))
    assert kaggle_sync.listed_files("o/d") == {"best.pt": 10, "last.pt": 20, "summary.json": None}
