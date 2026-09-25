"""Exercise the actual training CLI, saving and reloading an EMA checkpoint.

The tiny synthetic dataset tests the lifecycle only. It is not solar validation.
"""
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest
from ultralytics import YOLO


@pytest.mark.parametrize('fp32_validation,criterion_name', [(False,'chunked'),(True,'chunked'),(True,'roi')])
def test_training_cli_lifecycle(tmp_path, fp32_validation, criterion_name):
    dataset = tmp_path / "dataset"
    for fold in ("train", "val"):
        (dataset / "images" / fold).mkdir(parents=True)
        (dataset / "labels" / fold).mkdir(parents=True)
        for index in range(2):
            image = np.full((64, 64, 3), 180, dtype=np.uint8)
            image[8:56, 24:40] = 40
            cv2.imwrite(str(dataset / "images" / fold / f"{index}.jpg"), image)
            (dataset / "labels" / fold / f"{index}.txt").write_text(
                "0 0.375 0.125 0.625 0.125 0.625 0.875 0.375 0.875\n")
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(f"path: {dataset.as_posix()}\ntrain: images/train\nval: images/val\nnames: [filament]\n")
    seed = tmp_path / "random_seed.pt"
    YOLO("yolo11n-seg.yaml").save(seed)
    command = [sys.executable, "-m", "experiments.exp_032_chunked.src.train",
               "--data", str(data_yaml), "--weights", str(seed),
               "--project", str(tmp_path / "runs"), "--name", "smoke",
               "--device", "cpu", "--imgsz", "64", "--batch", "1",
               "--epochs", "1", "--chunk-size", "2", "--workers", "0"]
    if fp32_validation:
        command.append('--fp32-validation')
    command.extend(['--criterion',criterion_name])
    completed = subprocess.run(command, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    run = tmp_path / "runs" / "smoke"
    report = json.loads((run / "resolved_experiment.json").read_text())
    assert report["criterion"] == ('ChunkedSegmentationLoss' if criterion_name=='chunked' else 'ROISegmentationLoss')
    assert report["resolved"]["mask_ratio"] == 1
    assert report["resolved"]["imgsz"] == 64
    assert report['resolved']['warmup_bias_lr'] == 0.0
    assert report['validation_precision'] == ('fp32' if fp32_validation else 'upstream')
    assert (run / "weights" / "last.pt").is_file()
    restored = YOLO(run / "weights" / "last.pt")
    predictions = restored.predict(np.full((64, 64, 3), 180, np.uint8), imgsz=64, device="cpu", verbose=False)
    assert len(predictions) == 1
    if fp32_validation:
        probe = [sys.executable, '-m', 'scripts.probe_training', '--data', str(data_yaml),
                 '--weights', str(seed), '--output', str(tmp_path/'resource_probe'),
                 '--imgsz', '64', '--batch', '1', '--chunk-size', '2', '--device', 'cpu', '--steps', '2']
        probe.extend(['--criterion',criterion_name])
        completed = subprocess.run(probe, capture_output=True, text=True,
                                   encoding='utf-8', errors='replace', timeout=180)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        report = json.loads((tmp_path/'resource_probe/probe.json').read_text())
        assert report['status'] == 'complete'
        assert len(report['steps']) == 2
        assert report['fp32_validator_metrics']
        assert Path(report['reloaded_checkpoint']).is_file()
