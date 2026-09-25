"""Build the free Kaggle T4 kernel for exp141: weight averaging and a re-tuned view fusion.

Reads the pod's weights dataset `cosmicwitness/sol-exp139-mask2former-weights` and the
competition data. Writes soup_tta_report.json. It submits nothing and spends no credits.

Usage:
    python scripts/package_exp141_soup_tta.py
    cd kaggle/exp141_soup_tta_t4 && kaggle kernels push -p .
"""
from __future__ import annotations

import base64
import hashlib
import json

import package_exp139_gpu_tests as base
from package_exp139_gpu_tests import ANNOTATIONS_SHA256, ROOT, build_payload

KERNEL_ID = "cosmicwitness/sol-exp141-soup-tta-t4"
TITLE = "sol exp141 soup tta t4"
FILES = [
    "shared/__init__.py",
    "shared/data_split.py",
    "shared/utils.py",
    "experiments/exp_033_inference/__init__.py",
    "experiments/exp_033_inference/src/__init__.py",
    "experiments/exp_033_inference/src/sweep.py",
    "experiments/exp_033_inference/src/cached_tta.py",
    "experiments/exp_139_mask2former/__init__.py",
    "experiments/exp_139_mask2former/src/__init__.py",
    "experiments/exp_139_mask2former/src/data.py",
    "experiments/exp_139_mask2former/src/model.py",
    "experiments/exp_139_mask2former/src/evaluate.py",
    "experiments/exp_139_mask2former/src/fast_ops.py",
    "experiments/exp_139_mask2former/src/submit.py",
    "experiments/exp_141_soup_and_tta/__init__.py",
    "experiments/exp_141_soup_and_tta/src/__init__.py",
    "experiments/exp_141_soup_and_tta/src/soup_tta.py",
]

KERNEL = """import base64, hashlib, io, json, os, subprocess, sys, tarfile
from pathlib import Path
root = Path("/tmp/code"); root.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode("__PAYLOAD__")), mode="r:gz") as tar:
    tar.extractall(root)
manifest = __MANIFEST__
for name, digest in manifest.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
print("CODE_VERIFIED", len(manifest), flush=True)
def run(command):
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)
run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchvision", "torchaudio"])
run([sys.executable, "-m", "pip", "install", "-q", "torch==2.8.0", "--index-url", "https://download.pytorch.org/whl/cu128"])
run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6", "kernels", "pycocotools", "scipy"])
source = Path("/kaggle/input")
annotations = next(source.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
assert hashlib.sha256(annotations.read_bytes()).hexdigest() == "__ANNOTATIONS_SHA256__"
train_images = annotations.parent / "train_images"
weights = [p.parent for p in source.rglob("manifest.json") if "sol-exp139-mask2former-weights" in str(p)]
assert len(weights) == 1, [str(p) for p in weights]
env = dict(os.environ, PYTHONPATH=str(root), PYTHONUNBUFFERED="1", HF_HUB_ENABLE_HF_TRANSFER="0")
step = subprocess.run([sys.executable, "-u", "-m", "experiments.exp_141_soup_and_tta.src.soup_tta",
                       "--weights", str(weights[0]), "--annotations", str(annotations),
                       "--train-images", str(train_images), "--output", "/kaggle/working",
                       "--cache-dir", "/tmp/caches"], cwd=root, env=env)
print("EXIT", step.returncode, flush=True)
report = Path("/kaggle/working/soup_tta_report.json")
if report.exists():
    facts = json.loads(report.read_text())
    print("CONCLUSION", json.dumps(facts.get("conclusion")), flush=True)
print("KERNEL_COMPLETE", flush=True)
"""


def main() -> None:
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    base.FILES = FILES
    manifest = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
    payload = build_payload()
    kernel = (KERNEL.replace("__PAYLOAD__", base64.b64encode(payload).decode("ascii"))
              .replace("__MANIFEST__", json.dumps(manifest))
              .replace("__ANNOTATIONS_SHA256__", ANNOTATIONS_SHA256))
    directory = ROOT / "kaggle" / "exp141_soup_tta_t4"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kernel.py").write_text(kernel, encoding="utf-8")
    (directory / "kernel-metadata.json").write_text(json.dumps(dict(
        id=KERNEL_ID, title=TITLE, code_file="kernel.py", language="python", kernel_type="script",
        is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=True,
        machine_shape="NvidiaTeslaT4", competition_sources=["filament-segmentation-2026"],
        dataset_sources=["cosmicwitness/sol-exp139-mask2former-weights"], kernel_sources=[],
        model_sources=[]), indent=2))
    print(json.dumps(dict(directory=str(directory), kernel_bytes=len(kernel), payload_bytes=len(payload),
                          payload_sha256=hashlib.sha256(payload).hexdigest(), files=len(FILES)), indent=2))


if __name__ == "__main__":
    main()
