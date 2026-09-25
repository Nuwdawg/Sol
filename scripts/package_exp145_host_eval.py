"""Build the free Kaggle T4 kernel that scores exp139 epoch 40 with the host's notebook.

Runs `experiments.exp_145_host_evaluation.src.host_eval` on the canonical validation
split and writes host_eval.json, the two frames it scored, the notebook's own three plots
and the emitted validation masks the paper figures are drawn from. It submits nothing.

Usage:
    python scripts/package_exp145_host_eval.py
    cd kaggle/exp145_host_eval_t4 && kaggle kernels push -p .
"""
from __future__ import annotations

import base64
import hashlib
import json

import package_exp139_gpu_tests as base_module
from package_exp139_gpu_tests import ANNOTATIONS_SHA256, ROOT, build_payload

KERNEL_ID = "cosmicwitness/sol-exp145-host-eval-t4"
TITLE = "sol exp145 host eval t4"
NOTEBOOK = "experiments/exp_145_host_evaluation/host/self_evaluation_notebook.ipynb"
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
    "experiments/exp_145_host_evaluation/__init__.py",
    "experiments/exp_145_host_evaluation/src/__init__.py",
    "experiments/exp_145_host_evaluation/src/host_eval.py",
    NOTEBOOK,
]

KERNEL = r'''import base64, hashlib, io, json, os, subprocess, sys, tarfile
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
checkpoint = weights[0] / "model_e040.pt"
recorded = json.loads((weights[0] / "manifest.json").read_text())
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == recorded[checkpoint.name]["sha256"]
env = dict(os.environ, PYTHONPATH=str(root), PYTHONUNBUFFERED="1", HF_HUB_ENABLE_HF_TRANSFER="0", MPLBACKEND="Agg")
result = subprocess.run([sys.executable, "-u", "-m", "experiments.exp_145_host_evaluation.src.host_eval",
                         "--checkpoint", str(checkpoint), "--annotations", str(annotations),
                         "--train-images", str(train_images), "--notebook", str(root / "__NOTEBOOK__"),
                         "--weights", str(weights[0]), "--output", "/kaggle/working"], cwd=root, env=env)
print("HOST_EVAL_EXIT", result.returncode, flush=True)
print("KERNEL_COMPLETE", flush=True)
'''


def main() -> None:
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    base_module.FILES = FILES
    manifest = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
    payload = build_payload()
    kernel = (KERNEL.replace("__PAYLOAD__", base64.b64encode(payload).decode("ascii"))
              .replace("__MANIFEST__", json.dumps(manifest))
              .replace("__ANNOTATIONS_SHA256__", ANNOTATIONS_SHA256)
              .replace("__NOTEBOOK__", NOTEBOOK))
    directory = ROOT / "kaggle" / "exp145_host_eval_t4"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kernel.py").write_text(kernel, encoding="utf-8")
    (directory / "kernel-metadata.json").write_text(json.dumps(dict(
        id=KERNEL_ID, title=TITLE, code_file="kernel.py", language="python", kernel_type="script",
        is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=True,
        machine_shape="NvidiaTeslaT4", competition_sources=["filament-segmentation-2026"],
        dataset_sources=["cosmicwitness/sol-exp139-mask2former-weights"], kernel_sources=[],
        model_sources=[]), indent=2))
    print(json.dumps(dict(directory=str(directory), kernel_bytes=len(kernel),
                          payload_sha256=hashlib.sha256(payload).hexdigest(), files=len(FILES)), indent=2))


if __name__ == "__main__":
    main()
