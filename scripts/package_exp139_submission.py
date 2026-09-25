"""Build the free Kaggle T4 kernel that validates exp139's weights and writes a test submission.

The kernel carries its code inline (base64 tarball plus SHA256 manifest, as the exp119 and
exp139 test kernels do) and reads the pod's weights from the private dataset
`cosmicwitness/sol-exp139-mask2former-weights`. It installs the pod's torch 2.8.0 (CUDA 12.8)
and transformers 4.57.6, then runs `experiments.exp_139_mask2former.src.submit`.
It writes submission.csv, summary.json and review_sheet.jpg; it submits nothing.

Usage:
    python scripts/package_exp139_submission.py
    cd kaggle/exp139_submission_t4 && kaggle kernels push -p .
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from package_exp139_gpu_tests import ANNOTATIONS_SHA256, ROOT, build_payload

KERNEL_ID = "cosmicwitness/sol-exp139-submission-t4"
TITLE = "sol exp139 submission t4"
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
test_images = next(p for p in source.rglob("test_images") if p.is_dir())
weights = [p.parent for p in source.rglob("manifest.json") if "sol-exp139-mask2former-weights" in str(p)]
assert len(weights) == 1, [str(p) for p in weights]
print("WEIGHTS", sorted(p.name for p in weights[0].iterdir()), flush=True)
env = dict(os.environ, PYTHONPATH=str(root), PYTHONUNBUFFERED="1", HF_HUB_ENABLE_HF_TRANSFER="0")
result = subprocess.run([sys.executable, "-u", "-m", "experiments.exp_139_mask2former.src.submit",
                         "--weights", str(weights[0]), "--annotations", str(annotations),
                         "--train-images", str(train_images), "--test-images", str(test_images),
                         "--output", "/kaggle/working"], cwd=root, env=env)
print("SUBMIT_EXIT", result.returncode, flush=True)
print("KERNEL_COMPLETE", flush=True)
'''


def main() -> None:
    import package_exp139_gpu_tests as base
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    base.FILES = FILES
    manifest = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
    payload = build_payload()
    kernel = (KERNEL.replace("__PAYLOAD__", base64.b64encode(payload).decode("ascii"))
              .replace("__MANIFEST__", json.dumps(manifest))
              .replace("__ANNOTATIONS_SHA256__", ANNOTATIONS_SHA256))
    directory = ROOT / "kaggle" / "exp139_submission_t4"
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
