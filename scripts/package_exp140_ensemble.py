"""Build the free Kaggle T4 kernel for exp140: exp139 + exp119's two YOLO models, validation first.

Inputs, all attached to the kernel (nothing moves through the user's PC):
- competition data;
- `cosmicwitness/sol-exp139-mask2former-weights` (model_e040.pt + manifest);
- `cosmicwitness/sol-exp097-yolov8l-2048` (stage2_best.pt + manifest) for exp097's test views;
- `cosmicwitness/sol-exp002-pools` (exp002 validation pool and raw test views + manifest);
- kernel output `cosmicwitness/sol-exp098-exp097-evaluation-t4` (exp097 validation D4 pool).

The kernel writes ensemble_report.json and, only if the combination beats exp139 alone by
the margin, submission.csv. It submits nothing.

Usage:
    python scripts/package_exp140_ensemble.py
    cd kaggle/exp140_ensemble_t4 && kaggle kernels push -p .
"""
from __future__ import annotations

import base64
import hashlib
import json

import package_exp139_gpu_tests as base
from package_exp139_gpu_tests import ANNOTATIONS_SHA256, ROOT, build_payload

KERNEL_ID = "cosmicwitness/sol-exp140-ensemble-t4"
TITLE = "sol exp140 ensemble t4"
FILES = [
    "shared/__init__.py",
    "shared/data_split.py",
    "shared/utils.py",
    "experiments/exp_007_tta/src/dihedral.py",
    "experiments/exp_033_inference/__init__.py",
    "experiments/exp_033_inference/src/__init__.py",
    "experiments/exp_033_inference/src/sweep.py",
    "experiments/exp_033_inference/src/cached_tta.py",
    "experiments/exp_097_runpod_yolo/__init__.py",
    "experiments/exp_097_runpod_yolo/src/__init__.py",
    "experiments/exp_097_runpod_yolo/src/events.py",
    "experiments/exp_097_runpod_yolo/src/evaluate_ensemble.py",
    "experiments/exp_139_mask2former/__init__.py",
    "experiments/exp_139_mask2former/src/__init__.py",
    "experiments/exp_139_mask2former/src/data.py",
    "experiments/exp_139_mask2former/src/model.py",
    "experiments/exp_139_mask2former/src/evaluate.py",
    "experiments/exp_139_mask2former/src/fast_ops.py",
    "experiments/exp_139_mask2former/src/submit.py",
    "experiments/exp_140_three_way_ensemble/__init__.py",
    "experiments/exp_140_three_way_ensemble/src/__init__.py",
    "experiments/exp_140_three_way_ensemble/src/ensemble.py",
    "experiments/exp_140_three_way_ensemble/src/infer_exp139.py",
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
run([sys.executable, "-m", "pip", "install", "-q", "torch==2.8.0", "torchvision==0.23.0", "--index-url", "https://download.pytorch.org/whl/cu128"])
run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6", "kernels", "pycocotools", "scipy", "ultralytics==8.4.153"])
source = Path("/kaggle/input")
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
annotations = next(source.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
assert sha(annotations) == "__ANNOTATIONS_SHA256__"
train_images = annotations.parent / "train_images"
test_images = next(p for p in source.rglob("test_images") if p.is_dir())
def dataset(slug):
    found = [p.parent for p in source.rglob("manifest.json") if slug in str(p)]
    assert len(found) == 1, (slug, [str(p) for p in found])
    return found[0]
exp139 = dataset("sol-exp139-mask2former-weights")
exp097 = dataset("sol-exp097-yolov8l-2048")
pools = dataset("sol-exp002-pools")
for name, facts in json.loads((pools / "manifest.json").read_text()).items():
    assert sha(pools / name) == facts["sha256"], name
yolo = exp097 / "stage2_best.pt"
assert sha(yolo) == json.loads((exp097 / "manifest.json").read_text())["stage2_best.pt"]["sha256"]
val097 = [p for p in source.rglob("d4_pool_stage2_best.json")]
assert len(val097) == 1, [str(p) for p in val097]
print("INPUTS_VERIFIED", flush=True)
env = dict(os.environ, PYTHONPATH=str(root), PYTHONUNBUFFERED="1", HF_HUB_ENABLE_HF_TRANSFER="0")
caches = Path("/tmp/caches"); caches.mkdir(exist_ok=True)
step = subprocess.run([sys.executable, "-u", "-m", "experiments.exp_140_three_way_ensemble.src.infer_exp139",
                       "--weights", str(exp139), "--annotations", str(annotations),
                       "--train-images", str(train_images), "--test-images", str(test_images),
                       "--output", str(caches)], cwd=root, env=env)
print("INFER_EXIT", step.returncode, flush=True)
assert step.returncode == 0
step = subprocess.run([sys.executable, "-u", "-m", "experiments.exp_140_three_way_ensemble.src.ensemble",
                       "--annotations", str(annotations),
                       "--exp002-val-pool", str(pools / "exp002_val_pool.json"),
                       "--exp097-val-pool", str(val097[0]),
                       "--exp139-val-cache", str(caches / "exp139_val_cache.json"),
                       "--output", "/kaggle/working",
                       # The user asked for every untested variant on the board, so the gate
                       # is opened to any positive gain instead of the 0.005 decision margin.
                       "--margin", "0.0",
                       "--test-images", str(test_images),
                       "--exp002-test-views", str(pools / "exp002_test_views_0.json"), str(pools / "exp002_test_views_1.json"),
                       "--exp097-weights", str(yolo),
                       "--exp139-test-cache", str(caches / "exp139_test_cache.json")], cwd=root, env=env)
print("ENSEMBLE_EXIT", step.returncode, flush=True)
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
    directory = ROOT / "kaggle" / "exp140_ensemble_t4"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kernel.py").write_text(kernel, encoding="utf-8")
    (directory / "kernel-metadata.json").write_text(json.dumps(dict(
        id=KERNEL_ID, title=TITLE, code_file="kernel.py", language="python", kernel_type="script",
        is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=True,
        machine_shape="NvidiaTeslaT4", competition_sources=["filament-segmentation-2026"],
        dataset_sources=["cosmicwitness/sol-exp139-mask2former-weights", "cosmicwitness/sol-exp097-yolov8l-2048",
                         "cosmicwitness/sol-exp002-pools"],
        kernel_sources=["cosmicwitness/sol-exp098-exp097-evaluation-t4"], model_sources=[]), indent=2))
    print(json.dumps(dict(directory=str(directory), kernel_bytes=len(kernel), payload_bytes=len(payload),
                          payload_sha256=hashlib.sha256(payload).hexdigest(), files=len(FILES)), indent=2))


if __name__ == "__main__":
    main()
