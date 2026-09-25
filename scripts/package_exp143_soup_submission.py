"""Build the free Kaggle T4 kernel that writes a test submission from a chosen checkpoint.

No validation, no sweep, no review sheet: exp139 already fixed the emission rule
(confidence 0.85, no erosion, minimum area 100, single view, float32) and exp141 measures
whether the weight average beats the single checkpoint. This kernel only writes the CSV.

`--base` selects the weights: `soup` averages model_e040.pt and model_e050.pt, `e040` or
`e050` uses that checkpoint alone.

Usage:
    python scripts/package_exp143_soup_submission.py --base soup
    cd kaggle/exp143_soup_submission_t4 && kaggle kernels push -p .
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json

import package_exp139_gpu_tests as base_module
from package_exp139_gpu_tests import ANNOTATIONS_SHA256, ROOT, build_payload

KERNEL_ID = "cosmicwitness/sol-exp143-soup-submission-t4"
TITLE = "sol exp143 soup submission t4"
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
    "experiments/exp_139_mask2former/src/emit_test.py",
    "experiments/exp_139_mask2former/src/fast_ops.py",
    "experiments/exp_139_mask2former/src/submit.py",
    "experiments/exp_141_soup_and_tta/__init__.py",
    "experiments/exp_141_soup_and_tta/src/__init__.py",
    "experiments/exp_141_soup_and_tta/src/soup_tta.py",
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
test_images = next(p for p in source.rglob("test_images") if p.is_dir())
weights = [p.parent for p in source.rglob("manifest.json") if "sol-exp139-mask2former-weights" in str(p)]
assert len(weights) == 1, [str(p) for p in weights]
print("WEIGHTS", sorted(p.name for p in weights[0].iterdir()), flush=True)
sys.path.insert(0, str(root))

BASE = "__BASE__"
if BASE == "soup":
    import torch
    from experiments.exp_141_soup_and_tta.src.soup_tta import average_weights
    parts = [weights[0] / "model_e040.pt", weights[0] / "model_e050.pt"]
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in parts}
    recorded = json.loads((weights[0] / "manifest.json").read_text())
    for name, digest in digests.items():
        assert digest == recorded[name]["sha256"], name
    checkpoint = Path("/tmp/model_soup.pt")
    torch.save(average_weights(parts), checkpoint)
    print("SOUP", json.dumps(dict(parts=digests, bytes=checkpoint.stat().st_size)), flush=True)
else:
    checkpoint = weights[0] / f"model_{BASE}.pt"

env = dict(os.environ, PYTHONPATH=str(root), PYTHONUNBUFFERED="1", HF_HUB_ENABLE_HF_TRANSFER="0")
result = subprocess.run([sys.executable, "-u", "-m", "experiments.exp_139_mask2former.src.emit_test",
                         "--checkpoint", str(checkpoint), "--test-images", str(test_images),
                         "--output", "/kaggle/working", "--conf", "0.85", "--grow", "0",
                         "--min-area", "100", "--log", "/kaggle/working/events.log"],
                        cwd=root, env=env)
print("EMIT_EXIT", result.returncode, flush=True)
for csv in sorted(Path("/kaggle/working").glob("submission_*.csv")):
    rows = sum(1 for _ in csv.open()) - 1
    print("CSV", json.dumps(dict(name=csv.name, rows=rows, bytes=csv.stat().st_size,
                                 sha256=hashlib.sha256(csv.read_bytes()).hexdigest())), flush=True)
print("KERNEL_COMPLETE", flush=True)
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", choices=("soup", "e040", "e050"), default="soup")
    args = parser.parse_args()
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    base_module.FILES = FILES
    manifest = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
    payload = build_payload()
    kernel = (KERNEL.replace("__PAYLOAD__", base64.b64encode(payload).decode("ascii"))
              .replace("__MANIFEST__", json.dumps(manifest))
              .replace("__ANNOTATIONS_SHA256__", ANNOTATIONS_SHA256)
              .replace("__BASE__", args.base))
    directory = ROOT / "kaggle" / "exp143_soup_submission_t4"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kernel.py").write_text(kernel, encoding="utf-8")
    (directory / "kernel-metadata.json").write_text(json.dumps(dict(
        id=KERNEL_ID, title=TITLE, code_file="kernel.py", language="python", kernel_type="script",
        is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=True,
        machine_shape="NvidiaTeslaT4", competition_sources=["filament-segmentation-2026"],
        dataset_sources=["cosmicwitness/sol-exp139-mask2former-weights"], kernel_sources=[],
        model_sources=[]), indent=2))
    print(json.dumps(dict(directory=str(directory), base=args.base, kernel_bytes=len(kernel),
                          payload_sha256=hashlib.sha256(payload).hexdigest(), files=len(FILES)), indent=2))


if __name__ == "__main__":
    main()
