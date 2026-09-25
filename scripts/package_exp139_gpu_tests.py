"""Build the free Kaggle T4 kernel that tests exp139's speed-ups on a real GPU.

The kernel carries the code inside itself (base64 tarball plus a SHA256 manifest, the
exp119 kernel's pattern), so no dataset upload is needed per iteration. It installs the
pod's exact torch (2.8.0, CUDA 12.8) and transformers (4.57.6), then runs
`experiments.exp_139_mask2former.tests.test_equivalence --suite gpu` on the competition's
real training images. No RunPod credit is involved.

Usage:
    python scripts/package_exp139_gpu_tests.py
    cd kaggle/exp139_gpu_tests_t4 && kaggle kernels push -p .
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
KERNEL_ID = "cosmicwitness/sol-exp139-gpu-tests-t4"
TITLE = "sol exp139 gpu tests t4"
ANNOTATIONS_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
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
    "experiments/exp_139_mask2former/tests/__init__.py",
    "experiments/exp_139_mask2former/tests/reference_data.py",
    "experiments/exp_139_mask2former/tests/test_equivalence.py",
]

KERNEL = r'''import base64, hashlib, io, json, os, subprocess, sys, tarfile
from pathlib import Path
root = Path("/tmp/code"); root.mkdir(parents=True, exist_ok=True)   # outside /kaggle/working: not downloaded with the report
with tarfile.open(fileobj=io.BytesIO(base64.b64decode("__PAYLOAD__")), mode="r:gz") as tar:
    tar.extractall(root)
manifest = __MANIFEST__
for name, digest in manifest.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
print("CODE_VERIFIED", len(manifest), flush=True)

def run(command, **kwargs):
    print("RUN", " ".join(command), flush=True)
    return subprocess.run(command, check=True, **kwargs)

run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchvision", "torchaudio"])
run([sys.executable, "-m", "pip", "install", "-q", "torch==2.8.0", "--index-url", "https://download.pytorch.org/whl/cu128"])
run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6", "kernels", "ninja", "pycocotools", "scipy"])
facts = subprocess.run([sys.executable, "-c",
    "import json, torch, transformers, importlib.metadata as m; print(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda, "
    "transformers=transformers.__version__, kernels=m.version('kernels'), gpu=torch.cuda.get_device_name(0), "
    "capability=torch.cuda.get_device_capability(0))))"], capture_output=True, text=True)
print("ENVIRONMENT", facts.stdout.strip(), facts.stderr.strip()[-500:], flush=True)
nvcc = subprocess.run(["bash", "-lc", "nvcc --version | tail -n 2; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"],
                      capture_output=True, text=True)
print("TOOLCHAIN", nvcc.stdout.strip().replace("\n", " | "), nvcc.stderr.strip()[-300:], flush=True)
source = Path("/kaggle/input")
annotations = next(source.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
assert hashlib.sha256(annotations.read_bytes()).hexdigest() == "__ANNOTATIONS_SHA256__"
images = annotations.parent / "train_images"
assert images.is_dir(), images
env = dict(os.environ, PYTHONPATH=str(root), PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
output = Path("/kaggle/working/exp139_gpu_tests.json")
result = subprocess.run([sys.executable, "-u", "-m", "experiments.exp_139_mask2former.tests.test_equivalence",
                         "--suite", "gpu", "--annotations", str(annotations), "--images", str(images),
                         "--output", str(output), "--skip-loader"], cwd=root, env=env)
print("TEST_EXIT", result.returncode, flush=True)
if output.exists():
    print("REPORT_JSON", output.read_text(), flush=True)
print("KERNEL_COMPLETE", flush=True)
'''


def build_payload() -> bytes:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for name in FILES:
            data = (ROOT / name).read_bytes()
            info = tarfile.TarInfo(name=name)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
    compressed = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0) as stream:
        stream.write(archive.getvalue())
    return compressed.getvalue()


def main() -> None:
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    manifest = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
    payload = build_payload()
    kernel = (KERNEL.replace("__PAYLOAD__", base64.b64encode(payload).decode("ascii"))
              .replace("__MANIFEST__", json.dumps(manifest))
              .replace("__ANNOTATIONS_SHA256__", ANNOTATIONS_SHA256))
    directory = ROOT / "kaggle" / "exp139_gpu_tests_t4"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kernel.py").write_text(kernel, encoding="utf-8")
    (directory / "kernel-metadata.json").write_text(json.dumps(dict(
        id=KERNEL_ID, title=TITLE, code_file="kernel.py", language="python", kernel_type="script",
        is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=True,
        machine_shape="NvidiaTeslaT4", competition_sources=["filament-segmentation-2026"],
        dataset_sources=[], kernel_sources=[], model_sources=[]), indent=2))
    print(json.dumps(dict(directory=str(directory), kernel_bytes=len(kernel), payload_bytes=len(payload),
                          payload_sha256=hashlib.sha256(payload).hexdigest(), files=len(FILES)), indent=2))


if __name__ == "__main__":
    main()
