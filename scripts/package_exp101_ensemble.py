"""Package the Kaggle T4 kernel that scores an ensemble of exp097 checkpoints.

The checkpoints arrive through the private dataset the pod uploaded, and every
one has its SHA256 checked against the pod's manifest before inference. The
kernel reports each checkpoint alone and the pooled ensemble from the same
predictions, so the comparison is internal to one run.

Cost: eight dihedral views per checkpoint over 180 validation photographs at
2048. Two checkpoints take about an hour on a T4, three about ninety minutes.

Usage, with checkpoints ranked by exp098:
    python scripts/package_exp101_ensemble.py --checkpoints stage2_best.pt best.pt
    cd kaggle/exp101_exp097_ensemble_t4 && kaggle kernels push -p .
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DATASET = "cosmicwitness/sol-exp097-yolov8l-2048"
KERNEL_ID = "cosmicwitness/sol-exp101-exp097-ensemble-t4"
ULTRALYTICS = "8.4.153"
ANNOTATIONS_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
FILES = [
    "shared/__init__.py", "shared/data_split.py", "shared/utils.py",
    "experiments/exp_007_tta/src/dihedral.py",
    "experiments/exp_033_inference/__init__.py",
    "experiments/exp_033_inference/src/__init__.py",
    "experiments/exp_033_inference/src/sweep.py",
    "experiments/exp_033_inference/src/cached_tta.py",
    "experiments/exp_097_runpod_yolo/__init__.py",
    "experiments/exp_097_runpod_yolo/src/__init__.py",
    "experiments/exp_097_runpod_yolo/src/events.py",
    "experiments/exp_097_runpod_yolo/src/evaluate_ensemble.py",
    "data/exp033_cached_tta/candidates_0.5_0.5.json",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--link", type=float, default=0.5)
    parser.add_argument("--vote", type=float, default=0.5)
    parser.add_argument("--timeout", type=int, default=40000)
    args = parser.parse_args()

    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    manifest = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for name in FILES:
            tar.add(ROOT / name, arcname=name)
    payload = base64.b64encode(archive.getvalue()).decode("ascii")
    slug = WEIGHTS_DATASET.split("/")[1]
    rule = ["--link", repr(args.link), "--vote", repr(args.vote),
            "--checkpoints", *args.checkpoints]

    wrapper = f'''import base64,hashlib,io,json,os,subprocess,sys,tarfile
from pathlib import Path
os.environ['SOL_SOURCE_REVISION']={revision!r}
root=Path('/kaggle/working/code');root.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r})),mode='r:gz') as tar:tar.extractall(root)
manifest={manifest!r}
for name,digest in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
os.environ['PYTHONPATH']=str(root)
subprocess.check_call([sys.executable,'-m','pip','install','-q','pycocotools','ultralytics=={ULTRALYTICS}'])
import torch
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0),torch.cuda.get_device_name(0)
source=Path('/kaggle/input')
annotations=next(source.rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
assert hashlib.sha256(annotations.read_bytes()).hexdigest()=={ANNOTATIONS_SHA256!r}
images=annotations.parent/'train_images';assert images.is_dir()
weights=[p.parent for p in source.rglob('manifest.json') if {slug!r} in str(p)]
assert len(weights)==1,[str(p) for p in weights]
print('CHECKPOINTS_AVAILABLE',sorted(p.name for p in weights[0].glob('*.pt')),flush=True)
subprocess.run([sys.executable,'-u','-m','experiments.exp_097_runpod_yolo.src.evaluate_ensemble',
 '--weights-dir',str(weights[0]),'--annotations',str(annotations),'--images',str(images),
 '--baseline-pool',str(root/'data/exp033_cached_tta/candidates_0.5_0.5.json'),
 '--output','/kaggle/working/ensemble']+{rule!r},cwd=root,check=True,timeout={args.timeout})
print('KERNEL_COMPLETE',flush=True)
'''
    directory = ROOT / "kaggle" / "exp101_exp097_ensemble_t4"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kernel.py").write_text(wrapper, encoding="utf-8")
    (directory / "kernel-metadata.json").write_text(json.dumps({
        "id": KERNEL_ID,
        # Kaggle derives the slug from the title; keep the two slug-identical (CLAUDE.md 14).
        "title": "sol exp101 exp097 ensemble t4",
        "code_file": "kernel.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "competition_sources": ["filament-segmentation-2026"],
        "dataset_sources": [WEIGHTS_DATASET],
        "kernel_sources": [], "model_sources": [],
    }, indent=2), encoding="utf-8")
    print(json.dumps(dict(directory=str(directory), checkpoints=args.checkpoints,
                          link=args.link, vote=args.vote, kernel_bytes=len(wrapper)), indent=2))


if __name__ == "__main__":
    main()
