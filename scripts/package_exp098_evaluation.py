"""Package the exp098 Kaggle T4 kernel that scores exp097's checkpoints.

It reads the private weights dataset the RunPod pod uploaded (`dataset_sources`), so
the weights go pod -> Kaggle -> kernel and never touch the user's PC. Results stay
in the kernel output; only the log's one-line PARITY/SINGLE_VIEW/D4/SUMMARY events
are read back.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DATASET = "cosmicwitness/sol-exp097-yolov8l-2048"
ULTRALYTICS = "8.4.153"   # the version the pod trained with
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
    "experiments/exp_097_runpod_yolo/src/evaluate_checkpoints.py",
    "data/exp033_cached_tta/candidates_0.5_0.5.json",
]


def main() -> None:
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
assert hashlib.sha256(annotations.read_bytes()).hexdigest()=='5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1'
images=annotations.parent/'train_images';assert images.is_dir()
weights=[p.parent for p in source.rglob('manifest.json') if {slug!r} in str(p)]
assert len(weights)==1,[str(p) for p in weights]
subprocess.run([sys.executable,'-u','-m','experiments.exp_097_runpod_yolo.src.evaluate_checkpoints',
 '--weights-dir',str(weights[0]),'--annotations',str(annotations),'--images',str(images),
 '--baseline-pool',str(root/'data/exp033_cached_tta/candidates_0.5_0.5.json'),
 '--output','/kaggle/working/evaluation','--d4-top','2'],cwd=root,check=True,timeout=40000)
print('KERNEL_COMPLETE',flush=True)
'''
    directory = ROOT / "kaggle" / "exp098_exp097_evaluation_t4"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kernel.py").write_text(wrapper, encoding="utf-8")
    (directory / "kernel-metadata.json").write_text(json.dumps({
        "id": "cosmicwitness/sol-exp098-exp097-evaluation-t4",
        # Kaggle derives the slug from the title; keep the two slug-identical.
        "title": "sol exp098 exp097 evaluation t4",
        "code_file": "kernel.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "competition_sources": ["filament-segmentation-2026"],
        "dataset_sources": [WEIGHTS_DATASET],
        "kernel_sources": [], "model_sources": [],
    }, indent=2), encoding="utf-8")
    print(directory, len(wrapper))


if __name__ == "__main__":
    main()
