"""Package the exp091 T4 kernel that harvests D4 proposals on the train split.

The boundary refiner must learn to repair proposals whose error distribution
matches the ones it will repair at inference. That means real detections from
the same frozen detector, the same eight dihedral views and the same fusion
constants, not perturbed ground truth. The kernel reproduces a shipped slice of
the stored validation pool first and refuses to spend an hour if it cannot.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

root = Path(__file__).resolve().parents[1]
files = [
    "shared/__init__.py", "shared/data_split.py", "shared/utils.py",
    "experiments/exp_007_tta/src/dihedral.py",
    "scripts/harvest_train_proposals.py",
    "data/exp091_parity_slice.json",
]
missing = [name for name in files if not (root / name).exists()]
assert not missing, missing
manifest = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
archive = io.BytesIO()
with tarfile.open(fileobj=archive, mode="w:gz") as tar:
    for name in files:
        tar.add(root / name, arcname=name)
payload = base64.b64encode(archive.getvalue()).decode("ascii")
directory = root / "kaggle" / "exp091_train_harvest_t4"
directory.mkdir(parents=True, exist_ok=True)
wrapper = f'''import base64,hashlib,io,json,os,subprocess,sys,tarfile
from pathlib import Path
os.environ['SOL_SOURCE_REVISION']={revision!r}
root=Path('/kaggle/working/code');root.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r})),mode='r:gz') as tar:tar.extractall(root)
manifest={manifest!r}
for name,digest in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
for package in ('experiments','experiments/exp_007_tta','experiments/exp_007_tta/src','scripts'):
    (root/package/'__init__.py').touch()
(Path('/kaggle/working')/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
os.environ['PYTHONPATH']=str(root)
subprocess.check_call([sys.executable,'-m','pip','install','-q','pycocotools','ultralytics'])
import torch
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0),torch.cuda.get_device_name(0)
print('ACCELERATOR',torch.cuda.get_device_name(0),'torch',torch.__version__,flush=True)
source=Path('/kaggle/input')
annotations=next(source.rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
assert hashlib.sha256(annotations.read_bytes()).hexdigest()=='5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1'
images=annotations.parent/'train_images';assert images.is_dir()
weights=[p for p in source.rglob('best.pt') if p.parent.name=='checkpoints']
assert len(weights)==1,[str(p) for p in weights]
print('WEIGHTS',weights[0],hashlib.sha256(weights[0].read_bytes()).hexdigest(),flush=True)
subprocess.run([sys.executable,'-u','-m','scripts.harvest_train_proposals',
 '--weights',str(weights[0]),'--annotations',str(annotations),'--images',str(images),
 '--reference',str(root/'data/exp091_parity_slice.json'),
 '--output','/kaggle/working/harvest','--imgsz','2048','--floor-conf','0.25',
 '--max-det','100','--time-hours','2.4'],cwd=root,check=True,timeout=10200)
print('KERNEL_COMPLETE',flush=True)
'''
(directory / "kernel.py").write_text(wrapper, encoding="utf-8")
(directory / "kernel-metadata.json").write_text(json.dumps({
    "id": "cosmicwitness/sol-exp091-train-harvest-t4",
    # Kaggle derives the slug from the title; keep the two slug-identical.
    "title": "sol exp091 train harvest t4",
    "code_file": "kernel.py", "language": "python", "kernel_type": "script",
    "is_private": True, "enable_gpu": True, "enable_tpu": False,
    "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
    "competition_sources": ["filament-segmentation-2026"],
    "dataset_sources": [],
    "kernel_sources": ["cosmicwitness/sol-exp002-yolo-seg"],
    "model_sources": [],
}, indent=2), encoding="utf-8")
print(directory, len(wrapper))
