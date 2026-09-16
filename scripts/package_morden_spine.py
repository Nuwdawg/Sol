"""Package the bounded exp085 MORDEN-style T4 training and evaluation kernel."""

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
    "shared/__init__.py",
    "shared/data_split.py",
    "shared/utils.py",
    "experiments/exp_004_spine_tpu/src/decompose.py",
    "experiments/exp_033_inference/__init__.py",
    "experiments/exp_033_inference/src/sweep.py",
    "experiments/exp_085_morden_spine/__init__.py",
    "experiments/exp_085_morden_spine/src/__init__.py",
    "experiments/exp_085_morden_spine/src/model.py",
    "experiments/exp_085_morden_spine/src/prepare.py",
    "experiments/exp_085_morden_spine/src/train.py",
    "experiments/exp_085_morden_spine/src/evaluate.py",
    "third_party/MORDEN_LICENSE",
]
for name in files:
    if not (root / name).exists():
        raise FileNotFoundError(name)
manifest = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
archive = io.BytesIO()
with tarfile.open(fileobj=archive, mode="w:gz") as tar:
    for name in files:
        tar.add(root / name, arcname=name)
payload = base64.b64encode(archive.getvalue()).decode("ascii")

directory = root / "kaggle" / "exp085_morden_spine_t4"
directory.mkdir(parents=True, exist_ok=True)
wrapper = f'''import base64,hashlib,io,json,os,subprocess,sys,tarfile,time
from pathlib import Path
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
os.environ['SOL_SOURCE_REVISION']={revision!r}
root=Path('/kaggle/working/code');root.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r})),mode='r:gz') as tar:tar.extractall(root)
manifest={manifest!r}
for name,digest in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,(name,digest)
(Path('/kaggle/working')/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
os.environ['PYTHONPATH']=str(root)
subprocess.check_call([sys.executable,'-m','pip','install','-q','pycocotools'])
import torch
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0),torch.cuda.get_device_name(0)
print('ACCELERATOR',torch.cuda.get_device_name(0),'torch',torch.__version__,flush=True)
source=Path('/kaggle/input')
annotations=next(source.rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
assert hashlib.sha256(annotations.read_bytes()).hexdigest()=='5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1'
images=annotations.parent/'train_images';assert images.is_dir()
cache=Path('/kaggle/temp/morden_cache')
subprocess.run([sys.executable,'-u','-m','experiments.exp_085_morden_spine.src.prepare',
 '--annotations',str(annotations),'--images',str(images),'--output',str(cache),
 '--size','1024','--spine-width','5'],cwd=root,check=True,timeout=1800)
subprocess.run([sys.executable,'-u','-m','experiments.exp_085_morden_spine.src.train',
 '--cache',str(cache),'--output','/kaggle/working/training','--epochs','10',
 '--batch','1','--accumulate','2','--base-channels','16','--lr','0.0002',
 '--workers','2','--time-hours','4.25'],cwd=root,check=True,timeout=16800)
subprocess.run([sys.executable,'-u','-m','experiments.exp_085_morden_spine.src.evaluate',
 '--checkpoint','/kaggle/working/training/best.pt','--annotations',str(annotations),
 '--images',str(images),'--output','/kaggle/working/validation','--device','cuda:0','--tta'],
 cwd=root,check=True,timeout=3600)
print('KERNEL_COMPLETE',flush=True)
'''
(directory / "kernel.py").write_text(wrapper, encoding="utf-8")
(directory / "kernel-metadata.json").write_text(json.dumps({
    "id": "cosmicwitness/sol-exp085-morden-spine-t4",
    "title": "Sol exp085 MORDEN spine T4",
    "code_file": "kernel.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4",
    "competition_sources": ["filament-segmentation-2026"],
    "dataset_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}, indent=2), encoding="utf-8")
print(directory)
