"""Package the exp095 T4 kernel that probes Ultralytics inference settings.

Usage: python scripts/package_yolo_inference_probe.py [--parity-pq VALUE]

All outputs stay on Kaggle. The kernel prints one compact JSON summary to its
log, which is the only thing read back.
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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--parity-pq", type=float, default=None)
parser.add_argument("--parity-iou", type=float, default=0.60)
parser.add_argument("--parity-retina", type=int, default=1)
args = parser.parse_args()

root = Path(__file__).resolve().parents[1]
files = [
    "shared/__init__.py", "shared/data_split.py", "shared/utils.py",
    "experiments/exp_033_inference/__init__.py",
    "experiments/exp_033_inference/src/__init__.py",
    "experiments/exp_033_inference/src/sweep.py",
    "scripts/probe_yolo_inference_config.py",
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
parity = [] if args.parity_pq is None else ["--parity-pq", repr(args.parity_pq)]
parity += ["--parity-iou", repr(args.parity_iou), "--parity-retina", str(args.parity_retina)]
directory = root / "kaggle" / "exp095_yolo_inference_probe_t4"
directory.mkdir(parents=True, exist_ok=True)
wrapper = f'''import base64,hashlib,io,json,os,subprocess,sys,tarfile
from pathlib import Path
os.environ['SOL_SOURCE_REVISION']={revision!r}
root=Path('/kaggle/working/code');root.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r})),mode='r:gz') as tar:tar.extractall(root)
manifest={manifest!r}
for name,digest in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
(root/'scripts'/'__init__.py').touch()
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
digest=hashlib.sha256(weights[0].read_bytes()).hexdigest()
assert digest=='d9c6641edb1fec427212d0c35b008c923361e9fcf5004905a69c883714d93389',digest
import ultralytics;print('ULTRALYTICS',ultralytics.__version__,flush=True)
subprocess.run([sys.executable,'-u','-m','scripts.probe_yolo_inference_config',
 '--weights',str(weights[0]),'--annotations',str(annotations),'--images',str(images),
 '--output','/kaggle/working/probe']+{parity!r},cwd=root,check=True,timeout=10000)
print('KERNEL_COMPLETE',flush=True)
'''
(directory / "kernel.py").write_text(wrapper, encoding="utf-8")
(directory / "kernel-metadata.json").write_text(json.dumps({
    "id": "cosmicwitness/sol-exp095-yolo-inference-probe-t4",
    # Kaggle derives the slug from the title; keep the two slug-identical.
    "title": "sol exp095 yolo inference probe t4",
    "code_file": "kernel.py", "language": "python", "kernel_type": "script",
    "is_private": True, "enable_gpu": True, "enable_tpu": False,
    "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
    "competition_sources": ["filament-segmentation-2026"],
    "dataset_sources": [],
    "kernel_sources": ["cosmicwitness/sol-exp002-yolo-seg"],
    "model_sources": [],
}, indent=2), encoding="utf-8")
print(directory, len(wrapper), parity)
