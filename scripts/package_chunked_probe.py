"""Package a bounded T4 probe of native-resolution YOLO fine-tuning."""
import base64
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

root = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--roi-comparison',action='store_true')
args=parser.parse_args()
files = ['shared/__init__.py', 'shared/data_split.py', 'shared/chunked_mask_loss.py',
         'experiments/exp_002_yolo_seg/src/prepare_yolo.py',
         'experiments/exp_032_chunked/src/train.py', 'scripts/probe_training.py']
if args.roi_comparison:files.append('shared/roi_mask_loss.py')
slug='sol-exp070-roi-probe' if args.roi_comparison else 'sol-exp067-chunked-probe'
title='Sol exp070 roi probe' if args.roi_comparison else 'Sol exp067 chunked probe'
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
manifest = {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files}
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode='w:gz') as tar:
    for name in files:
        tar.add(root/name, arcname=name)
payload = base64.b64encode(buf.getvalue()).decode()
directory = root/('kaggle/exp070_roi_probe' if args.roi_comparison else 'kaggle/exp067_chunked_probe')
directory.mkdir(exist_ok=True)
wrapper = f'''import os,sys,io,tarfile,base64,subprocess,json,hashlib
from pathlib import Path
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
os.environ['SOL_SOURCE_REVISION']={revision!r}
root=Path('/kaggle/working/code');root.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r}))) as tar:tar.extractall(root)
manifest={manifest!r}
for name,sha in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
(root/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
subprocess.check_call([sys.executable,'-m','pip','install','-q','ultralytics==8.4.0','shapely','pycocotools'])
os.environ['PYTHONPATH']=str(root)
import torch
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
print('ACCELERATOR',torch.cuda.get_device_name(0),torch.__version__,flush=True)
source=Path('/kaggle/input')
weights=[p for p in source.rglob('best.pt') if p.parent.name=='checkpoints'];assert len(weights)==1
assert hashlib.sha256(weights[0].read_bytes()).hexdigest()=='d9c6641edb1fec427212d0c35b008c923361e9fcf5004905a69c883714d93389'
annotations=next(source.rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
assert hashlib.sha256(annotations.read_bytes()).hexdigest()=='5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1'
dataset='/kaggle/temp/yolo_ds'
subprocess.run([sys.executable,'-u','-m','experiments.exp_002_yolo_seg.src.prepare_yolo',
 '--annotations',str(annotations),'--images',str(next(source.rglob('train_images'))),
 '--output',dataset,'--polygon-offset','0.5'],cwd=root,check=True,timeout=600)
for criterion in {['chunked','roi'] if args.roi_comparison else ['chunked']!r}:
 output='/kaggle/working/'+('probe_'+criterion if {args.roi_comparison!r} else 'probe')
 subprocess.run([sys.executable,'-u','-m','scripts.probe_training','--data',dataset+'/data.yaml',
 '--weights',str(weights[0]),'--output',output,'--imgsz','2048','--criterion',criterion,
 '--batch','1','--chunk-size','8','--steps','5','--device','0'],cwd=root,check=True,timeout=1500)
'''
(directory/'kernel.py').write_text(wrapper, encoding='utf-8')
(directory/'kernel-metadata.json').write_text(json.dumps(dict(
    id='cosmicwitness/'+slug, title=title,
    code_file='kernel.py', language='python', kernel_type='script', is_private=True,
    enable_gpu=True, enable_tpu=False, enable_internet=True, machine_shape='NvidiaTeslaT4',
    competition_sources=['filament-segmentation-2026'], dataset_sources=[],
    kernel_sources=['cosmicwitness/sol-exp002-yolo-seg'], model_sources=[]), indent=2))
print(directory)
