"""Package a short low-rate ROI-loss continuation from the completed native model."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parents[1]
revision = __import__('subprocess').check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip()
files = ['shared/__init__.py', 'shared/data_split.py', 'shared/chunked_mask_loss.py',
         'shared/roi_mask_loss.py', 'experiments/exp_002_yolo_seg/src/prepare_yolo.py',
         'experiments/exp_032_chunked/src/train.py']
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode='w:gz') as tar:
    for name in files:
        tar.add(root/name, arcname=name)
payload = base64.b64encode(buf.getvalue()).decode()
directory = root/'kaggle/exp073_roi_continuation'
directory.mkdir(exist_ok=True)
wrapper = f'''import os,sys,io,tarfile,base64,subprocess,hashlib
from pathlib import Path
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
os.environ['SOL_SOURCE_REVISION']={revision!r}
root=Path('/kaggle/working/code');root.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r}))) as tar:tar.extractall(root)
subprocess.check_call([sys.executable,'-m','pip','install','-q','ultralytics==8.4.0','shapely','pycocotools'])
os.environ['PYTHONPATH']=str(root)
import torch
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
print('ACCELERATOR',torch.cuda.get_device_name(0),flush=True)
source=Path('/kaggle/input')
weights=[p for p in source.rglob('best.pt') if p.parent.name=='weights' and p.parent.parent.name=='native'];assert len(weights)==1
assert hashlib.sha256(weights[0].read_bytes()).hexdigest()=='efe905fd14c9089221acb80b8816e349ec2d82c9b2b9226cbad5d1d9f555e95a'
annotations=next(source.rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
assert hashlib.sha256(annotations.read_bytes()).hexdigest()=='5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1'
dataset='/kaggle/temp/yolo_ds'
subprocess.run([sys.executable,'-u','-m','experiments.exp_002_yolo_seg.src.prepare_yolo','--annotations',str(annotations),'--images',str(next(source.rglob('train_images'))),'--output',dataset,'--polygon-offset','0.5'],cwd=root,check=True,timeout=600)
subprocess.run([sys.executable,'-u','-m','experiments.exp_032_chunked.src.train','--data',dataset+'/data.yaml','--weights',str(weights[0]),'--project','/kaggle/working/runs','--name','roi_continuation','--imgsz','2048','--batch','1','--chunk-size','8','--criterion','roi','--device','0','--epochs','4','--time-hours','0.9','--lr0','0.0001','--fp32-validation'],cwd=root,check=True,timeout=5200)
'''
(directory/'kernel.py').write_text(wrapper, encoding='utf-8')
(directory/'kernel-metadata.json').write_text(json.dumps(dict(
    id='cosmicwitness/sol-exp073-roi-continuation', title='Sol exp073 roi continuation',
    code_file='kernel.py', language='python', kernel_type='script', is_private=True,
    enable_gpu=True, enable_tpu=False, enable_internet=True, machine_shape='NvidiaTeslaT4',
    competition_sources=['filament-segmentation-2026'], dataset_sources=[], model_sources=[],
    kernel_sources=['cosmicwitness/sol-exp068-native-mask-training']), indent=2))
print(directory)
