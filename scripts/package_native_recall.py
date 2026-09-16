"""Extend the completed native model's confidence sweep below its boundary optimum."""
import base64,hashlib,io,json,tarfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]
files=['shared/__init__.py','shared/data_split.py','shared/utils.py','shared/chunked_mask_loss.py',
       'experiments/exp_007_tta/src/dihedral.py','experiments/exp_033_inference/src/sweep.py',
       'experiments/exp_033_inference/src/cached_tta.py','scripts/existing_weight_trials.py',
       'scripts/evaluate_yolo_checkpoints.py']
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as tar:
    for name in files:tar.add(root/name,arcname=name)
payload=base64.b64encode(buf.getvalue()).decode()
sha=hashlib.sha256((root/'data/exp068_output/runs/native/weights/best.pt').read_bytes()).hexdigest()
d=root/'kaggle/exp071_native_recall';d.mkdir(exist_ok=True)
(d/'kernel.py').write_text(f'''import os,sys,io,tarfile,base64,subprocess,hashlib
from pathlib import Path
root=Path('/kaggle/working/code');root.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r}))) as tar:tar.extractall(root)
subprocess.check_call([sys.executable,'-m','pip','install','-q','ultralytics==8.4.0','pycocotools'])
os.environ['PYTHONPATH']=str(root)
import torch
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
print('ACCELERATOR',torch.cuda.get_device_name(0),flush=True)
source=Path('/kaggle/input')
weights=[p for p in source.rglob('best.pt') if p.parent.name=='weights' and p.parent.parent.name=='native'];assert len(weights)==1
assert hashlib.sha256(weights[0].read_bytes()).hexdigest()=={sha!r}
subprocess.run([sys.executable,'-u','-m','scripts.evaluate_yolo_checkpoints','--weights',str(weights[0]),
 '--annotations',str(next(source.rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))),
 '--images',str(next(source.rglob('train_images'))),'--output','/kaggle/working/validation',
 '--device','0','--floor','0.05','--force-d4'],cwd=root,check=True,timeout=3000)
''',encoding='utf-8')
(d/'kernel-metadata.json').write_text(json.dumps(dict(id='cosmicwitness/sol-exp071-native-recall',
 title='Sol exp071 native recall',code_file='kernel.py',language='python',kernel_type='script',
 is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,machine_shape='NvidiaTeslaT4',
 competition_sources=['filament-segmentation-2026'],dataset_sources=[],model_sources=[],
 kernel_sources=['cosmicwitness/sol-exp068-native-mask-training']),indent=2))
print(d)
