import base64
import io
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parents[1]
files=['shared/__init__.py','shared/data_split.py','shared/utils.py',
       'experiments/exp_033_inference/src/sweep.py','scripts/sam2_boundary_probe.py']
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w:gz') as tar:
    for name in files:tar.add(root/name,arcname=name)
    tar.add(root/'data/exp033_cached_tta/candidates_0.5_0.5.json',arcname='d4_candidates.json')
payload=base64.b64encode(b.getvalue()).decode()
d=root/'kaggle/exp066_sam2_boundary_probe';d.mkdir(exist_ok=True)
wrapper=f'''import sys,os,base64,tarfile,io,subprocess
from pathlib import Path
root=Path('/kaggle/working/code');root.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r}))) as tar:tar.extractall(root)
subprocess.check_call([sys.executable,'-m','pip','install','-q','ultralytics==8.4.0','pycocotools','pandas'])
os.environ['PYTHONPATH']=str(root)
import torch
assert torch.cuda.is_available()
print('ACCELERATOR',torch.cuda.get_device_name(0),flush=True)
source=Path('/kaggle/input')
subprocess.run([sys.executable,'-u','-m','scripts.sam2_boundary_probe','--cache',str(root/'d4_candidates.json'),
 '--annotations',str(next(source.rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))),
 '--images',str(next(source.rglob('train_images'))),'--output','/kaggle/working/results','--device','0'],
 cwd=root,check=True,timeout=7200)
'''
(d/'kernel.py').write_text(wrapper)
(d/'kernel-metadata.json').write_text(json.dumps(dict(id='cosmicwitness/sol-exp066-sam2-boundary-probe',
 title='Sol exp066 sam2 boundary probe',code_file='kernel.py',language='python',kernel_type='script',
 is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,machine_shape='NvidiaTeslaT4',
 competition_sources=['filament-segmentation-2026'],dataset_sources=[],kernel_sources=[],model_sources=[]),indent=2))
print(d)
