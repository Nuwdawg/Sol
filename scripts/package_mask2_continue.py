"""Package a two-epoch TPU continuation from exp046's checkpoint."""
from __future__ import annotations
import base64, io, json, tarfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
files = ['shared/__init__.py','shared/data_split.py','shared/query_mask_loss.py',
         'shared/mask2_checkpointing.py','kaggle/exp046_mask2_train/train.py']
b = io.BytesIO()
with tarfile.open(fileobj=b, mode='w:gz') as a:
    for name in files:
        a.add(root/name, arcname='train.py' if name.endswith('/train.py') else name)
payload = base64.b64encode(b.getvalue()).decode()
directory = root/'kaggle/exp058_mask2_continue'; directory.mkdir(exist_ok=True)
wrapper = f'''import os,sys,base64,io,tarfile,subprocess
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''; os.environ['PJRT_DEVICE']='TPU'
root=Path('/kaggle/working/code'); root.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r}))) as archive:
    archive.extractall(root)
subprocess.check_call([sys.executable,'-m','pip','install','-q','transformers==4.57.6','pycocotools','scipy'])
os.environ['PYTHONPATH']=str(root)
resume=next(Path('/kaggle/input').rglob('checkpoint.pt'))
subprocess.run([sys.executable,'-u',str(root/'train.py'),'--bf16','--checkpoint-activations',
    '--max-epochs','2','--seconds','27000','--resume',str(resume)],cwd=root,check=True,timeout=28800)
'''
(directory/'kernel.py').write_text(wrapper,encoding='utf-8')
metadata = dict(id='cosmicwitness/sol-exp058-mask2-continue',
    title='Sol exp058 mask2 TPU continuation', code_file='kernel.py', language='python',
    kernel_type='script', is_private=True, enable_gpu=False, enable_tpu=True,
    machine_shape='Tpu1VmV38', enable_internet=True, dataset_sources=[],
    competition_sources=['filament-segmentation-2026'],
    kernel_sources=['cosmicwitness/sol-exp046-mask2-train'], model_sources=[])
(directory/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(directory)
