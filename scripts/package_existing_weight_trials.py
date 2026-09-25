"""Build three independent CPU inference kernels from the reviewed local code."""
import base64
import io
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parents[1]
files=['shared/__init__.py','shared/data_split.py','shared/utils.py',
       'experiments/exp_007_tta/src/dihedral.py',
       'experiments/exp_033_inference/src/sweep.py',
       'experiments/exp_033_inference/src/cached_tta.py',
       'scripts/existing_weight_trials.py']
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
    for name in files:
        archive.add(root/name,arcname=name)
payload=base64.b64encode(buffer.getvalue()).decode()
jobs=[]
for number,mode,title in [(59,'soft','soft d4'),(60,'nms','mask nms'),(61,'weights','weight interpolation')]:
    slug=f'sol-exp{number:03d}-{title.replace(" ","-")}'
    directory=root/f'kaggle/exp{number:03d}_{mode}'
    directory.mkdir(exist_ok=True)
    wrapper=f'''import os,sys,base64,io,tarfile,subprocess
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONUNBUFFERED']='1'
root=Path('/kaggle/working/code');root.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r}))) as archive:
    archive.extractall(root)
subprocess.check_call([sys.executable,'-m','pip','install','-q','ultralytics==8.4.0','pycocotools','pandas'])
os.environ['PYTHONPATH']=str(root)
base=[p for p in Path('/kaggle/input').rglob('best.pt') if p.parent.name=='checkpoints']
assert len(base)==1,base
annotations=next(Path('/kaggle/input').rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
images=next(Path('/kaggle/input').rglob('train_images'))
command=[sys.executable,'-u','-m','scripts.existing_weight_trials','--mode',{mode!r},
    '--weights',str(base[0]),'--annotations',str(annotations),'--images',str(images),
    '--output','/kaggle/working/results']
if {mode!r}=='weights':
    donor=[p for p in Path('/kaggle/input').rglob('best.pt') if p.parent.name=='weights' and p.parent.parent.name=='frozen']
    assert len(donor)==1,donor
    command.extend(['--donor',str(donor[0])])
subprocess.run(command,cwd=root,check=True,timeout=40000)
'''
    (directory/'kernel.py').write_text(wrapper,encoding='utf-8')
    metadata=dict(id=f'cosmicwitness/{slug}',title=f'Sol exp{number:03d} {title}',
                  code_file='kernel.py',language='python',kernel_type='script',is_private=True,
                  enable_gpu=False,enable_tpu=False,enable_internet=True,dataset_sources=[],
                  competition_sources=['filament-segmentation-2026'],
                  kernel_sources=['cosmicwitness/sol-exp002-yolo-seg']+(['cosmicwitness/sol-exp031-frozen'] if mode=='weights' else []),
                  model_sources=[])
    (directory/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    jobs.append(dict(slug=metadata['id'],directory=str(directory),output=str(root/f'data/exp{number:03d}_output')))
Path('data/existing_weight_jobs.json').write_text(json.dumps(jobs,indent=2))
print(json.dumps(jobs,indent=2))
