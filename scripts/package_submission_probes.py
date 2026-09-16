"""One bounded Kaggle GPU job: three existing-weight probes plus exp058 evaluation."""
import base64
import io
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parents[1]
refs=root/'data/submission_probe_references';refs.mkdir(exist_ok=True)
for mode,path in [('soft','data/exp059_output/results/soft_0.4.json'),
                  ('nms','data/exp060_output/results/mask_0.3.json'),
                  ('weights','data/exp061_recovery/results/alpha_0.25.json')]:
    data=json.loads((root/path).read_text())
    names=sorted(data,key=lambda n:-len(data[n]))[:4]
    (refs/f'{mode}.json').write_text(json.dumps({n:data[n] for n in names}))
files=['shared/__init__.py','shared/data_split.py','shared/utils.py',
       'experiments/exp_007_tta/src/dihedral.py','experiments/exp_033_inference/src/sweep.py',
       'experiments/exp_033_inference/src/cached_tta.py','experiments/exp_046_mask2/src/evaluate.py',
       'scripts/existing_weight_trials.py','scripts/existing_weight_test.py']
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w:gz') as tar:
    for name in files:tar.add(root/name,arcname=name)
    for path in refs.glob('*.json'):tar.add(path,arcname='reference/'+path.name)
payload=base64.b64encode(buffer.getvalue()).decode()
directory=root/'kaggle/exp063_submission_probes';directory.mkdir(exist_ok=True)
wrapper=f'''import os,sys,io,tarfile,base64,subprocess,time
from pathlib import Path
root=Path('/kaggle/working/code');root.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r}))) as tar:tar.extractall(root)
subprocess.check_call([sys.executable,'-m','pip','install','-q','ultralytics==8.4.0','transformers==4.57.6','pycocotools','pandas'])
os.environ['PYTHONPATH']=str(root)
import torch
assert torch.cuda.is_available(),'GPU job must not silently fall back to CPU'
print('ACCELERATOR',torch.cuda.get_device_name(0),flush=True)
source=Path('/kaggle/input')
base=[p for p in source.rglob('best.pt') if p.parent.name=='checkpoints'];assert len(base)==1
donor=[p for p in source.rglob('best.pt') if p.parent.parent.name=='frozen'];assert len(donor)==1
images=next(source.rglob('train_images'));test=next(source.rglob('test_images'))
subprocess.run([sys.executable,'-u','-m','scripts.existing_weight_test','--weights',str(base[0]),
 '--donor',str(donor[0]),'--images',str(images),'--test-images',str(test),'--reference',str(root/'reference'),
 '--output','/kaggle/working/probes','--device','0'],cwd=root,check=True,timeout=9000)
checkpoints=list(source.rglob('checkpoint.pt'));assert len(checkpoints)==1
annotations=next(source.rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
subprocess.run([sys.executable,'-u','-m','experiments.exp_046_mask2.src.evaluate','--checkpoint',str(checkpoints[0]),
 '--annotations',str(annotations),'--images',str(images),'--test-images',str(test),
 '--output','/kaggle/working/mask2','--device','cuda:0','--force-submission'],cwd=root,check=True,timeout=3000)
'''
(directory/'kernel.py').write_text(wrapper)
metadata=dict(id='cosmicwitness/sol-exp063-submission-probes',title='Sol exp063 submission probes',
 code_file='kernel.py',language='python',kernel_type='script',is_private=True,
 enable_gpu=True,enable_tpu=False,enable_internet=True,dataset_sources=[],
 machine_shape='NvidiaTeslaT4',
 competition_sources=['filament-segmentation-2026'],
 kernel_sources=['cosmicwitness/sol-exp002-yolo-seg','cosmicwitness/sol-exp031-frozen',
                 'cosmicwitness/sol-exp058-mask2-tpu-continuation'],model_sources=[])
(directory/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
print(directory)
