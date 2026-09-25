"""Package bounded native YOLO fine-tuning only after a matching GPU lifecycle probe."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--probe-results', required=True)
args = p.parse_args()
probe_root = Path(args.probe_results)
probe = json.loads((probe_root/'probe/probe.json').read_text())
assert probe['status'] == 'complete' and len(probe['steps']) == 5
assert probe['fp32_validator_metrics'] and probe['reloaded_checkpoint']
assert probe['cuda_peak_bytes'] < 14*1024**3
assert all(s['mask_shape'] == [1,2048,2048] for s in probe['steps'])
resolved = json.loads((probe_root/'probe/resolved_experiment.json').read_text())
assert resolved['validation_precision'] == 'fp32'
assert resolved['source_sha256'] == hashlib.sha256((root/'shared/chunked_mask_loss.py').read_bytes()).hexdigest()
trainer_path = 'experiments/exp_032_chunked/src/train.py'
probed_source = (probe_root/'code'/trainer_path).read_bytes()
assert resolved['trainer_sha256'] == hashlib.sha256(probed_source).hexdigest()
current_source = (root/trainer_path).read_bytes()
# The manual probe uses full lr0; this CLI-only change removes the upstream
# AdamW bias warmup spike. Criterion, validation and reload code stay identical.
assert current_source == probed_source or current_source.replace(b'cos_lr=True, warmup_bias_lr=0.0,', b'cos_lr=True,') == probed_source
files = ['shared/__init__.py','shared/data_split.py','shared/utils.py','shared/chunked_mask_loss.py',
         'experiments/exp_002_yolo_seg/src/prepare_yolo.py','experiments/exp_032_chunked/src/train.py',
         'experiments/exp_007_tta/src/dihedral.py','experiments/exp_033_inference/src/sweep.py',
         'experiments/exp_033_inference/src/cached_tta.py',
         'scripts/existing_weight_trials.py','scripts/evaluate_yolo_checkpoints.py']
manifest = {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files}
revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
buf = io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as tar:
    for name in files:tar.add(root/name,arcname=name)
payload = base64.b64encode(buf.getvalue()).decode()
directory = root/'kaggle/exp068_chunked_train'
directory.mkdir(exist_ok=True)
wrapper = f'''import os,sys,io,tarfile,base64,subprocess,json,hashlib,time
from pathlib import Path
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
os.environ['SOL_SOURCE_REVISION']={revision!r}
root=Path('/kaggle/working/code');root.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r}))) as tar:tar.extractall(root)
manifest={manifest!r}
for name,sha in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
(root/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
(root/'probe_gate.json').write_text({json.dumps(probe)!r})
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
images=next(source.rglob('train_images'))
dataset='/kaggle/temp/yolo_ds'
subprocess.run([sys.executable,'-u','-m','experiments.exp_002_yolo_seg.src.prepare_yolo',
 '--annotations',str(annotations),'--images',str(images),'--output',dataset,
 '--polygon-offset','0.5'],cwd=root,check=True,timeout=600)
training=subprocess.Popen([sys.executable,'-u','-m','experiments.exp_032_chunked.src.train',
 '--data',dataset+'/data.yaml','--weights',str(weights[0]),'--project','/kaggle/working/runs',
 '--name','native','--imgsz','2048','--batch','1','--chunk-size','8','--device','0',
 '--epochs','8','--time-hours','2.5','--fp32-validation'],cwd=root)
deadline=time.monotonic()+10000
reported=False
while training.poll() is None:
 resolved=Path('/kaggle/working/runs/native/resolved_experiment.json')
 if resolved.exists() and not reported:
  print('RESOLVED_EXPERIMENT',resolved.read_text(),flush=True);reported=True
 if time.monotonic()>deadline:
  training.terminate();raise RuntimeError('training exceeded bounded runtime')
 time.sleep(5)
assert training.returncode==0,training.returncode
run=Path('/kaggle/working/runs/native')
subprocess.run([sys.executable,'-u','-m','scripts.evaluate_yolo_checkpoints','--weights',
 str(run/'weights/best.pt'),str(run/'weights/last.pt'),'--annotations',str(annotations),
 '--images',str(images),'--output','/kaggle/working/validation','--device','0'],
 cwd=root,check=True,timeout=2000)
'''
(directory/'kernel.py').write_text(wrapper,encoding='utf-8')
(directory/'kernel-metadata.json').write_text(json.dumps(dict(
    id='cosmicwitness/sol-exp068-native-mask-training',title='Sol exp068 native mask training',
    code_file='kernel.py',language='python',kernel_type='script',is_private=True,
    enable_gpu=True,enable_tpu=False,enable_internet=True,machine_shape='NvidiaTeslaT4',
    competition_sources=['filament-segmentation-2026'],dataset_sources=[],
    kernel_sources=['cosmicwitness/sol-exp002-yolo-seg'],model_sources=[]),indent=2))
print(directory)
