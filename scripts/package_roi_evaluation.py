"""Package canonical validation of the completed exp073 ROI continuation."""
import base64
import io
import json
from pathlib import Path
import tarfile


root = Path(__file__).resolve().parents[1]
files = [
    'shared/__init__.py',
    'shared/data_split.py',
    'shared/utils.py',
    'experiments/exp_007_tta/src/dihedral.py',
    'experiments/exp_033_inference/src/sweep.py',
    'experiments/exp_033_inference/src/cached_tta.py',
    'scripts/existing_weight_trials.py',
    'scripts/evaluate_yolo_checkpoints.py',
]
archive = io.BytesIO()
with tarfile.open(fileobj=archive, mode='w:gz') as tar:
    for name in files:
        tar.add(root / name, arcname=name)
payload = base64.b64encode(archive.getvalue()).decode()

directory = root / 'kaggle' / 'exp074_roi_evaluation'
directory.mkdir(exist_ok=True)
wrapper = f'''import base64,io,os,subprocess,sys,tarfile
from pathlib import Path

root = Path('/kaggle/working/code')
root.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r}))) as tar:
    tar.extractall(root)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'ultralytics==8.4.0', 'pycocotools'])
os.environ['PYTHONPATH'] = str(root)
import torch
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
print('ACCELERATOR', torch.cuda.get_device_name(0), flush=True)

source = Path('/kaggle/input')
weights = [p for p in source.rglob('best.pt')
           if p.parent.name == 'weights' and p.parent.parent.name == 'roi_continuation']
assert len(weights) == 1, weights
annotations = next(source.rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
images = next(source.rglob('train_images'))
subprocess.run([sys.executable, '-u', '-m', 'scripts.evaluate_yolo_checkpoints',
                '--weights', str(weights[0]), '--annotations', str(annotations),
                '--images', str(images), '--output', '/kaggle/working/validation',
                '--device', '0', '--force-d4'], cwd=root, check=True, timeout=3600)
'''
(directory / 'kernel.py').write_text(wrapper, encoding='utf-8')
(directory / 'kernel-metadata.json').write_text(json.dumps({
    'id': 'cosmicwitness/sol-exp074-roi-validation',
    'title': 'Sol exp074 ROI validation',
    'code_file': 'kernel.py',
    'language': 'python',
    'kernel_type': 'script',
    'is_private': True,
    'enable_gpu': True,
    'enable_tpu': False,
    'enable_internet': True,
    'machine_shape': 'NvidiaTeslaT4',
    'competition_sources': ['filament-segmentation-2026'],
    'dataset_sources': [],
    'model_sources': [],
    'kernel_sources': ['cosmicwitness/sol-exp073-roi-continuation'],
}, indent=2), encoding='utf-8')
print(directory)
