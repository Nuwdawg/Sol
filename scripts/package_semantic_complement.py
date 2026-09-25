"""Package frozen exp004 semantic-complement validation on a T4."""
import base64
import io
import json
from pathlib import Path
import tarfile


root = Path(__file__).resolve().parents[1]
files = [
    "shared/__init__.py",
    "shared/data_split.py",
    "shared/utils.py",
    "experiments/exp_033_inference/src/sweep.py",
    "scripts/evaluate_semantic_complement.py",
    "data/exp076_physical_rescue_v7/feature_stage.json",
]
archive = io.BytesIO()
with tarfile.open(fileobj=archive, mode="w:gz") as tar:
    for name in files:
        tar.add(root / name, arcname=name)
payload = base64.b64encode(archive.getvalue()).decode()

directory = root / "kaggle" / "exp081_semantic_support_shortlist_t4"
directory.mkdir(exist_ok=True)
wrapper = f'''import base64,io,os,subprocess,sys,tarfile
from pathlib import Path

root = Path('/kaggle/working/code')
root.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r}))) as tar:
    tar.extractall(root)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'segmentation-models-pytorch', 'pycocotools'])
os.environ['PYTHONPATH'] = str(root)
import torch
assert torch.cuda.is_available()
print('ACCELERATOR', torch.cuda.get_device_name(0), flush=True)

source = Path('/kaggle/input')
weights = [p for p in source.rglob('best.pt') if p.parent.name == 'checkpoints']
candidates = list(source.rglob('d4_candidates.json'))
annotations = list(source.rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
image_dirs = [p for p in source.rglob('train_images') if p.is_dir()]
assert len(weights) == 1, weights
assert len(candidates) == 1, candidates
assert annotations and image_dirs
subprocess.run([sys.executable, '-u', '-m', 'scripts.evaluate_semantic_complement',
                '--checkpoint', str(weights[0]), '--candidates', str(candidates[0]),
                '--annotations', str(annotations[0]), '--images', str(image_dirs[0]),
                '--output', '/kaggle/working/validation', '--support-only',
                '--candidate-audit', str(root / 'data/exp076_physical_rescue_v7/feature_stage.json')],
               cwd=root, check=True, timeout=7200)
'''
(directory / "kernel.py").write_text(wrapper, encoding="utf-8")
(directory / "kernel-metadata.json").write_text(json.dumps({
    "id": "cosmicwitness/sol-exp081-semantic-support-shortlist-t4",
    "title": "Sol exp081 semantic support shortlist T4",
    "code_file": "kernel.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4",
    "competition_sources": ["filament-segmentation-2026"],
    "dataset_sources": [],
    "model_sources": [],
    "kernel_sources": [
        "cosmicwitness/sol-exp004-tpu",
        "cosmicwitness/sol-exp066-sam2-boundary-probe",
    ],
}, indent=2), encoding="utf-8")
print(directory)
