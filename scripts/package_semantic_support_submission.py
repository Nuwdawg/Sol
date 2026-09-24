"""Package frozen semantic-support test inference on a T4."""
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
    "experiments/exp_033_inference/src/cached_tta.py",
    "scripts/evaluate_semantic_complement.py",
    "scripts/build_semantic_support_submission.py",
]
archive = io.BytesIO()
with tarfile.open(fileobj=archive, mode="w:gz") as tar:
    for name in files:
        tar.add(root / name, arcname=name)
payload = base64.b64encode(archive.getvalue()).decode()

directory = root / "kaggle" / "exp082_semantic_support_submission_t4"
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
source = Path('/kaggle/input')
weights = [p for p in source.rglob('best.pt') if p.parent.name == 'checkpoints']
image_dirs = [p for p in source.rglob('test_images') if p.is_dir()]
view_files = list(source.rglob('test_views.json'))
assert len(weights) == 1, weights
assert len(image_dirs) == 1, image_dirs
assert len(view_files) == 2, view_files
subprocess.run([sys.executable, '-u', '-m', 'scripts.build_semantic_support_submission',
                '--checkpoint', str(weights[0]),
                '--views', str(view_files[0]), str(view_files[1]),
                '--images', str(image_dirs[0]),
                '--output', '/kaggle/working/submission.csv'],
               cwd=root, check=True, timeout=3600)
'''
(directory / "kernel.py").write_text(wrapper, encoding="utf-8")
(directory / "kernel-metadata.json").write_text(json.dumps({
    "id": "cosmicwitness/sol-exp082-semantic-support-submission-t4",
    "title": "Sol exp082 semantic support submission T4",
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
        "cosmicwitness/sol-exp037-test-0",
        "cosmicwitness/sol-exp037-test-1",
    ],
}, indent=2), encoding="utf-8")
print(directory)
