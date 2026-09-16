"""Package the CPU Kaggle kernel that fuses exp002's and exp097's D4 pools.

No GPU and no inference: both pools already exist, so this is RLE arithmetic and
runs in minutes beside any GPU job. exp002's pool ships inside the payload;
exp097's arrives from exp098's kernel output through `kernel_sources`.

Usage:
    python scripts/package_exp102_pool_ensemble.py --checkpoint best.pt
    cd kaggle/exp102_pool_ensemble_cpu && kaggle kernels push -p .
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_KERNEL = "cosmicwitness/sol-exp098-exp097-evaluation-t4"
KERNEL_ID = "cosmicwitness/sol-exp102-pool-ensemble-cpu"
ANNOTATIONS_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
FILES = [
    "shared/__init__.py", "shared/data_split.py", "shared/utils.py",
    "experiments/exp_007_tta/src/dihedral.py",
    "experiments/exp_033_inference/__init__.py",
    "experiments/exp_033_inference/src/__init__.py",
    "experiments/exp_033_inference/src/sweep.py",
    "experiments/exp_033_inference/src/cached_tta.py",
    "experiments/exp_097_runpod_yolo/__init__.py",
    "experiments/exp_097_runpod_yolo/src/__init__.py",
    "experiments/exp_097_runpod_yolo/src/events.py",
    "experiments/exp_097_runpod_yolo/src/evaluate_ensemble.py",
    "experiments/exp_097_runpod_yolo/src/ensemble_pools.py",
    "data/exp033_cached_tta/candidates_0.5_0.5.json",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="best.pt",
                        help="which exp098 pool to pull, by checkpoint name")
    parser.add_argument("--link", type=float, default=0.5)
    parser.add_argument("--vote", type=float, default=0.5)
    args = parser.parse_args()

    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    manifest = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for name in FILES:
            tar.add(ROOT / name, arcname=name)
    payload = base64.b64encode(archive.getvalue()).decode("ascii")
    stem = Path(args.checkpoint).stem

    wrapper = f'''import base64,hashlib,io,json,os,subprocess,sys,tarfile
from pathlib import Path
os.environ['SOL_SOURCE_REVISION']={revision!r}
root=Path('/kaggle/working/code');root.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r})),mode='r:gz') as tar:tar.extractall(root)
manifest={manifest!r}
for name,digest in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
os.environ['PYTHONPATH']=str(root)
subprocess.check_call([sys.executable,'-m','pip','install','-q','pycocotools'])
source=Path('/kaggle/input')
annotations=next(source.rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
assert hashlib.sha256(annotations.read_bytes()).hexdigest()=={ANNOTATIONS_SHA256!r}
pools=sorted(source.rglob('d4_pool_{stem}.json'))
assert pools,'exp098 pool for {stem} is not mounted: '+str(sorted(p.name for p in source.rglob('d4_pool_*.json')))
print('POOLS',[str(p) for p in pools],flush=True)
subprocess.run([sys.executable,'-u','-m','experiments.exp_097_runpod_yolo.src.ensemble_pools',
 '--annotations',str(annotations),'--output','/kaggle/working/pool_ensemble',
 '--link',{str(args.link)!r},'--vote',{str(args.vote)!r},
 '--pool','exp002_d4='+str(root/'data/exp033_cached_tta/candidates_0.5_0.5.json'),
 '--pool','exp097_{stem}='+str(pools[0])],cwd=root,check=True,timeout=20000)
print('KERNEL_COMPLETE',flush=True)
'''
    directory = ROOT / "kaggle" / "exp102_pool_ensemble_cpu"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kernel.py").write_text(wrapper, encoding="utf-8")
    (directory / "kernel-metadata.json").write_text(json.dumps({
        "id": KERNEL_ID,
        # Kaggle derives the slug from the title; keep the two slug-identical (CLAUDE.md 14).
        "title": "sol exp102 pool ensemble cpu",
        "code_file": "kernel.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_tpu": False,
        # No `machine_shape`: Kaggle rejects a null value, and a CPU kernel has
        # no accelerator to name. Sending one fails the push with a bare
        # "Expecting value: line 1 column 1".
        "enable_internet": True,
        "competition_sources": ["filament-segmentation-2026"],
        "dataset_sources": [], "kernel_sources": [EVALUATION_KERNEL], "model_sources": [],
    }, indent=2), encoding="utf-8")
    print(json.dumps(dict(directory=str(directory), checkpoint=args.checkpoint,
                          link=args.link, vote=args.vote, kernel_bytes=len(wrapper)), indent=2))


if __name__ == "__main__":
    main()
