"""Package the CPU kernel that sweeps emission rules, including dilation, on exp097's pool.

The frozen scorer refuses positive `grow` because exp_005 measured that dilation
always lost - on a model trained at 1280. exp097's model is trained at 2048, its
optimum has already moved from erosion -1 to 0, and nothing has tested whether it
moved further. This kernel asks, on the D4 pool exp098 already computed, so it
needs no GPU and no inference.

It carries exp002's stored pool purely as a parity check: the same code must
reproduce 0.4558638687292126 at conf 0.35, erosion 1, area 300 before any new
number is believed.

Usage:
    python scripts/package_exp104_emission_grid.py --checkpoint best.pt
    cd kaggle/exp104_emission_grid_cpu && kaggle kernels push -p .
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
ENSEMBLE_KERNEL = "cosmicwitness/sol-exp102-pool-ensemble-cpu"
KERNEL_ID = "cosmicwitness/sol-exp104-emission-grid-cpu"
ANNOTATIONS_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
FILES = [
    "shared/__init__.py", "shared/data_split.py", "shared/utils.py",
    "experiments/exp_033_inference/__init__.py",
    "experiments/exp_033_inference/src/__init__.py",
    "experiments/exp_033_inference/src/sweep.py",
    "experiments/exp_097_runpod_yolo/__init__.py",
    "experiments/exp_097_runpod_yolo/src/__init__.py",
    "experiments/exp_097_runpod_yolo/src/events.py",
    "experiments/exp_097_runpod_yolo/src/sweep_emission.py",
    "data/exp033_cached_tta/candidates_0.5_0.5.json",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--grows", default="-2,-1,0,1,2")
    parser.add_argument("--areas", default="100,300,500,800")
    parser.add_argument("--confs", default="0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50")
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
ensemble=sorted(source.rglob('ensemble_pool.json'))
pools=ensemble if ensemble else sorted(source.rglob('d4_pool_{stem}.json'))
label='exp102_ensemble' if ensemble else 'exp097_{stem}'
assert pools,'no pool mounted: '+str(sorted(p.name for p in source.rglob('*pool*.json')))
print('POOL',label,str(pools[0]),flush=True)
subprocess.run([sys.executable,'-u','-m','experiments.exp_097_runpod_yolo.src.sweep_emission',
 '--annotations',str(annotations),'--output','/kaggle/working/emission_grid',
 '--pool',label+'='+str(pools[0]),
 '--parity-pool',str(root/'data/exp033_cached_tta/candidates_0.5_0.5.json'),
 '--grows='+{args.grows!r},'--areas='+{args.areas!r},'--confs='+{args.confs!r}],
 cwd=root,check=True,timeout=28000)
print('KERNEL_COMPLETE',flush=True)
'''
    directory = ROOT / "kaggle" / "exp104_emission_grid_cpu"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kernel.py").write_text(wrapper, encoding="utf-8")
    (directory / "kernel-metadata.json").write_text(json.dumps({
        "id": KERNEL_ID,
        # Kaggle derives the slug from the title; keep the two slug-identical (CLAUDE.md 14).
        "title": "sol exp104 emission grid cpu",
        "code_file": "kernel.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_tpu": False,
        # No `machine_shape` on a CPU kernel: a null value fails the push.
        "enable_internet": True,
        "competition_sources": ["filament-segmentation-2026"],
        "dataset_sources": [], "kernel_sources": [EVALUATION_KERNEL, ENSEMBLE_KERNEL], "model_sources": [],
    }, indent=2), encoding="utf-8")
    print(json.dumps(dict(directory=str(directory), checkpoint=args.checkpoint,
                          grows=args.grows, areas=args.areas, kernel_bytes=len(wrapper)), indent=2))


if __name__ == "__main__":
    main()
