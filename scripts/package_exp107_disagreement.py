"""Package the CPU kernel that tests true cross-architecture boundary disagreement.

The last untested trigger in the user's sequence. A 2026-09-16 local run passed
exp002's pool as both arguments and therefore measured self-consistency, not
disagreement; exp097's pool lives in exp098's kernel output, so the real test runs
on Kaggle where both are mountable. No GPU and no inference: the pools exist.

The question is narrow. Where two independently trained networks - yolo11m-seg at
1280 and YOLOv8l-seg at 2048 - disagree about a boundary, is that mask more likely
to be one of the near-threshold 1:1 failures? Disagreement is a *gating* signal: it
says a boundary is unreliable, not which way to move it. It is worth measuring
because exp102 showed these two models carry complementary information (+0.001340
pooled, against +0.000050 for three checkpoints of one run), while every trigger
derived from a single model's own geometry or semantics has failed.

Precision at operating points decides this, not AUC. Five triggers have now been
rejected on that distinction.

Usage:
    python scripts/package_exp107_disagreement.py
    cd kaggle/exp107_disagreement_cpu && kaggle kernels push -p .
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
KERNEL_ID = "cosmicwitness/sol-exp107-disagreement-cpu"
ANNOTATIONS_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
FILES = [
    "shared/__init__.py", "shared/data_split.py", "shared/utils.py",
    "experiments/exp_033_inference/__init__.py",
    "experiments/exp_033_inference/src/__init__.py",
    "experiments/exp_033_inference/src/sweep.py",
    "experiments/exp_097_runpod_yolo/__init__.py",
    "experiments/exp_097_runpod_yolo/src/__init__.py",
    "experiments/exp_097_runpod_yolo/src/events.py",
    "experiments/exp_106_support/src/audit_support.py",
    "data/exp033_cached_tta/candidates_0.5_0.5.json",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--min-area", type=int, default=100)
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
assert pools,'exp098 pool for {stem} not mounted: '+str(sorted(p.name for p in source.rglob('d4_pool_*.json')))
print('EXP097_POOL',str(pools[0]),flush=True)
exp002=str(root/'data/exp033_cached_tta/candidates_0.5_0.5.json')
# Audited pool is exp002's (the one whose 1:1 populations every earlier audit used),
# with exp097's pool as the independent second opinion.
subprocess.run([sys.executable,'-u','-m','experiments.exp_106_support.src.audit_support',
 '--pool',exp002,'--other-pool',str(pools[0]),'--annotations',str(annotations),
 '--output','/kaggle/working/disagreement','--radius',{str(args.radius)!r},
 '--conf',{str(args.conf)!r},'--min-area',{str(args.min_area)!r}],cwd=root,check=True,timeout=20000)
# And the reverse direction: audit exp097's masks with exp002 as the second opinion.
subprocess.run([sys.executable,'-u','-m','experiments.exp_106_support.src.audit_support',
 '--pool',str(pools[0]),'--other-pool',exp002,'--annotations',str(annotations),
 '--output','/kaggle/working/disagreement_reverse','--radius',{str(args.radius)!r},
 '--conf',{str(args.conf)!r},'--min-area',{str(args.min_area)!r}],cwd=root,check=True,timeout=20000)
print('KERNEL_COMPLETE',flush=True)
'''
    directory = ROOT / "kaggle" / "exp107_disagreement_cpu"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kernel.py").write_text(wrapper, encoding="utf-8")
    (directory / "kernel-metadata.json").write_text(json.dumps({
        "id": KERNEL_ID,
        # Kaggle derives the slug from the title; keep the two slug-identical (CLAUDE.md 14).
        "title": "sol exp107 disagreement cpu",
        "code_file": "kernel.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_tpu": False,
        # No `machine_shape` on a CPU kernel: a null value fails the push.
        "enable_internet": True,
        "competition_sources": ["filament-segmentation-2026"],
        "dataset_sources": [], "kernel_sources": [EVALUATION_KERNEL], "model_sources": [],
    }, indent=2), encoding="utf-8")
    print(json.dumps(dict(directory=str(directory), radius=args.radius,
                          kernel_bytes=len(wrapper)), indent=2))


if __name__ == "__main__":
    main()
