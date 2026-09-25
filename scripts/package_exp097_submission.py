"""Package the Kaggle T4 kernel that emits a submission from an exp097 checkpoint.

The rule comes from exp098's report, never from a guess here, and the kernel
refuses to emit unless it first reproduces exp098's validation PQ. The weights
arrive through the private dataset the pod uploaded, and exp098's own validation
pool arrives through `kernel_sources`, so the replay costs seconds instead of a
second forty-minute D4 pass. Nothing is downloaded to the user's PC.

Usage, with the numbers taken from exp098's SUMMARY line:
    python scripts/package_exp097_submission.py --checkpoint stage1_best.pt \
        --mode d4 --conf 0.35 --grow -1 --min-area 300 --expected-pq 0.4712
    cd kaggle/exp100_exp097_submission_t4 && kaggle kernels push -p .
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
WEIGHTS_DATASET = "cosmicwitness/sol-exp097-yolov8l-2048"
EVALUATION_KERNEL = "cosmicwitness/sol-exp098-exp097-evaluation-t4"
KERNEL_ID = "cosmicwitness/sol-exp100-exp097-submission-t4"
ULTRALYTICS = "8.4.153"   # the version the pod trained with
ANNOTATIONS_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
FILES = [
    "shared/__init__.py", "shared/data_split.py", "shared/utils.py",
    # exp_007_tta has no __init__.py files; it imports as a namespace package,
    # exactly as exp098's packager already relies on.
    "experiments/exp_007_tta/src/dihedral.py",
    "experiments/exp_033_inference/__init__.py",
    "experiments/exp_033_inference/src/__init__.py",
    "experiments/exp_033_inference/src/sweep.py",
    "experiments/exp_033_inference/src/cached_tta.py",
    "scripts/emit_exp097_submission.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="file name inside the weights dataset")
    parser.add_argument("--mode", choices=("single", "d4"), required=True)
    parser.add_argument("--conf", type=float, required=True)
    parser.add_argument("--grow", type=int, required=True)
    parser.add_argument("--min-area", type=int, required=True)
    parser.add_argument("--expected-pq", type=float, required=True)
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
    slug = WEIGHTS_DATASET.split("/")[1]
    rule = ["--mode", args.mode, "--conf", repr(args.conf), "--grow", str(args.grow),
            "--min-area", str(args.min_area), "--expected-pq", repr(args.expected_pq)]
    stem = Path(args.checkpoint).stem

    wrapper = f'''import base64,hashlib,io,json,os,subprocess,sys,tarfile
from pathlib import Path
os.environ['SOL_SOURCE_REVISION']={revision!r}
root=Path('/kaggle/working/code');root.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r})),mode='r:gz') as tar:tar.extractall(root)
manifest={manifest!r}
for name,digest in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
(root/'scripts'/'__init__.py').touch()
os.environ['PYTHONPATH']=str(root)
subprocess.check_call([sys.executable,'-m','pip','install','-q','pycocotools','ultralytics=={ULTRALYTICS}'])
import torch
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0),torch.cuda.get_device_name(0)
source=Path('/kaggle/input')
annotations=next(source.rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
assert hashlib.sha256(annotations.read_bytes()).hexdigest()=={ANNOTATIONS_SHA256!r}
train_images=annotations.parent/'train_images';assert train_images.is_dir()
test_images=next(p for p in source.rglob('test_images') if p.is_dir())
weights=[p.parent for p in source.rglob('manifest.json') if {slug!r} in str(p)]
assert len(weights)==1,[str(p) for p in weights]
checkpoint=weights[0]/{args.checkpoint!r}
assert checkpoint.is_file(),sorted(p.name for p in weights[0].glob('*.pt'))
pod=json.loads((weights[0]/'manifest.json').read_text())
digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert digest==pod[{args.checkpoint!r}]['sha256'],(digest,pod[{args.checkpoint!r}]['sha256'])
print('CHECKPOINT',{args.checkpoint!r},digest,flush=True)
pools=[p for p in source.rglob('d4_pool_{stem}.json')]
replay=['--replay-pool',str(pools[0])] if pools else []
print('REPLAY_POOL',[str(p) for p in pools],flush=True)
subprocess.run([sys.executable,'-u','-m','scripts.emit_exp097_submission',
 '--weights',str(checkpoint),'--annotations',str(annotations),
 '--train-images',str(train_images),'--test-images',str(test_images),
 '--output','/kaggle/working']+{rule!r}+replay,cwd=root,check=True,timeout=40000)
print('KERNEL_COMPLETE',flush=True)
'''
    directory = ROOT / "kaggle" / "exp100_exp097_submission_t4"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kernel.py").write_text(wrapper, encoding="utf-8")
    (directory / "kernel-metadata.json").write_text(json.dumps({
        "id": KERNEL_ID,
        # Kaggle derives the slug from the title; keep the two slug-identical (CLAUDE.md 14).
        "title": "sol exp100 exp097 submission t4",
        "code_file": "kernel.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "competition_sources": ["filament-segmentation-2026"],
        "dataset_sources": [WEIGHTS_DATASET],
        "kernel_sources": [EVALUATION_KERNEL], "model_sources": [],
    }, indent=2), encoding="utf-8")
    print(json.dumps(dict(directory=str(directory), checkpoint=args.checkpoint, mode=args.mode,
                          rule=dict(conf=args.conf, grow=args.grow, min_area=args.min_area),
                          expected_pq=args.expected_pq, kernel_bytes=len(wrapper)), indent=2))


if __name__ == "__main__":
    main()
