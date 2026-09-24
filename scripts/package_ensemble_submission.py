"""Package the Kaggle T4 kernel that emits the exp102 two-architecture ensemble submission.

Built on `package_exp097_submission.py`, which already works, with three changes:

- the tar carries `evaluate_ensemble` (for `fuse_sources`) and its `events` import,
  because importing that module executes its top level on the pod;
- exp002's pools arrive through a private dataset rather than being recomputed,
  since its D4 validation pool and raw test views were already saved locally;
- exp097's validation pool still arrives through `kernel_sources` from exp098, so
  the replay costs seconds rather than a second forty-minute D4 pass.

The rule comes from exp102's report and is never guessed here. The kernel refuses
to write a test row unless it first reproduces PQ 0.468146 on the canonical
validation split (CLAUDE.md section 5).

`experiments/__init__.py` is deliberately absent, matching the exp100 kernel that
already runs: `exp_007_tta` has no `__init__.py` either and imports as a namespace
package. Adding one here would diverge from the proven arrangement.

Usage:
    python scripts/package_ensemble_submission.py
    cd kaggle/exp119_ensemble_submission_t4 && kaggle kernels push -p .
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
POOLS_DATASET = "cosmicwitness/sol-exp002-pools"
EVALUATION_KERNEL = "cosmicwitness/sol-exp098-exp097-evaluation-t4"
KERNEL_ID = "cosmicwitness/sol-exp119-ensemble-submission-t4"
KERNEL_TITLE = "sol exp119 ensemble submission t4"
ULTRALYTICS = "8.4.153"
ANNOTATIONS_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
CHECKPOINT = "stage2_best.pt"
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
    "scripts/emit_ensemble_submission.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--grow", type=int, default=-1)
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--expected-pq", type=float, default=0.468146)
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
    weights_slug = WEIGHTS_DATASET.split("/")[1]
    pools_slug = POOLS_DATASET.split("/")[1]
    rule = ["--conf", repr(args.conf), "--grow", str(args.grow),
            "--min-area", str(args.min_area), "--expected-pq", repr(args.expected_pq)]

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
test_images=next(p for p in source.rglob('test_images') if p.is_dir())
weights=[p.parent for p in source.rglob('manifest.json') if {weights_slug!r} in str(p)]
assert len(weights)==1,[str(p) for p in weights]
checkpoint=weights[0]/{CHECKPOINT!r}
assert checkpoint.is_file(),sorted(p.name for p in weights[0].glob('*.pt'))
pod=json.loads((weights[0]/'manifest.json').read_text())
digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert digest==pod[{CHECKPOINT!r}]['sha256'],(digest,pod[{CHECKPOINT!r}]['sha256'])
print('CHECKPOINT',{CHECKPOINT!r},digest,flush=True)
pools=[p.parent for p in source.rglob('manifest.json') if {pools_slug!r} in str(p)]
assert len(pools)==1,[str(p) for p in pools]
declared=json.loads((pools[0]/'manifest.json').read_text())
for name,facts in declared.items():
    got=hashlib.sha256((pools[0]/name).read_bytes()).hexdigest()
    assert got==facts['sha256'],(name,got,facts['sha256'])
print('EXP002_POOLS',sorted(declared),flush=True)
val=[p for p in source.rglob('d4_pool_stage2_best.json')]
assert val,'exp098 validation pool not attached; kernel_sources missing'
print('EXP097_VAL_POOL',[str(p) for p in val],flush=True)
subprocess.run([sys.executable,'-u','-m','scripts.emit_ensemble_submission',
 '--weights',str(checkpoint),'--annotations',str(annotations),
 '--test-images',str(test_images),
 '--exp002-val-pool',str(pools[0]/'exp002_val_pool.json'),
 '--exp097-val-pool',str(val[0]),
 '--exp002-test-views',str(pools[0]/'exp002_test_views_0.json'),
                          str(pools[0]/'exp002_test_views_1.json'),
 '--output','/kaggle/working']+{rule!r},cwd=root,check=True,timeout=40000)
print('KERNEL_COMPLETE',flush=True)
'''
    directory = ROOT / "kaggle" / "exp119_ensemble_submission_t4"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kernel.py").write_text(wrapper, encoding="utf-8")
    (directory / "kernel-metadata.json").write_text(json.dumps({
        "id": KERNEL_ID,
        # Kaggle derives the slug from the title; keep the two slug-identical (CLAUDE.md 14).
        "title": KERNEL_TITLE,
        "code_file": "kernel.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "competition_sources": ["filament-segmentation-2026"],
        "dataset_sources": [WEIGHTS_DATASET, POOLS_DATASET],
        "kernel_sources": [EVALUATION_KERNEL], "model_sources": [],
    }, indent=2), encoding="utf-8")
    print(json.dumps(dict(directory=str(directory), checkpoint=CHECKPOINT,
                          rule=dict(conf=args.conf, grow=args.grow, min_area=args.min_area),
                          expected_pq=args.expected_pq, files=len(FILES),
                          kernel_bytes=len(wrapper)), indent=2))


if __name__ == "__main__":
    main()
