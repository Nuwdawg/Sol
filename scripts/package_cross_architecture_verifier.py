"""Package the Kaggle CPU kernel that tests cross-architecture proposal confirmation.

exp124 asks whether exp097 independently proposes a filament where exp002 emitted a
sub-threshold candidate. It needs no GPU and no inference: both D4 pools already
exist, so this is RLE arithmetic over 106 photographs.

Where each input comes from, and why nothing is downloaded:

- exp002's validation pool travels in the private `sol-exp002-pools` dataset, whose
  `manifest.json` carries a SHA256 per file and is verified on the pod.
- exp097's validation pool stays in exp098's kernel output and arrives through
  `kernel_sources`. It is never copied to the user's machine.

The tar is deliberately small. `sweep.py` imports only `shared.data_split` and
`shared.utils`, so `cached_tta` and `dihedral` are not needed here - the dependency
runs the other way. `shared/utils.py` wants pandas, which Kaggle provides.

CPU kernels must omit `machine_shape` entirely. Passing it as null made an earlier
push fail, and exp102, exp104 and exp107 all run without the key.

Usage:
    python scripts/package_cross_architecture_verifier.py
    cd kaggle/exp124_cross_architecture_cpu && kaggle kernels push -p .
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
POOLS_DATASET = "cosmicwitness/sol-exp002-pools"
EVALUATION_KERNEL = "cosmicwitness/sol-exp098-exp097-evaluation-t4"
KERNEL_ID = "cosmicwitness/sol-exp124-cross-architecture-cpu"
KERNEL_TITLE = "sol exp124 cross architecture cpu"
ANNOTATIONS_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
FILES = [
    "shared/__init__.py", "shared/data_split.py", "shared/utils.py",
    "experiments/exp_033_inference/__init__.py",
    "experiments/exp_033_inference/src/__init__.py",
    "experiments/exp_033_inference/src/sweep.py",
    "scripts/audit_cross_architecture_verifier.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floor", type=float, default=0.10)
    parser.add_argument("--grow", type=int, default=-1)
    parser.add_argument("--min-area", type=int, default=300)
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
    pools_slug = POOLS_DATASET.split("/")[1]
    rule = ["--floor", repr(args.floor), "--grow", str(args.grow),
            "--min-area", str(args.min_area)]

    wrapper = f'''import base64,hashlib,io,json,os,subprocess,sys,tarfile
from pathlib import Path
os.environ['SOL_SOURCE_REVISION']={revision!r}
root=Path('/kaggle/working/code');root.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r})),mode='r:gz') as tar:tar.extractall(root)
manifest={manifest!r}
for name,digest in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
(root/'scripts'/'__init__.py').touch()
os.environ['PYTHONPATH']=str(root)
subprocess.check_call([sys.executable,'-m','pip','install','-q','pycocotools'])
source=Path('/kaggle/input')
annotations=next(source.rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
assert hashlib.sha256(annotations.read_bytes()).hexdigest()=={ANNOTATIONS_SHA256!r}
pools=[p.parent for p in source.rglob('manifest.json') if {pools_slug!r} in str(p)]
assert len(pools)==1,[str(p) for p in pools]
declared=json.loads((pools[0]/'manifest.json').read_text())
for name,facts in declared.items():
    got=hashlib.sha256((pools[0]/name).read_bytes()).hexdigest()
    assert got==facts['sha256'],(name,got,facts['sha256'])
print('EXP002_POOLS',sorted(declared),flush=True)
partner=[p for p in source.rglob('d4_pool_stage2_best.json')]
assert partner,'exp098 validation pool not attached; kernel_sources missing'
print('EXP097_POOL',[str(p) for p in partner],flush=True)
subprocess.run([sys.executable,'-u','-m','scripts.audit_cross_architecture_verifier',
 '--exp002-pool',str(pools[0]/'exp002_val_pool.json'),
 '--exp097-pool',str(partner[0]),
 '--annotations',str(annotations),
 '--output','/kaggle/working']+{rule!r},cwd=root,check=True,timeout=30000)
print('KERNEL_COMPLETE',flush=True)
'''
    directory = ROOT / "kaggle" / "exp124_cross_architecture_cpu"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kernel.py").write_text(wrapper, encoding="utf-8")
    # No machine_shape key: a CPU kernel rejects it, including as null.
    (directory / "kernel-metadata.json").write_text(json.dumps({
        "id": KERNEL_ID,
        # Kaggle derives the slug from the title; keep the two slug-identical (CLAUDE.md 14).
        "title": KERNEL_TITLE,
        "code_file": "kernel.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": True,
        "competition_sources": ["filament-segmentation-2026"],
        "dataset_sources": [POOLS_DATASET],
        "kernel_sources": [EVALUATION_KERNEL], "model_sources": [],
    }, indent=2), encoding="utf-8")
    print(json.dumps(dict(directory=str(directory), files=len(FILES),
                          kernel_bytes=len(wrapper),
                          rule=dict(floor=args.floor, grow=args.grow,
                                    min_area=args.min_area)), indent=2))


if __name__ == "__main__":
    main()
