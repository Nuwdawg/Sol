"""Package the exp099 Kaggle T4 kernel: exp097's pod code run against Ultralytics 8.4.153.

The kernel embeds the exact payload bytes the pod verifies. Packaging refuses to
run unless those bytes hash to the SHA256 in kaggle/exp097_code_dataset/manifest.json,
the value the pod request carries, so a pass certifies the code the pod will run.
The smoke driver itself (experiments/exp_097_runpod_yolo/smoke/smoke.py) is embedded
beside it and is not part of the pod payload. Only the log's SMOKE_CHECK and
SMOKE_RESULT lines are read back.

Usage:
    python scripts/package_exp097_code.py        # first: rebuild the payload and manifest
    python scripts/package_exp099_smoke.py
    cd kaggle/exp099_exp097_smoke_t4 && kaggle kernels push -p .
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_ID = "cosmicwitness/sol-exp099-exp097-smoke-t4"
SMOKE = ROOT / "experiments" / "exp_097_runpod_yolo" / "smoke" / "smoke.py"
ULTRALYTICS = "8.4.153"   # identical to pod/bootstrap.sh


def build_payload() -> bytes:
    spec = importlib.util.spec_from_file_location(
        "package_exp097_code", ROOT / "scripts" / "package_exp097_code.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_payload()


def main() -> None:
    payload = build_payload()
    digest = hashlib.sha256(payload).hexdigest()
    manifest = json.loads((ROOT / "kaggle" / "exp097_code_dataset" / "manifest.json").read_text())
    if manifest["payload_sha256"] != digest:
        raise SystemExit("the sources differ from the built code dataset: "
                         "run scripts/package_exp097_code.py first")
    encoded_payload = base64.b64encode(payload).decode("ascii")
    encoded_smoke = base64.b64encode(SMOKE.read_bytes()).decode("ascii")
    wrapper = f'''import base64,hashlib,io,os,subprocess,sys,tarfile
from pathlib import Path
PAYLOAD_SHA256={digest!r}
payload=base64.b64decode({encoded_payload!r})
assert hashlib.sha256(payload).hexdigest()==PAYLOAD_SHA256,'payload hash mismatch'
root=Path('/kaggle/working/code');root.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as tar:tar.extractall(root)
smoke=root/'smoke.py';smoke.write_bytes(base64.b64decode({encoded_smoke!r}))
print('PAYLOAD',PAYLOAD_SHA256,flush=True)
subprocess.check_call([sys.executable,'-m','pip','install','-q','ultralytics=={ULTRALYTICS}','shapely','pycocotools'])
import torch
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0),'no T4'
print('DEVICE',torch.cuda.get_device_name(0),torch.__version__,flush=True)
code=subprocess.run([sys.executable,'-u',str(smoke)],cwd='/kaggle/working',env=dict(os.environ,PYTHONPATH=str(root))).returncode
print('KERNEL_COMPLETE',code,flush=True)
sys.exit(code)
'''
    directory = ROOT / "kaggle" / "exp099_exp097_smoke_t4"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kernel.py").write_text(wrapper, encoding="utf-8")
    (directory / "kernel-metadata.json").write_text(json.dumps({
        "id": KERNEL_ID,
        # Kaggle derives the slug from the title; keep the two slug-identical (CLAUDE.md 14).
        "title": "sol exp099 exp097 smoke t4",
        "code_file": "kernel.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "competition_sources": ["filament-segmentation-2026"],
        "dataset_sources": [], "kernel_sources": [], "model_sources": [],
    }, indent=2), encoding="utf-8")
    print(json.dumps(dict(directory=str(directory), payload_sha256=digest,
                          kernel_bytes=len(wrapper)), indent=2))


if __name__ == "__main__":
    main()
