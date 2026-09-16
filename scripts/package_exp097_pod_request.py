"""Generate the exact create-pod body for exp097, for review before anything is billed.

Nothing here calls RunPod. It writes `experiments/exp_097_runpod_yolo/pod/pod_request.json`,
which is passed verbatim to the RunPod MCP `create-pod` tool only after explicit approval.

Usage:
    python scripts/package_exp097_code.py            # prints SOL_CODE_SHA256
    python scripts/package_exp097_pod_request.py --code-sha256 <sha> --price-per-hour <live price>

The price must be the live figure read with `list-gpu-types` immediately before launch;
the pod uses it for its spend accounting and its budget watchdog.

Kaggle key: by user decision (2026-09-15) it is passed directly in the pod's
environment rather than through a RunPod secret. This file only ever holds a
placeholder; the real key is inserted into the create-pod call at launch and is
never written to the repository. An unreplaced placeholder fails loudly at the
launcher's first Kaggle download.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POD_DIR = ROOT / "experiments" / "exp_097_runpod_yolo" / "pod"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--code-sha256", required=True)
    parser.add_argument("--price-per-hour", type=float, required=True)
    parser.add_argument("--budget-usd", type=float, default=9.95)
    parser.add_argument("--gpu", default="NVIDIA L40S")
    parser.add_argument("--cloud", choices=("COMMUNITY", "SECURE"), default="COMMUNITY")
    parser.add_argument("--volume-gb", type=int, default=60)
    parser.add_argument("--disk-gb", type=int, default=30)
    parser.add_argument("--min-vcpu-per-gpu", type=int, default=8)
    parser.add_argument("--min-ram-per-gpu", type=int, default=40)
    parser.add_argument("--image", default="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404")
    parser.add_argument("--upload-dataset", default="cosmicwitness/sol-exp097-yolov8l-2048")
    parser.add_argument("--kaggle-key-placeholder", default="__INSERT_KAGGLE_KEY_AT_LAUNCH__",
                        help="the real key goes only into the create-pod call, never this file")
    args = parser.parse_args()

    if len(args.code_sha256) != 64:
        raise SystemExit("code sha256 must be 64 hex characters")
    launcher = (POD_DIR / "launcher.sh").read_text(encoding="utf-8")
    if "'" in launcher:
        raise SystemExit("launcher.sh must not contain single quotes; it is wrapped in bash -lc '...'")

    body = dict(
        name="sol-exp097-yolov8l-2048",
        image=args.image,
        cloud=args.cloud,
        gpu=dict(id=args.gpu, count=1, minCudaVersion="12.8",
                 minVcpuCountPerGpu=args.min_vcpu_per_gpu, minRamPerGpu=args.min_ram_per_gpu),
        disk=args.disk_gb,
        mounts=dict(persistent=dict(size=args.volume_gb, path="/workspace")),
        env={
            "KAGGLE_USERNAME": "cosmicwitness",
            "KAGGLE_KEY": args.kaggle_key_placeholder,
            "SOL_CODE_SHA256": args.code_sha256,
            "SOL_PRICE_PER_HOUR": f"{args.price_per_hour:.4f}",
            "SOL_BUDGET_USD": f"{args.budget_usd:.2f}",
            "SOL_UPLOAD_DATASET": args.upload_dataset,
            "SOL_BILLING_OFFSET_SECONDS": "300",
            # Ubuntu 24.04 marks the system Python externally managed (PEP 668); the
            # RunPod PyTorch image keeps torch there, so installs must go there too.
            "PIP_BREAK_SYSTEM_PACKAGES": "1",
        },
        args="bash -lc '" + launcher.strip() + "'",
    )
    target = POD_DIR / "pod_request.json"
    target.write_text(json.dumps(dict(body=body), indent=2), encoding="utf-8")
    print(json.dumps(dict(written=str(target), gpu=args.gpu, cloud=args.cloud,
                          price_per_hour=args.price_per_hour, budget_usd=args.budget_usd,
                          max_hours_at_price=round(args.budget_usd / args.price_per_hour, 2)),
                     indent=2))


if __name__ == "__main__":
    main()
