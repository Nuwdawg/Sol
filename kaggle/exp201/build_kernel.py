"""Build exp201's Kaggle kernel: exp139's recipe and the Swin-B voter trained on all labelled photographs.

The code payload is the recovered exp190b kernel code (final_decision/recovered_code/
sol-exp190b-session-long-t4), which trained Swin-S (seed 7) to 0.4677 and Swin-B (seed 11) to 0.4698 on
the canonical split over two T4 sessions. exp201 changes only the arguments: `--fold all` (train +
validation, 1,154 records), no in-run evaluation, fixed saved epochs, a 12-photograph in-sample sanity
check, and session 2 resumes session 1 (RESUME_FROM). The driver keeps every saved checkpoint.

    python kaggle/exp201/build_kernel.py --session 1     # writes kaggle/exp201/s1/{kernel.py,kernel-metadata.json}
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "final_decision" / "recovered_code" / "sol-exp190b-session-long-t4"
CACHE_FILES = ["experiments/exp_147_candidate_cache/__init__.py", "experiments/exp_147_candidate_cache/src/__init__.py",
               "experiments/exp_147_candidate_cache/src/cache.py"]   # session 2 only, from this repository
CACHE_BLOCK = """
# exp201 session 2: cache the final checkpoints on the 180 test photographs, as run.py's m2f stage does
# (Swin-S: full, dihedral and shift views; Swin-B: full view). Skipped late in the session.
elapsed = (time.time() - KERNEL_START) / 3600
print("CACHE_CHECK", round(elapsed, 2), flush=True)
if elapsed < 10.0:
    jobs = []
    for gpu, (name, modes) in enumerate((("swins", "full,d4,shift"), ("swinb", "full"))):
        found = sorted((Path("/kaggle/working") / name).glob("model_e[0-9][0-9][0-9].pt"))
        if not found:
            print("CACHE_SKIPPED", name, "no checkpoint", flush=True)
            continue
        out = Path("/kaggle/working") / ("cache_" + name)
        out.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", "-m", "experiments.exp_147_candidate_cache.src.cache", "--weights", str(found[-1]),
                   "--annotations", str(annotations), "--train-images", str(train_images), "--test-images", str(test_images),
                   "--output", str(out), "--no-val", "--test-modes", modes, "--tag", found[-1].stem]
        jobs.append((name, subprocess.Popen(command, cwd=root, env=gpu_env(gpu), stdout=open(out / "stdout.log", "w"),
                                            stderr=subprocess.STDOUT)))
        print("CACHE_STARTED", name, found[-1].name, modes, "gpu", gpu, flush=True)
    for name, job in jobs:
        print("CACHE_EXIT", name, job.wait(), flush=True)
else:
    print("CACHE_SKIPPED late in the session; cache on CPU instead", flush=True)
"""
RUNS = [["swins", "experiments.exp_173_quality_ema.src.train",
         ["--quality-target", "none", "--ema", "0", "--epoch-policy", "session", "--epochs", "50", "--seed", "7",
          "--fold", "all", "--eval-every", "0", "--save-epochs", "40", "45", "50", "--sanity-photographs", "12"]],
        ["swinb", "experiments.exp_173_quality_ema.src.train",
         ["--quality-target", "none", "--ema", "0", "--epoch-policy", "session", "--epochs", "32", "--seed", "11",
          "--pretrained", "facebook/mask2former-swin-base-IN21k-coco-instance", "--checkpoint", "all",
          "--micro-batch", "2", "--fold", "all", "--eval-every", "0", "--save-epochs", "28", "32",
          "--sanity-photographs", "12"]]]
CLEANUP_OLD = """    if is_session(extra):
        # Keep last.pt for the next session and only the best checkpoint beside it.
        best = [e for e in events(folder) if e.get("event") == "BEST"]
        keep = f"model_e{best[-1]['epoch']:03d}.pt" if best else None
        for path in folder.glob("model_e[0-9][0-9][0-9].pt"):
            if path.name != keep:
                path.unlink()
        print("SESSION_KEPT", name, sorted(p.name for p in folder.glob("*.pt")), flush=True)"""
CLEANUP_NEW = """    if is_session(extra):
        # exp201: no in-run evaluation, so there is no BEST event; keep last.pt and every saved epoch.
        print("SESSION_KEPT", name, sorted(p.name for p in folder.glob("*.pt")), flush=True)"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    driver = (CODE / "kernel_driver.py").read_text()
    manifest = json.loads(re.search(r"manifest = (\{.*?\})\n", driver).group(1))
    for name, digest in manifest.items():
        assert hashlib.sha256((CODE / name).read_bytes()).hexdigest() == digest, name
    if args.session == 2:
        for name in CACHE_FILES:
            manifest[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name in sorted(manifest):
            tar.add((ROOT / name) if name in CACHE_FILES else (CODE / name), arcname=name)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    text = driver.replace("<payload: the files in this folder, verified against the manifest below>", payload)
    old_manifest = re.search(r"manifest = (\{.*?\})\n", text).group(1)
    text = text.replace(old_manifest, json.dumps(manifest))
    text = text.replace("import base64, hashlib, io, json, os, subprocess, sys, tarfile\n",
                        "import base64, hashlib, io, json, os, subprocess, sys, tarfile, time\nKERNEL_START = time.time()\n", 1)
    if args.session == 2:
        assert text.count('print("KERNEL_COMPLETE", flush=True)') == 1
        text = text.replace('print("KERNEL_COMPLETE", flush=True)', CACHE_BLOCK + 'print("KERNEL_COMPLETE", flush=True)')
    text = text.replace("# Driver of the Kaggle kernel, recovered read-only; the base64 code payload is replaced by\n"
                        "# the files in this folder (each matches the SHA256 manifest below).\n",
                        "# exp201 session %d: kaggle/exp201/build_kernel.py (exp190b's driver, all-data arguments).\n" % args.session)
    old_runs = re.search(r"^RUNS = .*$", text, flags=re.M).group(0)
    text = text.replace(old_runs, "RUNS = " + json.dumps(RUNS))
    resume = "" if args.session == 1 else "sol-exp201-alldata-s1-t4"
    text = re.sub(r'^RESUME_FROM = ".*"$', f'RESUME_FROM = "{resume}"', text, flags=re.M)
    assert text.count(CLEANUP_OLD) == 1
    text = text.replace(CLEANUP_OLD, CLEANUP_NEW)
    slug = f"sol-exp201-alldata-s{args.session}-t4"
    out = ROOT / "kaggle" / "exp201" / f"s{args.session}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernel.py").write_text(text)
    meta = dict(id=f"cosmicwitness/{slug}", title=slug.replace("-", " "), code_file="kernel.py", language="python",
                kernel_type="script", is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=True,
                dataset_sources=[], competition_sources=["filament-segmentation-2026"],
                kernel_sources=[] if args.session == 1 else ["cosmicwitness/sol-exp201-alldata-s1-t4"],
                model_sources=[], machine_shape="NvidiaTeslaT4")
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(dict(kernel=str(out / "kernel.py"), chars=len(text), payload_sha256=hashlib.sha256(buffer.getvalue()).hexdigest(),
                          resume=resume, runs=[r[0] for r in RUNS])))


if __name__ == "__main__":
    main()
