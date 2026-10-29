# Driver of the Kaggle kernel, recovered read-only; the base64 code payload is replaced by
# the files in this folder (each matches the SHA256 manifest below).
import base64, hashlib, io, json, os, subprocess, sys, tarfile
from pathlib import Path
root = Path("/tmp/code"); root.mkdir(parents=True, exist_ok=True)   # outside /kaggle/working
with tarfile.open(fileobj=io.BytesIO(base64.b64decode("<payload: the files in this folder, verified against the manifest below>")), mode="r:gz") as tar:
    tar.extractall(root)
manifest = {"experiments/exp_033_inference/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_033_inference/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_033_inference/src/cached_tta.py": "f412a55a51afdc81347690bd47bae15d60766b6fc18e4b11490e194f3263b0c6", "experiments/exp_033_inference/src/sweep.py": "9923412adb27609fe2d70a51e7baa80559addf99ca85999c65cd8fc062b6c01b", "experiments/exp_139_mask2former/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_139_mask2former/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_139_mask2former/src/data.py": "9b3054ddcc819f99c24780c5ff06d502d1133514a8b33547ae08de3a40e6f6e9", "experiments/exp_139_mask2former/src/evaluate.py": "df143da11fe4b3546bd86e764612d6f592c4252fd38b642c73dc098424bf1817", "experiments/exp_139_mask2former/src/fast_ops.py": "6c92c64061e6139074e29a4789f7f65399533277bd8c98c8cc0a567797beaaa4", "experiments/exp_139_mask2former/src/model.py": "f3c20a18972e442c5a988b3955c9e7a95f50a53ceac1d6b873e638ae533f0c79", "experiments/exp_173_quality_ema/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_173_quality_ema/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_173_quality_ema/src/quality.py": "0fe016cdfc4090fefc971a6156c64b3f8a1bd6acf410a18f23d51d878f805e2d", "experiments/exp_173_quality_ema/src/train.py": "b2e8efe26f75a4127d0d7170e013670dfe0521286a10baa74c94efbaeb89ad80", "shared/__init__.py": "a5d8360a9c1a06dc42b0b1c336e0cfaa7e7edd4605ee67daf1049bc0798edb36", "shared/data_split.py": "f534c584da409642b0462d48c5c88f1658e6ce9f310b22996ca4461369ce780e", "shared/utils.py": "97c2ffb5c4b0b870775769988295533ddf5142499080f41df4fe31b2b6e17a44"}
for name, digest in manifest.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
print("CODE_VERIFIED", len(manifest), flush=True)

def run(command, **kwargs):
    print("RUN", " ".join(command), flush=True)
    return subprocess.run(command, check=True, **kwargs)

source = Path("/kaggle/input")
annotations = next(source.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
assert hashlib.sha256(annotations.read_bytes()).hexdigest() == "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
train_images = annotations.parent / "train_images"
test_images = next(p for p in source.rglob("test_images") if p.is_dir())
env = dict(os.environ, PYTHONPATH=str(root), PYTHONUNBUFFERED="1", HF_HUB_ENABLE_HF_TRANSFER="0")
run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchvision", "torchaudio"])
run([sys.executable, "-m", "pip", "install", "-q", "torch==2.8.0", "--index-url", "https://download.pytorch.org/whl/cu128"])
run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6", "kernels", "pycocotools", "scipy"])

import time
RUNS = [["long", "experiments.exp_173_quality_ema.src.train", ["--quality-target", "none", "--ema", "0", "--epoch-policy", "session", "--epochs", "50", "--seed", "7"]], ["longb", "experiments.exp_173_quality_ema.src.train", ["--quality-target", "none", "--ema", "0", "--epoch-policy", "session", "--epochs", "40", "--seed", "11", "--pretrained", "facebook/mask2former-swin-base-IN21k-coco-instance", "--checkpoint", "all", "--micro-batch", "2"]]]
RESUME_FROM = "sol-exp190a-session-long-t4"
HEAD = "experiments.exp_180_quality_head.src.train"
import shutil
def is_session(extra):
    return any(a == "--epoch-policy" and b == "session" for a, b in zip(extra, extra[1:]))
common = ["--annotations", str(annotations), "--images", str(train_images), "--fp16",
          "--batch", "8", "--micro-batch", "2", "--checkpoint", "whole", "--msda", "hub",
          "--workers", "3", "--no-final-d4", "--sweep-processes", "2", "--probe-steps", "20",
          "--epoch-policy", "fit", "--epochs", "32", "--min-epochs", "16",
          "--price-per-hour", "1.0", "--budget-usd", "10.4", "--reserve-usd", "0.6",
          "--fold", "train", "--eval-every", "4"]

def gpu_env(gpu):
    return dict(env, CUDA_VISIBLE_DEVICES=str(gpu), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

def events(path):
    path = Path(path) / "events.log"
    found = []
    if path.exists():
        for line in path.read_text().splitlines():
            tag, _, payload = line.partition(" ")
            try:
                item = json.loads(payload) if payload else {}
            except ValueError:
                continue
            item["event"] = tag
            found.append(item)
    return found

REFINE = "experiments.exp_193_point_refine.src.train"
if any(module == REFINE for _, module, _ in RUNS):
    check = subprocess.run([sys.executable, "-u", "-m", "experiments.exp_193_point_refine.src.check_loss"],
                           cwd=root, env=dict(env, CUDA_VISIBLE_DEVICES="")).returncode
    print("CHECK_REFINE_EXIT", check, flush=True)
    if check != 0:
        raise SystemExit("exp193 check failed; nothing was trained")

if any(module == HEAD for _, module, _ in RUNS):
    check = subprocess.run([sys.executable, "-u", "-m", "experiments.exp_180_quality_head.src.check_loss"],
                           cwd=root, env=dict(env, CUDA_VISIBLE_DEVICES="")).returncode
    print("CHECK_EXIT", check, flush=True)
    if check != 0:
        raise SystemExit("exp180 loss check failed; nothing was trained")

for name, module, extra in RUNS:
    smoke = Path("/kaggle/working/smoke_" + name)
    smoke.mkdir(parents=True, exist_ok=True)
    tiny = ["--model", "tiny", "--batch", "2", "--micro-batch", "1", "--max-steps", "3", "--eval-every", "1",
            "--eval-limit", "2", "--no-preload", "--workers", "0", "--sweep-processes", "1", "--checkpoint", "none",
            "--epoch-policy", "fixed", "--budget-usd", "0"]
    if module.endswith("exp_177_student.src.student"):
        tiny += ["--links", "/tmp/smoke_links_" + name]
    code = subprocess.run([sys.executable, "-u", "-m", module, "--output", str(smoke)] + common + extra + tiny,
                          cwd=root, env=gpu_env(0)).returncode
    requalified = 0
    if module == HEAD and code == 0:
        requalified = subprocess.run([sys.executable, "-u", "-m", HEAD, "requalify", "--output", str(smoke),
                                      "--annotations", str(annotations), "--processes", "1"], cwd=root, env=env).returncode
    reports = sorted(p.name for p in smoke.glob("*_sweep.json"))
    print("SMOKE", name, code, requalified, reports, flush=True)
    if code != 0 or requalified != 0 or not reports or (module == HEAD and not any("q1" in r for r in reports)):
        raise SystemExit("smoke test " + name + " failed; the long runs were not started")

processes = {}
for gpu, (name, module, extra) in enumerate(RUNS):
    out = Path("/kaggle/working") / name
    if RESUME_FROM:
        # exp190: continue the previous session's run from its last.pt, schedule and all.
        found = [p.parent for p in source.rglob("last.pt") if RESUME_FROM in str(p) and p.parent.name == name]
        assert len(found) == 1, (name, [str(p) for p in source.rglob("last.pt")])
        shutil.copytree(found[0], out, dirs_exist_ok=True)
        extra = extra + ["--resume"]
        print("RESUMING", name, "from", found[0], sorted(p.name for p in out.iterdir())[:40], flush=True)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "stdout.log", "w")
    processes[name] = (subprocess.Popen([sys.executable, "-u", "-m", module, "--output", str(out)] + common + extra,
                                        cwd=root, env=gpu_env(gpu), stdout=log, stderr=subprocess.STDOUT), module)
    print("STARTED", name, module, "gpu", gpu, extra, flush=True)

while any(p.poll() is None for p, _ in processes.values()):
    time.sleep(300)
    for name in processes:
        found = events(Path("/kaggle/working") / name)
        if found:
            print("LAST", name, json.dumps(found[-1])[:400], flush=True)
for name, (p, module) in processes.items():
    print("EXIT", name, p.returncode, flush=True)
    folder = Path("/kaggle/working") / name
    if module == HEAD:
        done = subprocess.run([sys.executable, "-u", "-m", HEAD, "requalify", "--output", str(folder),
                               "--annotations", str(annotations), "--processes", "4"], cwd=root, env=env).returncode
        print("REQUALIFY_EXIT", name, done, flush=True)
    for e in events(folder):
        if e.get("event") in ("SCHEDULE", "EVAL", "BEST", "SUMMARY", "ERROR", "SANITY", "PROBE"):
            print("RESULT", name, json.dumps(e)[:600], flush=True)
    extra = next(r[2] for r in RUNS if r[0] == name)
    patterns = ("*_cache.json", "*.partial") if is_session(extra) else ("*_cache.json", "last.pt", "*.partial")
    for pattern in patterns:
        for path in folder.glob(pattern):
            path.unlink()
    if is_session(extra):
        # Keep last.pt for the next session and only the best checkpoint beside it.
        best = [e for e in events(folder) if e.get("event") == "BEST"]
        keep = f"model_e{best[-1]['epoch']:03d}.pt" if best else None
        for path in folder.glob("model_e[0-9][0-9][0-9].pt"):
            if path.name != keep:
                path.unlink()
        print("SESSION_KEPT", name, sorted(p.name for p in folder.glob("*.pt")), flush=True)
print("KERNEL_COMPLETE", flush=True)
