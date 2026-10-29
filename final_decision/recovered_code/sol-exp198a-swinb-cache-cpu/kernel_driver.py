# Driver of the Kaggle kernel, recovered read-only; the base64 code payload is replaced by
# the files in this folder (each matches the SHA256 manifest below).
import base64, hashlib, io, json, os, subprocess, sys, tarfile
from pathlib import Path
root = Path("/tmp/code"); root.mkdir(parents=True, exist_ok=True)   # outside /kaggle/working
with tarfile.open(fileobj=io.BytesIO(base64.b64decode("<payload: the files in this folder, verified against the manifest below>")), mode="r:gz") as tar:
    tar.extractall(root)
manifest = {"experiments/exp_033_inference/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_033_inference/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_033_inference/src/cached_tta.py": "f412a55a51afdc81347690bd47bae15d60766b6fc18e4b11490e194f3263b0c6", "experiments/exp_033_inference/src/sweep.py": "9923412adb27609fe2d70a51e7baa80559addf99ca85999c65cd8fc062b6c01b", "experiments/exp_139_mask2former/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_139_mask2former/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_139_mask2former/src/data.py": "9b3054ddcc819f99c24780c5ff06d502d1133514a8b33547ae08de3a40e6f6e9", "experiments/exp_139_mask2former/src/evaluate.py": "df143da11fe4b3546bd86e764612d6f592c4252fd38b642c73dc098424bf1817", "experiments/exp_139_mask2former/src/fast_ops.py": "6c92c64061e6139074e29a4789f7f65399533277bd8c98c8cc0a567797beaaa4", "experiments/exp_139_mask2former/src/model.py": "f3c20a18972e442c5a988b3955c9e7a95f50a53ceac1d6b873e638ae533f0c79", "experiments/exp_147_candidate_cache/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_147_candidate_cache/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_147_candidate_cache/src/cache.py": "39b86fc4dee0a7d18565dfcdb79ceb4eca186628111bd827974e07762b0e2fa4", "shared/__init__.py": "a5d8360a9c1a06dc42b0b1c336e0cfaa7e7edd4605ee67daf1049bc0798edb36", "shared/data_split.py": "f534c584da409642b0462d48c5c88f1658e6ce9f310b22996ca4461369ce780e", "shared/utils.py": "97c2ffb5c4b0b870775769988295533ddf5142499080f41df4fe31b2b6e17a44"}
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
run([sys.executable, "-m", "pip", "install", "-q", "torch==2.8.0", "--index-url", "https://download.pytorch.org/whl/cpu"])
run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6", "pycocotools", "scipy"])
SOURCE, RUN, CHECKPOINT, TAG, SUBSET, TEST = ["sol-exp190b-session-long-t4", "longb", "model_e032.pt", "swinb_e032_s0", "0/5", False]
hits = [p for p in source.rglob(CHECKPOINT) if SOURCE in str(p) and p.parent.name == RUN]
assert len(hits) == 1, [str(p) for p in source.rglob(CHECKPOINT)]
print("CHECKPOINT", hits[0], hits[0].stat().st_size, flush=True)
import torch
print("THREADS", torch.get_num_threads(), flush=True)
command = [sys.executable, "-u", "-m", "experiments.exp_147_candidate_cache.src.cache", "--weights", str(hits[0]),
           "--annotations", str(annotations), "--train-images", str(train_images), "--output", "/kaggle/working",
           "--val-modes", "full,d4", "--test-modes", "full,d4", "--no-parity", "--gpus", "0", "--tag", TAG,
           "--subset", SUBSET]
if TEST:
    command += ["--test-images", str(test_images)]
code = subprocess.run(command, cwd=root, env=env).returncode
print("CACHE_EXIT", code, sorted(p.name for p in Path("/kaggle/working").glob(TAG + "*")), flush=True)
for name in (TAG + "_summary.json",):
    path = Path("/kaggle/working") / name
    if path.exists():
        print("SUMMARY", path.read_text()[:1500], flush=True)
print("KERNEL_COMPLETE", flush=True)
