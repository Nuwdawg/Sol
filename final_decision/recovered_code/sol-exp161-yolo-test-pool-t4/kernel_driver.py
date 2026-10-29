# Driver of the Kaggle kernel, recovered read-only; the base64 code payload is replaced by
# the files in this folder (each matches the SHA256 manifest below).
import base64, hashlib, io, json, os, subprocess, sys, tarfile
from pathlib import Path
root = Path("/tmp/code"); root.mkdir(parents=True, exist_ok=True)   # outside /kaggle/working
with tarfile.open(fileobj=io.BytesIO(base64.b64decode("<payload: the files in this folder, verified against the manifest below>")), mode="r:gz") as tar:
    tar.extractall(root)
manifest = {"experiments/exp_007_tta/src/dihedral.py": "190f8ba462816a8c878afa4d81ccce338c0da0f618524f918b58cc39fb50646f", "experiments/exp_033_inference/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_033_inference/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_033_inference/src/cached_tta.py": "f412a55a51afdc81347690bd47bae15d60766b6fc18e4b11490e194f3263b0c6", "experiments/exp_033_inference/src/sweep.py": "9923412adb27609fe2d70a51e7baa80559addf99ca85999c65cd8fc062b6c01b", "experiments/exp_161_yolo_test_pool/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_161_yolo_test_pool/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_161_yolo_test_pool/src/pool.py": "eaf3fe25b1c5bc16c9435bd1b5d47d74ff5c1f75eae542a7a533e997bf7c0a03", "shared/__init__.py": "a5d8360a9c1a06dc42b0b1c336e0cfaa7e7edd4605ee67daf1049bc0798edb36", "shared/data_split.py": "f534c584da409642b0462d48c5c88f1658e6ce9f310b22996ca4461369ce780e", "shared/utils.py": "97c2ffb5c4b0b870775769988295533ddf5142499080f41df4fe31b2b6e17a44"}
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
run([sys.executable, "-m", "pip", "install", "-q", "pycocotools", "ultralytics==8.4.153"])
weights = [p.parent for p in source.rglob("manifest.json") if "sol-exp097-yolov8l-2048" in str(p)]
assert len(weights) == 1, [str(p) for p in weights]
checkpoint = weights[0] / "stage2_best.pt"
pod = json.loads((weights[0] / "manifest.json").read_text())
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert digest == pod["stage2_best.pt"]["sha256"], (digest, pod["stage2_best.pt"]["sha256"])
print("CHECKPOINT", digest, flush=True)
out = Path("/kaggle/working")
processes = []
for shard in range(2):
    processes.append(subprocess.Popen([sys.executable, "-u", "-m", "experiments.exp_161_yolo_test_pool.src.pool",
                                       "--weights", str(checkpoint), "--annotations", str(annotations),
                                       "--train-images", str(train_images), "--test-images", str(test_images),
                                       "--output", str(out), "--shard", str(shard), "--shards", "2"],
                                      cwd=root, env=dict(env, CUDA_VISIBLE_DEVICES=str(shard))))
codes = [p.wait() for p in processes]
print("EXIT", codes, flush=True)
sys.path.insert(0, str(root))
from experiments.exp_161_yolo_test_pool.src.pool import compare
for split in ("val", "test"):
    merged = {}
    for shard in range(2):
        part = out / f"yolov8l_{split}_pool_shard{shard}.json"
        merged.update(json.loads(part.read_text()))
        part.unlink()
    (out / f"yolov8l_{split}_pool.json").write_text(json.dumps(merged))
    print("POOL", split, len(merged), sum(len(v) for v in merged.values()), flush=True)
reference = [p for p in source.rglob("d4_pool_stage2_best.json") if "sol-exp098-exp097-evaluation-t4" in str(p)]
assert len(reference) == 1, [str(p) for p in reference]
facts = compare(json.loads((out / "yolov8l_val_pool.json").read_text()), json.loads(reference[0].read_text()))
(out / "parity.json").write_text(json.dumps(facts, indent=1))
print("PARITY", json.dumps(facts), flush=True)
print("KERNEL_COMPLETE", flush=True)
