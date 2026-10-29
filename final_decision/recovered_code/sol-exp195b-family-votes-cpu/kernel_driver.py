# Driver of the Kaggle kernel, recovered read-only; the base64 code payload is replaced by
# the files in this folder (each matches the SHA256 manifest below).
import base64, hashlib, io, json, os, subprocess, sys, tarfile
from pathlib import Path
root = Path("/tmp/code"); root.mkdir(parents=True, exist_ok=True)   # outside /kaggle/working
with tarfile.open(fileobj=io.BytesIO(base64.b64decode("<payload: the files in this folder, verified against the manifest below>")), mode="r:gz") as tar:
    tar.extractall(root)
manifest = {"experiments/exp_033_inference/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_033_inference/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_033_inference/src/cached_tta.py": "f412a55a51afdc81347690bd47bae15d60766b6fc18e4b11490e194f3263b0c6", "experiments/exp_033_inference/src/sweep.py": "9923412adb27609fe2d70a51e7baa80559addf99ca85999c65cd8fc062b6c01b", "experiments/exp_149_evidence_fusion/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_149_evidence_fusion/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_149_evidence_fusion/src/evidence.py": "c7f23c2e18e9a87012f787d2d87c814bf310a8fd662759cb27e519832574d27b", "experiments/exp_157_shift_views/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_157_shift_views/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_157_shift_views/src/views.py": "32c5d3419a0a69a8e3e3eaf3c6f556cf7c9d99d18f628cf374aefe508e646df9", "experiments/exp_159_yolo_votes/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_159_yolo_votes/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_159_yolo_votes/src/votes.py": "2514f54d0cc12e2191d9fb0d19c29e9d3688890f56725b07c649bbf6a8526495", "experiments/exp_166_yolo_mask_votes/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_166_yolo_mask_votes/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_166_yolo_mask_votes/src/mask_votes.py": "a601316d5bc3480399cd0f77927dca4e21aa7880c54fca8f182d6ea1df4e5494", "experiments/exp_168_joint_tuning/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_168_joint_tuning/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_168_joint_tuning/src/joint.py": "06a42420090be9e07eab107e425c072f7a8aa11a381740085287ec7e17fea57a", "experiments/exp_185_m2f_votes/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_185_m2f_votes/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_185_m2f_votes/src/votes.py": "970921de01415803650f04f0a13afe3e3e514cb097bafed1e38ad89fcc78d522", "experiments/exp_195_family_votes/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_195_family_votes/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_195_family_votes/src/votes.py": "75c71fb8f40e281883d807ceb1706f17c027dedd83e94a78f906a5d079c478dd", "shared/__init__.py": "a5d8360a9c1a06dc42b0b1c336e0cfaa7e7edd4605ee67daf1049bc0798edb36", "shared/data_split.py": "f534c584da409642b0462d48c5c88f1658e6ce9f310b22996ca4461369ce780e", "shared/solar_disk.py": "25b30f7a9583ad9994a30467d9b13918a8156922850e51dec4763b630ab9e9c9", "shared/utils.py": "97c2ffb5c4b0b870775769988295533ddf5142499080f41df4fe31b2b6e17a44"}
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
def one(name, where):
    hits = [p for p in source.rglob(name) if where in str(p)]
    assert len(hits) == 1, (name, where, [str(p) for p in hits])
    return str(hits[0])
NEW = {"rfdetr1152": ["sol-exp194b-rfdetr-train-t4", "rfdetr_r1152_val_cache.json"]}
extra = []
for tag, (kernel, filename) in NEW.items():
    extra += ["--new", tag + "=" + one(filename, kernel)]
for flags in ([], ["--with-masks"]):
    out = Path("/kaggle/working") / ("with_masks" if flags else "scores_only")
    step = subprocess.run([sys.executable, "-u", "-m", "experiments.exp_195_family_votes.src.votes",
                           "--cache", one("val_cache.json", "sol-exp147-candidate-cache-t4"),
                           "--shift", one("shift_val_cache.json", "sol-exp157-shift-cache-t4"),
                           "--yolo", "yolov8l=" + one("d4_pool_stage2_best.json", "sol-exp098-exp097-evaluation-t4"),
                           "--yolo", "yolo11m=" + one("exp002_val_pool.json", "sol-exp002-pools"),
                           "--exp168-report", one("exp168_report.json", "sol-exp168-joint-tuning-cpu"),
                           "--annotations", str(annotations), "--output", str(out)] + extra + flags,
                          cwd=root, env=env)
    print("EXIT", flags, step.returncode, flush=True)
print("KERNEL_COMPLETE", flush=True)
