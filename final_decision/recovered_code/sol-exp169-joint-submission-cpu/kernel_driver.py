# Driver of the Kaggle kernel, recovered read-only; the base64 code payload is replaced by
# the files in this folder (each matches the SHA256 manifest below).
import base64, hashlib, io, json, os, subprocess, sys, tarfile
from pathlib import Path
root = Path("/tmp/code"); root.mkdir(parents=True, exist_ok=True)   # outside /kaggle/working
with tarfile.open(fileobj=io.BytesIO(base64.b64decode("<payload: the files in this folder, verified against the manifest below>")), mode="r:gz") as tar:
    tar.extractall(root)
manifest = {"experiments/exp_033_inference/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_033_inference/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_033_inference/src/cached_tta.py": "f412a55a51afdc81347690bd47bae15d60766b6fc18e4b11490e194f3263b0c6", "experiments/exp_033_inference/src/sweep.py": "9923412adb27609fe2d70a51e7baa80559addf99ca85999c65cd8fc062b6c01b", "experiments/exp_149_evidence_fusion/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_149_evidence_fusion/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_149_evidence_fusion/src/evidence.py": "c7f23c2e18e9a87012f787d2d87c814bf310a8fd662759cb27e519832574d27b", "experiments/exp_152_rule_and_emit/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_152_rule_and_emit/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_152_rule_and_emit/src/emit.py": "919199b78b4909def470bdc0b2553036e3b5e591ad293c14bc30ebed90c6d04c", "experiments/exp_154_view_masks/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_154_view_masks/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_154_view_masks/src/view_masks.py": "fdbd14a344991f265365dcc184e6f21414a34581672e74dc011d8bbcde16c40e", "experiments/exp_156_review/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_156_review/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_156_review/src/review.py": "bef5d46f518c9b60508b2851550cd3fcd4d467e3b0c636249b9ceddc7cdd23e7", "experiments/exp_157_shift_views/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_157_shift_views/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_157_shift_views/src/views.py": "32c5d3419a0a69a8e3e3eaf3c6f556cf7c9d99d18f628cf374aefe508e646df9", "experiments/exp_159_yolo_votes/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_159_yolo_votes/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_159_yolo_votes/src/votes.py": "2514f54d0cc12e2191d9fb0d19c29e9d3688890f56725b07c649bbf6a8526495", "experiments/exp_162_final_submission/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_162_final_submission/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_162_final_submission/src/final.py": "a9daec59db51858e749ed4a3a99ab34b03af004011623f79f7c6ed95f924e95f", "experiments/exp_166_yolo_mask_votes/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_166_yolo_mask_votes/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_166_yolo_mask_votes/src/mask_votes.py": "a601316d5bc3480399cd0f77927dca4e21aa7880c54fca8f182d6ea1df4e5494", "experiments/exp_168_joint_tuning/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_168_joint_tuning/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_168_joint_tuning/src/joint.py": "06a42420090be9e07eab107e425c072f7a8aa11a381740085287ec7e17fea57a", "experiments/exp_169_joint_submission/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_169_joint_submission/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "experiments/exp_169_joint_submission/src/joint_submit.py": "2de6a575b9190c18fc93892a8a84c90cd37a71396545d1186bb61c467be3f048", "shared/__init__.py": "a5d8360a9c1a06dc42b0b1c336e0cfaa7e7edd4605ee67daf1049bc0798edb36", "shared/data_split.py": "f534c584da409642b0462d48c5c88f1658e6ce9f310b22996ca4461369ce780e", "shared/solar_disk.py": "25b30f7a9583ad9994a30467d9b13918a8156922850e51dec4763b630ab9e9c9", "shared/utils.py": "97c2ffb5c4b0b870775769988295533ddf5142499080f41df4fe31b2b6e17a44"}
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
out = Path("/kaggle/working")
base = one("submission.csv", "sol-exp139-submission-t4")
views = sorted(str(p) for p in source.rglob("exp002_test_views_*.json") if "sol-exp002-pools" in str(p))
assert len(views) == 2, views
step = subprocess.run([sys.executable, "-u", "-m", "experiments.exp_169_joint_submission.src.joint_submit",
                       "--report", one("exp168_report.json", "sol-exp168-joint-tuning-cpu"),
                       "--val-cache", one("val_cache.json", "sol-exp147-candidate-cache-t4"),
                       "--val-shift", one("shift_val_cache.json", "sol-exp157-shift-cache-t4"),
                       "--test-cache", one("e040_test_cache.json", "sol-exp151a-e040-test-cache-t4"),
                       "--test-shift", one("shift_test_cache.json", "sol-exp157-shift-cache-t4"),
                       "--val-yolo", "yolov8l=" + one("d4_pool_stage2_best.json", "sol-exp098-exp097-evaluation-t4"),
                       "--val-yolo", "yolo11m=" + one("exp002_val_pool.json", "sol-exp002-pools"),
                       "--test-yolo", "yolov8l=" + one("yolov8l_test_pool.json", "sol-exp161-yolo-test-pool-t4"),
                       "--test-views-002", *views, "--base-csv", base,
                       "--annotations", str(annotations), "--output", str(out)], cwd=root, env=env)
print("EXIT exp169", step.returncode, flush=True)
for candidate in sorted(out.glob("submission_*.csv")):
    step = subprocess.run([sys.executable, "-u", "-m", "experiments.exp_156_review.src.review", "--base", base,
                           "--candidate", str(candidate), "--images", str(test_images),
                           "--output", str(out / "review" / candidate.stem)], cwd=root, env=env)
    print("EXIT review", candidate.name, step.returncode, flush=True)
print("KERNEL_COMPLETE", flush=True)
