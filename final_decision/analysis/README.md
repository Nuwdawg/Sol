# CPU analyses behind the decision brief

All read existing predictions only; no model was run, no labels of the test set were used. Each
script that scores anything first reproduces a recorded number exactly and stops if it cannot.

Inputs (downloaded read-only from the Kaggle account `cosmicwitness`; not committed here because
they are large or are competition data):

| name below | source |
|---|---|
| `ANN` | competition file `MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json` (SHA256 `5da9e92b...`, read from the competition archive by HTTP range, see `httpzip.py`) |
| `LISTING` | competition file listing (`competitions/data/list`), names only |
| `VAL147` | `val_cache.json` of kernel `sol-exp147-candidate-cache-t4` |
| `SHIFTVAL` | `shift_val_cache.json` of `sol-exp157-shift-cache-t4` |
| `V8LVAL` / `V8LTEST` | `evaluation/d4_pool_stage2_best.json` of `sol-exp098-exp097-evaluation-t4` / `yolov8l_test_pool.json` of `sol-exp161-yolo-test-pool-t4` |
| `Y11VAL`, `Y11TEST0/1` | `kaggle/exp002_pools_dataset/` on branch `exp/002-yolo-seg` |
| `TEST151` | `e040_test_cache.json` of `sol-exp151a-e040-test-cache-t4` |
| `BASECSV` / `NEWCSV` | `parity_deployed.csv` / `submission_joint.csv` of `sol-exp169-joint-submission-cpu` |
| `PAYLOAD` | `../recovered_code/sol-exp169-joint-submission-cpu` |

Packages: numpy, pycocotools, opencv-python-headless, scipy (a plain venv). Scripts insert the
repository root `/home/user/Sol` into `sys.path` for `shared/` and `experiments/exp_033_inference`;
change that line if the checkout lives elsewhere.

| script | command | output | key result |
|---|---|---|---|
| `temporal_train_test.py` | `python -I temporal_train_test.py LISTING outputs/split_names.json` | printed; split names | nearest training photo within 1 day: val 29.2%, test 29.4% |
| `diagnose_exp139.py` | `python -I diagnose_exp139.py ANN VAL147 outputs/diagnose_exp139.json` | `outputs/diagnose_exp139.json` | parity 0.470524999 exact; error taxonomy; rubric quantities; bootstrap |
| `committee_ceiling.py` | `python -I committee_ceiling.py ANN VAL147 outputs/committee_ceiling.json` | `outputs/committee_ceiling.json` | two-annotator committees 0.295-0.329 vs exp139 0.454 on 81 held-out records |
| `removed_instances.py` | `python -I removed_instances.py BASECSV NEWCSV TEST151 outputs/removed_instances.json` | `outputs/removed_instances.json` | exp169 deletes 101 and reshapes 13 of exp139's 1,192 test instances |
| `veto_check.py` | `python -I veto_check.py PAYLOAD ANN VAL147 SHIFTVAL V8LVAL Y11VAL outputs/veto_check.json` | `outputs/veto_check.json` | exp169 replay 0.48894859995 exact; what it drops on validation; guard variants |
| `veto_mechanism.py` | `python -I veto_mechanism.py ANN VAL147 V8LVAL Y11VAL outputs/veto_check.json outputs/removed_instances.json BASECSV TEST151 V8LTEST Y11TEST0 Y11TEST1 outputs/veto_mechanism.json PAYLOAD` | `outputs/veto_mechanism.json` | drops are low YOLO confidence, not fragmentation (57 of 69) |
| `render_dropped.py` | `python -I render_dropped.py <archive URL> ANN outputs/veto_check.json outputs/removed_instances.json BASECSV NEWCSV review VAL147` | `review/*.jpg` | native-resolution crops of the largest drops |

The guard numbers in the brief are the `(3000, 0.85)` row of `veto_check.json["protect"]`:
PQ 0.48835 vs 0.48895 without it. Other rows in that table were measured too and were not used
to choose the cut.
