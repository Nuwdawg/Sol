# Reproduce the two final picks

Nothing here trains a model: every checkpoint is stored in `weights/` and is only loaded.

- **First pick:** the remnant + rescue file,
  `final_decision/results/remnant_rescue/submission_exp169_remnant_rescue.csv` (1,149 instances,
  SHA256 `F9423B74429F34130BF70C9421B2416AAFD3C3852D9E465FC9F117C0FADA162D`). It is exp169's rule with
  exp199's large-filament guard, a remnant filter and a Swin-B rescue.
- **Second pick:** exp139's single-view file (1,192 instances, SHA256 `CCA9C06E...`). The `rule`
  stage rebuilds it on the way as `work/rule/parity_deployed.csv`.
- Rebuilt and checked along the way: exp169 + guard, `final_decision/results/submission_exp169_guarded.csv`
  (1,150 instances, SHA256 `EC00408E7F8ED78A350EEC50FCC1F96824A9851D972310F9408B4B64D7D966A7`).

| stage | what it does | where | time |
|---|---|---|---|
| `weights` | rebuild three checkpoints from their parts and check their SHA256 | `weights/assemble.py` | 1 min |
| `m2f` | Mask2Former Swin-S (exp139 epoch 40) on the 180 test photographs, 11 views each (identity, 7 dihedral, 3 half-stride shifts) | `experiments/exp_147_candidate_cache` | ~40 min, 2 x T4 |
| `yolo` | YOLOv8l (exp097) and YOLO11m (exp002), 8 dihedral views, on validation and test; each validation pool is checked against the stored one | `experiments/exp_161_yolo_test_pool` | ~30 min, 2 x T4 |
| `rule` | exp169's vote. It first replays its validation PQ 0.48894859995 from the stored validation predictions, and checks that the Mask2Former path rebuilds exp139's submitted file | `experiments/exp_169_joint_submission` | ~17 min, CPU |
| `guard` | re-add the 9 large Mask2Former filaments the vote deleted | `experiments/exp_199_swinb_votes/src/guard.py` | 1 min |
| `check` | compare exp169 + guard and exp139's file with the submitted ones (SHA256, else instance by instance) | `reproduce/run.py` | seconds |
| `swinb` | Swin-B (exp190b epoch 32) full-view candidates on the 180 test photographs, class probability >= 0.5 | `experiments/exp_200_test_truth/src/swinb_infer.py` | ~4 h, 4 CPU cores |
| `remnant_rescue` | first rebuild this run's exp169 + guard file byte for byte, then drop instances keeping under 70% of their vote mask and rescue borderline ones (0.60-0.68) that Swin-B confirms at >= 0.95; compare with the stored first pick | `experiments/exp_200_test_truth/src/combo_test.py` | ~25 min, CPU |

```bash
pip uninstall -y torchaudio      # on Kaggle: the image's copy is built for another torch
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python reproduce/run.py --data /path/to/MAGFiLO_1.0_Kaggle_2026 --work reproduce/work
```

On Kaggle `--data` can be left out: `run.py` finds the competition folder under `/kaggle/input`
the way the original kernels did. The notebook `reproduce_final_submission.ipynb` runs the same
stages one cell at a time, with an explanation of each. It works from inside the repository or
from the folder that holds a clone named `Sol`.

## What has been verified, and what not yet

- **Verified on CPU (2026-11-03): the default run's last two stages.** On the stored model outputs
  (below), `run.py --stages check remnant_rescue` reported exp169 + guard identical (`EC00408E...`),
  exp139's file identical (`CCA9C06E...`), and the first pick identical (`F9423B74...`).
- **Verified on CPU (2026-10-29).** The `rule` and `guard` stages were run on the model outputs that
  produced the submissions: the exp151a and exp157 test caches, the exp161 YOLOv8l pool and exp002's
  YOLO11m views. They reproduce exp169's submitted file byte for byte (SHA256 `CC15EA41...`), and
  then the guarded file (`EC00408E...`). Run through `run.py` itself, with those outputs placed where
  the GPU stages write them, the rule, guard and check stages end with `CHECK identical: true`. The
  weights rebuild byte-identical to the Kaggle originals.
- **Smoke-tested on CPU (2026-10-29): `m2f` and `yolo`, one photograph each.** `run.py --stages m2f
  yolo --limit 1` ran the stored weights with the pinned packages (torch 2.8.0+cpu, torchvision
  0.23.0, transformers 4.57.6, ultralytics 8.4.153). Every output matched the stored one
  (`smoke_test.log`):
  - Mask2Former, test photograph `20110120105534Ch`: all 132 candidates of the 11 views match the
    T4 cache, with the same count per view, mask IoU 1.0 and scores within 1e-5. This ran on
    PyTorch's deformable-attention loop, not the hub kernel the T4 used.
  - YOLOv8l and YOLO11m, validation photograph `20110220082634Lh`: 3 and 4 instances, IoU 1.0, scores
    within 3e-7 of the stored pools.
  - Time: Mask2Former took 8.3 min per photograph on 4 CPU cores, about 25 hours for all 180.
- **Not yet run: `m2f` and `yolo` on all 180 photographs**, which needs a GPU (about 70 min on two
  T4s). Two things may still stop the result being byte-identical:
  - exp002's YOLO11m views were made with an older Ultralytics patch release on CPU, and exp033
    found boundary pixels can move between releases (the smoke photograph did not move);
  - GPU arithmetic may differ slightly.

  `check` then reports instance-level agreement instead of a hash match. The `M2F_KERNEL` lines
  say whether each shard ran the deformable-attention build of the original caches.

## Stored inputs (`artefacts/`, SHA256 in `artefacts/SHA256SUMS`)

These are the validation predictions exp169 was tuned on, used only for its validation replay, plus
the two submitted files used as references. `exp168_report.json` holds the rule's constants.
`exp147_val_cache.json` and `exp157_shift_val_cache.json` are Mask2Former's validation candidates.
`exp098_yolov8l_val_pool.json` and `exp002_yolo11m_val_pool.json` are the YOLO validation pools.
`exp139_submission.csv` and `exp169_submission.csv` are the two submitted files.
