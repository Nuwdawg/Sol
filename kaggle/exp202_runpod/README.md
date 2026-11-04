# exp202 (optional, the user's decision): exp139's exact H100 recipe on all 707 labelled photographs

**Status: not launched.** It spends RunPod credit, so it runs only if the user says so.

**Why this, after exp201.** exp201 retrained on all data with the free T4 recipe (batch 2) and printed
0.40 (`final_decision/results/exp201/README.md`). exp139, the best single model (public 0.41), was
trained on an H100 with batch 8 in one forward pass. The open question is whether exp139's own recipe
gains from the 18% more data. **Honest odds of printing 0.42: modest, perhaps 1 in 4.**

**Cost.** exp139's run took 109 min on an H100 NVL at $2.59/h, about $4.80. All data is 1,154 records
against 974, so about 130 min, or **about $5.60** of the $12 RunPod credit. The pod's budget watchdog
stops it at the cap, whatever the trainer is doing.

## Launch (the user, with their own keys; this repository never holds a key)

0. **Upload the new code payload first.** exp139's payload (`0d8b0c6f...`, 17 Sep) predates the fix for
   exp142's all-data crash: the budget projection divided by `--eval-every 0`. Rebuilt from this
   repository (`scripts/package_exp139_code.py`, reproducible), it is in `code_dataset/`:
   `sol_exp139_code.payload`, 49,858 bytes, SHA256
   `d4b93620d4869b540fe3e6fe084c67b9640e9cd033974ce46571161d6d513851`. From inside that folder
   (CLAUDE.md section 15):
   ```bash
   kaggle datasets version -p . -m "exp202: all-data code" -r skip
   ```
   Then list the dataset's files and check that `sol_exp139_code.payload` is there under that name.
   - CPU rehearsal of this code's all-data path (2026-11-04: `--model tiny --max-steps 2 --fold all
     --eval-every 0 --sanity-photographs 2`): it loaded 1,154 records, projected the budget,
     saved the epoch checkpoint, ran the sanity check and exited 0.
1. Take `experiments/exp_139_mask2former/pod/pod_request.json`. Change these `env` values:
   - `SOL_CODE_SHA256`: `d4b93620d4869b540fe3e6fe084c67b9640e9cd033974ce46571161d6d513851`;
   - `SOL_FOLD`: `all` (train + validation);
   - `SOL_EVAL_EVERY`: `0` (no held-out photographs left to score);
   - `SOL_SAVE_EPOCHS`: `40 45 50`;
   - `SOL_SANITY`: `12` (an in-sample check that only detects a broken run);
   - `SOL_BUDGET_USD`: `6.80`;
   - `SOL_UPLOAD_DATASET`: a new private dataset name, e.g. `cosmicwitness/sol-exp202-alldata-weights`.

   Everything else (the image, the GPU, the launcher) stays as in exp139's request.
2. Insert the Kaggle key where the template says `__INSERT_KAGGLE_KEY_AT_LAUNCH__`, then create the pod
   on RunPod (H100 NVL, community cloud). It downloads the code dataset, checks its SHA256, trains 50
   epochs, uploads `model_e040/45/50.pt` to the dataset above, and stops itself.

## After it finishes (this session's part)

1. Cache `model_e050.pt` on the 180 test photographs (11 views). This runs on CPU, about 13 h
   (`reproduce/run.py`'s exp201 stage takes any checkpoint; or a free Kaggle T4, about 40 min).
2. Build the file with `experiments/exp_201_alldata/src/build.py` and the old Swin-B as the rescue
   voter. Review it at native resolution, then upload it once.
3. Final-pick rule: as for exp201 (`kaggle/exp201/README.md`). It is kept only if it prints 0.42, or
   0.41 alongside the first pick.
