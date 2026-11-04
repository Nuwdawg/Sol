# exp201: exp139's model and the Swin-B voter retrained on all 707 labelled photographs (free Kaggle T4)

**Why.** Every inference-time change to the existing models now prints 0.41. Six files score 0.41
from validation 0.483-0.491 (`final_decision/results/remnant_rescue/README.md`). The rule adds +0.018
to whatever model it is given, so a better base model is the remaining lever. exp201 trains the same two
models on train + validation: 1,154 annotation records instead of 974, about 18% more.

**Cost.** None in money: Kaggle's free GPU quota, two sessions of about 10.5 h each (about 21 of the 30
weekly hours). Nothing here uses RunPod.

**Code.** `kernel.py` is exp190b's driver and its recovered code (17 files, each checked against its
SHA256 inside the kernel). That pair trained Swin-S seed 7 to 0.4677 and Swin-B seed 11 to 0.4698 over
two T4 sessions. Only the arguments change:
- `--fold all` and `--eval-every 0`: no held-out set remains.
- `--save-epochs 40 45 50` for Swin-S (50-epoch schedule) and `28 32` for Swin-B (32-epoch schedule).
- a 12-photograph in-sample sanity check, which only detects a broken run.
- the driver keeps every saved checkpoint.

Each session stops itself at 9.8 h of training, so it completes rather than being cancelled
(CLAUDE.md section 8). Session 2 resumes from session 1's `last.pt`. A CPU rehearsal of the smoke
path, with these arguments and the pinned code, loaded 1,154 records, wrote the sweep report and the
sanity check, and exited 0.

## How to run (the one step this session may not do: write to your Kaggle account)

From a machine with your Kaggle API token:

```bash
kaggle kernels push -p kaggle/exp201/s1
```

When session 1 shows COMPLETE (about 10.5 h later):

```bash
kaggle kernels push -p kaggle/exp201/s2
```

Or on kaggle.com: New Notebook, paste `s1/kernel.py`, and title it exactly `sol exp201 alldata s1 t4`;
session 2 finds session 1's output by that name. Then set Accelerator to GPU T4 x2 and Internet on, add
the competition data, and Save Version (Save & Run All).

## What happens after

Session 2 ends with `swins/model_e050.pt` (and e040, e045) and `swinb/model_e032.pt`. It then caches
them on the 180 test photographs (GPU 0: Swin-S full, dihedral and shift views; GPU 1: Swin-B full view)
with this repository's `cache.py`, unless the session is already past 10 h. A CPU test of that path,
on the recovered code with a checkpoint written by the recovered trainer, produced the expected views.
Then:
1. (only if session 2 skipped it) cache the new models on CPU;
2. build exp169's rule + guard + remnant filter + Swin-B rescue on the new models, with no constant
   changed:
   ```bash
   python -m experiments.exp_201_alldata.src.build --swins-cache OUT/cache_swins/model_e050_test_cache.json --swinb OUT/cache_swinb/model_e032_test_cache.json --yolo yolov8l=WORK/yolo_yolov8l/test_pool.json --yolo yolo11m=WORK/yolo_yolo11m/test_pool.json --output exp201_file
   ```
   (`OUT` = session 2's output, `WORK` = `reproduce/run.py --stages weights yolo`). Rehearsed on
   exp139 e40's test cache and the old Swin-B candidates (2026-11-03), the same command rebuilt all
   three deployed files byte for byte: exp139's single-view file (`CCA9C06E...`), exp169 + guard
   (`EC00408E...`) and the remnant + rescue file (`F9423B74...`);
3. review the changed instances at native resolution and upload once.

There is no held-out set left, so the board is the only judge. Honest odds of printing 0.42: about
1 in 4. Swin-S trained on a T4 reached 0.4677 against exp139's 0.4705, so 18% more data has to more
than cover that difference first.

## Status

- **Session 1: COMPLETE** (pushed by the user, started 2026-11-03 20:34 UTC; read back from its log on
  2026-11-04 06:47 UTC):
  - data: 1,154 records, `--fold all`;
  - Swin-S: 4.35 s/step, stopped on budget at 9.82 h, kept `last.pt` and `model_e029.pt`;
  - Swin-B: 6.95 s/step, stopped on budget at 9.82 h, kept `last.pt` and `model_e017.pt`;
  - in-sample sanity check on 12 photographs: PQ 0.452 (Swin-S) and 0.451 (Swin-B), mid-schedule.
    This only shows the run is not broken.
- **Session 2:** to push now. It needs about 21 more Swin-S epochs (about 7.1 h) and 15 more Swin-B
  epochs (about 8.3 h), then the test caching.
