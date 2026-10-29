# Sol: solar filament instance segmentation

Final model: a native-resolution Mask2Former (exp139, `facebook/mask2former-swin-small-coco-instance`,
100 queries). Validation PQ 0.4705 on the canonical split; public leaderboard 0.41.
This branch also holds the book (`book/`) and the paper (`paper/`).

Every other experiment lives on its own branch `exp/NNN-*`, which works when checked out alone.
The experiment log is `docs/ACTIVE_EXPERIMENT.md` (from exp033) and `docs/leaderboard-analysis.md` (up to exp031).
The appendix `book/chapters/appB_ledger.tex` lists every experiment with its decisive result.

## Reproduce the final submission

The final submission is exp169's vote over exp139's Mask2Former, two YOLO models and eleven views of each
photograph, with exp199's guard that keeps large Mask2Former filaments (validation PQ 0.4889, public 0.41):
`final_decision/results/submission_exp169_guarded.csv`. The weights are stored in `weights/`. To rebuild the
file without training, run `python reproduce/run.py --data <MAGFiLO_1.0_Kaggle_2026> --work reproduce/work`,
or open the notebook `reproduce/reproduce_final_submission.ipynb`. See `reproduce/README.md` for the stages and
what has been verified, and `final_decision/` for how this submission was chosen.

## Reproduce the final model

1. Install the dependencies: `pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128`, then `pip install -r requirements.txt`.
2. Put the competition data outside git (the `data/` folder is ignored).
3. Train: `python -m experiments.exp_139_mask2former.src.train --annotations <train json> --images <image dir> --output <run dir>`.
   Run `--help` for the schedule, crop and precision options. The pod launcher is `experiments/exp_139_mask2former/pod/bootstrap.sh`.
4. Emit the test submission: `python -m experiments.exp_139_mask2former.src.emit_test --checkpoint <ckpt> --test-images <dir> --output submission.csv`.
   Its defaults are the validated rule: confidence 0.85, no erosion, minimum area 100, single view, float32.
5. Kaggle kernels: `kaggle/exp139_submission_t4/` emits a submission and `kaggle/exp139_gpu_tests_t4/` runs the GPU tests.
6. Tests: `python -m pytest tests experiments/exp_139_mask2former/tests`.

## Build the book and the paper

In `book/` (and likewise in `paper/`): `pdflatex main.tex`, `bibtex main`, then `pdflatex main.tex` three times.
Figures rebuild with `python book/make_book_figures.py` and `python paper/make_figures.py`; the image figures need the data.
The Kaggle kernels `kaggle/book_build_cpu/` and `kaggle/paper_build_cpu/` run the same build.

`CLAUDE.md` holds the working notes that Chapter 14 of the book cites.

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python book/make_book_figures.py`
- `python -m experiments.exp_139_mask2former.src.emit_test --help`
- `python -m experiments.exp_139_mask2former.src.evaluate --help`
- `python -m experiments.exp_139_mask2former.src.submit --help`
- `python -m experiments.exp_139_mask2former.src.train --help`
- `python -m experiments.exp_139_mask2former.tests.test_accumulation --help`
- `python -m experiments.exp_139_mask2former.tests.test_equivalence --help`
- `python paper/make_figures.py`
- `python scripts/download_kernel.py`
- `python scripts/filter_submission_area.py`
- `python scripts/launch_detached.py`
- `python scripts/monitor_local_job.py`
- `python scripts/monitor_training.py`
- `python scripts/package_book.py`
- `python scripts/package_deliverables.py`
- `python scripts/package_exp139_code.py`
- `python scripts/package_exp139_gpu_tests.py`
- `python scripts/package_exp139_submission.py`
- `python scripts/package_paper.py`
- `python scripts/review_submission_changes.py`
- `python scripts/stream_kernel.py`
- `python scripts/submit.py`
- `python scripts/watch_kernel.py`
- Kaggle kernel: `cd kaggle/book_build_cpu && kaggle kernels push -p .`
- Kaggle kernel: `cd kaggle/exp139_gpu_tests_t4 && kaggle kernels push -p .`
- Kaggle kernel: `cd kaggle/exp139_submission_t4 && kaggle kernels push -p .`
- Kaggle kernel: `cd kaggle/paper_build_cpu && kaggle kernels push -p .`
- Kaggle dataset folder: `kaggle/book_src_dataset`
- Kaggle dataset folder: `kaggle/exp139_code_dataset`
- Kaggle dataset folder: `kaggle/paper_src_dataset`
- Tests: `python -m pytest tests/test_download_resume.py tests/test_pq_matches_official.py tests/test_training_monitor.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_033_inference/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/cached_tta.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/sweep.py` from `exp/033-inference`
- `experiments/exp_097_runpod_yolo/__init__.py` from `exp/097-runpod-yolo`
- `experiments/exp_097_runpod_yolo/src/__init__.py` from `exp/097-runpod-yolo`
- `experiments/exp_097_runpod_yolo/src/events.py` from `exp/097-runpod-yolo`
- `experiments/exp_097_runpod_yolo/src/kaggle_sync.py` from `exp/097-runpod-yolo`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)
- `experiments/exp_002_yolo_seg/src/prepare_yolo.py` from `exp/002-yolo-seg`
- `experiments/exp_097_runpod_yolo/src/fetch_data.py` from `exp/097-runpod-yolo`
- `experiments/exp_128_detector_retrain/__init__.py` from `exp/128-detector-retrain`
- `experiments/exp_128_detector_retrain/src/__init__.py` from `exp/128-detector-retrain`
- `experiments/exp_128_detector_retrain/src/prepare_dataset.py` from `exp/128-detector-retrain`
- `experiments/exp_128_detector_retrain/src/preprocess.py` from `exp/128-detector-retrain`
- `kaggle/exp046_mask2_train/train.py` from `exp/046-mask2`
- `scripts/evaluate_mask2_fusion.py` from `exp/048-mask2-eval`
- `scripts/evaluate_public_fusion.py` from `exp/033-inference`
- `shared/mask2_checkpointing.py` from shared module (original in `main`)
- `shared/query_mask_loss.py` from shared module (original in `main`)

## Experiment log

- `docs/ACTIVE_EXPERIMENT.md` line 3171: exp139 - a properly trained Mask2Former (built 2026-09-17, not yet run)
- `docs/ACTIVE_EXPERIMENT.md` line 3527: exp145: the host's self-evaluation notebook on exp139 epoch 40 (2026-09-21)

## Dependencies not in requirements.txt

The root `requirements.txt` pins the final pipeline (torch, transformers, kernels, ultralytics) and lists its other packages. Older experiments also import `pytest`, `requests`, `shapely` and `torch_xla`. The exp001 list is `requirements-exp001.txt`.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
