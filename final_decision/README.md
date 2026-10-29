# Final experiment decision (2026-10-29)

Result of the task in `sol_final_experiment_prompt.md`: one primary experiment, one fallback, and
the evidence for both. The plan was approved and run on 2026-10-29, on CPU only: see `results/RESULTS.md`. Nothing
was trained or submitted.

| file | what it is |
|---|---|
| `decision_brief.pdf` | the 4-page decision brief (source: `decision_brief.html`, rendered with headless Chromium) |
| `HANDOFF.md` | step-by-step instructions for the agent who runs the experiment, with every parity gate and the kill criterion |
| `ledger.md` | the full ledger: all branches plus the 52 Kaggle-only experiments exp146-exp198, and the new measurements |
| `rules_and_forum.md` | competition rules and organizer rulings that constrain the plan, with forum topic ids |
| `analysis/` | the CPU-only analyses behind the diagnosis: scripts, outputs, review crops (see `analysis/README.md`) |
| `recovered_code/` | code of the key Kaggle kernels (exp151a, 161, 169, 190b, 195b, 198a), each file checked against its kernel's SHA256 manifest |
| `figure_dropped_filaments.jpg` | Figure 1 of the brief |
| `results/` | **outcome of the run (2026-10-29)**: pilot killed, guarded exp169 file ready to submit (`results/RESULTS.md`) |

Short version: exp169 (validation 0.4889, public 0.41) stays the base. Primary: test whether the
already-trained Swin-B epoch-32 Mask2Former helps as a second voter, on cached validation
predictions, at zero cost; keep it only at +0.002 cross-validated PQ. Fallback and final step:
stop the vote from deleting large Mask2Former filaments (9 on test). Then make the submission
reproducible from this repository, which the rules require for any award.
