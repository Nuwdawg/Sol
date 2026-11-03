# Pipeline of the first final pick (draft for the technical report)

Report-ready text for the 4-page technical report (Overleaf template). It describes
`submission_exp169_remnant_rescue.csv`, the first final pick, which `reproduce/run.py` rebuilds byte
for byte from the weights in `weights/`. If the report so far describes only exp139, replace or extend
its method section with this one. The second pick, exp139's single-view file, is stage 1 below on
its own.

## Data

The 707 labelled photographs are split by photograph, stratified by observing site
(`shared/data_split.py`): 601 for training (974 annotation records) and 106 for validation (180
records; a photograph has one to three annotators). Every validation number below is the
competition's PQ computed per annotation record and pooled, as the organisers' self-evaluation
notebook does. Our evaluator agrees with it to 1e-10.

## Models (all trained on the 601 training photographs only)

- **Mask2Former, Swin-S backbone (exp139).** Initialised from `facebook/mask2former-swin-small-coco-instance`,
  100 queries. Trained at native 2,048 px resolution: native 1,024 crops plus 25% whole frames,
  targets rasterised with pycocotools at native resolution, dihedral flips/rotations and gain/gamma
  augmentation. AdamW, learning rate 1e-4 (backbone x 0.1), weight decay 0.05 (none on norms,
  embeddings and position tables), gradient clip 0.01, 500 warm-up steps, then cosine decay; 50
  epochs at batch 8. Epoch 40 is used. Inference is float32 on the whole frame.
- **Two YOLO segmentation models** as independent detectors: YOLOv8l-seg (exp097, trained at 2,048 px)
  and YOLO11m-seg (exp002, trained at 1,280 px). Each is run in the 8 dihedral views (confidence
  floor 0.25), and the views are fused by mask IoU.
- **Mask2Former, Swin-B backbone (exp190b, epoch 32)**, ImageNet-21k / COCO-instance initialisation,
  used only as a second opinion on borderline instances (step 4).

## Inference (no learned post-processing; every constant chosen on validation)

1. **Views.** exp139 sees each test photograph 11 ways: as is, in the 7 other dihedral transforms, and
   shifted by half its 4-pixel output stride (2 px) in three directions. Each view's masks are
   mapped back to the original frame.
2. **Votes.** Every full-view candidate (score s = class probability x mean mask probability) is
   matched to its best partner (IoU > 0.5) in each of the 10 other views and in each YOLO pool.
   Its mask is the pixel-wise majority of itself, its view partners and its YOLO partners (eroded by
   1 px). Its score is s^0.3 x m^0.7, where m = (sum of partner scores over the 11 views + 4 x the
   sum of the two YOLO supports) / (11 + 4 x 2). A missing partner counts 0.
3. **Selection.** Keep candidates scoring at least 0.68.
4. **Rescue.** Also keep candidates scoring in [0.60, 0.68) whose Swin-B full-view partner (class
   probability >= 0.5, IoU > 0.5) scores at least 0.95. Without YOLO support a candidate's score
   cannot reach 0.68, so this band holds filaments that only the Mask2Former models see.
5. **Painting (disjoint instances).** Instances are painted in descending score, and each pixel goes
   to the highest-scoring instance covering it. An instance is dropped if it keeps fewer than 100 px,
   or less than 70% of its own vote mask (a lower-ranked duplicate of a neighbour).
6. **Guard.** Large filaments the vote deleted are put back: exp139's single-view instances (score
   >= 0.85) of at least 3,000 px that the final instances cover by less than 10%.

The result has no overlapping masks (checked before writing): 1,149 instances on 179 of the 180
test photographs.

## Results

| step | validation PQ (per record) | SQ / RQ | public |
|---|---|---|---|
| exp139, single view, threshold 0.85 | 0.47052 | 0.686 / 0.686 | 0.41 |
| + views and YOLO votes (steps 1-3) | 0.48895 | 0.691 / 0.708 | 0.41 |
| + guard (step 6) | 0.48893 | | 0.41 |
| + remnant filter and Swin-B rescue (steps 4-5): **first pick** | **0.49095** | | 0.41 |

- The vote's constants (view set, YOLO weight 4, exponent 0.3, threshold 0.68) were chosen by
  two-fold cross-validation over photographs: 0.48707 cross-validated against 0.48895 in-sample.
- **Transfer check.** Applied unchanged to the Swin-B model (8 views), the vote of steps 1-3 raises
  its validation PQ from 0.46796 to 0.48290 (+0.0149). That is 81% of the gain it gives exp139
  (+0.0184), so the gain is mostly not tuned to one model.
- **The rubric's quantitative items** (validation, per record, the self-evaluation notebook's
  definitions: every ground-truth / prediction pair with IoU > 0;
  `analysis/outputs/rubric_quantities.json`):

  | file | PQ | mean IoU | mean Dice | GT overlapped by 2+ predictions (1:n) | predictions overlapping 2+ GT (n:1) |
  |---|---|---|---|---|---|
  | exp139 single view (second pick) | 0.47052 | 0.6045 | 0.7337 | 37 of 1,325 | 40 of 1,242 |
  | exp169 + guard | 0.48893 | 0.6203 | 0.7476 | 30 of 1,325 | 30 of 1,199 |
  | **remnant + rescue (first pick)** | **0.49095** | **0.6225** | **0.7502** | **26 of 1,325** | **29 of 1,199** |

  The first pick is better on every quantitative item. The vote removes spurious fragments, and the
  remnant filter removes lower-ranked duplicates of filaments already held by another instance.
- **What did not help** (validation, per record): learned rescorers (logistic, boosted trees, CNN),
  boundary snapping to image intensity, mask erosion or dilation, other mask-fusion rules
  (expected-IoU-optimal level, signed-distance averaging, STAPLE), photograph- or instance-dependent
  thresholds, scale and tile views, and further Mask2Former or YOLO models as voters.

## Reproduction and cost

`reproduce/run.py` (or `reproduce/reproduce_final_submission.ipynb`) rebuilds both final picks from
the stored weights, stage by stage. Each stage checks itself against a recorded number:
- the vote first replays its validation PQ;
- the Mask2Former path must rebuild exp139's submitted file;
- the final files are compared with the submitted ones.

Time: the Mask2Former and YOLO stages take about 70 min on two T4 GPUs; the rule, guard and rescue
stages take about 45 min of CPU, plus about 4 h of CPU for Swin-B's full-view pass.
