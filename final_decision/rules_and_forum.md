# Competition rules and organizer rulings that constrain the next step

Read 2026-10-29 through the Kaggle API (competition pages and the discussion forum of
`filament-segmentation-2026`, 43 topics). `P:` = competition page, `F:n` = forum topic id
(`https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/n`). Quotes are the
organizer's (Azim Ahmadzadeh) unless marked. Other participants' posts are summarised, not quoted.

## Dates and submissions
- Kaggle deadline 2026-11-15 06:00 UTC (competition API). Reports and code: "Nov 15, 2026,
  11:59 pm CDT" (P:Important Dates). Winners announced Nov 30.
- At most 5 submissions per day. "You may select up to two (2) Final Submissions for judging"
  (P:rules, section 2.2.b). If two different strategies are selected, both must be reproducible
  (F:747960).
- The public board scores about 50% of the 180 test images and prints two decimals
  (P:Submission File; competition API `leaderboardPercentage` 50).

## How the final ranking is decided (P:Evaluation, P:Prizes)
- 70% quantitative: PQ (leaderboard), the distribution of Dice scores, the distribution of IoU
  scores, and the distribution of one-to-many and many-to-one relations (host self-evaluation
  notebook; copy on branch `exp/145-host-evaluation`). 30% qualitative: pipeline description,
  "apparent morphology of predicted segmentations", code quality.
- Organizers on quality (F:724503, 2026-10-26): "we actually look at the exact segmentation on
  several examples we find to be important"; on feasibility: run in reasonable time, "without
  over-complicating the task". Efficiency is a filter, not a competing factor (F:749526).
- Prize money is proportional to score squared (P:Prizes).

## The leaderboard above about 0.5
- 82 of the ~800 listed teams sit at exactly 0.55; the top is 0.70 (leaderboard API, 2026-10-29).
  Participants link the 0.55 cluster to a public notebook that uses the leaked test annotations
  (F:743958, F:740400, F:745880). The public MAGFiLO release contains annotations for all 180 test
  images (F:728062, F:729396).
- Organizers: "Yes, we take a score of 0.41 very seriously" (F:743958); "We will clean the
  leaderboard after the deadline, so that credit is given to genuine submissions" (F:724503,
  Oct 23 review of the top 250); "A PQ greater than 3.5 [sic, 0.35] still has significant merit".
- Honest participants report validation 0.46-0.49 and public 0.40-0.41 (F:745880, F:744765).

## Reproducibility (P:Final Submission; F:724503, Oct 23)
- Public Git repository, `requirements.txt` with versions, a notebook showing the whole pipeline,
  "The committee should not require any additional file to reproduce the submitted predictions."
- "For ML/DL models, the model's weights must be stored in the repo, and be loaded in the
  notebook". Submissions whose weights are not in the repository "will not be considered for the
  award" (F:724503).
- A 4-page report (Overleaf template), submitted with the form; AI use is allowed but the team
  must understand and defend every component (F:724503; P:rules section 2.8.b).

## Data
- Allowed for training only: unlabelled GONG H-alpha frames, excluding the test images and
  anything within +-1 hour of their timestamps (F:744047, F:748045, F:748498); copying training
  labels onto neighbouring archive frames of the same site (F:748045); GONG magnetograms
  (F:743521); other instruments' observations (F:729396).
- Allowed as supervision: the dataset's `spine`, `bbox`, `area` and `category_id` (F:736543).
- Not allowed: any extra ground truth, e.g. SWEFil (F:746318) or the public MAGFiLO release
  (F:729396); any input other than the single test image at inference, such as neighbouring frames
  (F:743385, F:744047).
- The organizers want all real filaments detected, annotated or not; tiny unclear detections may
  be cleaned by post-processing (F:735256).

## Not answered by the organizers
- How test images with several annotators are scored (F:729487 unanswered; F:753043 declined).
- Whether metadata in the file name (time, site) may be used at inference (F:753665, open).
