# Visual review of the 0.39 D4 submission

Reviewed2026-09-13 following the user's host-guidance instruction.

Compared exp057(confidence0.385, minimum area100, public0.38) with
exp062(confidence0.35, minimum area300, public0.39). Both use the same weights,
D4 fusion and one-pixel erosion. Therefore the public gain does not isolate
confidence from the area cutoff.

The complete prediction comparison found82 added and37 removed instances, with
no changed boundaries among matching instances. Each changed instance has zero
overlap with the other submission's masks. Six examples were selected
deterministically: largest, median and smallest added/removed instance by area.
This is a diagnostic sample, not an estimate of the fraction of correct changes.

Artifacts: `data/exp062_quality_review/review_manifest.json`, `case_1.png` through
`case_6.png`, and `contact_sheet.png`. Original observations and contours are
shown at native crop coordinates, with identical1st-to99th-percentile display
contrast across each row. No test annotations were accessed.

## Observations, not test labels

- **20161015144954Bh, added9856px:** the new outline follows a broad, faint,
  elongated dark region near the limb. There is visible structural support for
  the detection, although its diffuse boundary remains uncertain.
- **20120402105814Th, added1094px:** the new mask follows a distinct curved dark
  structure previously omitted. This is a convincing example of recovering
  useful image structure by admitting a lower-confidence detection.
- **20140309170314Mh, added307px:** a compact dark feature is detected. Its
  physical identity is less clear from this image alone; darkness is not enough
  to establish that it is a filament.
- **20210127110130Ch, removed299px:** the previous outline follows a coherent,
  narrow, curved dark structure. The300px cutoff removes a plausible filament
  just one pixel below its threshold. This is a concrete quality concern.
- **20130213063454Uh, removed247px:** a compact dark feature is removed. Its
  identity is ambiguous; this example does not establish a quality improvement.
- **20121220110614Th, removed122px:** the small component disappears while the
  surrounding larger filament masks remain. The local region contains multiple
  thin structures, so removal needs more justification than area alone.

## Decision

Keep exp062 as the verified public best, but do not promote its300px cutoff as
the scientifically preferred filter. The sample supports trying improved recall
while preserving small coherent masks. Prioritize **confidence0.35/minimum
area100** as the next controlled D4 probe, holding all other settings unchanged.
Compare it directly with exp057 to isolate confidence, and with exp062 to isolate
the area cutoff. The already-prepared confidence0.30/area300 candidate remains
unsubmitted and is lower priority.

These visual judgments do not override the official validation metric or create
test-set ground truth. Continue examining representative predictions from the
four currently authorized exp063 probes when their outputs arrive.
