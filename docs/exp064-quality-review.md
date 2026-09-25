# exp064: lower-confidence D4 probe

Compared with exp062 (public0.39), the candidate preserves all1233 masks and adds
131, with no altered boundaries or removed instances. All180 test photographs
were processed;177 have predictions. All1364 submitted masks pass overlap checks.

Reviewed the deterministic largest, median and smallest additions in
`data/exp064_quality_review/contact_sheet.png`:

- `20120622063154Uh`,12354px: elongated faint dark structure near the limb;
  image support is visible, but the broad boundary and faint extension are uncertain.
- `20110408083214Th`,830px: coherent curved dark hook with visible image support.
- `20111017113714Th`,304px: compact dark feature beside a bright active region.
  Its identity is ambiguous; possible sunspot contamination is a quality concern.

These are visual observations, not test labels or a precision estimate. Submit
as the already-authorized exploratory recall probe; do not claim that lowering
the confidence threshold improves scientific prediction quality from score alone.
Canonical validation falls from0.455864 to approximately0.450091, so this tests
the public response to a nearby existing-weight rule rather than a proven gain.

SHA256: `8b36a8f982e160ae78fff143c49e160e67ee2a810f47c76813d3565b7f76b22c`.
