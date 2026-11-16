# CIE A Level topical tests: Maths 9709 and Physics 9702

These tests collect the hardest questions on each syllabus topic from Cambridge International
past papers. All papers, mark schemes and examiner reports were downloaded from PapaCambridge.

| Papers | Years covered |
|---|---|
| Maths P2 Pure 2, P3 Pure 3, P4 Mechanics; Physics P4, P5 | **9 years**: Feb/Mar 2018 to May/Jun 2026 |
| All other papers | **5 years**: Feb/Mar 2022 to May/Jun 2026 |
| Physics P4 topics 12, 14, 16, 19, 21 (extra) | hand-checked questions from Oct/Nov 2004 to Oct/Nov 2017 |

Oct/Nov 2026 has not been sat yet.

The PDFs are **not committed**, because this repository is public and the papers are
Cambridge copyright material (`.gitignore`). Rebuild them with `tools/`, as described below.
`index.csv` lists every question in every test with its original reference.

## What a test looks like

- Each page holds the question content only, cropped from the original paper as vector
  graphics. Nothing else from the exam page is carried over:
  - no page header or footer, and no page numbers
  - no "DO NOT WRITE IN THIS MARGIN" bars, barcodes or corner marks
  - no copyright notices, no blank or "additional" pages
  - no PapaCambridge watermark or trace text
- Nothing is added either: no title band and no running header.
- Questions are renumbered 1, 2, 3, ... and keep their original answer space.
- Within each test, questions run from the least hard to the hardest of the selection.
- Each test has a companion file:
  - `*_mark_scheme.pdf` holds the official mark-scheme rows for each question, labelled
    with the test question number and the original reference, e.g. `9709/32/M/J/18 Q6`.
  - `*_answers.pdf` is the answer key for multiple-choice tests.

## Contents

The last column of the per-test tables (at the end of this file) counts the unique questions
on that topic in the years covered. Where it is below 40, the test contains every question
there is. A test can hold a few more questions than that count, because strongly two-topic
questions are allowed into both topics (`cross_topic` in `index.csv`).

| Paper | Years | Tests | Questions per test |
|---|---|---|---|
| 9709 P1 Pure 1 | 2022-26 | 8 topics (1.1-1.8) | 33-40 |
| 9709 P2 Pure 2 | 2018-26 | 6 topics (2.1-2.6) | 40 each |
| 9709 P3 Pure 3 | 2018-26 | 9 topics (3.1-3.9) | 40 each |
| 9709 P4 Mechanics | 2018-26 | 5 topics (4.1-4.5) | 40 each |
| 9709 P5 Probability & Statistics 1 | 2022-26 | 5 topics (5.1-5.5) | 34-40 |
| 9709 P6 Probability & Statistics 2 | 2022-26 | 5 topics (6.1-6.5) | 30-40 |
| 9702 P1 Multiple choice | 2022-26 | syllabus topics 1-11 | 40 each |
| 9702 P2 AS structured | 2022-26 | syllabus topics 1-11 | 16-35 |
| 9702 P4 A Level structured | 2018-26 (2004-26 for topics 12, 14, 16, 19, 21) | syllabus topics 12-25 | 18-40 |
| 9702 P5 Planning, analysis and evaluation | 2018-26 | planning; data analysis | 40 each |

Paper 3 (Physics practical) is excluded, because its questions need apparatus.

**Five thin Physics P4 topics use older papers too.** A Paper 4 has about 10 questions covering
14 topics, and variants 41 and 43 are the same paper, so over 2018-2026 Motion in a circle had
only 11 questions, Alternating currents 23, Temperature 24, Thermodynamics 26 and Capacitance 28.
For these five topics the pool also takes Paper 4 questions from Oct/Nov 2004 to Oct/Nov 2017.
Each of those 100 questions was read and checked against the current syllabus before it was
admitted; they are listed in `tools/old_p4_whitelist.json` with the test each may appear in.
Questions on transformers, power transmission or modulation were left out (no longer on the
syllabus), as were 2002-2004 papers whose mark scheme is missing or only a scanned image.
Their difficulty uses the same statistics as the 2018-2026 questions, so no other test changes.

Circular motion is mostly examined inside orbit questions, so a Paper 4 question on gravitational
or electric fields also counts for Motion in a circle (as cross-topic) when its mark scheme uses
the centripetal force (`centripetal`, `ω²`, `v²/r`) and the question involves an orbit or circle.

| Topic | 2018-2026 only | With 2004-2017 (40 / 50 version) |
|---|---:|---:|
| Motion in a circle | 11 | 40 / 50 |
| Temperature | 24 | 40 / 44 |
| Thermodynamics | 26 | 40 / 43 |
| Capacitance | 28 | 40 / 49 |
| Alternating currents | 23 | 37 / 37 |

Three other Physics P4 tests stay below 40: Astronomy and cosmology (18, on the syllabus only
since 2022), Ideal gases (34) and Quantum physics (39).

## Maths by subtopic

Maths P2, P3 and P4 (Mechanics) are also available split into subtopics, in the style of
Past Paper Penguin's 9709 topical pages (pastpaperpenguin.com). Each question belongs to exactly
one subtopic, and each subtopic test holds that subtopic's hardest questions over 2018-2026 (up to
50), least hard first. The files sit inside their syllabus-topic folder, for example
`Maths_9709/P3_Pure_Mathematics_3/3.5_Integration/3.5.2_Integration_by_Parts.pdf`.
`index_subtopics.csv` lists every question with its source.

| Paper | Subtopics | Questions | Names |
|---|---:|---:|---|
| P2 Pure 2 | 14 | 309 | Past Paper Penguin has no Pure 2 page: Pure 3 names where the content is shared, plus "Integration of Exponentials and 1/(ax+b)" and "Trapezium Rule" |
| P3 Pure 3 | 23 | 613 | Past Paper Penguin's 22 Pure 3 subtopics, plus "Trapezium Rule" (2 questions) |
| P4 Mechanics | 16 | 413 (of 436; three subtopics have more than 50) | Past Paper Penguin's 16 Mechanics subtopics |

Subtopics are assigned by rules on the question and mark-scheme text (`tools/subtopics.py`).
Every list was then read through, and 79 questions were moved by hand
(`tools/subtopic_overrides.json`). Mark schemes rarely name the integration method, so most of
those moves are integration questions (by parts, substitution, trig). About a fifth of the older
question texts were extracted without spaces, so the Mechanics keywords are matched with
whitespace removed.

Some subtopics are small because Cambridge rarely sets them on their own since 2018, for example
Connected Particles: Vertical (2 questions) and Pure 3 Trapezium Rule (2).

## Older papers: content no longer on the syllabus

The 2018-2021 papers were checked against the current syllabuses (9709 for 2026-27, 9702 for
2025-27). Questions are excluded when a substantial part tests removed content. The 66
excluded questions are listed in `tools/off_syllabus.json`.

| Paper | Excluded content | Questions | Source of the rule |
|---|---|---:|---|
| Physics P4 | Communication, Electronics, NMRI, band theory, transformers | 55 | 9702 syllabus "Changes to this syllabus" page; transformers no longer in topic 21 |
| Maths P3 | Equations of planes (P3 vectors now covers lines only) | 11 | 9709 syllabus, topic 3.7 |

No exclusions were needed for Maths P2, P4 or Physics P5. Physics P5 is a skills paper, and its
2018-2021 contexts are ordinary laboratory set-ups.

May/Jun 2020 papers are included. That series was cancelled, so no examiner report exists for
it.

## How questions were chosen

1. **Download.** The script fetched 557 question papers, 553 mark schemes and 72 examiner
   reports (including Physics Paper 4 back to 2002).
2. **Split into questions.** Question numbers are found in sequence at the left margin. Each
   question is cropped page by page between the exam's header and footer.
   - 427 written papers: the question marks add up to the total printed on the cover.
   - 28 older Physics P4 and P5 papers: the cover prints no total, so the marks were checked
     against the paper's fixed total (100 and 30).
   - 35 MCQ papers: each yields 40 questions.
3. **De-duplicate.** Cambridge reuses identical papers across time-zone variants. For
   example, 9702/41 and /43 are the same paper, as are 9709/22, /23 and /25. Only one copy of
   each question is used. Copies are matched on the question text and, where the text extracted
   differently (a lost θ symbol), on a near-identical mark scheme from the same session.
4. **Classify by topic.** Classification uses keyword evidence from the question and its mark
   scheme. Physics multiple-choice papers also use Cambridge's syllabus-order layout.
   Uncertain cases were checked by hand, and so was every Physics P2/P4 question that could be
   selected. The corrections are in `tools/topic_overrides.json`.
5. **Score difficulty.** Each question gets a weighted sum of z-scores, computed within its
   paper type:

   | Paper type | Weights |
   |---|---|
   | Maths | 0.40 marks, 0.25 position in paper, 0.15 length, 0.20 examiner report |
   | Physics P2/P4 | 0.45 marks, 0.20 length, 0.35 examiner report |
   | Physics P5 | 0.60 examiner report, 0.40 length |
   | Physics MCQ | 0.50 examiner report, 0.20 numerical data in the stem, 0.15 length, 0.15 diagram content |

   The examiner-report term balances cues such as "very few candidates" and "found this
   challenging" against "well answered" and "straightforward". It also uses the explicit
   lists, e.g. "Candidates found Questions 10, 20 and 22 particularly difficult".
6. **Select.** The 40 highest-scoring unique questions on each topic go into the test. For the
   nine-year papers, every year contributes roughly in proportion to its share of the pool.

## Limits

- **"Hardest" is an estimate.** Cambridge publishes no per-question success rates. The score
  combines marks, position, length and examiner commentary.
- **Examiner reports are missing for some sessions.** PapaCambridge has none for May/Jun and
  Oct/Nov 2024-2025, Feb/Mar 2024, May/Jun 2026, or the cancelled May/Jun 2020 series.
  Questions from those papers get a neutral examiner term.
- **Topics overlap.** Many Cambridge questions span two topics. Each question is filed under
  its main topic, and appears in a second test only when the evidence is strong.
- **Two mark schemes are incomplete.** The 9702/21/M/J/26 mark scheme is not on PapaCambridge,
  and the 9709/33/M/J/24 mark scheme stops at Q10. The mark-scheme file says so where those
  questions appear.

## Rebuild

This needs Python 3 with `pymupdf` and `fonttools`.

```bash
export SP=/path/to/scratch            # downloads and intermediate files
mkdir -p $SP/papers $SP/work
cat tools/wanted*.txt | xargs -P 12 -I{} tools/dl.sh {} $SP/papers
python3 tools/segment.py    $SP/work/seg $SP/papers/97*_qp_*.pdf
python3 tools/ms_segment.py $SP/work/ms  $SP/papers/97*_ms_*.pdf
python3 tools/analyze.py    $SP
python3 tools/er.py         $SP
python3 tools/select_q.py   $SP       # 9-year window for 9709 P2/P3/P4 and 9702 P4/P5, 5-year otherwise,
                                      # plus the reviewed 2004-2017 P4 questions in tools/old_p4_whitelist.json
# add --fifty to select up to 50 questions per topic for 9709 P2/P3/P4 and 9702 P4/P5
python3 tools/make_tests.py $SP .     # writes Maths_9709/, Physics_9702/, index.csv
# Maths by subtopic (after select_q.py --fifty):
python3 tools/subtopics.py      $SP   # one subtopic per question -> work/subtopics.json
python3 tools/subtopic_tests.py $SP   # hardest 50 per subtopic   -> work/selection_sub.json
SELECTION=work/selection_sub.json SUMMARY=work/build_summary_sub.json python3 tools/make_tests.py $SP out_sub
python3 tools/leakscan.py   .         # checks no exam header/footer text survived
```

## Per-test counts

**Maths 9709 / P1 Pure Mathematics 1** (papers 2022-2026)

| Test | Questions | Pages | Unique questions on this topic |
|---|---:|---:|---:|
| 1.1 Quadratics | 33 | 40 | 32 |
| 1.2 Functions | 40 | 70 | 57 |
| 1.3 Coordinate geometry | 38 | 64 | 37 |
| 1.4 Circular measure | 34 | 54 | 33 |
| 1.5 Trigonometry | 40 | 52 | 40 |
| 1.6 Series | 40 | 57 | 78 |
| 1.7 Differentiation | 40 | 69 | 47 |
| 1.8 Integration | 40 | 73 | 55 |

**Maths 9709 / P2 Pure Mathematics 2** (papers 2018-2026)

| Test | Questions | Pages | Unique questions on this topic |
|---|---:|---:|---:|
| 2.1 Algebra | 40 | 71 | 72 |
| 2.2 Logarithmic and exponential functions | 40 | 49 | 38 |
| 2.3 Trigonometry | 40 | 77 | 55 |
| 2.4 Differentiation | 40 | 71 | 58 |
| 2.5 Integration | 40 | 67 | 44 |
| 2.6 Numerical solution of equations | 40 | 76 | 42 |

**Maths 9709 / P3 Pure Mathematics 3** (papers 2018-2026)

| Test | Questions | Pages | Unique questions on this topic |
|---|---:|---:|---:|
| 3.1 Algebra | 40 | 70 | 88 |
| 3.2 Logarithmic and exponential functions | 40 | 41 | 62 |
| 3.3 Trigonometry | 40 | 76 | 76 |
| 3.4 Differentiation | 40 | 71 | 64 |
| 3.5 Integration | 40 | 78 | 76 |
| 3.6 Numerical solution of equations | 40 | 78 | 58 |
| 3.7 Vectors | 40 | 79 | 50 |
| 3.8 Differential equations | 40 | 79 | 61 |
| 3.9 Complex numbers | 40 | 69 | 78 |

**Maths 9709 / P4 Mechanics** (papers 2018-2026)

| Test | Questions | Pages | Unique questions on this topic |
|---|---:|---:|---:|
| 4.1 Forces and equilibrium | 40 | 72 | 95 |
| 4.2 Kinematics of motion in a straight line | 40 | 81 | 123 |
| 4.3 Momentum | 40 | 67 | 48 |
| 4.4 Newton s laws of motion | 40 | 81 | 53 |
| 4.5 Energy work and power | 40 | 80 | 117 |

**Maths 9709 / P5 Probability and Statistics 1** (papers 2022-2026)

| Test | Questions | Pages | Unique questions on this topic |
|---|---:|---:|---:|
| 5.1 Representation of data | 40 | 66 | 40 |
| 5.2 Permutations and combinations | 40 | 80 | 64 |
| 5.3 Probability | 34 | 54 | 28 |
| 5.4 Discrete random variables | 39 | 56 | 38 |
| 5.5 The normal distribution | 40 | 76 | 59 |

**Maths 9709 / P6 Probability and Statistics 2** (papers 2022-2026)

| Test | Questions | Pages | Unique questions on this topic |
|---|---:|---:|---:|
| 6.1 The Poisson distribution | 40 | 65 | 44 |
| 6.2 Linear combinations of random variables | 30 | 43 | 30 |
| 6.3 Continuous random variables | 32 | 59 | 32 |
| 6.4 Sampling and estimation | 40 | 45 | 45 |
| 6.5 Hypothesis tests | 40 | 69 | 61 |

**Physics 9702 / P1 Multiple Choice** (papers 2022-2026)

| Test | Questions | Pages | Unique questions on this topic |
|---|---:|---:|---:|
| T01 Physical quantities and units | 40 | 12 | 119 |
| T02 Kinematics | 40 | 18 | 111 |
| T03 Dynamics | 40 | 17 | 118 |
| T04 Forces density and pressure | 40 | 17 | 152 |
| T05 Work energy and power | 40 | 13 | 99 |
| T06 Deformation of solids | 40 | 17 | 93 |
| T07 Waves | 40 | 18 | 175 |
| T08 Superposition | 40 | 16 | 114 |
| T09 Electricity | 40 | 15 | 119 |
| T10 D.C. circuits | 40 | 20 | 115 |
| T11 Particle physics | 40 | 12 | 101 |

**Physics 9702 / P2 AS Structured** (papers 2022-2026)

| Test | Questions | Pages | Unique questions on this topic |
|---|---:|---:|---:|
| T01 Physical quantities and units | 23 | 36 | 23 |
| T02 Kinematics | 16 | 34 | 16 |
| T03 Dynamics | 26 | 54 | 22 |
| T04 Forces density and pressure | 31 | 66 | 25 |
| T05 Work energy and power | 16 | 30 | 10 |
| T06 Deformation of solids | 25 | 53 | 21 |
| T07 Waves | 21 | 42 | 17 |
| T08 Superposition | 24 | 44 | 23 |
| T09 Electricity | 25 | 52 | 16 |
| T10 D.C. circuits | 25 | 56 | 23 |
| T11 Particle physics | 35 | 42 | 35 |

**Physics 9702 / P4 A Level Structured** (papers 2018-2026; topics 12, 14, 16, 19, 21 also 2004-2017)

| Test | Questions | Pages | Unique questions on this topic |
|---|---:|---:|---:|
| T12 Motion in a circle | 40 | 81 | 20 (+40 orbit questions) |
| T13 Gravitational fields | 40 | 78 | 42 |
| T14 Temperature | 40 | 70 | 40 |
| T15 Ideal gases | 34 | 65 | 29 |
| T16 Thermodynamics | 40 | 63 | 40 |
| T17 Oscillations | 40 | 87 | 46 |
| T18 Electric fields | 40 | 78 | 41 |
| T19 Capacitance | 40 | 79 | 46 |
| T20 Magnetic fields | 40 | 84 | 58 |
| T21 Alternating currents | 37 | 72 | 36 |
| T22 Quantum physics | 39 | 67 | 38 |
| T23 Nuclear physics | 40 | 67 | 39 |
| T24 Medical physics | 40 | 61 | 41 |
| T25 Astronomy and cosmology | 18 | 27 | 16 |

**Physics 9702 / P5 Planning Analysis Evaluation** (papers 2018-2026)

| Test | Questions | Pages | Unique questions on this topic |
|---|---:|---:|---:|
| Data analysis and evaluation | 40 | 160 | 46 |
| Planning | 40 | 120 | 46 |
