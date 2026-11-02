# CIE A Level topical tests: Maths 9709 and Physics 9702

These tests collect the hardest questions on each syllabus topic from Cambridge International
past papers. All papers, mark schemes and examiner reports were downloaded from PapaCambridge.

| Papers | Years covered |
|---|---|
| Maths P3 Pure 3, Maths P4 Mechanics, Physics P4 | **9 years**: Feb/Mar 2018 to May/Jun 2026 |
| All other papers | **5 years**: Feb/Mar 2022 to May/Jun 2026 |

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
| 9709 P2 Pure 2 | 2022-26 | 6 topics (2.1-2.6) | 23-37 |
| 9709 P3 Pure 3 | 2018-26 | 9 topics (3.1-3.9) | 40 each |
| 9709 P4 Mechanics | 2018-26 | 5 topics (4.1-4.5) | 40 each |
| 9709 P5 Probability & Statistics 1 | 2022-26 | 5 topics (5.1-5.5) | 34-40 |
| 9709 P6 Probability & Statistics 2 | 2022-26 | 5 topics (6.1-6.5) | 30-40 |
| 9702 P1 Multiple choice | 2022-26 | syllabus topics 1-11 | 40 each |
| 9702 P2 AS structured | 2022-26 | syllabus topics 1-11 | 16-35 |
| 9702 P4 A Level structured | 2018-26 | syllabus topics 12-25 | 11-40 |
| 9702 P5 Planning, analysis and evaluation | 2022-26 | planning; data analysis | 26 each |

Paper 3 (Physics practical) is excluded, because its questions need apparatus.

**Why some Physics P4 tests stay below 40, even over nine years.** A Paper 4 has about 10
questions covering 14 topics, so most topics appear at most once per paper. Variants 41 and 43
are the same paper in every May/Jun and Oct/Nov session, which leaves about 5 distinct papers a
year. Some topics are also rarely examined on their own:

| Topic | Questions | Reason |
|---|---:|---|
| Motion in a circle | 11 | usually examined inside gravitational-field questions |
| Astronomy and cosmology | 18 | only on the syllabus since 2022 |
| Alternating currents | 23 | |
| Temperature | 24 | |
| Thermodynamics | 26 | |
| Capacitance | 28 | |
| Ideal gases | 34 | |
| Quantum physics | 39 | |

## Older papers: content no longer on the syllabus

The 2018-2021 papers were checked against the current syllabuses (9709 for 2026-27, 9702 for
2025-27). Questions are excluded when a substantial part tests removed content. The 66
excluded questions are listed in `tools/off_syllabus.json`.

| Paper | Excluded content | Questions | Source of the rule |
|---|---|---:|---|
| Physics P4 | Communication, Electronics, NMRI, band theory, transformers | 55 | 9702 syllabus "Changes to this syllabus" page; transformers no longer in topic 21 |
| Maths P3 | Equations of planes (P3 vectors now covers lines only) | 11 | 9709 syllabus, topic 3.7 |

May/Jun 2020 papers are included. That series was cancelled, so no examiner report exists for
it.

## How questions were chosen

1. **Download.** The script fetched 434 question papers, 433 mark schemes and 38 examiner
   reports.
2. **Split into questions.** Question numbers are found in sequence at the left margin. Each
   question is cropped page by page between the exam's header and footer.
   - 385 written papers: the question marks add up to the total printed on the cover.
   - 14 older Physics P4 papers: the cover prints no total, so the marks were checked against
     Paper 4's fixed 100.
   - 35 MCQ papers: each yields 40 questions.
3. **De-duplicate.** Cambridge reuses identical papers across time-zone variants. For
   example, 9702/41 and /43 are the same paper, as are 9709/22, /23 and /25. Only one copy of
   each question is used.
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
cat tools/wanted.txt tools/wanted_2018_2021.txt | xargs -P 12 -I{} tools/dl.sh {} $SP/papers
python3 tools/segment.py    $SP/work/seg $SP/papers/97*_qp_*.pdf
python3 tools/ms_segment.py $SP/work/ms  $SP/papers/97*_ms_*.pdf
python3 tools/analyze.py    $SP
python3 tools/er.py         $SP
python3 tools/select_q.py   $SP       # 9-year window for 9709 P3/P4 and 9702 P4, 5-year otherwise
python3 tools/make_tests.py $SP .     # writes Maths_9709/, Physics_9702/, index.csv
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

**Maths 9709 / P2 Pure Mathematics 2** (papers 2022-2026)

| Test | Questions | Pages | Unique questions on this topic |
|---|---:|---:|---:|
| 2.1 Algebra | 36 | 53 | 36 |
| 2.2 Logarithmic and exponential functions | 27 | 31 | 18 |
| 2.3 Trigonometry | 37 | 56 | 33 |
| 2.4 Differentiation | 34 | 52 | 34 |
| 2.5 Integration | 30 | 47 | 29 |
| 2.6 Numerical solution of equations | 23 | 41 | 22 |

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

**Physics 9702 / P4 A Level Structured** (papers 2018-2026)

| Test | Questions | Pages | Unique questions on this topic |
|---|---:|---:|---:|
| T12 Motion in a circle | 11 | 23 | 10 |
| T13 Gravitational fields | 40 | 78 | 42 |
| T14 Temperature | 24 | 46 | 20 |
| T15 Ideal gases | 34 | 65 | 29 |
| T16 Thermodynamics | 26 | 43 | 23 |
| T17 Oscillations | 40 | 87 | 46 |
| T18 Electric fields | 40 | 78 | 41 |
| T19 Capacitance | 28 | 54 | 25 |
| T20 Magnetic fields | 40 | 84 | 58 |
| T21 Alternating currents | 23 | 46 | 22 |
| T22 Quantum physics | 39 | 67 | 38 |
| T23 Nuclear physics | 40 | 67 | 39 |
| T24 Medical physics | 40 | 61 | 41 |
| T25 Astronomy and cosmology | 18 | 27 | 16 |

**Physics 9702 / P5 Planning Analysis Evaluation** (papers 2022-2026)

| Test | Questions | Pages | Unique questions on this topic |
|---|---:|---:|---:|
| Planning | 26 | 78 | 26 |
| Data analysis and evaluation | 26 | 104 | 26 |
