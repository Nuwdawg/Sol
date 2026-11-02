# CIE A Level topical tests: Maths 9709 and Physics 9702

These tests collect the hardest questions on each syllabus topic from the last five years of
Cambridge International past papers. They run from Feb/Mar 2022 to May/Jun 2026; Oct/Nov 2026
has not been sat yet. All papers, mark schemes and examiner reports were downloaded from
PapaCambridge.

The PDFs are **not committed**, because this repository is public and the papers are
Cambridge copyright material (`.gitignore`). Rebuild them with `tools/`, as described below.
`index.csv` lists every question in every test with its original reference.

## What a test looks like

- Each page holds the question content only, cropped from the original paper as vector
  graphics. Nothing else from the exam page is carried over:
  - no page header or footer, and no page numbers
  - no "DO NOT WRITE IN THIS MARGIN" bars or barcodes
  - no copyright notices, no blank or "additional" pages
  - no PapaCambridge watermark or trace text
- Nothing is added either: no title band and no running header.
- Questions are renumbered 1, 2, 3, ... and keep their original answer space.
- Within each test, questions run from the least hard to the hardest of the selection.
- Each test has a companion file:
  - `*_mark_scheme.pdf` holds the official mark-scheme rows for each question, labelled
    with the test question number and the original reference, e.g. `9709/12/M/J/23 Q7`.
  - `*_answers.pdf` is the answer key for multiple-choice tests.

## Contents

The last column counts the unique questions on that topic across all five years. Where it
is below 40, the test contains every question there is. A test can hold a few more questions
than that count because strongly two-topic questions are allowed into both topics
(`cross_topic` in `index.csv`).

| Paper | Tests | Questions per test |
|---|---|---|
| 9709 P1 Pure 1 | 8 topics (1.1-1.8) | 33-40 |
| 9709 P2 Pure 2 | 6 topics (2.1-2.6) | 23-37 |
| 9709 P3 Pure 3 | 9 topics (3.1-3.9) | 33-40 |
| 9709 P4 Mechanics | 5 topics (4.1-4.5) | 34-40 |
| 9709 P5 Probability & Statistics 1 | 5 topics (5.1-5.5) | 34-40 |
| 9709 P6 Probability & Statistics 2 | 5 topics (6.1-6.5) | 30-40 |
| 9702 P1 Multiple choice | syllabus topics 1-11 | 40 each |
| 9702 P2 AS structured | syllabus topics 1-11 | 16-35 |
| 9702 P4 A Level structured | syllabus topics 12-25 | 9-27 |
| 9702 P5 Planning, analysis and evaluation | planning; data analysis | 26 each |

Paper 3 (Physics practical) is excluded, because its questions need apparatus.
Full per-test counts are at the end of this file.

## How questions were chosen

1. **Download.** The script fetched 350 question papers, 349 mark schemes and 16 examiner
   reports (9709 P1-P6; 9702 P1, P2, P4, P5).
2. **Split into questions.** Question numbers are found in sequence at the left margin.
   Each question is cropped page by page between the exam's header and footer. Every
   written paper's question marks add up to the total printed on its cover, and every MCQ
   paper yields 40 questions.
3. **De-duplicate.** Cambridge reuses identical papers across time-zone variants. For
   example, 9702/41 and /43 are the same paper, as are 9709/22, /23 and /25. The 3,786
   extracted questions reduce to about 3,470 unique ones, and only one copy of each is used.
4. **Classify by topic.** Classification uses keyword evidence from the question and its
   mark scheme, such as "Integrate to obtain", "partial fractions" or "Hall voltage".
   - Physics multiple-choice papers also use the fact that Cambridge orders questions by
     syllabus topic.
   - Uncertain cases were checked by hand, and so was every selected Physics P2/P4
     question. 137 corrections are in `tools/topic_overrides.json`.
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
6. **Select.** The 40 highest-scoring unique questions on each topic go into the test.

## Limits

- **"Hardest" is an estimate.** Cambridge publishes no per-question success rates. The
  score combines marks, position, length and examiner commentary.
- **Examiner reports are patchy.** Reports exist for 8 of the 14 sessions. PapaCambridge
  has none for May/Jun and Oct/Nov 2024-2025, May/Jun 2026, or Feb/Mar 2024. Questions
  from those papers get a neutral examiner term.
- **Topics overlap.** Many Cambridge questions span two topics. Each question is filed
  under its main topic, and appears in a second test only when the evidence is strong.
- **Two mark schemes are incomplete.** The 9702/21/M/J/26 mark scheme is not on
  PapaCambridge, and the 9709/33/M/J/24 mark scheme stops at Q10. The mark-scheme file
  says so where those questions appear.

## Rebuild

This needs Python 3 with `pymupdf` and `fonttools`.

```bash
export SP=/path/to/scratch            # downloads and intermediate files
mkdir -p $SP/papers $SP/work
cat tools/wanted.txt | xargs -P 12 -I{} tools/dl.sh {} $SP/papers
python3 tools/segment.py    $SP/work/seg $SP/papers/97*_qp_*.pdf
python3 tools/ms_segment.py $SP/work/ms  $SP/papers/97*_ms_*.pdf
python3 tools/analyze.py    $SP
python3 tools/er.py         $SP
python3 tools/select_q.py   $SP
python3 tools/make_tests.py $SP .     # writes Maths_9709/, Physics_9702/, index.csv
python3 tools/leakscan.py   .         # checks no exam header/footer text survived
```

## Per-test counts

**Maths 9709 / P1 Pure Mathematics 1**

| Test | Questions | Pages | Unique questions on this topic, 2022-2026 |
|---|---:|---:|---:|
| 1.1 Quadratics | 33 | 40 | 32 |
| 1.2 Functions | 40 | 70 | 57 |
| 1.3 Coordinate geometry | 38 | 64 | 37 |
| 1.4 Circular measure | 34 | 54 | 33 |
| 1.5 Trigonometry | 40 | 52 | 40 |
| 1.6 Series | 40 | 57 | 78 |
| 1.7 Differentiation | 40 | 69 | 47 |
| 1.8 Integration | 40 | 73 | 55 |

**Maths 9709 / P2 Pure Mathematics 2**

| Test | Questions | Pages | Unique questions on this topic, 2022-2026 |
|---|---:|---:|---:|
| 2.1 Algebra | 36 | 53 | 36 |
| 2.2 Logarithmic and exponential functions | 27 | 31 | 18 |
| 2.3 Trigonometry | 37 | 56 | 33 |
| 2.4 Differentiation | 34 | 52 | 34 |
| 2.5 Integration | 30 | 47 | 29 |
| 2.6 Numerical solution of equations | 23 | 41 | 22 |

**Maths 9709 / P3 Pure Mathematics 3**

| Test | Questions | Pages | Unique questions on this topic, 2022-2026 |
|---|---:|---:|---:|
| 3.1 Algebra | 40 | 63 | 58 |
| 3.2 Logarithmic and exponential functions | 37 | 37 | 36 |
| 3.3 Trigonometry | 40 | 71 | 39 |
| 3.4 Differentiation | 40 | 61 | 39 |
| 3.5 Integration | 40 | 75 | 42 |
| 3.6 Numerical solution of equations | 33 | 59 | 32 |
| 3.7 Vectors | 35 | 69 | 35 |
| 3.8 Differential equations | 35 | 66 | 35 |
| 3.9 Complex numbers | 40 | 50 | 53 |

**Maths 9709 / P4 Mechanics**

| Test | Questions | Pages | Unique questions on this topic, 2022-2026 |
|---|---:|---:|---:|
| 4.1 Forces and equilibrium | 40 | 61 | 47 |
| 4.2 Kinematics of motion in a straight line | 40 | 77 | 67 |
| 4.3 Momentum | 34 | 55 | 33 |
| 4.4 Newton s laws of motion | 40 | 66 | 31 |
| 4.5 Energy work and power | 40 | 70 | 66 |

**Maths 9709 / P5 Probability and Statistics 1**

| Test | Questions | Pages | Unique questions on this topic, 2022-2026 |
|---|---:|---:|---:|
| 5.1 Representation of data | 40 | 66 | 40 |
| 5.2 Permutations and combinations | 40 | 80 | 64 |
| 5.3 Probability | 34 | 54 | 28 |
| 5.4 Discrete random variables | 39 | 56 | 38 |
| 5.5 The normal distribution | 40 | 76 | 59 |

**Maths 9709 / P6 Probability and Statistics 2**

| Test | Questions | Pages | Unique questions on this topic, 2022-2026 |
|---|---:|---:|---:|
| 6.1 The Poisson distribution | 40 | 65 | 44 |
| 6.2 Linear combinations of random variables | 30 | 43 | 30 |
| 6.3 Continuous random variables | 32 | 59 | 32 |
| 6.4 Sampling and estimation | 40 | 45 | 45 |
| 6.5 Hypothesis tests | 40 | 69 | 61 |

**Physics 9702 / P1 Multiple Choice**

| Test | Questions | Pages | Unique questions on this topic, 2022-2026 |
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

**Physics 9702 / P2 AS Structured**

| Test | Questions | Pages | Unique questions on this topic, 2022-2026 |
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

**Physics 9702 / P4 A Level Structured**

| Test | Questions | Pages | Unique questions on this topic, 2022-2026 |
|---|---:|---:|---:|
| T12 Motion in a circle | 9 | 19 | 8 |
| T13 Gravitational fields | 22 | 45 | 22 |
| T14 Temperature | 18 | 32 | 15 |
| T15 Ideal gases | 19 | 37 | 16 |
| T16 Thermodynamics | 16 | 25 | 15 |
| T17 Oscillations | 25 | 53 | 25 |
| T18 Electric fields | 24 | 48 | 23 |
| T19 Capacitance | 18 | 36 | 16 |
| T20 Magnetic fields | 27 | 55 | 26 |
| T21 Alternating currents | 18 | 37 | 17 |
| T22 Quantum physics | 24 | 42 | 24 |
| T23 Nuclear physics | 23 | 43 | 21 |
| T24 Medical physics | 20 | 33 | 19 |
| T25 Astronomy and cosmology | 18 | 26 | 16 |

**Physics 9702 / P5 Planning Analysis Evaluation**

| Test | Questions | Pages | Unique questions on this topic, 2022-2026 |
|---|---:|---:|---:|
| Planning | 26 | 78 | 26 |
| Data analysis and evaluation | 26 | 104 | 26 |
