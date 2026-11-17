"""Maths 9709 topicals organised the way Past Paper Penguin organises its 9709 topical pages.

* Pure Mathematics 2 and 3 form ONE list, in the order of Past Paper Penguin's Pure 3 page
  (pastpaperpenguin.com/pure-3-topical-exam-questions-and-worked-solutions, read 2026-11-17).
  Pure 2 questions go into the same subtopics as the Pure 3 questions on the same content.
  Two rows are added after the integration rows for Pure 2 content that page has no row for.
* Mechanics follows the order of the Mechanics 1 page.
* A subtopic with more than 25 questions is split into Set A, Set B, ... of near-equal size.
* Every question starts on a new page.
* For the 606 questions that appear in Past Paper Penguin's own sets (Pure 3 and Mechanics, 2020 to
  Feb/Mar 2025), their placement is used (ppp_placements.json, read from their PDFs by ppp_read.py).
  Every other question (2018-2019, May/Jun 2025 onwards, all of Pure 2) is placed by subtopics.py,
  whose rules follow the same conventions.
Within a subtopic the hardest questions are kept (up to 50, as before), least hard first, so the
last set holds the hardest ones.

Reads work/selection.json and work/subtopics.json; writes work/selection_ppp.json.
usage: ppp_sets.py SCRATCH
"""
import sys, os, json, re, math

PURE = [  # (name, subtopic codes from subtopics.py)
    ("Division & Factor/Remainder Theorems", ["2.1.1", "3.1.1"]),
    ("Modulus Equations and Inequalities", ["2.1.2", "3.1.2"]),
    ("Exps and Logs: Equations and Inequalities", ["2.2.1", "3.2.1"]),
    ("Transforming Log Graphs to Linear", ["2.2.2", "3.2.2"]),
    ("Trig: Compound and Double Angles", ["2.3.1", "3.3.1"]),
    ("Rsin(x+a) and Rcos(x+a)", ["2.3.2", "3.3.2"]),
    ("Implicit Differentiation", ["2.4.2", "3.4.2"]),
    ("Parametric Differentiation", ["2.4.3", "3.4.3"]),
    ("Product and Quotient Rule", ["2.4.1", "3.4.1"]),
    ("Integration of Trig Functions", ["2.5.1", "3.5.1"]),
    ("Integration by Parts", ["3.5.2"]),
    ("Integration by Substitution", ["3.5.3"]),
    ("Partial Fractions and Integration", ["3.5.4", "2.5.4"]),     # incl. divide-then-integrate (as Past Paper Penguin does)
    ("Integration of Exponentials and 1/(ax+b)", ["2.5.2"]),        # added: Pure 2 only
    ("Trapezium Rule", ["2.5.3", "3.5.5"]),                          # added: not on the Pure 3 page
    ("Differential Equations", ["3.8.1"]),
    ("Differential Equations & Modelling", ["3.8.2"]),
    ("Iteration with Calculus", ["2.6.1", "3.6.1"]),
    ("Iteration with Graph Sketching or Radians", ["2.6.2", "3.6.2"]),
    ("Binomial Expansion", ["3.1.3"]),
    ("Partial Fractions and Binomial Expansion", ["3.1.4"]),
    ("Vectors", ["3.7.1"]),
    ("Complex Numbers: Equations and Roots", ["3.9.1"]),
    ("Complex Numbers: Equations and Loci", ["3.9.2"]),
]
ADDED = {"Integration of Exponentials and 1/(ax+b)", "Trapezium Rule"}
MECH = [
    ("Kinematic Graphs", ["4.2.1"]), ("Suvat: Horizontal", ["4.2.2"]), ("Suvat: Vertical", ["4.2.3"]),
    ("Resolving Coplanar Forces", ["4.1.1"]), ("Friction: Horizontal", ["4.1.2"]), ("Friction: On an Incline", ["4.1.3"]),
    ("Connected Particles: Horizontal or Inclined", ["4.4.1"]), ("Connected Particles: Vertical", ["4.4.2"]),
    ("Pulleys", ["4.4.3"]), ("Pulleys and Work/Energy", ["4.5.4"]), ("General Motion in a Straight Line", ["4.2.4"]),
    ("Momentum & Collisions", ["4.3.1"]), ("Collisions with Friction or Energy", ["4.3.2"]), ("Work and Energy", ["4.5.1"]),
    ("Power and F=ma", ["4.5.2"]), ("Power and F=ma with Connected Particles", ["4.5.3"]),
]
CAP, SET_MAX = 50, 25

def safe(s):
    s = s.replace("/", "-").replace(":", "")
    return re.sub(r"[^A-Za-z0-9.()+=&-]+", "_", s).strip("_")

if __name__ == "__main__":
    SP = sys.argv[1]
    Q = json.load(open(os.path.join(SP, "work/selection.json")))["questions"]
    ST = json.load(open(os.path.join(SP, "work/subtopics.json")))
    here = os.path.dirname(os.path.abspath(__file__))
    PL = json.load(open(os.path.join(here, "ppp_placements.json")))
    rowname = {c: n for n, cs in PURE + MECH for c in cs}
    ROW = {}
    for i, c in ST.items():
        ours = rowname[c]
        theirs = PL.get(i)
        ROW[i] = theirs if theirs else [ours]          # a few questions sit in two of their subtopics
    print("placed by Past Paper Penguin:", sum(1 for i in ST if PL.get(i) and ROW[i] != [rowname[ST[i]]]), "differ from our rules;",
          sum(1 for i in ST if PL.get(i)), "covered;", sum(1 for i in ST if len(ROW[i]) > 1), "in two subtopics")
    tests, rows = [], []
    for folder, kind, spec in (("Maths_9709/Pure_Mathematics_2_and_3", "P23", PURE), ("Maths_9709/Mechanics", "P4", MECH)):
        for n, (name, codes) in enumerate(spec, 1):
            pool = [Q[i] for i in ST if name in ROW[i]]
            chosen = sorted(pool, key=lambda r: -r["difficulty"])[:CAP]
            chosen.sort(key=lambda r: r["difficulty"])
            k = max(1, math.ceil(len(chosen) / SET_MAX))
            cut = [round(j * len(chosen) / k) for j in range(k + 1)]
            sets = []
            for j in range(k):
                letter = "ABCDEFGH"[j]
                qs = [r["id"] for r in chosen[cut[j]:cut[j + 1]]]
                fname = f"{n:02d}_{safe(name)}_Set_{letter}"
                tests.append({"subj": "9709", "kind": kind, "topic": f"{n:02d}", "title": f"{name} - Set {letter}",
                              "slug": f"9709_{kind}_{n:02d}_{letter}", "folder": folder, "name": fname, "newpage": True,
                              "pool_primary": len(pool), "pool_secondary": 0, "questions": qs, "secondary": []})
                sets.append((letter, len(qs)))
            rows.append((folder.split("/")[1], n, name, len(pool), len(chosen), sets,
                         sum(1 for r in chosen if r["comp"] == 2), sum(1 for r in chosen if r["comp"] == 3)))
    used = {i for t in tests for i in t["questions"]}
    json.dump({"tests": tests, "questions": {i: Q[i] for i in used}, "rows": rows, "added": sorted(ADDED)},
              open(os.path.join(SP, "work/selection_ppp.json"), "w"))
    for fo, n, name, pool, ch, sets, p2, p3 in rows:
        mix = f"  (P2 {p2}, P3 {p3})" if fo.startswith("Pure") else ""
        print(f"{fo[:4]} {n:02d} {name:45s} pool {pool:3d} chosen {ch:2d}  sets {' '.join(f'{l}:{c}' for l, c in sets)}{mix}")
    print(len(tests), "sets;", len(used), "questions")
