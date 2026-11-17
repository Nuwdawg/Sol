"""CONTENTS.txt for the Past Paper Penguin-style maths sets.  usage: ppp_contents.py SCRATCH OUT_DIR"""
import json, os, sys, collections
SP, OUT = sys.argv[1], sys.argv[2]
S = json.load(open(os.path.join(SP, "work/selection_ppp.json")))
B = {name: (n, npg) for folder, name, n, npg, _ in json.load(open(os.path.join(SP, "work/build_summary_ppp.json")))}
L = ["Cambridge A Level Maths 9709 - topical questions organised like Past Paper Penguin",
     "",
     "Pure Mathematics 2 and 3 are ONE list, in the order of Past Paper Penguin's Pure 3 topical page;",
     "Pure 2 questions sit in the same subtopics as Pure 3 questions on the same content.",
     "Mechanics follows the order of their Mechanics 1 page.",
     "A subtopic with more than 25 questions is split into Set A and Set B. Every question starts on a new",
     "page. Each set has a mark scheme file (Past Paper Penguin's 'ANSWERS' column) with every question's",
     "source paper, e.g. 9709/22/O/N/23 Q6 is Pure 2, 9709/32/... is Pure 3.",
     "",
     "Papers: Feb/Mar 2018 to May/Jun 2026. Up to 50 hardest questions per subtopic, least hard first,",
     "so Set B holds the hardest. Questions that Past Paper Penguin's own sets contain (Pure 3 and Mechanics,",
     "2020 to Feb/Mar 2025) are in the subtopic they chose; a few are in two subtopics, as on their site.",
     "* = row added for Pure 2 content that Past Paper Penguin's Pure 3 page has no row for.", ""]
rows = S["rows"]
cur = None
for fo, n, name, pool, ch, sets, p2, p3 in rows:
    if fo != cur:
        L += ["", {"Pure_Mathematics_2_and_3": "PURE MATHEMATICS 2 AND 3", "Mechanics": "MECHANICS"}[fo], f"{'TOPIC':48s} QUESTIONS / ANSWERS"]
        cur = fo
    star = "*" if name in S["added"] else " "
    mix = f"   (Pure 2: {p2}, Pure 3: {p3})" if fo.startswith("Pure") else ""
    setinfo = " . ".join(f"Set {l} ({c} q)" for l, c in sets)
    L.append(f"{n:02d} {name + star:45s} {setinfo}{mix}")
open(os.path.join(OUT, "CONTENTS.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
