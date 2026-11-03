"""Score question difficulty and pick the hardest 40 per topic.

Difficulty is a weighted sum of within-paper-type z-scores:
  maths 9709        0.40 marks + 0.25 position-in-paper + 0.15 length + 0.20 examiner-report
  physics P2 / P4   0.45 marks + 0.20 length + 0.35 examiner-report
  physics P5        0.60 examiner-report + 0.40 length
  physics P1 (MCQ)  0.50 examiner-report + 0.20 numeric data in stem + 0.15 length + 0.15 diagram content
Examiner-report evidence = (hard cues - easy cues) / (hard + easy + 3), plus the
explicit "found Questions x, y particularly difficult / straightforward" lists.
Papers without a published report get a neutral (0) examiner term.
Output: work/selection.json
"""
import sys, os, json, re, math, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pymupdf, wmclean, topics

SP = sys.argv[1]
R = json.load(open(os.path.join(SP, "work/questions.json")))
ER = json.load(open(os.path.join(SP, "work/er.json")))
# Year window: nine years (Feb/Mar 2018 - May/Jun 2026) for Maths P2, P3, P4 (Mechanics) and Physics P4, P5;
# five years (Feb/Mar 2022 - May/Jun 2026) for everything else
NINE_YEAR = {("9709", 2), ("9709", 3), ("9709", 4), ("9702", 4), ("9702", 5)}
R = [r for r in R if r["year"] >= (2018 if (r["subj"], r["comp"]) in NINE_YEAR else 2022)]
def _datafile(name, fallback):
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    return here if os.path.exists(here) else os.path.join(SP, "work", fallback)
TARGET = 40

# ---------------- features
def words(t):
    return len(re.findall(r"[A-Za-z]{2,}", topics.norm(t)))

def stem_of_mcq(t):
    n = topics.norm(t)
    m = re.search(r"\sA\s", n)
    return n[:m.start()] if m else n

fig_cache = {}
def drawings_in(r):
    """Number of vector drawings/images inside the question's crop (diagram richness)."""
    paper = r["paper"]
    if paper not in fig_cache:
        d = pymupdf.open(os.path.join(SP, "papers", paper + ".pdf"))
        wmclean.clean_doc(d)
        fig_cache.clear()
        fig_cache[paper] = (d, {})
    d, pc = fig_cache[paper]
    n = 0
    for p, x0, y0, x1, y1 in r["segs"]:
        if p not in pc:
            pc[p] = [dr["rect"] for dr in d[p].get_drawings()] + [pymupdf.Rect(b["bbox"]) for b in d[p].get_text("dict")["blocks"] if b["type"] == 1]
        n += sum(1 for g in pc[p] if g.y0 >= y0 - 1 and g.y1 <= y1 + 1 and g.width + g.height > 3)
    return n

for r in R:
    r["f_words"] = words(r["text"])
    r["f_relpos"] = (r["q"] - 1) / max(1, r["N"] - 1)
    e = ER.get(r["paper"])
    if e is None:
        r["f_er"] = None
    else:
        c = e.get(str(r["q"]))
        if c is None:
            r["f_er"] = -0.15 if (r["subj"] == "9702" and r["comp"] == 1) else None
        else:
            h, s = c["hard"], c["easy"]
            r["f_er"] = (h - s) / (h + s + 3.0)
            r["er_note"] = ("listed as hard" if c.get("hard_list") else "listed as easy" if c.get("easy_list") else "")
    if r["subj"] == "9702" and r["comp"] == 1:
        stem = stem_of_mcq(r["text"])
        r["f_nums"] = len(re.findall(r"\d+(?:\.\d+)?", stem))
        r["f_words"] = words(stem)
        r["f_figs"] = drawings_in(r)

# ---------------- z-scores within paper type
def group(r):
    return (r["subj"], r["comp"])

def zfill(key):
    by = collections.defaultdict(list)
    for r in R:
        v = r.get(key)
        if v is not None:
            by[group(r)].append(v)
    stats = {g: (sum(v) / len(v), math.sqrt(sum((x - sum(v) / len(v)) ** 2 for x in v) / max(1, len(v) - 1)) or 1.0) for g, v in by.items()}
    for r in R:
        v = r.get(key)
        m, s = stats.get(group(r), (0, 1))
        r["z_" + key[2:]] = 0.0 if v is None else max(-3, min(3, (v - m) / s))

for k in ("f_words", "f_relpos", "f_er", "f_nums", "f_figs"):
    zfill(k)
for r in R:
    r["z_marks"] = 0.0
by = collections.defaultdict(list)
for r in R:
    by[group(r)].append(r["marks"])
for r in R:
    v = by[group(r)]; m = sum(v) / len(v); s = math.sqrt(sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)) or 1.0
    r["z_marks"] = (r["marks"] - m) / s

for r in R:
    if r["subj"] == "9709":
        d = 0.40 * r["z_marks"] + 0.25 * r["z_relpos"] + 0.15 * r["z_words"] + 0.20 * r["z_er"]
    elif r["comp"] in (2, 4):
        d = 0.45 * r["z_marks"] + 0.20 * r["z_words"] + 0.35 * r["z_er"]
    elif r["comp"] == 5:
        d = 0.60 * r["z_er"] + 0.40 * r["z_words"]
    else:
        d = 0.50 * r["z_er"] + 0.20 * r["z_nums"] + 0.15 * r["z_words"] + 0.15 * r["z_figs"]
    r["difficulty"] = round(d, 4)

# ---------------- de-duplicate: CIE reuses identical papers/questions across variants
def shingles(t):
    c = re.sub(r"[^a-z0-9]+", "", topics.norm(t).lower())
    return {c[i:i + 8] for i in range(max(1, len(c) - 7))}

SH = {r["id"]: shingles(r["text"]) for r in R}
canon = {}
by_slot = collections.defaultdict(list)
for r in R:
    by_slot[(r["subj"], r["comp"])].append(r)
for slot, rs in by_slot.items():
    reps = []
    # prefer the copy that has examiner-report evidence, then the lowest variant
    for r in sorted(rs, key=lambda r: (r["f_er"] is None, r["paper"])):
        s = SH[r["id"]]
        hit = None
        for rep in reps:
            t = SH[rep["id"]]
            if len(s) > 12 and len(s & t) / (len(s | t) or 1) > 0.75:
                hit = rep; break
        if hit:
            canon[r["id"]] = hit["id"]
            hit.setdefault("dup_of_this", []).append(r["id"])
        else:
            canon[r["id"]] = r["id"]; reps.append(r)
# ---------------- optional manual topic overrides (id -> topic code)
ovf = _datafile("topic_overrides.json", "overrides.json")
OV = json.load(open(ovf)) if os.path.exists(ovf) else {}
names = {}
for comp, spec in topics.MATH.items():
    for code, name, _ in spec:
        names[("9709", code)] = name
for code, name, _ in topics.PHYS:
    names[("9702", code)] = name
for code, name in topics.PHYS_P5:
    names[("9702", code)] = name
RID = {r["id"]: r for r in R}
for qid, code in OV.items():
    if qid not in RID:
        print("override for unknown id", qid); continue
    r = RID[canon[qid]]
    r["topic"] = code; r["topic_name"] = names[(r["subj"], code)]; r["topic_override"] = True

# questions on content removed from the current syllabus (id -> reason), e.g. pre-2022 Physics communication/electronics
off_f = _datafile("off_syllabus.json", "off_syllabus.json")
OFF = json.load(open(off_f)) if os.path.exists(off_f) else {}
for qid, why in OFF.items():
    if qid in RID:
        RID[canon[qid]]["off_syllabus"] = why
UNIQUE = [r for r in R if canon[r["id"]] == r["id"] and not r.get("off_syllabus")]
print("excluded as off-syllabus:", sum(1 for r in R if r.get("off_syllabus")))
print("unique questions:", len(UNIQUE), "of", len(R))

# ---------------- tests
tests = []
def add_test(subj, kind, comp_filter, topic_code, title, slug):
    prim = [r for r in UNIQUE if r["subj"] == subj and comp_filter(r) and r["topic"] == topic_code]
    sec = [r for r in UNIQUE if r["subj"] == subj and comp_filter(r) and r["topic"] != topic_code
           and topic_code in r.get("topics_2nd", []) and not r.get("topic_override") and kind != "P1-MCQ"]
    ranked = sorted([(r["difficulty"], r, 1) for r in prim] + [(r["difficulty"] - 0.4, r, 2) for r in sec],
                    key=lambda x: -x[0])
    chosen = [(r, k) for d, r, k in ranked[:TARGET]]
    chosen.sort(key=lambda x: x[0]["difficulty"])      # within a test: hard -> hardest
    tests.append({"subj": subj, "kind": kind, "topic": topic_code, "title": title, "slug": slug,
                  "pool_primary": len(prim), "pool_secondary": len(sec),
                  "questions": [r["id"] for r, k in chosen], "secondary": [r["id"] for r, k in chosen if k == 2]})

for comp, spec in topics.MATH.items():
    for code, name, _ in spec:
        add_test("9709", f"P{comp}", lambda r, c=comp: r["comp"] == c, code, f"Paper {comp} - {code} {name}",
                 f"9709_P{comp}_{code.replace('.', '-')}_{re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')}")
for code, name, _ in topics.PHYS[:11]:
    add_test("9702", "P1-MCQ", lambda r: r["comp"] == 1, code, f"Paper 1 (MCQ) - Topic {int(code)} {name}",
             f"9702_P1-MCQ_{code}_{re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')}")
for code, name, _ in topics.PHYS[:11]:
    add_test("9702", "P2", lambda r: r["comp"] == 2, code, f"Paper 2 (structured) - Topic {int(code)} {name}",
             f"9702_P2_{code}_{re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')}")
for code, name, _ in topics.PHYS[11:]:
    add_test("9702", "P4", lambda r: r["comp"] == 4, code, f"Paper 4 (structured) - Topic {int(code)} {name}",
             f"9702_P4_{code}_{re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')}")
for code, name in topics.PHYS_P5:
    add_test("9702", "P5", lambda r: r["comp"] == 5, code, f"Paper 5 - {name}",
             f"9702_P5_{code}_{re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')}")

json.dump({"tests": tests, "questions": {r["id"]: r for r in R}}, open(os.path.join(SP, "work/selection.json"), "w"))
for t in tests:
    print(f'{t["slug"]:62s} unique pool {t["pool_primary"]:4d} (+{t["pool_secondary"]:3d} cross-topic)  chosen {len(t["questions"])} ({len(t["secondary"])} cross)')
print(len(tests), "tests;", sum(len(t["questions"]) for t in tests), "questions")
