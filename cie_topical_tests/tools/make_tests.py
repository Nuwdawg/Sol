"""Build the topical test PDFs (questions only, no exam headers) and their mark-scheme/answer PDFs.

usage: make_tests.py SCRATCH OUT_DIR [slug-prefix ...]
"""
import sys, os, json, re, csv, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pymupdf, wmclean, ms_segment

SP, OUT = sys.argv[1], sys.argv[2]
ONLY = sys.argv[3:]
S = json.load(open(os.path.join(SP, "work/selection.json")))
Q = S["questions"]

A4P = (595.28, 841.89)
A4L = (841.89, 595.28)
SERIES = {"m": "F/M", "s": "M/J", "w": "O/N"}
COMP_NAME = {
    ("9709", 1): "P1 Pure Mathematics 1", ("9709", 2): "P2 Pure Mathematics 2", ("9709", 3): "P3 Pure Mathematics 3",
    ("9709", 4): "P4 Mechanics", ("9709", 5): "P5 Probability and Statistics 1", ("9709", 6): "P6 Probability and Statistics 2",
}

def ref(r):
    """CIE-style reference, e.g. 9709/12/M/J/23 Q7"""
    subj, sess, _, cv = r["paper"].split("_")
    return f"{subj}/{cv}/{SERIES[sess[0]]}/{sess[1:]} Q{r['q']}"

# ------------------------------------------------------------------ sources
_qp, _ms = {}, {}
def qp_doc(paper):
    if paper not in _qp:
        d = pymupdf.open(os.path.join(SP, "papers", paper + ".pdf"))
        wmclean.clean_doc(d)
        _qp[paper] = d
    return _qp[paper]

def ms_doc(paper):
    if paper not in _ms:
        _ms[paper] = ms_segment.normalized(os.path.join(SP, "papers", paper + ".pdf"))
    return _ms[paper]

def save_small(doc, path):
    try:
        doc.subset_fonts(verbose=False)
    except Exception as e:
        print("  (font subsetting skipped:", e, ")")
    doc.save(path, garbage=4, deflate=True, clean=True)

# ------------------------------------------------------------------ question test
def build_questions(items, path, font):
    _qp.clear()                                   # fresh copies: number redactions are per test
    by_page = collections.defaultdict(list)
    for r in items:
        by_page[(r["paper"], r["segs"][0][0])].append(r["num_bbox"])
    for (paper, pno), boxes in by_page.items():
        pg = qp_doc(paper)[pno]
        for b in boxes:
            pg.add_redact_annot(pymupdf.Rect(b[0] - 0.5, b[1] + 1.0, b[2] + 0.3, b[3] - 1.0))
        pg.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                            text=pymupdf.PDF_REDACT_TEXT_REMOVE)
    out = pymupdf.open()
    page, y = None, 0.0
    TOP, BOT, GAPQ, GAPS = 40.0, 40.0, 20.0, 6.0
    for k, r in enumerate(items, 1):
        src = qp_doc(r["paper"])
        for si, (pno, x0, y0, x1, y1) in enumerate(r["segs"]):
            h = y1 - y0
            gap = (GAPQ if si == 0 else GAPS) if page is not None else 0.0
            if page is None or y + gap + h > A4P[1] - BOT:
                page = out.new_page(width=A4P[0], height=A4P[1]); y, gap = TOP, 0.0
            y += gap
            page.show_pdf_page(pymupdf.Rect(x0, y, x1, y + h), src, pno, clip=pymupdf.Rect(x0, y0, x1, y1))
            if si == 0:
                nb = r["num_bbox"]
                page.insert_text((nb[0], y + (r["num_base"] - y0)), str(k), fontsize=r["num_size"], fontname=font)
            y += h
    out.set_metadata({"title": os.path.splitext(os.path.basename(path))[0].replace("_", " "), "author": "", "creator": "", "producer": ""})
    save_small(out, path)
    return out.page_count

# ------------------------------------------------------------------ mark schemes
def build_markscheme(items, path, title):
    out = pymupdf.open()
    W, H = A4L
    M, LBL = 30.0, 18.0
    page, y = None, 0.0
    def new_page():
        p = out.new_page(width=W, height=H)
        return p, 28.0
    for k, r in enumerate(items, 1):
        label = f"Question {k}   ({ref(r)}, {r['marks']} marks)"
        segs = r.get("ms_segs") or []
        if segs:
            sw0 = ms_doc(r["ms_paper"])[segs[0][0]].rect.width
            sc0 = min(1.0, (W - 2 * M) / (sw0 - 60.0))
            need = LBL + min((segs[0][4] - segs[0][2]) * sc0, H - 24 - 28 - LBL)
        else:
            need = LBL + 20
        if page is None or y + need > H - 24:
            page, y = new_page()
        page.insert_text((M + 4, y + 12), label, fontsize=10.5, fontname="hebo", color=(0.1, 0.1, 0.1))
        y += LBL
        if not segs:
            page.insert_text((M + 4, y + 12), "Mark scheme for this question is not included in the published source file.",
                             fontsize=10, fontname="helv", color=(0.35, 0.35, 0.35))
            y += 26
            continue
        src = ms_doc(r["ms_paper"])
        for pno, x0, y0, x1, y1 in segs:
            h = y1 - y0
            sw = src[pno].rect.width
            x0, x1 = 30.0, sw - 30.0
            scale = min(1.0, (W - 2 * M) / (x1 - x0))
            th = h * scale
            if y + th > H - 24:
                page, y = new_page()
            tw = (x1 - x0) * scale
            page.show_pdf_page(pymupdf.Rect(M, y, M + tw, y + th), src, pno, clip=pymupdf.Rect(x0, y0, x1, y1))
            y += th + 4
        y += 10
    out.set_metadata({"title": title, "author": "", "creator": "", "producer": ""})
    save_small(out, path)
    return out.page_count

def build_mcq_key(items, path, title):
    out = pymupdf.open()
    W, H = A4P
    page = out.new_page(width=W, height=H)
    y = 50.0
    page.insert_text((50, y), "Answers", fontsize=13, fontname="hebo"); y += 26
    page.insert_text((50, y), "Question", fontsize=9.5, fontname="hebo")
    page.insert_text((130, y), "Answer", fontsize=9.5, fontname="hebo")
    page.insert_text((200, y), "Source", fontsize=9.5, fontname="hebo"); y += 6
    page.draw_line((50, y), (420, y), color=(0.6, 0.6, 0.6), width=0.6); y += 14
    for k, r in enumerate(items, 1):
        page.insert_text((62, y), str(k), fontsize=10, fontname="helv")
        page.insert_text((142, y), r.get("key") or "-", fontsize=10, fontname="hebo")
        page.insert_text((200, y), ref(r), fontsize=10, fontname="helv")
        y += 17
    out.set_metadata({"title": title, "author": "", "creator": "", "producer": ""})
    out.save(path, garbage=4, deflate=True, clean=True)

# ------------------------------------------------------------------ main
def folder_and_name(t):
    if t["subj"] == "9709":
        comp = int(t["kind"][1])
        folder = os.path.join("Maths_9709", COMP_NAME[("9709", comp)].replace(" ", "_"))
        name = f'{t["topic"]}_{t["title"].split(t["topic"] + " ", 1)[1]}'
    else:
        sub = {"P1-MCQ": "P1_Multiple_Choice", "P2": "P2_AS_Structured", "P4": "P4_A_Level_Structured",
               "P5": "P5_Planning_Analysis_Evaluation"}[t["kind"]]
        folder = os.path.join("Physics_9702", sub)
        if t["kind"] == "P5":
            name = "Planning" if t["topic"] == "P1" else "Data_analysis_and_evaluation"
        else:
            name = f'T{t["topic"]}_' + t["title"].split(f'Topic {int(t["topic"])} ', 1)[1]
    name = re.sub(r"[^A-Za-z0-9.]+", "_", name).strip("_")
    return folder, name

rows = []
summary = []
for t in S["tests"]:
    if ONLY and not any(t["slug"].startswith(p) for p in ONLY):
        continue
    items = [Q[i] for i in t["questions"]]
    folder, name = folder_and_name(t)
    os.makedirs(os.path.join(OUT, folder), exist_ok=True)
    qpath = os.path.join(OUT, folder, name + ".pdf")
    font = "tibo" if t["subj"] == "9709" else "hebo"
    npg = build_questions(items, qpath, font)
    if t["kind"] == "P1-MCQ":
        apath = os.path.join(OUT, folder, name + "_answers.pdf")
        build_mcq_key(items, apath, name + " answers")
    else:
        apath = os.path.join(OUT, folder, name + "_mark_scheme.pdf")
        build_markscheme(items, apath, name + " mark scheme")
    for k, r in enumerate(items, 1):
        rows.append({"test": f"{folder}/{name}.pdf", "question": k, "source": ref(r), "marks": r["marks"],
                     "topic": r["topic"], "cross_topic": "yes" if r["id"] in t["secondary"] else "",
                     "difficulty_score": r["difficulty"],
                     "examiner_report": (r.get("er_note") or ("commented" if r.get("f_er") is not None else "")),
                     "answer": r.get("key") or ""})
    summary.append((folder, name, len(items), npg, t["pool_primary"]))
    print(f"{folder}/{name}: {len(items)} questions, {npg} pages", flush=True)

if not ONLY:
    with open(os.path.join(OUT, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    json.dump(summary, open(os.path.join(SP, "work/build_summary.json"), "w"))
