"""Split CIE question papers into questions.

For each paper: find question-number spans (1..N in order, at the left margin),
compute per-question page segments with exam headers/footers/margins excluded,
and record text and marks.  Output: one JSON per paper.
"""
import sys, os, re, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pymupdf, wmclean

X0, X1 = 39.5, 556.0          # horizontal content window (excludes margin bars/text and the stroked corner marks)
FOOT_RX = re.compile(r"(©|UCLES|Cambridge University Press|Turn over|97\d\d/\d\d/[A-Z]/[A-Z]/\d\d)")
SKIP_RX = re.compile(r"(BLANK PAGE|Additional [Pp]age|If you use the following lined page|"
                     r"Permission to reproduce items|Local Examinations Syndicate)")
MARK_RX = re.compile(r"\[(\d{1,2})\]")
TOTAL_RX = re.compile(r"\[Total:\s*(\d{1,3})\]")

def page_items(page):
    """Horizontal text lines (with chars) and graphics with bboxes."""
    lines, graphics = [], []
    d = page.get_text("rawdict")
    for b in d["blocks"]:
        if b["type"] == 1:
            graphics.append(pymupdf.Rect(b["bbox"]))
            continue
        for l in b["lines"]:
            if abs(l["dir"][1]) > 0.01:
                continue                                   # rotated margin text
            spans, chars = [], []
            for sp in l["spans"]:
                t = "".join(c["c"] for c in sp["chars"])
                if not t.strip():
                    continue
                spans.append({"text": t, "bbox": sp["bbox"], "size": sp["size"], "font": sp["font"],
                              "origin": sp["origin"]})
                chars.extend((c["c"], c["bbox"][0], c["bbox"][2]) for c in sp["chars"])
            if not spans:
                continue
            lines.append({"bbox": pymupdf.Rect(l["bbox"]), "spans": spans, "chars": chars,
                          "text": "".join(sp["text"] for sp in spans)})
    for dr in page.get_drawings():
        r = dr["rect"]
        if r.width == 0 and r.height == 0:
            continue
        graphics.append(pymupdf.Rect(r))
    return lines, graphics

COPY_RX = re.compile(r"(Permission to reproduce|To avoid the issue of disclosure|Copyright Acknowledgements|"
                     r"is the name of our awarding body|Local Examinations Syndicate|is part of Cambridge)")

def cut_lines(page, lines, graphics):
    """Return (top, bottom) y-limits of the content area for this page."""
    H, W = page.rect.height, page.rect.width
    top = 52.0
    for l in lines:
        r, sz = l["bbox"], l["spans"][0]["size"]
        if r.y0 >= 76:
            continue
        lone_num = re.fullmatch(r"\s*\d{1,2}\s*", l["text"]) and 0.42 * W < r.x0 < 0.56 * W and r.y1 <= 63
        barcode = sz < 6.6 and r.width > 100 and r.y1 <= 75
        if barcode or lone_num or r.y1 <= 52 or (r.y1 <= 62 and sz < 9):
            top = max(top, r.y1 + 0.5)                     # page number, barcode text/label
    for g in graphics:
        if g.y0 < 58 and (g.x1 < 62 or g.x0 > W - 66) and g.height < 30:
            top = max(top, g.y1 + 2.0)                     # corner marks (stroke ~2.8pt), black square
        elif g.y1 <= 52:
            top = max(top, g.y1 + 0.5)
    bottom = H - 30.0
    for l in lines:
        r, sz = l["bbox"], l["spans"][0]["size"]
        if r.y0 < H - 110:
            continue
        if FOOT_RX.search(l["text"]) or sz < 6.6 or (sz < 7.6 and r.width > 150) or COPY_RX.search(l["text"]):
            bottom = min(bottom, r.y0 - 0.5)
    for g in graphics:
        if g.y0 > H - 70 and (g.x1 < 62 or g.x0 > W - 66):
            bottom = min(bottom, g.y0 - 2.0)               # bottom corner marks (stroke ~2.8pt)
    return top, bottom

def qnum_of_line(l, top, bottom):
    """If line l starts with a question number, return (n, bbox_of_digits)."""
    ch = [c for c in l["chars"]]
    i = 0
    while i < len(ch) and ch[i][0].isspace():
        i += 1
    j = i
    while j < len(ch) and ch[j][0].isdigit():
        j += 1
    if j == i or j - i > 2:
        return None
    x0 = ch[i][1]
    if not (46.5 <= x0 <= 53.5) or l["spans"][0]["size"] < 10:
        return None
    r = l["bbox"]
    if r.y0 < top - 1 or r.y1 > bottom + 1:
        return None
    k = j
    while k < len(ch) and ch[k][0].isspace():
        k += 1
    if k < len(ch) and ch[k][1] < 64:
        return None                                        # ordinary text that happens to start with digits
    n = int("".join(c[0] for c in ch[i:j]))
    sp0 = l["spans"][0]
    return n, pymupdf.Rect(x0, r.y0, ch[j - 1][2], r.y1), sp0["origin"][1], sp0["size"]

def find_qnums(doc, info):
    """Ordered list of (qnum, page, bbox) for question starts (must run 1, 2, 3, ...)."""
    cands = []
    for pno, (lines, graphics, top, bottom, skip) in info.items():
        if skip:
            continue
        for l in lines:
            res = qnum_of_line(l, top, bottom)
            if res:
                cands.append((pno, res[1].y0, res[0], res[1], res[2], res[3]))
    cands.sort(key=lambda c: (c[0], c[1]))
    out, expect = [], 1
    for pno, y, n, r, base, size in cands:
        if n == expect:
            out.append((n, pno, r, base, size)); expect += 1
    return out

def is_skip_page(pno, lines, graphics, top, bottom):
    if pno == 0:
        return True
    body = [l for l in lines if top <= l["bbox"].y0 and l["bbox"].y1 <= bottom + 1 and X0 <= l["bbox"].x0]
    txt = " ".join(l["text"] for l in body)
    if re.search(r"BLANK PAGE", txt) and len(txt) < 80:
        return True
    if re.search(r"(Additional [Pp]age|If you use the following lined page)", txt):
        return True
    return False

def content_extent(lines, graphics, y0, y1, top, bottom):
    """Tight vertical extent of content within [y0, y1] on a page (x-window applied)."""
    lo, hi = None, None
    for l in lines:
        r = l["bbox"]
        if r.y1 <= y0 + 0.5 or r.y0 >= y1 - 0.5 or r.x1 < X0 or r.x0 > X1:
            continue
        if r.y0 < top - 0.5 or r.y1 > bottom + 1.5:
            continue
        lo = r.y0 if lo is None else min(lo, r.y0)
        hi = r.y1 if hi is None else max(hi, r.y1)
    for g in graphics:
        if g.y1 <= y0 + 0.5 or g.y0 >= y1 - 0.5 or g.x1 < X0 or g.x0 > X1:
            continue
        if g.y0 < top - 0.5 or g.y1 > bottom + 1.5:
            continue
        lo = g.y0 if lo is None else min(lo, g.y0)
        hi = g.y1 if hi is None else max(hi, g.y1)
    return lo, hi

def start_top(lines, graphics, nb):
    """Top of the first row of a question: expand the number's line upward through
    touching items to its right (tall fractions, column vectors, diagrams)."""
    lo, hi = nb.y0, nb.y1
    items = [l["bbox"] for l in lines] + list(graphics)
    changed = True
    while changed:
        changed = False
        for r in items:
            if r.x0 < nb.x1 - 1 or r.x1 < X0 or r.x0 > X1:
                continue
            if r.y0 < lo - 0.2 and r.y1 > lo - 1.0 and r.y0 < hi and r.y0 >= nb.y0 - 25:
                lo = r.y0; changed = True
    return lo

def segment(path):
    doc = pymupdf.open(path)
    wmclean.clean_doc(doc)
    info = {}
    for pno in range(len(doc)):
        page = doc[pno]
        lines, graphics = page_items(page)
        top, bottom = cut_lines(page, lines, graphics)
        # cut copyright block on last pages
        for l in lines:
            if COPY_RX.search(l["text"]):
                bottom = min(bottom, l["bbox"].y0 - 4)
            # in-body page-turn notes ("Turn over", "[Question 7 continues on the next page.]")
            if re.fullmatch(r"\s*\[?\s*(Turn over|Question \d+ continues on the next page\.?)\s*\]?\s*", l["text"]) and l["bbox"].y0 > 150:
                bottom = min(bottom, l["bbox"].y0 - 2)
        skip = is_skip_page(pno, lines, graphics, top, bottom)
        info[pno] = (lines, graphics, top, bottom, skip)
    qn = find_qnums(doc, info)
    cover = doc[0].get_text()
    m = re.search(r"total (?:number of )?marks? for this paper (?:is|will be) (\d+)", cover)
    paper_total = int(m.group(1)) if m else None
    questions = []
    tops = [start_top(info[pno][0], info[pno][1], r) for (n, pno, r, base, size) in qn]
    for i, (n, pno, r, base, size) in enumerate(qn):
        if i + 1 < len(qn):
            end_p, end_y = qn[i + 1][1], tops[i + 1] - 1.5
        else:
            # last question: run to the last non-skip page
            end_p = pno
            for p in range(pno + 1, len(doc)):
                if info[p][4]:
                    break
                end_p = p
            end_y = None
        segs, texts = [], []
        for p in range(pno, end_p + 1):
            lines, graphics, top, bottom, skip = info[p]
            if skip:
                continue
            y0 = tops[i] - 2 if p == pno else top
            y1 = end_y if (p == end_p and end_y is not None) else bottom
            y0 = max(y0, top)
            y1 = min(y1, bottom)
            if y1 - y0 < 4:
                continue
            lo, hi = content_extent(lines, graphics, y0, y1, top, bottom)
            if lo is None:
                continue
            seg = [p, X0, max(y0, lo - 2), X1, min(y1, hi + 3)]
            segs.append(seg)
            body = [l for l in lines if seg[2] - 1 <= (l["bbox"].y0 + l["bbox"].y1) / 2 <= seg[4] + 1
                    and l["bbox"].x0 < X1 and l["bbox"].x1 > X0]
            body.sort(key=lambda l: (round(l["bbox"].y0 / 3), l["bbox"].x0))
            texts.append("\n".join(l["text"] for l in body))
        text = "\n".join(texts)
        total = TOTAL_RX.findall(text)
        marks = [int(x) for x in MARK_RX.findall(text)]
        q = {"q": n, "page": pno, "num_bbox": [round(v, 2) for v in r], "num_base": round(base, 2),
             "num_size": round(size, 2),
             "segs": [[s[0]] + [round(v, 2) for v in s[1:]] for s in segs],
             "marks_parts": marks, "marks": int(total[-1]) if total else sum(marks),
             "text": text}
        questions.append(q)
    return {"paper": os.path.basename(path)[:-4], "n_pages": len(doc), "paper_total": paper_total,
            "questions": questions}

if __name__ == "__main__":
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    for path in sys.argv[2:]:
        res = segment(path)
        with open(os.path.join(out_dir, res["paper"] + ".json"), "w") as f:
            json.dump(res, f)
        s = sum(q["marks"] for q in res["questions"])
        flag = "" if res["paper_total"] == s else "  <-- MISMATCH"
        print(f'{res["paper"]}: {len(res["questions"])} q, marks {s}/{res["paper_total"]}{flag}', flush=True)
