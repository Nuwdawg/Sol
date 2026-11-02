"""Split CIE mark schemes into per-question regions (and MCQ answer keys)."""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pymupdf, wmclean

LABEL_RX = re.compile(r"^\s*[.Q]?\s*(\d{1,2})\s*((\([a-z]{1,2}\)|\([ivx]{1,4}\))\s*)*$")

def normalized(path):
    """Watermark-free copy with every page unrotated in displayed orientation."""
    doc = pymupdf.open(path)
    wmclean.clean_doc(doc)
    for p in doc:
        if p.rotation:
            p.remove_rotation()
    return doc

def lines_of(page):
    res = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"]:
            continue
        for l in b["lines"]:
            if abs(l["dir"][1]) > 0.01:
                continue
            t = "".join(s["text"] for s in l["spans"])
            if t.strip():
                res.append((pymupdf.Rect(l["bbox"]), t))
    return res

def page_window(page, lines):
    """Content window between the mark-scheme running header and footer (positions vary by year)."""
    H = page.rect.height
    top, bottom = 0.0, H
    for r, t in lines:
        if r.y0 < 0.12 * H and re.search(r"(Mark Scheme|PUBLISHED|^\s*\d{4}/\d{2}\s*$|(January|February|March|May|June|October|November)[/ ].*20\d\d)", t):
            top = max(top, r.y1 + 1)
        if r.y0 > 0.4 * H and re.search(r"(©|UCLES|Cambridge University Press|Page \d+ of \d+)", t):
            bottom = min(bottom, r.y0 - 1)
    if top == 0.0:
        top = 48.0
    if bottom == H:
        bottom = H - 40.0
    return top, bottom

def segment_ms(path):
    doc = normalized(path)
    info, heads, labels = {}, [], []
    started = False
    col = (0, 0)
    for pno, page in enumerate(doc):
        lines = lines_of(page)
        top, bottom = page_window(page, lines)
        info[pno] = (lines, top, bottom)
        for r, t in lines:
            if r.y0 < top - 1 or r.y1 > bottom + 1:
                continue
            if t.strip().startswith("Question") and r.x0 < 160:
                row = [t2.strip() for r2, t2 in lines if abs(r2.y0 - r.y0) < 4 and r2.x0 > r.x1]
                row += t.strip().split()[1:]
                if any(x in ("Answer", "Marks", "Answers", "Mark") for x in row):
                    heads.append((pno, r.y0, r)); started = True
                    col = (r.x0 - 25, r.x1 + 5)
                    continue
            if started and col[0] <= r.x0 and r.x1 <= col[1] + 6 and r.x0 < col[1]:
                m = LABEL_RX.match(t)
                if m:
                    labels.append((pno, r.y0, int(m.group(1)), t.strip(), r))
    labels.sort(key=lambda x: (x[0], x[1]))
    starts, expect = [], 1
    for pno, y, n, txt, r in labels:
        if n == expect or (n == expect + 1 and expect > 1):
            starts.append((n, pno, r)); expect = n + 1
    # include a table header row immediately above a question start
    def start_y(pno, r):
        y = r.y0
        for hp, hy, hr in heads:
            if hp == pno and 0 <= r.y0 - hy < 40:
                y = min(y, hy)
        return y - 3
    qs = []
    last_page = max(info)
    for i, (n, pno, r) in enumerate(starts):
        sy = start_y(pno, r)
        if i + 1 < len(starts):
            ep, ey = starts[i + 1][1], start_y(starts[i + 1][1], starts[i + 1][2]) - 1
        else:
            ep, ey = last_page, None
        segs, texts = [], []
        for p in range(pno, ep + 1):
            lines, top, bottom = info[p]
            y0 = sy if p == pno else top
            y1 = ey if (p == ep and ey is not None) else bottom
            body = [(lr, t) for lr, t in lines if lr.y0 >= y0 - 1 and lr.y1 <= y1 + 1 and top <= lr.y0 and lr.y1 <= bottom + 1]
            if not body:
                continue
            lo = min(lr.y0 for lr, t in body); hi = max(lr.y1 for lr, t in body)
            # include table rules/drawings: extend to drawings within the window
            for d in doc[p].get_drawings():
                g = d["rect"]
                if g.y0 >= y0 - 1 and g.y1 <= y1 + 1 and g.y0 >= top - 1 and g.y1 <= bottom + 1:
                    lo = min(lo, g.y0); hi = max(hi, g.y1)
            W = doc[p].rect.width
            segs.append([p, 30.0, round(max(y0, lo - 2), 2), round(W - 30.0, 2), round(min(y1, hi + 2), 2)])
            body.sort(key=lambda x: (round(x[0].y0 / 3), x[0].x0))
            texts.append("\n".join(t for lr, t in body))
        qs.append({"q": n, "segs": segs, "text": "\n".join(texts)})
    return doc, qs

def mcq_key(path):
    doc = normalized(path)
    key = {}
    for page in doc:
        lines = lines_of(page)
        nums = [(r, int(t.strip())) for r, t in lines if re.fullmatch(r"\s*\d{1,2}\s*", t) and r.x0 < 200]
        lets = [(r, t.strip()) for r, t in lines if re.fullmatch(r"\s*[A-D]\s*", t)]
        for r, n in nums:
            row = [(lr.x0, L) for lr, L in lets if abs(lr.y0 - r.y0) < 4 and lr.x0 > r.x1]
            if row and 1 <= n <= 40 and n not in key:
                key[n] = sorted(row)[0][1]
    return key

if __name__ == "__main__":
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    for path in sys.argv[2:]:
        name = os.path.basename(path)[:-4]
        if name.startswith("9702") and name.split("_")[-1][0] == "1":
            k = mcq_key(path)
            json.dump({"paper": name, "key": k}, open(os.path.join(out_dir, name + ".json"), "w"))
            print(f"{name}: key {len(k)}/40", flush=True)
        else:
            doc, qs = segment_ms(path)
            json.dump({"paper": name, "questions": qs}, open(os.path.join(out_dir, name + ".json"), "w"))
            print(f"{name}: {len(qs)} q", flush=True)
