"""Read Past Paper Penguin set PDFs: (subtopic, set) -> list of (paper id, question number).
usage: ppp_read.py PDF_DIR OUT_JSON"""
import pymupdf, glob, re, sys, os, json
SER = {"F/M": "m", "M/J": "s", "O/N": "w"}
out = {}
for f in sorted(glob.glob(os.path.join(sys.argv[1], "Q-*.pdf"))):
    d = pymupdf.open(f)
    items, cur = [], None
    for p in d:
        t = p.get_text()
        m = re.search(r"97(09)/(\d\d)/([A-Z])/?([A-Z])/?(\d\d)", t)
        ref = f"9709_{SER[m.group(3) + '/' + m.group(4)]}{m.group(5)}_{m.group(2)}" if m else None
        qn = None
        for b in p.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                s = "".join(x["text"] for x in l["spans"]).strip()
                mm = re.match(r"^(\d{1,2})(\s|$)", s)
                if mm and l["bbox"][0] < 75 and l["bbox"][1] < 200 and qn is None:
                    qn = int(mm.group(1))
        if ref and qn:
            cur = (ref, qn); items.append(cur)
        elif ref is None and len(t.strip()) < 80:
            items.append(("img", None))
    name = os.path.basename(f)[2:-4]
    out[name] = items
json.dump(out, open(sys.argv[2], "w"), indent=0)
print(sum(len([i for i in v if i[0] != "img"]) for v in out.values()), "questions read;",
      sum(len([i for i in v if i[0] == "img"]) for v in out.values()), "image-only pages")
