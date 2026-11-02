import sys, os, re, glob, pymupdf
root = sys.argv[1]
PAT = re.compile(r"(UCLES|Cambridge University Press|Turn over|DO NOT WRITE|BLANK PAGE|Additional page|Permission to reproduce|PapaCambridge|papacambridge|97(09|02)/\d\d/[A-Z]/[A-Z]/\d\d|Page \d+ of \d+|PUBLISHED|Mark Scheme)", re.I)
hits = 0; pages = 0
for f in sorted(glob.glob(os.path.join(root, "**/*.pdf"), recursive=True)):
    is_q = not (f.endswith("_mark_scheme.pdf") or f.endswith("_answers.pdf"))
    d = pymupdf.open(f)
    for p in d:
        pages += 1
        t = p.get_text()
        if not is_q:
            t = re.sub(r"Question \d+ +\(97\d\d/\d\d/[A-Z]/[A-Z]/\d\d Q\d+, \d+ marks\)", "", t)
            t = re.sub(r"97\d\d/\d\d/[A-Z]/[A-Z]/\d\d Q\d+", "", t)
        for m in PAT.finditer(t):
            hits += 1
            if hits <= 15: print(os.path.relpath(f, root), p.number, repr(m.group(0)))
print("pages scanned", pages, "hits", hits)
