"""Remove the PapaCambridge watermark, site footer and trace text from a paper.

PapaCambridge stamps 2026 files like this (per page content stream):
    q  [q ...tiled /FormXob.pcm watermark draws (alpha .055)... Q]
       [q 0 0 W H re W n  ...original page content...  Q]  Q
    q 0 0 W H re W n ...inline watermark paths (gRLs0-0), site footer, invisible trace text... Q
We keep exactly one block: the full-page-clipped block that contains no
watermark marker. That block is the original exam page.
"""
import re

MARKERS = (b"FormXob.pcm", b"gRLs0", b"PapaCambridge", b"papacambridge")
CLIP = re.compile(rb"^q\s*\n\s*0(\.0)? 0(\.0)? \d+(\.\d+)? \d+(\.\d+)? re\s*\n\s*W\s*\n\s*n\b")

def blocks(s: bytes):
    """Yield (start, end) byte ranges of every q..Q block (q and Q on own lines)."""
    out, stack, pos = [], [], 0
    for ln in s.split(b"\n"):
        tok = ln.strip()
        if tok == b"q":
            stack.append(pos + ln.index(b"q"))
        elif tok == b"Q":
            if not stack:
                raise ValueError("unbalanced Q")
            st = stack.pop()
            out.append((st, pos + ln.index(b"Q") + 1))
        pos += len(ln) + 1
    if stack:
        raise ValueError("unbalanced q")
    return out

def clean_doc(doc):
    n = 0
    for page in doc:
        xrefs = page.get_contents()
        s = b"\n".join(doc.xref_stream(x) for x in xrefs)
        if not any(m in s for m in MARKERS):
            continue
        cands = [(a, b) for a, b in blocks(s)
                 if CLIP.match(s[a:b]) and not any(m in s[a:b] for m in MARKERS)]
        if len(cands) != 1:
            raise ValueError(f"page {page.number}: {len(cands)} clean content blocks")
        a, b = cands[0]
        doc.update_stream(xrefs[0], s[a:b] + b"\n")
        for x in xrefs[1:]:
            doc.update_stream(x, b"")
        n += 1
    return n
