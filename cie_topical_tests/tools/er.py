"""Parse CIE Principal Examiner Reports into per-question difficulty evidence.

Returns {paper_id: {qnum: {"hard": float, "easy": float, "text": str}}}
where paper_id is like 9709_s23_qp_12.  For MCQ papers the general comments
("Candidates found Questions 10, 20 and 22 particularly difficult ...") are
also used.
"""
import sys, os, re, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pymupdf, wmclean, unicodedata

HARD = [
    (r"most (challenging|difficult|demanding)", 3), (r"(very|particularly|extremely|quite) (challenging|difficult|demanding)", 3),
    (r"proved (to be )?(challenging|difficult|demanding)", 3), (r"found (this|it|the question|this question|this part)( to be)? (challenging|difficult|demanding|hard)", 3),
    (r"\bchallenging\b", 1.5), (r"\bdifficult(y|ies)?\b", 1.2), (r"\bdemanding\b", 1.5),
    (r"very few", 3), (r"\bfew candidates\b", 2), (r"only a (small )?(minority|few|handful|small number)", 2.5),
    (r"a (small )?minority of candidates (were able|managed|scored|gained|obtained|achieved|realised|recognised|answered|knew|gave|produced|could)", 2),
    (r"(less|fewer) than half", 2.5), (r"rarely", 2), (r"seldom", 2), (r"poorly (answered|attempted|done)", 3),
    (r"not (well|often) answered|less well answered|not answered well", 2.5), (r"struggled", 2), (r"many candidates (were unable|did not|failed|could not|omitted|struggled|lost|made)", 1.5),
    (r"common (error|mistake|misconception)", 0.7), (r"misconception|confus", 1), (r"(were|was) (often|frequently) (omitted|incorrect)|no response", 1.5),
    (r"discriminat", 1.2), (r"only the (most able|strongest|best)|stronger candidates|more able|most able", 1.5),
    (r"(incorrect|wrong) (answer|option|response)s? (was|were) (chosen|selected|popular)", 1.5), (r"more candidates (chose|selected)", 2),
]
EASY = [
    (r"very well answered|answered very well|extremely well", 3), (r"(generally|usually|mostly) well (answered|done|attempted)", 2.5),
    (r"\bwell answered\b|answered well|well done\b", 2), (r"(nearly|almost) all", 2.5),
    (r"(most|the majority of|a large majority of|a high proportion of|many) candidates (were able|scored|gained|obtained|answered|achieved|found|completed|knew|recognised|correctly|identified|gave|could|made a good)", 1.8),
    (r"straightforward|accessible|good start|high (level of )?success|successfully", 1.5), (r"full marks were (often|frequently|commonly)", 2),
    (r"confident(ly)?|familiar", 0.8), (r"mostly answered (this question )?correctly|answered correctly by (most|the majority)", 2.5),
]
HARD_RX = [(re.compile(p, re.I), w) for p, w in HARD]
EASY_RX = [(re.compile(p, re.I), w) for p, w in EASY]

def doc_text(path):
    d = pymupdf.open(path)
    wmclean.clean_doc(d)
    t = "\n".join(p.get_text() for p in d)
    t = unicodedata.normalize("NFKC", t)
    # drop running headers
    t = re.sub(r"Cambridge International (Advanced Subsidiary and Advanced Level|AS & A Level)[^\n]*\n[^\n]*(Mathematics|Physics)[^\n]*\n[^\n]*Principal Examiner Report for Teachers[^\n]*\n(©[^\n]*\n)?", "\n", t)
    return t

def parse(path):
    name = os.path.basename(path)[:-4]                 # 9709_s23_er
    subj, sess, _ = name.split("_")
    t = doc_text(path)
    parts = re.split(r"\n\s*Paper (97\d\d)/(\d\d)\s*\n", t)
    out = {}
    for i in range(1, len(parts) - 2, 3):
        comp_var, body = parts[i + 1], parts[i + 2]
        pid = f"{subj}_{sess}_qp_{comp_var}"
        res = {}
        # per-question comments
        chunks = re.split(r"\n\s*Questions? (\d{1,2})(?:\s*(?:and|&|-|–|to)\s*(\d{1,2}))?\s*\n", "\n" + body)
        for j in range(1, len(chunks) - 2, 3):
            q1 = int(chunks[j]); q2 = int(chunks[j + 1]) if chunks[j + 1] else q1
            txt = chunks[j + 2]
            for q in range(q1, q2 + 1):
                res.setdefault(q, {"text": ""})["text"] += " " + re.sub(r"\s+", " ", txt)
        # MCQ general-comment lists: "found Questions 10, 20 and 22 particularly difficult, but ... 1, 3 and 5 straightforward"
        flat = re.sub(r"\s+", " ", body)
        LIST = r"Questions? ((?:\d{1,2}(?:,\s*|\s+and\s+|\s*&\s*)?){2,})"
        ms = list(re.finditer(LIST, flat))
        for k, m in enumerate(ms):
            qs = [int(x) for x in re.findall(r"\d{1,2}", m.group(1))]
            start = max(flat.rfind(".", 0, m.start()), ms[k - 1].end() if k else 0, flat.rfind(", but", 0, m.start()))
            nxt = ms[k + 1].start() if k + 1 < len(ms) else len(flat)
            end = min(x for x in (flat.find(".", m.end()), nxt, len(flat)) if x >= 0)
            ctx = (flat[start:m.start()] + " " + flat[m.end():end]).lower()
            hard = re.search(r"difficult|challenging|demanding|\bhard\b|fewer|least|poorly|weak", ctx)
            easy = re.search(r"straightforward|\beasy|easier|well answered|accessible|most candidates|high proportion|correctly by|well done", ctx)
            tag = "hard_list" if hard and not easy else "easy_list" if easy and not hard else None
            if tag:
                for q in qs:
                    res.setdefault(q, {"text": ""})[tag] = True
        for q, r in res.items():
            h = sum(w * min(len(rx.findall(r["text"])), 3) for rx, w in HARD_RX)
            e = sum(w * min(len(rx.findall(r["text"])), 3) for rx, w in EASY_RX)
            if r.get("hard_list"):
                h += 4
            if r.get("easy_list"):
                e += 4
            r["hard"], r["easy"] = round(h, 2), round(e, 2)
            r["text"] = r["text"].strip()[:1500]
        out[pid] = res
    return out

if __name__ == "__main__":
    SP = sys.argv[1]
    allr = {}
    for f in sorted(glob.glob(os.path.join(SP, "papers/97*_er.pdf"))):
        r = parse(f)
        allr.update(r)
        print(os.path.basename(f), len(r), "papers", sum(len(v) for v in r.values()), "question comments")
    json.dump(allr, open(os.path.join(SP, "work/er.json"), "w"))
