"""Compare our subtopic for each question with Past Paper Penguin's placement.
usage: ppp_compare.py SCRATCH  -> prints agreement, writes work/ppp_truth.json (our id -> PPP row name)"""
import json, re, sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppp_sets
SP = sys.argv[1]
P = json.load(open(os.path.join(SP, "work/ppp_assign.json")))
ALLQ = json.load(open(os.path.join(SP, "work/selection.json")))["questions"]
ST = json.load(open(os.path.join(SP, "work/subtopics.json")))
canon = {}
for r in ALLQ.values():
    canon.setdefault(r["id"], r["id"])
    for d in r.get("dup_of_this", []):
        canon[d] = r["id"]
def key(s):
    return re.sub(r"[^a-z]", "", s.lower())
rowname = {}
for name, codes in ppp_sets.PURE + ppp_sets.MECH:
    rowname[key(name)] = name
    for c in codes:
        rowname[c] = name
truth, missing = {}, []
for setname, items in P.items():
    base = re.sub(r"-(A|B|C)?-?(P3|M1)$", "", setname)
    base = re.sub(r"-(A|B|C)$", "", base)
    name = rowname.get(key(base))
    assert name, setname
    for ref, qn in items:
        if ref == "img":
            continue
        qid = f"{ref}_q{qn:02d}"
        cid = canon.get(qid)
        if cid is None:
            missing.append(qid); continue
        truth.setdefault(cid, set()).add(name)
agree = sum(1 for k, v in truth.items() if k in ST and rowname[ST[k]] in v)
inpool = sum(1 for k in truth if k in ST)
print(f"PPP questions matched to ours: {len(truth)} (not found: {len(missing)}); in our subtopic pool: {inpool}; same subtopic: {agree}")
conf = collections.Counter()
for k, v in truth.items():
    if k in ST and rowname[ST[k]] not in v:
        conf[(rowname[ST[k]], " / ".join(sorted(v)))] += 1
for (a, b), n in conf.most_common(60):
    print(f"{n:3d}  ours: {a:42s} PPP: {b}")
print("multi-placed by PPP:", sum(1 for v in truth.values() if len(v) > 1))
print("not found examples:", missing[:10])
json.dump({k: sorted(v) for k, v in truth.items()}, open(os.path.join(SP, "work/ppp_truth.json"), "w"), indent=0)
