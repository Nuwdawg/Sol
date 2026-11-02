"""Merge segmentation + mark schemes, classify every question into a syllabus topic.

Output: work/questions.json (list of question records).
"""
import sys, os, json, glob, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import topics

SP = sys.argv[1]

def dp_monotone(E, lam):
    """Topic sequence maximising sum E[i][t] - lam * (#backward steps)."""
    n, T = len(E), len(E[0])
    best = [E[0][:]]
    back = [[-1] * T]
    for i in range(1, n):
        row, br = [], []
        # running max over t' <= t for forward moves
        pref, pref_arg = [], []
        m, a = -1e9, -1
        for t in range(T):
            if best[i - 1][t] > m:
                m, a = best[i - 1][t], t
            pref.append(m); pref_arg.append(a)
        gm = max(best[i - 1]); ga = best[i - 1].index(gm)
        for t in range(T):
            fwd, bwd = pref[t], gm - lam
            if fwd >= bwd:
                row.append(fwd + E[i][t]); br.append(pref_arg[t])
            else:
                row.append(bwd + E[i][t]); br.append(ga)
        best.append(row); back.append(br)
    t = best[-1].index(max(best[-1]))
    path = [t]
    for i in range(n - 1, 0, -1):
        t = back[i][t]
        path.append(t)
    return path[::-1]

def margin(s):
    o = sorted(s, reverse=True)
    return (o[0] - o[1]) / (o[0] + 1.0) if len(o) > 1 else 1.0

records = []
for f in sorted(glob.glob(os.path.join(SP, "work/seg/*.json"))):
    d = json.load(open(f))
    paper = d["paper"]
    subj, sess, _, cv = paper.split("_")
    comp, var = int(cv[0]), cv[1]
    msf = os.path.join(SP, "work/ms", paper.replace("_qp_", "_ms_") + ".json")
    ms = json.load(open(msf)) if os.path.exists(msf) else None
    msq = {}
    if ms and "questions" in ms:
        msq = {m["q"]: m for m in ms["questions"]}
    key = ms.get("key", {}) if ms else {}
    qs = d["questions"]
    N = len(qs)
    recs = []
    for q in qs:
        m = msq.get(q["q"])
        r = {
            "id": f'{subj}_{sess}_{cv}_q{q["q"]:02d}', "paper": paper, "subj": subj, "sess": sess,
            "year": 2000 + int(sess[1:]), "series": {"m": "Feb/Mar", "s": "May/Jun", "w": "Oct/Nov"}[sess[0]],
            "comp": comp, "variant": var, "q": q["q"], "N": N,
            "marks": 1 if (subj == "9702" and comp == 1) else q["marks"],
            "marks_parts": q["marks_parts"], "text": q["text"], "ms_text": m["text"] if m else "",
            "segs": q["segs"], "num_bbox": q["num_bbox"], "num_base": q["num_base"], "num_size": q["num_size"],
            "ms_paper": paper.replace("_qp_", "_ms_") if m else None, "ms_segs": m["segs"] if m else [],
            "key": key.get(str(q["q"])) if key else None,
        }
        recs.append(r)
    # ---- classify
    if subj == "9709":
        spec = topics.MATH[comp]
        for r in recs:
            s = topics.scores(spec, r["text"], r["ms_text"])
            i = max(range(len(s)), key=lambda k: s[k])
            r["topic"], r["topic_name"] = spec[i][0], spec[i][1]
            r["topic_scores"] = [round(x, 2) for x in s]; r["topic_conf"] = round(margin(s), 3)
            top = max(s)
            r["topics_2nd"] = [spec[k][0] for k in range(len(s)) if k != i and s[k] >= 0.75 * top and s[k] >= 6]
    elif comp == 5:
        for r in recs:
            code, name = topics.PHYS_P5[0] if r["q"] == 1 else topics.PHYS_P5[1]
            r["topic"], r["topic_name"], r["topic_scores"], r["topic_conf"] = code, name, [], 1.0
            r["topics_2nd"] = []
    else:
        spec = topics.PHYS[:11] if comp in (1, 2) else topics.PHYS[11:]
        S = [topics.scores(spec, r["text"], r["ms_text"], 0.35) for r in recs]
        E = [[x / (max(s) + 1e-6) for x in s] for s in S]
        path = dp_monotone(E, 50.0 if comp == 1 else 0.25)
        for r, s, t in zip(recs, S, path):
            r["topic"], r["topic_name"] = spec[t][0], spec[t][1]
            r["topic_scores"] = [round(x, 2) for x in s]
            r["topic_conf"] = round(margin(s), 3)
            r["topic_argmax"] = spec[max(range(len(s)), key=lambda k: s[k])][0]
            top = max(s)
            r["topics_2nd"] = [spec[k][0] for k in range(len(s)) if spec[k][0] != r["topic"] and s[k] >= 0.75 * top and s[k] >= 6]
    records.extend(recs)

json.dump(records, open(os.path.join(SP, "work/questions.json"), "w"))
import collections
c = collections.Counter((r["subj"], r["comp"] if r["subj"] == "9709" else ("MCQ" if r["comp"] == 1 else r["comp"]), r["topic"], r["topic_name"]) for r in records)
for k in sorted(c, key=str):
    print(k, c[k])
print(len(records), "questions")
