"""One test per Maths subtopic: the hardest questions of that subtopic (up to 50), least hard first.

Reads work/selection.json (question records with difficulty, duplicates already merged) and
work/subtopics.json (from subtopics.py). Writes work/selection_sub.json in the format make_tests.py reads.

usage: subtopic_tests.py SCRATCH [N]
"""
import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import topics, subtopics

SP = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 50
S = json.load(open(os.path.join(SP, "work/selection.json")))
Q = S["questions"]
ST = json.load(open(os.path.join(SP, "work/subtopics.json")))
TOPIC_NAME = {code: name for comp, spec in topics.MATH.items() for code, name, _ in spec}

tests = []
for comp in (2, 3, 4):
    for code, name, parent in subtopics.SUB[comp]:
        pool = [Q[i] for i, c in ST.items() if c == code]
        ranked = sorted(pool, key=lambda r: -r["difficulty"])[:N]
        ranked.sort(key=lambda r: r["difficulty"])
        tests.append({"subj": "9709", "kind": f"P{comp}", "topic": parent, "sub": code, "sub_name": name,
                      "topic_name": TOPIC_NAME[parent], "title": f"{code} {name}",
                      "slug": f"9709_P{comp}_{code.replace('.', '-')}_{re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')}",
                      "pool_primary": len(pool), "pool_secondary": 0,
                      "questions": [r["id"] for r in ranked], "secondary": []})
used = {i for t in tests for i in t["questions"]}
json.dump({"tests": tests, "questions": {i: Q[i] for i in used}}, open(os.path.join(SP, "work/selection_sub.json"), "w"))
for t in tests:
    print(f'{t["sub"]:6s} {t["sub_name"]:45s} pool {t["pool_primary"]:3d}  chosen {len(t["questions"])}')
print(len(tests), "tests;", sum(len(t["questions"]) for t in tests), "questions")
