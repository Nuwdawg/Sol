"""Temporal proximity of test and validation photographs to training photographs.

Reproduces shared.data_split.make_split from file names alone (it uses only the sorted
file names and the site code), then measures, for each validation and each test photograph,
the gap to the nearest photograph in (a) the 601-photo training fold and (b) all 707 labelled
photographs. Input: the competition file listing (names only, no labels, no images).
"""
import datetime as dt
import json
import sys
from collections import defaultdict

import numpy as np

listing = json.load(open(sys.argv[1]))
train_names, test_names = [], []
for f in listing:
    n = f["name"]
    base = n.rsplit("/", 1)[-1]
    if "/train/train_images/" in n:
        train_names.append(base)
    elif "/test/test_images/" in n:
        test_names.append(base)
train_names.sort(); test_names.sort()
print("train photos", len(train_names), "test photos", len(test_names))

def site(n):
    return n.rsplit(".", 1)[0][-2:]

by_site = defaultdict(list)
for s in sorted(train_names):
    by_site[site(s)].append(s)
rng = np.random.default_rng(2026)
tr, va = [], []
for st in sorted(by_site):
    stems = by_site[st]
    rng.shuffle(stems)
    n_val = max(1, round(len(stems) * 0.15))
    va.extend(stems[:n_val]); tr.extend(stems[n_val:])
print("canonical split: train", len(tr), "val", len(va))

def t(n):
    return dt.datetime.strptime(n[:14], "%Y%m%d%H%M%S")

def gaps(queries, pool):
    pt = np.array(sorted(t(p).timestamp() for p in pool))
    out = []
    for q in queries:
        x = t(q).timestamp()
        i = np.searchsorted(pt, x)
        cands = [abs(pt[j] - x) for j in (i - 1, i) if 0 <= j < len(pt)]
        out.append(min(cands) / 86400.0)
    return np.array(out)

def table(name, g):
    row = [name, f"n={len(g)}"]
    for d in (1 / 24, 0.5, 1, 2, 3, 7):
        row.append(f"<={d:g}d {100 * np.mean(g <= d):5.1f}%")
    row.append(f"median {np.median(g):.2f}d")
    print("  ".join(row))

table("val  -> train601", gaps(va, tr))
table("test -> train601", gaps(test_names, tr))
table("test -> all707  ", gaps(test_names, train_names))
table("test -> val106  ", gaps(test_names, va))
table("val  -> val(other)", np.array([gaps([v], [w for w in va if w != v])[0] for v in va]))
# same-site neighbours
def gaps_site(queries, pool):
    out = []
    for q in queries:
        same = [p for p in pool if site(p) == site(q)]
        out.append(gaps([q], same)[0] if same else np.inf)
    return np.array(out)
table("val  -> train601 same site", gaps_site(va, tr))
table("test -> train601 same site", gaps_site(test_names, tr))
from collections import Counter
print("sites train", Counter(site(n) for n in train_names))
print("sites test ", Counter(site(n) for n in test_names))
print("years train", sorted(Counter(n[:4] for n in train_names).items()))
print("years test ", sorted(Counter(n[:4] for n in test_names).items()))
json.dump({"train": sorted(tr), "val": sorted(va), "test": test_names}, open(sys.argv[2], "w"), indent=0)
