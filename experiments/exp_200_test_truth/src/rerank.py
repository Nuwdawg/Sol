"""exp200k: rank exp169's candidates by "would any annotator draw this", judged by cross-validation.

Under pooled one-to-one truth a candidate counts as a true positive if any annotator drew it. exp139
was trained per annotator, so its scores rank a filament one annotator of three drew below one all
three drew. exp169's score separates the candidates that match pooled truth with AUC 0.862, and a
perfect ranking of the same candidates would give pooled PQ 0.4925 against 0.4138.

A classifier is trained on validation candidates (exp169's score at least 0.30) with the label
"its vote mask matches some annotator's filament at IoU > 0.5". Features come from the caches only:
the rule's score and its parts, the per-decoder-layer class probabilities, mask softness (areas
at logits -1 and +1), YOLO support, shape, and a 16-dimensional PCA of the decoder embedding.
Judged on exp168's two folds (seed 2026): train on one half, choose the threshold on that half, score
the other half's pooled PQ, and compare with exp169's rule on the same halves.

    python -m experiments.exp_200_test_truth.src.rerank --annotations ... --ingredients val_ingredients.pkl
        --cache exp147_val_cache.json --output rerank.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from pycocotools import mask as mu
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_149_evidence_fusion.src import evidence as E
from experiments.exp_168_joint_tuning.src.joint import scores
from experiments.exp_200_test_truth.src.hypotheses import as_bytes
from experiments.exp_200_test_truth.src.tune import greedy, painted_prefix, pq

FLOOR = 0.30                                   # exp169-score floor of the candidates considered
GRID = tuple(round(t, 3) for t in np.arange(0.05, 0.951, 0.025))


def shape(crop: "E.Crop") -> tuple[float, float, float]:
    ys, xs = np.nonzero(crop.mask)
    if len(ys) < 3:
        return 0.0, 0.0, 0.0
    h, w = crop.mask.shape
    cov = np.cov(np.stack([ys, xs]).astype(float))
    ev = np.sort(np.linalg.eigvalsh(cov))
    elong = float(np.sqrt(ev[1] / max(ev[0], 1e-6)))
    return float(len(ys) / (h * w)), float(max(h, w) / max(min(h, w), 1)), elong


def candidate_table(prep: dict, cache: dict, names: list) -> tuple[list, np.ndarray, list]:
    """(photo, index) keys, a feature matrix and the crops, for every candidate above FLOOR."""
    keys, rows, crops = [], [], []
    for n in names:
        p, full = prep[n], cache[n]["full"]
        s169 = scores(p, 4.0, 0.3)
        mean = p["total"] / p["views"]
        for i, c in enumerate(full):
            if s169[i] < FLOOR:
                continue
            area = max(c["area"], 1)
            low = mu.area(E.as_rle(c["rle_low"])) / area
            high = mu.area(E.as_rle(c["rle_high"])) / area
            layers = np.asarray(c["layers"], float)
            fill, aspect, elong = shape(p["crops"][i])
            rows.append([s169[i], c["score"], c["p"], c["m"], float(np.asarray(mean).ravel()[i]),
                         *[float(y[i]) for y in p["yolo"]], np.log(area), low, high, layers.min(), layers.mean(),
                         layers.std(), layers[-1] - layers[0], fill, np.log(aspect), np.log(elong + 1e-6)])
            keys.append((n, i))
            crops.append(p["crops"][i])
    return keys, np.asarray(rows, float), crops


def labels(keys, crops, passes) -> np.ndarray:
    out = np.zeros(len(keys), int)
    by_photo = {}
    for k, (n, _) in enumerate(keys):
        by_photo.setdefault(n, []).append(k)
    for n, ks in by_photo.items():
        flat = [t for truth in passes[n] for t in truth]
        if not flat:
            continue
        rles = []
        for k in ks:
            c = crops[k]
            full = np.zeros((E.SIZE, E.SIZE), np.uint8)
            h, w = c.mask.shape
            full[c.y0:c.y0 + h, c.x0:c.x0 + w] = c.mask
            rles.append(mu.encode(np.asfortranarray(full)))
        iou = np.asarray(mu.iou(rles, flat, [0] * len(flat))).reshape(len(rles), len(flat))
        out[ks] = (iou.max(1) > 0.5).astype(int)
    return out


def pooled_rows(by_photo: dict, crops, passes, area: int = 100, grid=GRID) -> dict:
    """{threshold: {photo: [iou, tp, fp, fn]}} for candidates painted in descending new score."""
    out = {t: {} for t in grid}
    for n, items in by_photo.items():
        ks, kept = painted_prefix([(score, crops[k]) for k, score in items], area)
        flat = [t for truth in passes[n] for t in truth]
        iou = (np.asarray(mu.iou(kept, flat, [0] * len(flat))).reshape(len(kept), len(flat))
               if kept and flat else np.zeros((len(kept), len(flat))))
        for t in grid:
            k = sum(1 for s in ks if s >= t)
            tp, total = greedy(iou[:k])
            out[t][n] = np.array([total, tp, k - tp, len(flat) - tp])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--ingredients", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, _ = records_for(args.annotations)
    names = sorted(records)
    passes = {n: [[as_bytes(t) for t in truth] for _, truth in sorted(records[n])] for n in names}
    prep = pickle.loads(args.ingredients.read_bytes())
    cache = E.load_cache(args.cache)
    keys, X, crops = candidate_table(prep, cache, names)
    embed = np.asarray([cache[n]["full"][i]["embedding"] for n, i in keys], float)
    X = np.hstack([X, PCA(16, random_state=0).fit_transform(embed)])     # unsupervised, no labels
    y = labels(keys, crops, passes)
    print(f"candidates {len(y)}, matching some annotator {y.mean():.3f}", flush=True)
    rng = np.random.default_rng(2026)
    order = rng.permutation(len(names))
    folds = [set(names[i] for i in order[:len(names) // 2]), set(names[i] for i in order[len(names) // 2:])]
    photo = np.array([n for n, _ in keys])
    models = {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)),
        "boosting": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                           l2_regularization=1.0, random_state=0),
    }
    def by_photo(photos, score):
        return {n: [(k, score[k]) for k in np.nonzero(photo == n)[0]] for n in photos}

    def out_of_fold(make, photos):
        """Probabilities for `photos` from models that never saw them (inner two-way split)."""
        inner = sorted(photos)
        perm = np.random.default_rng(7).permutation(len(inner))
        halves = [set(inner[i] for i in perm[:len(inner) // 2]), set(inner[i] for i in perm[len(inner) // 2:])]
        prob = np.zeros(len(y))
        for fit_on, apply_to in ((halves[0], halves[1]), (halves[1], halves[0])):
            mask = np.isin(photo, list(fit_on))
            prob[np.isin(photo, list(apply_to))] = make().fit(X[mask], y[mask]).predict_proba(
                X[np.isin(photo, list(apply_to))])[:, 1]
        return prob

    report = {}
    for name, make in models.items():
        held_sums, base_sums, aucs = [], [], []
        for train, held in ((folds[0], folds[1]), (folds[1], folds[0])):
            tr = np.isin(photo, list(train)); he = np.isin(photo, list(held))
            inner = pooled_rows(by_photo(train, out_of_fold(make, train)), crops, passes)
            best = max(GRID, key=lambda t: pq(sum(inner[t].values())))          # threshold never sees `held`
            prob = make().fit(X[tr], y[tr]).predict_proba(X)[:, 1]
            pos, neg = prob[he & (y == 1)], prob[he & (y == 0)]
            r = np.empty(len(pos) + len(neg)); r[np.argsort(np.concatenate([pos, neg]))] = np.arange(1, len(r) + 1)
            aucs.append(float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))))
            held_sums.append(sum(pooled_rows(by_photo(held, prob), crops, passes)[best].values()))
            base_sums.append(sum(pooled_rows(by_photo(held, X[:, 0]), crops, passes, grid=(0.68,))[0.68].values()))
            print(f"{name}: held-out AUC {aucs[-1]:.3f}, threshold {best}, held pooled "
                  f"{pq(held_sums[-1]):.4f} vs exp169 {pq(base_sums[-1]):.4f}", flush=True)
        report[name] = dict(cv_pooled=pq(sum(held_sums)), exp169=pq(sum(base_sums)), auc=aucs)
        print(f"{name}: CV pooled {report[name]['cv_pooled']:.4f} against exp169 {report[name]['exp169']:.4f}",
              flush=True)
    s169 = X[:, 0]
    pos, neg = s169[y == 1], s169[y == 0]
    r = np.empty(len(y)); r[np.argsort(np.concatenate([pos, neg]))] = np.arange(1, len(y) + 1)
    report["exp169_auc"] = float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    print("exp169 score AUC", round(report["exp169_auc"], 3), flush=True)
    args.output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
