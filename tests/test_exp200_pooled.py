"""Unit tests for exp200's building blocks: pooled training targets, pooled matching, uncovered additions.

These confirm the functions do what their docstrings say on small hand-made masks. Whether the ideas
help is measured on validation (experiments/exp_200_test_truth/README.md), not here.
"""

from __future__ import annotations

import numpy as np
from pycocotools import mask as mu

from experiments.exp_139_mask2former.src.data import pooled_targets
from experiments.exp_200_test_truth.src.additions import add_uncovered
from experiments.exp_200_test_truth.src.matching import one_to_one

SIZE = 2048


def box(y0, x0, y1, x1):
    m = np.zeros((SIZE, SIZE), np.uint8)
    m[y0:y1, x0:x1] = 1
    return mu.encode(np.asfortranarray(m))


def record(image_id, name, rles):
    return dict(image_id=image_id, file_name=name, rles=rles, boxes=[mu.toBbox(r).tolist() for r in rles])


def test_single_annotator_photographs_pass_through():
    r = record("a", "p.jpeg", [box(0, 0, 10, 10)])
    assert pooled_targets([r]) == [r]


def test_overlapping_versions_collapse_and_minority_filaments_stay():
    # Annotators 1 and 2 drew the same filament with different extents; only annotator 2 drew a second one.
    common_1, common_2, minority = box(100, 100, 120, 200), box(102, 100, 122, 210), box(500, 500, 520, 600)
    third = box(101, 100, 121, 205)                        # annotator 3, closest to the other two
    pooled = pooled_targets([record("1", "p.jpeg", [common_1]), record("2", "p.jpeg", [common_2, minority]),
                             record("3", "p.jpeg", [third])])
    assert len(pooled) == 1 and pooled[0]["file_name"] == "p.jpeg"
    kept = pooled[0]["rles"]
    assert len(kept) == 2                                  # one version of the shared filament, plus the minority one
    assert any(r["counts"] == third["counts"] for r in kept)   # the version that agrees best with the others
    assert any(r["counts"] == minority["counts"] for r in kept)


def test_one_to_one_counts_each_version_once():
    a, b = box(0, 0, 20, 100), box(1, 0, 21, 100)          # two annotators' versions of one filament
    result = one_to_one([box(0, 0, 20, 100)], [a, b])
    assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 1)


def test_add_uncovered_skips_covered_and_paints_unclaimed():
    final = [box(0, 0, 50, 50)]
    entries = [dict(score=0.95, rle=box(10, 10, 40, 40)),   # inside a kept instance: skipped
               dict(score=0.92, rle=box(45, 0, 200, 50)),   # 5 of 155 rows overlap (3%): added, overlap removed
               dict(score=0.50, rle=box(300, 300, 400, 400))]   # below the threshold
    added = add_uncovered(final, entries, 0.90)
    assert len(added) == 1
    assert mu.area(mu.merge([added[0], final[0]], intersect=True)) == 0
    assert mu.area(added[0]) == (200 - 50) * 50
