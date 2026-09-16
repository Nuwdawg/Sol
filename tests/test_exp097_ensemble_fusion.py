"""The ensemble's fusion must be the proven D4 fusion, changed in exactly one way.

`cached_tta.fuse` produced the 0.39 submission. `evaluate_ensemble.fuse_sources`
rewrites it so the support denominator is passed in, because pooling two models
means sixteen views rather than eight, and the original divides by a hardcoded
eight. A rewrite of scoring code that sits under every ensemble number is worth
pinning: these tests fail if the clustering, the pixel vote or the score changes
for any reason other than that denominator.

They run on the stored exp002 D4 pool, which is real RLE data from the pipeline
in question, not a synthetic case built to satisfy the assumption under test
(CLAUDE.md section 6).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.exp_033_inference.src.cached_tta import fuse as reference_fuse
from experiments.exp_097_runpod_yolo.src.evaluate_ensemble import VIEWS, fuse_sources

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data" / "exp033_cached_tta" / "candidates_0.5_0.5.json"
LINK, VOTE = 0.5, 0.5


def entries_from(pool: dict, photographs: int) -> list[list[tuple[float, dict]]]:
    return [[(entry["score"], entry["rle"]) for entry in pool[name]]
            for name in sorted(pool)[:photographs]]


def counts_of(instance: dict) -> str:
    counts = instance["rle"]["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else counts


def test_eight_sources_reproduces_the_shipped_fusion():
    if not POOL.is_file():
        pytest.skip("cached D4 pool is not present")
    pool = json.loads(POOL.read_text())
    compared = 0
    for entries in entries_from(pool, 8):
        shipped = reference_fuse(entries, LINK, VOTE)
        rewritten = fuse_sources(entries, LINK, VOTE, VIEWS)
        assert len(shipped) == len(rewritten)
        for old, new in zip(shipped, rewritten):
            assert new["score"] == pytest.approx(old["score"], abs=1e-12)
            assert counts_of(new) == counts_of(old)
            assert new["rle"]["size"] == old["rle"]["size"]
        compared += len(shipped)
    assert compared > 0


def test_more_sources_only_lowers_scores_and_never_moves_a_mask():
    """Pooling a second model must not re-cluster anything; it only dilutes support."""
    if not POOL.is_file():
        pytest.skip("cached D4 pool is not present")
    pool = json.loads(POOL.read_text())
    for entries in entries_from(pool, 4):
        eight = fuse_sources(entries, LINK, VOTE, VIEWS)
        sixteen = fuse_sources(entries, LINK, VOTE, 2 * VIEWS)
        assert len(eight) == len(sixteen)
        for narrow, wide in zip(eight, sixteen):
            assert counts_of(wide) == counts_of(narrow)     # identical masks
            assert wide["score"] < narrow["score"]          # strictly less support
