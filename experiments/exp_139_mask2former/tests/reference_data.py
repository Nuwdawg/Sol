"""Frozen copy of exp139's first data loader (RLE decode per sample, photometrics on CPU).

Kept only so the faster loader can be tested against it sample by sample. Do not edit.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mu
from torch.utils.data import Dataset

from experiments.exp_139_mask2former.src.data import NATIVE, read_gray, to_pixels


class ReferenceSamples(Dataset):
    """Native-resolution crops and whole frames with exact dihedral augmentation."""

    def __init__(self, records: list[dict], images: Path, min_piece: int = 40,
                 seed: int = 2026, augment: bool = True, rotate90: bool = True,
                 cache: int = 64):
        self.records, self.images = records, Path(images)
        self.min_piece, self.seed, self.augment, self.rotate90 = min_piece, seed, augment, rotate90
        self.cache_size = cache
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _gray(self, name: str) -> np.ndarray:
        image = self._cache.get(name)
        if image is None:
            image = read_gray(self.images / name)
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[name] = image
        return image

    @staticmethod
    def _window(box, crop: int, rng, jitter: float) -> tuple[int, int]:
        x, y, w, h = box
        cx = x + w / 2 + rng.uniform(-jitter, jitter) * crop
        cy = y + h / 2 + rng.uniform(-jitter, jitter) * crop
        x0 = int(np.clip(round(cx - crop / 2), 0, NATIVE - crop))
        y0 = int(np.clip(round(cy - crop / 2), 0, NATIVE - crop))
        return x0, y0

    def _masks(self, record: dict, x0: int, y0: int, crop: int) -> list[np.ndarray]:
        kept = []
        for rle, (bx, by, bw, bh) in zip(record["rles"], record["boxes"]):
            if bx + bw < x0 or by + bh < y0 or bx > x0 + crop or by > y0 + crop:
                continue
            piece = mu.decode(rle)[y0:y0 + crop, x0:x0 + crop]
            if int(piece.sum()) >= (self.min_piece if crop < NATIVE else 1):
                kept.append(piece)
        return kept

    def __getitem__(self, key):
        record_index, extent, epoch, slot, last, parts = key
        record = self.records[record_index]
        rng = np.random.default_rng([self.seed, epoch, slot, record_index])
        gray = self._gray(record["file_name"])
        if extent >= NATIVE:
            x0 = y0 = 0
            masks = self._masks(record, 0, 0, NATIVE)
        else:
            masks = []
            # The anchor filament's centre is always inside its own crop, so a retry
            # happens only when that filament is under `min_piece` pixels.
            for jitter in (0.35, 0.2, 0.0):
                anchor = record["boxes"][int(rng.integers(len(record["boxes"])))]
                x0, y0 = self._window(anchor, extent, rng, jitter)
                masks = self._masks(record, x0, y0, extent)
                if masks:
                    break
        if not masks:
            raise RuntimeError(f"no filament in sample of {record['image_id']} at {extent}")
        patch = gray[y0:y0 + extent, x0:x0 + extent]
        stack = np.stack(masks)
        if self.augment:
            if rng.random() < 0.5:
                patch, stack = patch[:, ::-1], stack[:, :, ::-1]
            if rng.random() < 0.5:
                patch, stack = patch[::-1, :], stack[:, ::-1, :]
            if self.rotate90:
                turns = int(rng.integers(4))
                if turns:
                    patch, stack = np.rot90(patch, turns), np.rot90(stack, turns, axes=(1, 2))
            gamma = math.exp(rng.uniform(math.log(0.8), math.log(1.25)))
            gain = rng.uniform(0.85, 1.15)
            patch = np.clip(255.0 * (patch.astype(np.float32) / 255.0) ** gamma * gain, 0, 255)
        # Masks cross the worker boundary as uint8, a quarter of float32; the trainer
        # converts them on the GPU.
        return (to_pixels(patch), torch.from_numpy(np.ascontiguousarray(stack)),
                torch.zeros(len(stack), dtype=torch.long), (bool(last), int(parts), int(extent)))
