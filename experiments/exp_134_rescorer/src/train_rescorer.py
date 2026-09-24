"""Train the rescorer: does a real filament lie under this sub-threshold proposal?

Reuses `exp_025_verifier.src.model.Verifier` unchanged - a deliberately small
2-channel CNN on a 128 px crop. exp025's own note records that two larger networks
in this project overfit while validation stood still, and the dataset here is
~5,300 examples at a 19.3% positive rate, so capacity is not the lever.

**exp025's trainer is deliberately not reused.** It imports `torch_xla` at module
scope and will not import anywhere except a TPU kernel, and its `CropSet` expects
separate memory-mapped `.npy` files rather than the single `.npz` written here.
Only `model.py` is shared.

Two decisions carried over because they were paid for:

- **Selection by average precision, not loss.** The emission rule that consumes
  this model sweeps a threshold, so the ordering is what matters. exp012 selected
  a refiner on loss and picked a checkpoint worse than its own epoch two.
- **Dihedral augmentation.** The solar disk has no canonical orientation, so all
  eight views are exact - no interpolation and no invented pixels.

What this model must beat, from the ceiling analysis:

| rescorer captures | at precision | validation PQ | public |
|---|---|---|---|
| 86 of 172 | 50% | 0.4661 | 0.39-0.40 |
| 120 of 172 | 70% | 0.4816 | 0.40-0.41 |
| 172 of 172 | 100% | 0.5048 | 0.43-0.44 |

Break-even for admitting a sub-threshold proposal is about 35% precision, and
admitting blind gives 19-25%. So average precision is the headline number here,
but the decision is made later by `evaluate_recall` on official PQ.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from experiments.exp_025_verifier.src.model import Verifier, positive_weighted_loss


class CropSet(Dataset):
    """Crops from `harvest_crops.py`'s npz, with exact dihedral augmentation."""

    def __init__(self, path: Path, augment: bool, indices: np.ndarray | None = None):
        payload = np.load(path)
        self.image = payload["image"]
        self.mask = payload["mask"]
        self.label = payload["label"].astype(np.float32)
        self.indices = np.arange(len(self.label)) if indices is None else indices
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        index = int(self.indices[position])
        image = self.image[index].astype(np.float32) / 255.0
        mask = self.mask[index].astype(np.float32)
        if mask.max() > 1.0:
            mask = mask / 255.0
        if self.augment:
            if np.random.rand() < 0.5:
                image, mask = image[:, ::-1], mask[:, ::-1]
            turns = np.random.randint(4)
            if turns:
                image, mask = np.rot90(image, turns), np.rot90(mask, turns)
            image = np.clip(image * np.random.uniform(0.85, 1.15), 0.0, 1.0)
        stacked = np.stack([np.ascontiguousarray(image), np.ascontiguousarray(mask)])
        return torch.from_numpy(stacked), torch.tensor(self.label[index])


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """AP by the ranking definition, so it matches how the threshold is swept."""
    order = np.argsort(-scores)
    ranked = labels[order]
    positives = ranked.sum()
    if positives == 0:
        return float("nan")
    cumulative = np.cumsum(ranked)
    precision = cumulative / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / positives)


def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    if k <= 0 or k > len(scores):
        return float("nan")
    order = np.argsort(-scores)[:k]
    return float(labels[order].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", type=Path, required=True,
                        help="train_crops.npz from harvest_crops.py")
    parser.add_argument("--index", type=Path, required=True,
                        help="train_index.json, used to hold out whole photographs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--holdout", type=float, default=0.2,
                        help="fraction of TRAINING photographs held out to select the epoch")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    index = json.loads(args.index.read_text())
    labels = np.load(args.crops)["label"].astype(np.float32)
    if len(index) != len(labels):
        raise SystemExit(f"index has {len(index)} rows but crops have {len(labels)}")

    # Hold out whole PHOTOGRAPHS, never rows: two records of one photograph share
    # the same crops, so a row-wise split would leak the image across the boundary.
    photographs = sorted({row["photograph"] for row in index})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(photographs)
    cut = max(1, int(len(photographs) * args.holdout))
    held = set(photographs[:cut])
    is_held = np.array([row["photograph"] in held for row in index])
    fit_rows = np.nonzero(~is_held)[0]
    held_rows = np.nonzero(is_held)[0]
    print(f"photographs {len(photographs)}, held out {len(held)}")
    print(f"fit rows {len(fit_rows)} ({labels[fit_rows].mean():.1%} positive), "
          f"held rows {len(held_rows)} ({labels[held_rows].mean():.1%} positive)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = Verifier(width=args.width).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    base_rate = float(labels[fit_rows].mean())
    pos_weight = (1 - base_rate) / max(base_rate, 1e-6)
    print(f"positive weight {pos_weight:.2f}")

    fit_loader = DataLoader(CropSet(args.crops, True, fit_rows), batch_size=args.batch,
                            shuffle=True, num_workers=0, drop_last=False)
    held_loader = DataLoader(CropSet(args.crops, False, held_rows), batch_size=256,
                             shuffle=False, num_workers=0)

    history, best = [], dict(ap=-1.0, epoch=-1)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch, target in fit_loader:
            batch, target = batch.to(device), target.to(device)
            optimiser.zero_grad()
            loss = positive_weighted_loss(model(batch), target, pos_weight)
            loss.backward()
            optimiser.step()
            total += float(loss) * len(target)
        model.eval()
        scores, truths = [], []
        with torch.no_grad():
            for batch, target in held_loader:
                scores.append(torch.sigmoid(model(batch.to(device))).cpu().numpy())
                truths.append(target.numpy())
        scores = np.concatenate(scores)
        truths = np.concatenate(truths)
        ap = average_precision(scores, truths)
        row = dict(epoch=epoch, loss=round(total / max(len(fit_rows), 1), 5),
                   held_ap=round(ap, 4),
                   p_at_50=round(precision_at_k(scores, truths, 50), 4),
                   p_at_100=round(precision_at_k(scores, truths, 100), 4),
                   p_at_200=round(precision_at_k(scores, truths, 200), 4))
        history.append(row)
        print(f"epoch {epoch:3d}  loss {row['loss']:.4f}  held AP {ap:.4f}  "
              f"P@50 {row['p_at_50']:.3f}  P@100 {row['p_at_100']:.3f}", flush=True)
        # Average precision, not loss: the consumer sweeps a threshold (exp012).
        if ap > best["ap"]:
            best = dict(ap=float(ap), epoch=epoch, **{k: row[k] for k in
                        ("p_at_50", "p_at_100", "p_at_200")})
            torch.save(dict(state=model.state_dict(), width=args.width, epoch=epoch,
                            held_ap=float(ap)), args.output / "rescorer_best.pt")

    report = dict(experiment_id="exp134", base_rate=round(base_rate, 4),
                  pos_weight=round(pos_weight, 3), held_photographs=len(held),
                  fit_rows=int(len(fit_rows)), held_rows=int(len(held_rows)),
                  best=best, history=history)
    (args.output / "train_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nbest epoch {best['epoch']} at held AP {best['ap']:.4f}")
    print(f"blind admission would give {base_rate:.1%}; break-even is about 35%.")
    print("The decision is made by evaluate_recall on official PQ, not by AP.")


if __name__ == "__main__":
    main()
