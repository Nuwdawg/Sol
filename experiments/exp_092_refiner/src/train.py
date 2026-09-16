"""Train the boundary refiner on (proposal, annotation record) crop pairs.

Selection is by mean crop IoU against annotation records on the validation
proposals, which is the segmentation-quality term the competition metric
multiplies. The final promotion decision is still exact canonical PQ, measured
separately by `evaluate.py`; a crop metric only chooses the checkpoint.

The seed IoU of the unrefined proposal is reported alongside every measurement,
so an epoch that fails to beat the input it was given is visible immediately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from experiments.exp_092_refiner.src.model import RefinerNet, parameter_count


class CropDataset(Dataset):
    def __init__(self, directory: Path, augment: bool):
        self.features = np.load(directory / "features.npy", mmap_mode="r")
        self.targets = np.load(directory / "targets.npy", mmap_mode="r")
        self.meta = json.loads((directory / "meta.json").read_text())
        self.augment = augment
        assert len(self.features) == len(self.targets) == len(self.meta)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        features = np.asarray(self.features[index], dtype=np.float32) / 255.0
        target = np.asarray(self.targets[index], dtype=np.float32)
        if self.augment:
            k = int(np.random.randint(4))
            if k:
                features = np.rot90(features, k, axes=(1, 2))
                target = np.rot90(target, k)
            if np.random.rand() < .5:
                features = features[:, :, ::-1]
                target = target[:, ::-1]
        features = np.ascontiguousarray(features)
        target = np.ascontiguousarray(target)
        return torch.from_numpy((features - 0.5) / 0.25), torch.from_numpy(target)[None]


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    total = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * intersection + 1) / (total + 1)).mean()


@torch.no_grad()
def measure(model, loader, device, seeds: np.ndarray) -> dict:
    model.eval()
    refined, position = [], 0
    for features, target in loader:
        features, target = features.to(device), target.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            predicted = (torch.sigmoid(model(features)) > .5).float()
        intersection = (predicted * target).sum(dim=(1, 2, 3))
        union = ((predicted + target) > 0).float().sum(dim=(1, 2, 3))
        refined.extend((intersection / union.clamp(min=1)).float().cpu().numpy().tolist())
        position += len(features)
    refined = np.asarray(refined)
    crossed = int(((seeds <= .5) & (refined > .5)).sum())
    lost = int(((seeds > .5) & (refined <= .5)).sum())
    return dict(refined_iou=float(refined.mean()), seed_iou=float(seeds.mean()),
                gain=float(refined.mean() - seeds.mean()), crossed_half=crossed,
                lost_half=lost, examples=int(refined.size))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--base", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--time-hours", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    train_set = CropDataset(args.train_cache, augment=True)
    val_set = CropDataset(args.val_cache, augment=False)
    val_seeds = np.asarray([row["seed_iou"] for row in val_set.meta])
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = RefinerNet(base=args.base).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    criterion = nn.BCEWithLogitsLoss()
    print(f"train={len(train_set)} val={len(val_set)} parameters={parameter_count(model)} "
          f"seed_iou={val_seeds.mean():.5f}", flush=True)

    history, best, started = [], -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.
        for step, (features, target) in enumerate(train_loader, start=1):
            features, target = features.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(features)
                loss = criterion(logits, target) + dice_loss(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimiser); scaler.update()
            total += float(loss.detach())
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss={total/step:.5f}",
                      flush=True)
        schedule.step()
        stats = measure(model, val_loader, device, val_seeds)
        row = dict(epoch=epoch, train_loss=total / max(1, len(train_loader)),
                   elapsed_hours=(time.time() - started) / 3600,
                   lr=optimiser.param_groups[0]["lr"], **stats)
        history.append(row)
        print("EPOCH", json.dumps(row), flush=True)
        if stats["refined_iou"] > best:
            best = stats["refined_iou"]
            torch.save(dict(model=model.state_dict(), epoch=epoch, base=args.base,
                            metrics=stats), args.output / "best.pt")
        torch.save(dict(model=model.state_dict(), epoch=epoch, base=args.base,
                        metrics=stats), args.output / "last.pt")
        (args.output / "status.json").write_text(json.dumps(
            dict(status="running", history=history, best_refined_iou=best), indent=2))
        if (time.time() - started) / 3600 > args.time_hours:
            print("TIME_LIMIT", epoch, flush=True)
            break
    (args.output / "status.json").write_text(json.dumps(
        dict(status="complete", history=history, best_refined_iou=best,
             seed_iou=float(val_seeds.mean())), indent=2))
    print(json.dumps(dict(status="complete", best_refined_iou=best,
                          seed_iou=float(val_seeds.mean()))), flush=True)


if __name__ == "__main__":
    main()
