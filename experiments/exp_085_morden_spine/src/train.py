"""Train the compact MORDEN-style mask/spine network on one T4."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from experiments.exp_085_morden_spine.src.model import MordenSpineNet, parameter_count


class CachedRecords(Dataset):
    def __init__(self, root: str, fold: str, augment: bool):
        self.root = Path(root)
        self.fold = fold
        self.augment = augment
        self.length = len(json.loads((self.root / f"{fold}_ids.json").read_text()))
        self._images = self._masks = self._spines = self._mapping = None

    def _open(self) -> None:
        if self._images is None:
            self._images = np.load(self.root / f"{self.fold}_images.npy", mmap_mode="r")
            self._masks = np.load(self.root / f"{self.fold}_masks.npy", mmap_mode="r")
            self._spines = np.load(self.root / f"{self.fold}_spines.npy", mmap_mode="r")
            self._mapping = np.load(self.root / f"{self.fold}_record_to_photo.npy", mmap_mode="r")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        self._open()
        image = np.array(self._images[int(self._mapping[index])], copy=True)
        mask = np.array(self._masks[index], copy=True)
        spine = np.array(self._spines[index], copy=True)
        if self.augment:
            if np.random.random() < 0.5:
                image, mask, spine = image[:, :, ::-1], mask[:, ::-1], spine[:, ::-1]
            if np.random.random() < 0.5:
                image, mask, spine = image[:, ::-1, :], mask[::-1, :], spine[::-1, :]
            turns = np.random.randint(4)
            if turns:
                image = np.rot90(image, turns, axes=(1, 2))
                mask = np.rot90(mask, turns)
                spine = np.rot90(spine, turns)
        image = np.ascontiguousarray(image).astype(np.float32) / 255.0
        image = (image - 0.5) / 0.25
        return (
            torch.from_numpy(image),
            torch.from_numpy(np.ascontiguousarray(mask)[None].astype(np.float32)),
            torch.from_numpy(np.ascontiguousarray(spine)[None].astype(np.float32)),
        )


def focal_dice(logits: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = torch.sigmoid(logits)
    pt = probability * target + (1.0 - probability) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    focal = (alpha_t * (1.0 - pt).square() * bce).mean()
    intersection = (probability * target).sum((1, 2, 3))
    cardinality = probability.sum((1, 2, 3)) + target.sum((1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (cardinality + 1.0)).mean()
    return 0.6 * focal + 0.4 * dice


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.inference_mode()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    loss_sum = mask_inter = mask_total = spine_inter = spine_total = 0.0
    batches = 0
    for images, masks, spines in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        spines = spines.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            mask_logits, spine_logits = model(images)
            loss = focal_dice(mask_logits, masks, 0.66) + 0.25 * focal_dice(spine_logits, spines, 0.85)
        mask_prediction = torch.sigmoid(mask_logits) >= 0.35
        spine_prediction = torch.sigmoid(spine_logits) >= 0.35
        mask_truth, spine_truth = masks > 0.5, spines > 0.5
        mask_inter += float((mask_prediction & mask_truth).sum().item()) * 2.0
        mask_total += float(mask_prediction.sum().item() + mask_truth.sum().item())
        spine_inter += float((spine_prediction & spine_truth).sum().item()) * 2.0
        spine_total += float(spine_prediction.sum().item() + spine_truth.sum().item())
        loss_sum += float(loss.item())
        batches += 1
    return {
        "loss": loss_sum / max(1, batches),
        "mask_dice_035": mask_inter / max(1.0, mask_total),
        "spine_dice_035": spine_inter / max(1.0, spine_total),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--time-hours", type=float, default=4.25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--checkpoint", default=None,
                        help="optional exp085 checkpoint used to continue the same architecture")
    args = parser.parse_args()

    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"Nvidia T4 required, found {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    model = MordenSpineNet(base_channels=args.base_channels).to(device)
    source_checkpoint = None
    if args.checkpoint:
        source_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if int(source_checkpoint["base_channels"]) != args.base_channels:
            raise ValueError("checkpoint base_channels does not match requested architecture")
        model.load_state_dict(source_checkpoint["model"], strict=True)
    print(
        f"RESOLVED device={torch.cuda.get_device_name(0)} base={args.base_channels} "
        f"parameters={parameter_count(model)} batch={args.batch} accumulate={args.accumulate} "
        f"epochs={args.epochs} lr={args.lr} time_hours={args.time_hours} "
        f"checkpoint_epoch={None if source_checkpoint is None else source_checkpoint['epoch']}",
        flush=True,
    )
    train_loader = DataLoader(
        CachedRecords(args.cache, "train", True), batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        CachedRecords(args.cache, "val", False), batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(args.cache) / "manifest.json").read_text())
    started = time.monotonic()
    best_dice = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        train_loss = 0.0
        for step, (images, masks, spines) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            spines = spines.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                mask_logits, spine_logits = model(images)
                loss = focal_dice(mask_logits, masks, 0.66) + 0.25 * focal_dice(spine_logits, spines, 0.85)
                scaled_loss = loss / args.accumulate
            scaler.scale(scaled_loss).backward()
            if step % args.accumulate == 0 or step == len(train_loader):
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)
            train_loss += float(loss.detach().item())
            if step % 200 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss={train_loss / step:.5f}", flush=True)
        scheduler.step()
        validation = validate(model, val_loader, device)
        elapsed_hours = (time.monotonic() - started) / 3600.0
        row = {
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader),
            **validation,
            "lr": optimiser.param_groups[0]["lr"],
            "elapsed_hours": elapsed_hours,
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        history.append(row)
        state = {
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "epoch": epoch,
            "base_channels": args.base_channels,
            "image_size": manifest["size"],
            "channels": manifest["channels"],
            "validation": validation,
        }
        torch.save(state, output / "last.pt")
        if validation["mask_dice_035"] > best_dice:
            best_dice = validation["mask_dice_035"]
            torch.save(state, output / "best.pt")
        status = {
            "status": "running",
            "resolved": vars(args),
            "cache_manifest": manifest,
            "best_mask_dice_035": best_dice,
            "history": history,
        }
        (output / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print("EPOCH " + json.dumps(row, sort_keys=True), flush=True)
        if elapsed_hours >= args.time_hours:
            print("time budget reached", flush=True)
            break

    status["status"] = "complete"
    (output / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"COMPLETE best_mask_dice_035={best_dice:.6f}", flush=True)


if __name__ == "__main__":
    main()
