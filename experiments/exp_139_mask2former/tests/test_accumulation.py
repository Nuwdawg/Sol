"""Is a step split into micro-batches the same as one big batch? (exp139 / exp141 budget plan)

transformers normalises each forward pass by that pass's own mask count and class-weight
sum, so the usual "divide by the number of parts" accumulation does NOT reproduce a big
batch when the parts hold different numbers of filaments. `train.step_losses` scales each
micro-batch by its exact share instead.

The loss samples points at random, and a big batch and its parts would draw different
points, which would hide the effect under sampling noise. This test therefore replaces
`torch.rand` with a fixed pattern that depends only on the point index, so every mask is
sampled at the same places in both runs. What remains is the normalisation alone.

    python -m experiments.exp_139_mask2former.tests.test_accumulation --annotations ... --images ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.exp_139_mask2former.src.data import FilamentSamples, collate, load_records
from experiments.exp_139_mask2former.src.fast_ops import install_batched_loss
from experiments.exp_139_mask2former.src.model import build_model
from experiments.exp_139_mask2former.src.train import capture_loss_dict, step_shares

REAL_RAND = torch.rand


def fixed_rand(*size, **kwargs):
    """Deterministic stand-in: every row gets the same point pattern, so a mask is sampled
    identically whether it sits in a big batch or in a part of one."""
    shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list)) else tuple(size)
    if len(shape) == 3 and shape[-1] == 2:
        generator = torch.Generator().manual_seed(11)
        pattern = REAL_RAND(1, shape[1], 2, generator=generator)
        return pattern.expand(shape[0], shape[1], 2).contiguous().to(kwargs.get("device", "cpu"))
    return REAL_RAND(*size, **kwargs)


def run(model, captured, groups: list, scaling: str) -> tuple[float, dict]:
    counts = [step_shares(classes, model.config.num_queries, model.config.no_object_weight)
              for _, _, classes in groups]
    total_masks = sum(count[0] for count in counts)
    total_weight = sum(count[1] for count in counts)
    model.zero_grad(set_to_none=True)
    value = 0.0
    for (pixels, masks, classes), (micro_masks, micro_weight) in zip(groups, counts):
        model(pixel_values=pixels, mask_labels=masks, class_labels=classes)
        if scaling == "exact":
            mask_share, class_share = micro_masks / total_masks, micro_weight / total_weight
        else:
            mask_share = class_share = 1.0 / len(groups)
        loss = sum(term * (class_share if "cross_entropy" in key else mask_share)
                   for key, term in captured["dict"].items())
        loss.backward()
        value += float(loss.detach())
    return value, {name: parameter.grad.detach().clone()
                   for name, parameter in model.named_parameters() if parameter.grad is not None}


def relative_norm(first: dict, second: dict) -> float:
    """||b - a|| / ||a|| over every gradient at once. A worst-element ratio is dominated by
    parameters whose gradient is near zero, so it says nothing useful here."""
    difference = sum(float(((second[name] - reference) ** 2).sum()) for name, reference in first.items())
    scale = sum(float((reference ** 2).sum()) for reference in first.values())
    return (difference ** 0.5) / max(scale ** 0.5, 1e-12)


def silence_stochastic_depth(model) -> int:
    """Drop path and dropout draw their own randomness, which would differ between the runs
    and hide the normalisation under noise. Both are switched off for the comparison."""
    changed = 0
    for module in model.modules():
        if hasattr(module, "drop_prob") and getattr(module, "drop_prob"):
            module.drop_prob = 0.0
            changed += 1
        if isinstance(module, torch.nn.Dropout) and module.p:
            module.p = 0.0
            changed += 1
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--extent", type=int, default=512)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    torch.manual_seed(3)
    records = load_records(args.annotations, "train")
    dataset = FilamentSamples(records, args.images, preload=False, augment=False)
    # Four samples with deliberately different filament counts, so the shares differ.
    order = sorted(range(len(records)), key=lambda index: -len(records[index]["rles"]))
    picks = [order[0], order[1], order[-1], order[-2]]
    items = [dataset[(index, args.extent, 0, position, True, 1)] for position, index in enumerate(picks)]
    batches = []
    for start, stop in ((0, 4), (0, 2), (2, 4)):
        pixels, masks, classes, _, _ = collate(items[start:stop])
        batches.append((pixels.float().div(255).unsqueeze(1).expand(-1, 3, -1, -1),
                        [m.float() for m in masks], classes))
    whole, first, second = batches
    print("filaments per sample", [len(c) for _, _, c in [whole]][0] and [len(m) for m in whole[1]], flush=True)

    model = build_model("tiny").train()
    print("stochastic layers silenced:", silence_stochastic_depth(model), flush=True)
    install_batched_loss(model)
    captured = capture_loss_dict(model)
    torch.rand = fixed_rand
    try:
        big_value, big_grads = run(model, captured, [whole], "exact")
        exact_value, exact_grads = run(model, captured, [first, second], "exact")
        naive_value, naive_grads = run(model, captured, [first, second], "naive")
    finally:
        torch.rand = REAL_RAND
    exact_loss = abs(exact_value - big_value) / abs(big_value)
    naive_loss = abs(naive_value - big_value) / abs(big_value)
    exact_grad = relative_norm(big_grads, exact_grads)
    naive_grad = relative_norm(big_grads, naive_grads)
    print(f"one batch           loss {big_value:.6f}")
    print(f"split, exact shares loss {exact_value:.6f}  relative difference {exact_loss:.2e}  "
          f"gradient norm difference {exact_grad:.2e}")
    print(f"split, 1/parts      loss {naive_value:.6f}  relative difference {naive_loss:.2e}  "
          f"gradient norm difference {naive_grad:.2e}")
    passed = exact_loss <= args.tolerance and exact_grad <= 1e-4
    print("RESULT " + ("exact accumulation reproduces one big batch" if passed else "FAILED"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
