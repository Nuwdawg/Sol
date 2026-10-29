"""exp173: IoU-aware class targets for Mask2Former, and an exponential moving average of weights.

Why the class target changes. The deployed emission rule keeps a candidate when its score
clears a floor, and the pooled PQ rewards a kept candidate only through the IoU it reaches
with an annotated filament. Adding one candidate changes the PQ denominator by 1/2 whether it
matches or not (a match turns a false negative into a true positive; a miss adds a false
positive), and the numerator by its IoU when it matches. Keeping it raises PQ exactly when

    P(match) x E[IoU | match]  >  PQ / 2,

so the score that ranks candidates best is an estimate of E[IoU x match]. Mask2Former trains
its class head towards 1 for every matched query, whatever the quality of its mask, so its
probability estimates P(match) alone. Rank-DETR (Pu et al., NeurIPS 2023) and Stable-DINO
(Liu et al., ICCV 2023) supervise the class score with the localisation quality of the
matched prediction instead; here the target of a matched query is the IoU of its own mask
with the annotation it was matched to, measured on the stride-4 mask grid and detached.

The loss keeps HF's weighting and normalisation: matched queries weigh 1, unmatched ones
`no_object_weight`, and the sum is divided by the sum of those weights, the quantity the
trainer's micro-batch shares already assume. For a matched query the two terms of the soft
cross-entropy carry the same weight, so the minimiser is p(filament) = IoU.
"""
from __future__ import annotations

import copy
import types

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def gt_on_mask_grid(mask_labels: list[torch.Tensor], size: tuple[int, int]) -> list[torch.Tensor]:
    """Each image's annotated masks reduced to the prediction grid by area averaging, then
    thresholded at one half: the binary mask the prediction grid can best represent."""
    out = []
    for masks in mask_labels:
        if masks.numel() == 0:
            out.append(torch.zeros((0, *size), dtype=torch.bool, device=masks.device))
            continue
        pooled = F.adaptive_avg_pool2d(masks[:, None].float(), size)[:, 0]
        out.append(pooled > 0.5)
    return out


@torch.no_grad()
def matched_iou(masks_queries_logits: torch.Tensor, gt_small: list[torch.Tensor], indices) -> list[torch.Tensor]:
    """IoU of each matched query's binary mask (logit > 0) with its matched annotation."""
    values = []
    for i, (src, tgt) in enumerate(indices):
        if len(src) == 0:
            values.append(torch.zeros(0, device=masks_queries_logits.device))
            continue
        pred = masks_queries_logits[i, src.to(masks_queries_logits.device)] > 0
        gt = gt_small[i][tgt.to(gt_small[i].device)]
        inter = (pred & gt).flatten(1).sum(1).float()
        union = (pred | gt).flatten(1).sum(1).float()
        values.append(torch.where(union > 0, inter / union.clamp(min=1), torch.zeros_like(union)))
    return values


def quality_loss_labels(criterion, class_queries_logits: torch.Tensor, indices, ious: list[torch.Tensor]) -> dict:
    """Soft cross-entropy: matched queries towards (IoU, 1 - IoU), unmatched towards no-object."""
    logits = class_queries_logits.float()
    batch, queries, classes = logits.shape
    if classes != 2:
        raise SystemExit(f"quality targets assume one foreground class, got {classes - 1}")
    log_p = logits.log_softmax(-1)
    target = torch.zeros(batch, queries, device=logits.device)
    matched = torch.zeros(batch, queries, dtype=torch.bool, device=logits.device)
    for i, ((src, _), iou) in enumerate(zip(indices, ious)):
        if len(src):
            src = src.to(logits.device)
            target[i, src] = iou.to(logits.device).clamp(0.0, 1.0)
            matched[i, src] = True
    eos = float(criterion.eos_coef)
    matched_loss = -(target * log_p[..., 0] + (1.0 - target) * log_p[..., 1])
    unmatched_loss = -eos * log_p[..., 1]
    per_query = torch.where(matched, matched_loss, unmatched_loss)
    normaliser = matched.sum().float() + eos * (~matched).sum().float()
    return {"loss_cross_entropy": per_query.sum() / normaliser.clamp(min=1e-12)}


def quality_loss_forward(self, masks_queries_logits, class_queries_logits, mask_labels, class_labels,
                         auxiliary_predictions=None):
    """exp139's batched loss (fast_ops.batched_loss_forward) with the class term replaced.

    Matching, point draws and mask losses are the same operations in the same order; only
    `loss_labels` becomes `quality_loss_labels`, fed by each layer's own mask IoUs.
    """
    from experiments.exp_139_mask2former.src import fast_ops   # needs transformers
    layers = [(masks_queries_logits, class_queries_logits)]
    if auxiliary_predictions is not None:
        layers += [(aux["masks_queries_logits"], aux["class_queries_logits"]) for aux in auxiliary_predictions]
    device = masks_queries_logits.device
    batch_size, num_queries = masks_queries_logits.shape[:2]
    matched = sum(min(num_queries, len(labels)) for labels in class_labels)
    oversampled_points = int(self.num_points * self.oversample_ratio)
    random_points = self.num_points - int(self.importance_sample_ratio * self.num_points)

    draws = []
    for _ in layers:
        matcher_points = [torch.rand(1, self.matcher.num_points, 2, device=device) for _ in range(batch_size)]
        oversampled = torch.rand(matched, oversampled_points, 2, device=device)
        extra = torch.rand(matched, random_points, 2, device=device) if random_points > 0 else None
        draws.append((matcher_points, oversampled, extra))

    with torch.no_grad():
        costs = [fast_ops._cost_matrix(self.matcher, masks[i], classes[i], mask_labels[i], class_labels[i], points[i])
                 for (masks, classes), (points, _, _) in zip(layers, draws) for i in range(batch_size)]
        flat = torch.cat([cost.reshape(-1).float() for cost in costs]).cpu()
    indices_per_layer, offset = [], 0
    for layer in range(len(layers)):
        indices = []
        for i in range(batch_size):
            shape = costs[layer * batch_size + i].shape
            size = shape[0] * shape[1]
            rows, columns = linear_sum_assignment(flat[offset:offset + size].view(shape))
            offset += size
            indices.append((torch.as_tensor(rows, dtype=torch.int64), torch.as_tensor(columns, dtype=torch.int64)))
        indices_per_layer.append(indices)

    gt_small = gt_on_mask_grid(mask_labels, tuple(masks_queries_logits.shape[-2:]))
    losses: dict[str, torch.Tensor] = {}
    for layer, ((masks, classes), (_, oversampled, extra), indices) in enumerate(
            zip(layers, draws, indices_per_layer)):
        num_masks = self.get_num_masks(class_labels, device=class_labels[0].device)
        ious = matched_iou(masks, gt_small, indices)
        part = {**fast_ops._loss_masks(self, masks, mask_labels, indices, num_masks, oversampled, extra),
                **quality_loss_labels(self, classes, indices, ious)}
        losses.update(part if layer == 0 else {f"{key}_{layer - 1}": value for key, value in part.items()})
    return losses


def install_quality_loss(model) -> None:
    model.criterion.forward = types.MethodType(quality_loss_forward, model.criterion)


class WeightAverage:
    """Exponential moving average of the weights, kept in float32 on the training device.

    The decay warms up as min(decay, (1 + step) / (10 + step)), so the first steps do not
    freeze the average at the pretrained weights. Buffers are copied, not averaged.
    """

    def __init__(self, model, decay: float):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step: int) -> float:
        decay = min(self.decay, (1.0 + step) / (10.0 + step))
        for average, current in zip(self.model.parameters(), model.parameters()):
            average.mul_(decay).add_(current.detach().float(), alpha=1.0 - decay)
        for average, current in zip(self.model.buffers(), model.buffers()):
            average.copy_(current)
        return decay
