"""exp139: faster Mask2Former training with the same mathematics.

Two replacements for transformers 4.57.6 code. Each must pass an equality test against
the code it replaces before a run uses it (`verify_deformable_kernel`,
`verify_batched_loss`); neither changes the model, the data, the loss or the optimiser.

1. Multi-scale deformable attention. `modeling_mask2former.multi_scale_deformable_attention`
   (line 799) is a pure-PyTorch loop of `grid_sample` calls. HF's own Deformable DETR maps
   the same operation to the prebuilt CUDA kernel `kernels-community/deformable-detr`
   (transformers/integrations/hub_kernels.py); this module wires that kernel into
   Mask2Former. Precision matches the reference: under autocast the reference samples in
   float32 (`grid_sample` promotes to its widest input, and the sampling grid is float32),
   so the kernel is called on float32 inputs and returns float32.

   Measured on a Kaggle T4 (2026-09-17): output, value gradient and weight gradient agree
   with the PyTorch loop to about 2e-6 relative. The gradient with respect to sampling
   locations disagreed by up to 1.2% at 2048 px, in a few elements. Bilinear sampling is
   piecewise linear, so that gradient jumps where a sampling point crosses a pixel centre,
   and float32 rounding of the coordinate decides which side a point on the edge falls.
   `verify_deformable_kernel` therefore compares against a float64 run of the PyTorch
   loop and requires: every error below 1e-4 away from those edges, and every larger
   error within `kink_margin` pixels of one. HF's own float32 loop is scored the same way.

   A local build of the MSDA source bundled in transformers (kernels/deta) failed to
   compile on the T4 image and was removed.

2. Hungarian matching. `Mask2FormerHungarianMatcher.forward` (line 475) copies one cost
   matrix to the CPU per image per decoder layer: 10 layers x 8 images = 80 GPU stalls
   per step. `batched_loss_forward` computes every cost matrix first, copies them in
   one transfer, then solves. Random draws are made up front in exactly the order HF's
   recursion makes them (per layer: one matcher draw per image, then the two point
   sampling draws). On CPU the loss and all gradients are identical to the bit. On a GPU
   the loss is identical to the bit and gradients differ only as much as two runs of
   HF's own loss differ, because `grid_sample`'s CUDA backward is not deterministic.
"""
from __future__ import annotations

import types

import torch
from scipy.optimize import linear_sum_assignment

from transformers.models.mask2former import modeling_mask2former as m2f

REFERENCE_MSDA = m2f.multi_scale_deformable_attention
HUB_REPO = "kernels-community/deformable-detr"
IM2COL_STEP = 64


# ------------------------------------------------------------------ item 1: kernel ---
def _level_start_index(shapes: torch.Tensor) -> torch.Tensor:
    return torch.cat((shapes.new_zeros((1,)), (shapes[:, 0] * shapes[:, 1]).cumsum(0)[:-1]))


def load_hub_kernel(revision: str | None = None):
    """The prebuilt kernel HF maps to Deformable DETR. Returns (callable, description)."""
    from kernels import get_kernel
    kernel = get_kernel(HUB_REPO, revision=revision) if revision else get_kernel(HUB_REPO)
    layer = kernel.layers.MultiScaleDeformableAttention()

    def run(value, shapes, shapes_list, level_start, locations, weights):
        return layer(value, shapes, shapes_list, level_start, locations, weights, IM2COL_STEP)

    return run, dict(source="hub", repo=HUB_REPO, revision=revision,
                     module=getattr(kernel, "__name__", None), path=str(getattr(kernel, "__file__", "")))


def make_kernel_msda(run):
    """Adapter from Mask2Former's call signature to the kernel's."""

    def kernel_msda(value, value_spatial_shapes, sampling_locations, attention_weights):
        shapes_list = [(int(height), int(width)) for height, width in value_spatial_shapes]
        shapes = torch.as_tensor(shapes_list, dtype=torch.long, device=value.device)
        with torch.autocast(value.device.type, enabled=False):
            return run(value.float().contiguous(), shapes, shapes_list, _level_start_index(shapes),
                       sampling_locations.float().contiguous(), attention_weights.float().contiguous())

    return kernel_msda


def install_deformable_kernel(source: str, revision: str | None = None) -> dict:
    """Replace the pure-PyTorch deformable attention for this process. `pytorch` restores it."""
    if source == "pytorch":
        m2f.multi_scale_deformable_attention = REFERENCE_MSDA
        return dict(source="pytorch")
    if source != "hub":
        raise SystemExit(f"unknown deformable attention source: {source}")
    run, description = load_hub_kernel(revision)
    m2f.multi_scale_deformable_attention = make_kernel_msda(run)
    return description


def verify_deformable_kernel(device: torch.device, extents=((1024, 2), (2048, 1)), tolerance: float = 1e-4,
                             kink_margin: float = 1e-3, seed: int = 0) -> list[dict]:
    """Installed kernel against a float64 run of HF's PyTorch loop, on training shapes.

    Mask2Former's pixel decoder attends from every token of the 1/8, 1/16 and 1/32 maps,
    8 heads of width 32, 4 points per level. Sampling points include positions just
    outside the map. Errors are max |candidate - truth| / max |truth| per tensor. For the
    location gradient, elements within `kink_margin` pixels of a pixel-centre crossing are
    reported separately, and every error above `tolerance` must lie within that margin.
    """
    kernel = m2f.multi_scale_deformable_attention
    if kernel is REFERENCE_MSDA:
        raise SystemExit("verify_deformable_kernel called while the reference is installed")
    rows = []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for extent, batch in extents:
        shapes = [(extent // stride, extent // stride) for stride in (8, 16, 32)]
        tokens = sum(height * width for height, width in shapes)
        value = torch.randn(batch, tokens, 8, 32, generator=generator, dtype=torch.float64)
        locations = torch.rand(batch, tokens, 8, 3, 4, 2, generator=generator, dtype=torch.float64) * 1.2 - 0.1
        weights = torch.rand(batch, tokens, 8, 3, 4, generator=generator, dtype=torch.float64)
        weights = weights / weights.sum((-1, -2), keepdim=True)
        grad_output = torch.randn(batch, tokens, 8 * 32, generator=generator, dtype=torch.float64)
        results = {}
        for name, function, dtype in (("truth", REFERENCE_MSDA, torch.float64),
                                      ("kernel", kernel, torch.float64),
                                      ("hf_float32", REFERENCE_MSDA, torch.float32)):
            # A fresh leaf per run: `.to` returns the same object when nothing changes.
            inputs = [tensor.detach().to(device=device, dtype=dtype).clone().requires_grad_(True)
                      for tensor in (value, locations, weights)]
            output = function(inputs[0], shapes, inputs[1], inputs[2])
            output.backward(grad_output.to(device=device, dtype=output.dtype))
            results[name] = [output.detach().double()] + [tensor.grad.double() for tensor in inputs]
            del inputs, output
            if device.type == "cuda":
                torch.cuda.empty_cache()
        grid = locations.to(device)
        widths = torch.tensor([width for _, width in shapes], dtype=torch.float64, device=device).view(1, 1, 1, 3, 1)
        heights = torch.tensor([height for height, _ in shapes], dtype=torch.float64, device=device).view(1, 1, 1, 3, 1)
        x = grid[..., 0] * widths - 0.5
        y = grid[..., 1] * heights - 0.5
        distance = torch.minimum((x - x.round()).abs(), (y - y.round()).abs()).unsqueeze(-1).expand(*grid.shape)
        near = distance < kink_margin
        row = dict(extent=extent, batch=batch, tokens=tokens, share_near_kink=float(near.double().mean()))
        truth = results["truth"]
        for name in ("kernel", "hf_float32"):
            errors = {}
            for label, reference, candidate in zip(("output", "grad_value", "grad_locations", "grad_weights"),
                                                   truth, results[name]):
                difference = (candidate - reference).abs() / reference.abs().max().clamp_min(1e-300)
                errors[label] = float(difference.max())
                if label == "grad_locations":
                    flagged = difference > tolerance
                    errors["grad_locations_away_from_kinks"] = float(difference[~near].max())
                    errors["grad_locations_flagged"] = int(flagged.sum())
                    errors["grad_locations_flagged_max_kink_distance"] = (
                        float(distance[flagged].max()) if bool(flagged.any()) else 0.0)
            row[name] = errors
        found = row["kernel"]
        row["passed"] = (found["output"] <= tolerance and found["grad_value"] <= tolerance
                         and found["grad_weights"] <= tolerance
                         and found["grad_locations_away_from_kinks"] <= tolerance
                         and found["grad_locations_flagged_max_kink_distance"] < kink_margin)
        rows.append(row)
        del results, grid, x, y, distance, near
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------- item 2: batched matching ---
def _cost_matrix(matcher, masks_queries_logits, class_queries_logits, mask_labels, class_labels,
                 point_coordinates):
    """`Mask2FormerHungarianMatcher.forward`'s body for one image, operation for operation,
    with the random point draw passed in and without the CPU copy."""
    pred_probs = class_queries_logits.softmax(-1)
    pred_mask = masks_queries_logits
    cost_class = -pred_probs[:, class_labels]
    target_mask = mask_labels.to(pred_mask)
    target_mask = target_mask[:, None]
    pred_mask = pred_mask[:, None]
    target_coordinates = point_coordinates.repeat(target_mask.shape[0], 1, 1)
    target_mask = m2f.sample_point(target_mask, target_coordinates, align_corners=False).squeeze(1)
    pred_coordinates = point_coordinates.repeat(pred_mask.shape[0], 1, 1)
    pred_mask = m2f.sample_point(pred_mask, pred_coordinates, align_corners=False).squeeze(1)
    cost_mask = m2f.pair_wise_sigmoid_cross_entropy_loss(pred_mask, target_mask)
    cost_dice = m2f.pair_wise_dice_loss(pred_mask, target_mask)
    cost_matrix = matcher.cost_mask * cost_mask + matcher.cost_class * cost_class + matcher.cost_dice * cost_dice
    cost_matrix = torch.minimum(cost_matrix, torch.tensor(1e10))
    cost_matrix = torch.maximum(cost_matrix, torch.tensor(-1e10))
    return torch.nan_to_num(cost_matrix, 0)


def _sample_points(criterion, logits, oversampled, extra):
    """`Mask2FormerLoss.sample_points_using_uncertainty` with its two draws passed in."""
    num_boxes = logits.shape[0]
    num_points_sampled = int(criterion.num_points * criterion.oversample_ratio)
    point_coordinates = oversampled
    point_logits = m2f.sample_point(logits, point_coordinates, align_corners=False)
    point_uncertainties = criterion.calculate_uncertainty(point_logits)
    num_uncertain_points = int(criterion.importance_sample_ratio * criterion.num_points)
    num_random_points = criterion.num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_points_sampled * torch.arange(num_boxes, dtype=torch.long, device=logits.device)
    idx += shift[:, None]
    point_coordinates = point_coordinates.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)
    if num_random_points > 0:
        point_coordinates = torch.cat([point_coordinates, extra], dim=1)
    return point_coordinates


def _loss_masks(criterion, masks_queries_logits, mask_labels, indices, num_masks, oversampled, extra):
    """`Mask2FormerLoss.loss_masks` with the point draws passed in."""
    src_idx = criterion._get_predictions_permutation_indices(indices)
    tgt_idx = criterion._get_targets_permutation_indices(indices)
    pred_masks = masks_queries_logits[src_idx]
    target_masks, _ = criterion._pad_images_to_max_in_batch(mask_labels)
    target_masks = target_masks[tgt_idx]
    pred_masks = pred_masks[:, None]
    target_masks = target_masks[:, None]
    with torch.no_grad():
        point_coordinates = _sample_points(criterion, pred_masks, oversampled, extra)
        point_labels = m2f.sample_point(target_masks, point_coordinates, align_corners=False).squeeze(1)
    point_logits = m2f.sample_point(pred_masks, point_coordinates, align_corners=False).squeeze(1)
    losses = {
        "loss_mask": m2f.sigmoid_cross_entropy_loss(point_logits, point_labels, num_masks),
        "loss_dice": m2f.dice_loss(point_logits, point_labels, num_masks),
    }
    del pred_masks
    del target_masks
    return losses


def batched_loss_forward(self, masks_queries_logits, class_queries_logits, mask_labels, class_labels,
                         auxiliary_predictions=None):
    """Drop-in for `Mask2FormerLoss.forward` with one CPU transfer for all matching."""
    layers = [(masks_queries_logits, class_queries_logits)]
    if auxiliary_predictions is not None:
        layers += [(aux["masks_queries_logits"], aux["class_queries_logits"]) for aux in auxiliary_predictions]
    device = masks_queries_logits.device
    batch_size, num_queries = masks_queries_logits.shape[:2]
    matched = sum(min(num_queries, len(labels)) for labels in class_labels)
    oversampled_points = int(self.num_points * self.oversample_ratio)
    random_points = self.num_points - int(self.importance_sample_ratio * self.num_points)

    # 1. Every random draw, in the order HF's recursion makes them.
    draws = []
    for _ in layers:
        matcher_points = [torch.rand(1, self.matcher.num_points, 2, device=device) for _ in range(batch_size)]
        oversampled = torch.rand(matched, oversampled_points, 2, device=device)
        extra = torch.rand(matched, random_points, 2, device=device) if random_points > 0 else None
        draws.append((matcher_points, oversampled, extra))

    # 2. Every cost matrix on the GPU, then one copy to the CPU.
    with torch.no_grad():
        costs = [_cost_matrix(self.matcher, masks[i], classes[i], mask_labels[i], class_labels[i], points[i])
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

    # 3. Losses per layer, with HF's key order so the final sum adds in the same order.
    losses: dict[str, torch.Tensor] = {}
    for layer, ((masks, classes), (_, oversampled, extra), indices) in enumerate(
            zip(layers, draws, indices_per_layer)):
        num_masks = self.get_num_masks(class_labels, device=class_labels[0].device)
        part = {**_loss_masks(self, masks, mask_labels, indices, num_masks, oversampled, extra),
                **self.loss_labels(classes, class_labels, indices)}
        losses.update(part if layer == 0 else {f"{key}_{layer - 1}": value for key, value in part.items()})
    return losses


def install_batched_loss(model) -> None:
    model.criterion.forward = types.MethodType(batched_loss_forward, model.criterion)


def restore_hf_loss(model) -> None:
    model.criterion.forward = types.MethodType(m2f.Mask2FormerLoss.forward, model.criterion)


def _worst_relative(first: dict, second: dict) -> float:
    worst = 0.0
    for name, reference in first.items():
        scale = float(reference.abs().max())
        if scale > 0:
            worst = max(worst, float((second[name] - reference).abs().max()) / scale)
    return worst


def verify_batched_loss(model, pixels, mask_labels, class_labels, autocast_dtype=None) -> dict:
    """HF's loss twice (the control) and the batched loss once: same seed, same inputs.

    Passes when all three losses are identical to the bit and the batched gradients differ
    from HF's no more than HF's differ from themselves: identical to the bit where the
    control is (CPU), within five times the control's spread where it is not (GPU).
    """
    runs = []
    for batched in (False, False, True):
        install_batched_loss(model) if batched else restore_hf_loss(model)
        model.zero_grad(set_to_none=True)
        torch.manual_seed(1234)
        with torch.autocast(pixels.device.type, dtype=autocast_dtype or torch.float32,
                            enabled=autocast_dtype is not None):
            loss = model(pixel_values=pixels, mask_labels=mask_labels, class_labels=class_labels).loss
        loss.backward()
        runs.append((loss.detach().clone(), {name: parameter.grad.detach().float().clone()
                                             for name, parameter in model.named_parameters()
                                             if parameter.grad is not None}))
    install_batched_loss(model)
    (loss_a, grads_a), (loss_b, grads_b), (loss_c, grads_c) = runs
    control_identical = all(torch.equal(grads_a[name], grads_b[name]) for name in grads_a)
    batched_identical = set(grads_a) == set(grads_c) and all(torch.equal(grads_a[name], grads_c[name])
                                                             for name in grads_a)
    control = _worst_relative(grads_a, grads_b)
    batched = _worst_relative(grads_a, grads_c)
    losses_identical = bool(torch.equal(loss_a, loss_b) and torch.equal(loss_a, loss_c))
    passed = losses_identical and (batched_identical if control_identical else batched <= 5 * control)
    return dict(loss_hf=float(loss_a), loss_hf_again=float(loss_b), loss_batched=float(loss_c),
                losses_identical=losses_identical, parameters=len(grads_a),
                control_gradients_identical=control_identical, batched_gradients_identical=batched_identical,
                control_worst_relative=control, batched_worst_relative=batched, passed=passed)
