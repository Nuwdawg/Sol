"""exp139: Mask2Former construction, optimiser groups and native-resolution inference.

Inference follows Mask2Former's own instance rule (Cheng et al. 2022,
`instance_inference`): score = class probability x mean mask probability inside
the binary mask. Masks come out at 1/4 of the input and are upsampled bilinearly
to the 2048 frame before thresholding at logit 0, as the HF post-processor does.
Every query above a low floor becomes a candidate {score, rle}; the operating
point is chosen afterwards by the same exp033 sweep that scored every earlier
model, so a Mask2Former number and a YOLO number are comparable by construction.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mu

from experiments.exp_139_mask2former.src.data import NATIVE, gpu_pixels

PRETRAINED = "facebook/mask2former-swin-small-coco-instance"
D4 = [(k, mirror) for mirror in (False, True) for k in range(4)]   # exp_007 order


def build_model(kind: str = "pretrained", config: dict | None = None):
    """`pretrained` downloads COCO weights; `config` rebuilds offline; `tiny` is random."""
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, SwinConfig
    if config is not None:
        return Mask2FormerForUniversalSegmentation(Mask2FormerConfig.from_dict(config))
    if kind == "tiny":
        backbone = SwinConfig(embed_dim=24, depths=[1, 1, 1, 1], num_heads=[1, 2, 4, 8],
                              window_size=7, out_features=["stage1", "stage2", "stage3", "stage4"])
        return Mask2FormerForUniversalSegmentation(Mask2FormerConfig(
            backbone_config=backbone, num_labels=1, num_queries=16, encoder_layers=1,
            decoder_layers=2, hidden_dim=64, mask_feature_size=64, feature_size=64,
            dim_feedforward=128))
    if kind != "pretrained":
        raise ValueError(kind)
    return Mask2FormerForUniversalSegmentation.from_pretrained(
        PRETRAINED, num_labels=1, ignore_mismatched_sizes=True)


def parameter_groups(model, lr: float, weight_decay: float, backbone_multiplier: float):
    """Mask2Former's `build_optimizer`: backbone lr x multiplier; no decay on norm
    layers, embeddings, Swin relative position tables and absolute position embeds.

    Parameter names were read from transformers 4.57.6: the Swin backbone lives under
    `model.pixel_level_module.encoder`, the query and level embeddings are nn.Embedding.
    """
    norm_types = (torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.BatchNorm2d)
    groups: dict[tuple[bool, bool], list] = {(b, d): [] for b in (True, False) for d in (True, False)}
    seen = set()
    for module_name, module in model.named_modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            full = f"{module_name}.{name}"
            backbone = full.startswith("model.pixel_level_module.encoder.")
            decay = not (isinstance(module, (*norm_types, torch.nn.Embedding))
                         or "relative_position_bias_table" in name or "absolute_pos_embed" in name)
            groups[(backbone, decay)].append(parameter)
    counts = {f"{'backbone' if b else 'head'}_{'decay' if d else 'plain'}":
              sum(p.numel() for p in params) for (b, d), params in groups.items()}
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if sum(counts.values()) != total or not counts["backbone_decay"] or not counts["head_decay"]:
        raise SystemExit(f"parameter grouping is wrong: {counts}, total {total}")
    return [dict(params=groups[(b, d)], lr=lr * (backbone_multiplier if b else 1.0),
                 weight_decay=weight_decay if d else 0.0)
            for (b, d) in groups if groups[(b, d)]], counts


def backbone_checkpointing(model, enabled: bool) -> None:
    """Recompute Swin stage activations in the backward pass instead of storing them.

    Exact: non-reentrant checkpointing replays the stage with the saved RNG state, so
    drop-path draws match. In transformers 4.57.6 the checkpointed unit is `SwinStage`,
    a `GradientCheckpointingLayer` that reads its flag on every call, so the flag can
    change between micro-batches: whole 2048 frames pay the recompute, crops do not.
    """
    from transformers.modeling_layers import GradientCheckpointingLayer
    encoder = model.model.pixel_level_module.encoder
    if enabled and not getattr(encoder, "_sol_checkpoint_ready", False):
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        encoder._sol_checkpoint_ready = True
    layers = [module for module in encoder.modules() if isinstance(module, GradientCheckpointingLayer)]
    if enabled and not layers:
        raise SystemExit("backbone has no checkpointable layers; the transformers version changed")
    for module in layers:
        module.gradient_checkpointing = enabled


def encode(binary: np.ndarray) -> dict:
    rle = mu.encode(np.asfortranarray(binary.astype(np.uint8)))
    return dict(size=rle["size"], counts=rle["counts"].decode("ascii"))


@torch.inference_mode()
def predict(model, gray: np.ndarray, device: torch.device, bf16: bool, floor: float = 0.02,
            min_pixels: int = 40, chunk: int = 8) -> list[tuple[float, np.ndarray]]:
    """Candidates for one view as (score, binary mask) in that view's frame."""
    pixels = gpu_pixels(torch.from_numpy(np.ascontiguousarray(gray))[None], None, device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
        output = model(pixel_values=pixels)
    probabilities = output.class_queries_logits[0].float().softmax(-1)[:, 0]
    logits = output.masks_queries_logits[0]
    keep = torch.nonzero(probabilities >= floor).flatten()
    candidates = []
    for start in range(0, len(keep), chunk):
        queries = keep[start:start + chunk]
        upsampled = F.interpolate(logits[queries][None].float(), size=gray.shape[:2],
                                  mode="bilinear", align_corners=False)[0]
        binary = upsampled > 0
        area = binary.flatten(1).sum(1)
        inside = (upsampled.sigmoid() * binary).flatten(1).sum(1) / area.clamp(min=1)
        scores = probabilities[queries] * inside
        binary = binary.to(torch.uint8).cpu().numpy()
        for j in range(len(queries)):
            if int(area[j]) >= min_pixels:
                candidates.append((float(scores[j]), binary[j]))
    return candidates


def predict_photograph(model, gray: np.ndarray, device: torch.device, bf16: bool,
                       d4: bool = False, floor: float = 0.02) -> list[dict]:
    """Candidates in the native frame. With `d4`, the exp033 eight-view vote fusion."""
    if gray.shape != (NATIVE, NATIVE):
        raise ValueError(f"expected a {NATIVE}x{NATIVE} photograph, got {gray.shape}")
    if not d4:
        return [dict(score=score, rle=encode(binary))
                for score, binary in predict(model, gray, device, bf16, floor)]
    from experiments.exp_033_inference.src.cached_tta import fuse
    entries = []
    for k, mirror in D4:
        view = np.ascontiguousarray(np.rot90(np.fliplr(gray) if mirror else gray, k))
        for score, binary in predict(model, view, device, bf16, floor):
            restored = np.rot90(binary, -k)
            if mirror:
                restored = np.fliplr(restored)
            entries.append((score, encode(restored)))
    return fuse(entries, 0.5, 0.5)
