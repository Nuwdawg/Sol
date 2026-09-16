"""Activation checkpointing for Mask2Former without changing state-dict names."""
from functools import wraps
import inspect

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

LAYER_TYPES = {'SwinLayer', 'Mask2FormerPixelDecoderEncoderLayer',
               'Mask2FormerMaskedAttentionDecoderLayer'}


def enable_fp32_sampling():
    """Keep deformable sampling values and coordinates in matching FP32 dtypes.

    XLA autocast leaves coordinates in FP32 while projecting values to BF16;
    its grid_sample requires equal dtypes. Preserve coordinate precision here.
    """
    from transformers.models.mask2former import modeling_mask2former as mask2
    original = mask2.multi_scale_deformable_attention
    if getattr(original, '_sol_fp32_sampling', False):
        return True
    @wraps(original)
    def sample(value, value_spatial_shapes, sampling_locations, attention_weights):
        with torch.autocast(value.device.type, enabled=False):
            return original(value.float(), value_spatial_shapes,
                            sampling_locations.float(), attention_weights.float())
    sample._sol_fp32_sampling = True
    mask2.multi_scale_deformable_attention = sample
    return True


def enable_checkpointing(model, backend='cpu'):
    if backend == 'tpu':
        from torch_xla.utils.checkpoint import checkpoint
    else:
        from torch.utils.checkpoint import checkpoint
    wrapped = []
    for name, module in model.named_modules():
        if type(module).__name__ not in LAYER_TYPES or getattr(module, '_sol_checkpointed', False):
            continue
        original = module.forward
        signature = inspect.signature(original)
        def make_forward(original, signature, module):
            @wraps(original)
            def forward(*args, **kwargs):
                if not module.training or not torch.is_grad_enabled():
                    return original(*args, **kwargs)
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                names = tuple(bound.arguments)
                values, structure = tree_flatten(tuple(bound.arguments.values()))
                # Pass all tensors explicitly: reentrant checkpointing cannot
                # propagate gradients through tensors captured only in closures.
                def run(*values):
                    return original(**dict(zip(names, tree_unflatten(list(values), structure))))
                return checkpoint(run, *values, use_reentrant=True, preserve_rng_state=True)
            return forward
        module.forward = make_forward(original, signature, module)
        module._sol_checkpointed = True
        wrapped.append(name)
    if not wrapped:
        raise ValueError('No expected Mask2Former layers found')
    return wrapped
