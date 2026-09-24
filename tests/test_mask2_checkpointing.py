import copy

import torch
import pytest
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, SwinConfig

from shared.mask2_checkpointing import enable_checkpointing, enable_fp32_sampling


@pytest.mark.parametrize('bf16', [False, True])
def test_checkpointed_model_preserves_forward_gradients_and_state_keys(bf16, monkeypatch):
    from transformers.models.mask2former import modeling_mask2former as mask2
    monkeypatch.setattr(mask2,'multi_scale_deformable_attention',mask2.multi_scale_deformable_attention)
    if bf16:
        enable_fp32_sampling()
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    backbone = SwinConfig(embed_dim=32, depths=[1,1,1,1], num_heads=[1,2,4,8],
                          window_size=4, drop_path_rate=0,
                          out_features=['stage1','stage2','stage3','stage4'])
    config = Mask2FormerConfig(backbone_config=backbone, num_labels=1,
                               encoder_layers=1, decoder_layers=2, num_queries=32, dropout=0)
    reference = Mask2FormerForUniversalSegmentation(config).train()
    checkpointed = copy.deepcopy(reference)
    names = enable_checkpointing(checkpointed)
    assert len(names) == 6
    assert list(reference.state_dict()) == list(checkpointed.state_dict())
    image = torch.randn(1,3,128,128)
    losses = []
    for model in (reference, checkpointed):
        torch.manual_seed(2026)
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
            output = model(pixel_values=image)
            loss = output.class_queries_logits.float().square().mean()+output.masks_queries_logits.float().square().mean()
        loss.backward()
        losses.append(loss.detach())
    torch.testing.assert_close(*losses, rtol=1e-5, atol=1e-6)
    for (name,a), (_,b) in zip(reference.named_parameters(),checkpointed.named_parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=1e-5,msg=name)


def test_mixed_dtype_sampling_matches_explicit_fp32_outputs_and_gradients(monkeypatch):
    from transformers.models.mask2former import modeling_mask2former as mask2
    original = mask2.multi_scale_deformable_attention
    monkeypatch.setattr(mask2,'multi_scale_deformable_attention',original)
    torch.manual_seed(2026)
    value = torch.randn(1,5,2,4,dtype=torch.bfloat16,requires_grad=True)
    locations = torch.rand(1,3,2,2,2,2,requires_grad=True)
    weights = torch.rand(1,3,2,2,2,dtype=torch.bfloat16,requires_grad=True)
    shapes = [(2,2),(1,1)]
    expected = original(value.float(),shapes,locations,weights.float())
    enable_fp32_sampling()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        actual = mask2.multi_scale_deformable_attention(value,shapes,locations,weights)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    reference_grads = torch.autograd.grad(expected.square().mean(),(value,locations,weights))
    actual_grads = torch.autograd.grad(actual.square().mean(),(value,locations,weights))
    for actual,expected in zip(actual_grads,reference_grads):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
