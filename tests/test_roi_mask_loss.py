"""Check loss and prototype/coefficient gradients against native FP32 masks."""
import pytest
import torch
import torch.nn.functional as F
from shared.chunked_mask_loss import ChunkedSegmentationLoss
from shared.roi_mask_loss import ROISegmentationLoss


@pytest.mark.parametrize('overlap',[False,True])
@pytest.mark.parametrize('anchors',[7,57])
@pytest.mark.parametrize('empty',[False,True])
def test_native_roi_parity(overlap,anchors,empty):
    torch.set_num_threads(4);torch.manual_seed(2026)
    b,c,h,w=2,4,44,60
    proto=torch.randn(b,c,h//4,w//4,requires_grad=True)
    coefficients=torch.randn(b,anchors,c,requires_grad=True)
    fg=torch.ones(b,anchors,dtype=torch.bool);fg[1]=False
    if empty:fg[:]=False
    labels=torch.randint(0,4,(b,h,w))
    masks=labels.float() if overlap else torch.stack([labels[i]==j for i in range(b) for j in (1,2,3)]).float()
    idx=torch.randint(0,3,(b,anchors))
    boxes=torch.tensor([1.3,2.6,42.7,31.4]).repeat(b,anchors,1)
    boxes[:,0]=torch.tensor([0.,0.,float(w),float(h)])
    batch_idx=torch.tensor([0,0,0,1,1,1]);size=torch.tensor([h,w])
    reference=object.__new__(ChunkedSegmentationLoss);reference.overlap=overlap;reference.chunk_size=8
    candidate=object.__new__(ROISegmentationLoss);candidate.overlap=overlap
    args=(fg,masks,idx,boxes,batch_idx)
    expected=reference.calculate_segmentation_loss(*args,F.interpolate(proto,(h,w),mode='bilinear',align_corners=False),coefficients,size)
    actual=candidate.calculate_segmentation_loss(*args,proto,coefficients,size)
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
    ga=torch.autograd.grad(actual,(proto,coefficients),retain_graph=True)
    ge=torch.autograd.grad(expected,(proto,coefficients))
    for a,e in zip(ga,ge):torch.testing.assert_close(a,e,rtol=3e-4,atol=3e-6)
