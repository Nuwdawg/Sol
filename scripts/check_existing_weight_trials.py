"""Numerical compatibility checks plus a real-photo native inference probe."""
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import ops
from pycocotools import mask as mu
from scripts.existing_weight_trials import decode_planes, infer, mask_nms, interpolate, encode, soft_fuse, TRANSFORMS, apply_transform, invert_transform

torch.set_num_threads(4); cv2.setNumThreads(1)
torch.manual_seed(17)
for n in (4, 51):
    proto = torch.randn(8, 13, 13)
    coeff = torch.randn(n, 8)
    boxes = torch.rand(n, 4)*20
    boxes[:, 2:] += boxes[:, :2]+1
    with torch.inference_mode():
        expected = ops.process_mask_native(proto, coeff, boxes, (64,64)).numpy()
        actual = np.stack([b for b, _ in decode_planes(proto, coeff, boxes, (64,64))])
    assert np.array_equal(expected, actual), n
print('Native mask decoding matches both Ultralytics crop branches', flush=True)
a=np.zeros((20,20),np.uint8);a[2:18,3:5]=1
b=np.zeros_like(a);b[2:18,10:12]=1
entries=[dict(score=.9,rle=encode(a)),dict(score=.8,rle=encode(a)),dict(score=.7,rle=encode(b))]
assert [e['score'] for e in mask_nms(entries,.5)]==[.9,.7]
x={'a':torch.tensor([1.,3.]),'n':torch.tensor(2)}
y={'a':torch.tensor([3.,5.]),'n':torch.tensor(4)}
assert torch.equal(interpolate(x,y,0)['a'],x['a'])
assert torch.equal(interpolate(x,y,1)['a'],y['a'])
assert torch.equal(interpolate(x,y,.5)['a'],torch.tensor([2.,4.]))
assert interpolate(x,y,1)['n']==2
assert x['a'][0]==1
print('Mask NMS and interpolation checks passed',flush=True)
root=Path('data/competition/unpacked/MAGFiLO_1.0_Kaggle_2026/train')
cache=json.loads(Path('data/exp033_cpu_output/results/identity_candidates.json').read_text())
name=sorted(cache)[0]
gray=cv2.imread(str(root/'train_images'/name),cv2.IMREAD_GRAYSCALE)
if gray.ndim==3 and gray.shape[2]==1:
    gray=gray[:,:,0]
image=np.repeat(gray[:,:,None],3,axis=2)
weights='data/checkpoints/exp002_verified.pt'
started=time.monotonic()
model=YOLO(weights)
with torch.inference_mode():
    standard=model.predict(image,imgsz=2048,conf=.25,iou=.6,max_det=100,retina_masks=True,device='cpu',verbose=False)[0]
reference=[dict(score=float(s),rle=encode(m)) for s,m in zip(standard.boxes.conf,standard.masks.data.numpy()) if m.sum()>=40]
del model
model=YOLO(weights)
captured=infer(model,image,soft=True)
assert len(reference)==len(captured)
assert np.max(np.abs(np.array([e['score'] for e in reference])-np.array([e['score'] for e in captured])))<1e-6
ious=np.asarray(mu.iou([e['rle'] for e in reference],[e['rle'] for e in captured],[0]*len(captured)))
assert np.diag(ious).min()>.9999, np.diag(ious)
for transform in TRANSFORMS:
    assert np.array_equal(invert_transform(apply_transform(gray,*transform),*transform),gray)
fused=soft_fuse(captured,(.5,))['0.5']
assert fused and all(mu.area(e['rle'])>=40 for e in fused)
historical=[e for e in cache[name] if e['score']>=.25]
hist_iou=np.asarray(mu.iou([e['rle'] for e in historical],[e['rle'] for e in captured],[0]*len(captured)))
assert len(historical)==len(captured)
assert np.min(np.max(hist_iou,axis=1))>.9
raw=infer(model,image,iou=1.,max_det=1000)
print(json.dumps(dict(photograph=name,native_instances=len(captured),raw_proposals=len(raw),
                     native_min_iou=float(np.diag(ious).min()),historical_min_iou=float(hist_iou.max(axis=1).min()),
                     mask_nms_kept=len(mask_nms(raw,.5)),seconds=time.monotonic()-started)),flush=True)
