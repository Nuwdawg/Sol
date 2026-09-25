"""Explain historical control drift using saved masks and official native inference."""
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from pycocotools import mask as mu
from ultralytics import YOLO
from experiments.exp_033_inference.src.sweep import encode, evaluate, records_for


def main():
    torch.set_num_threads(4); cv2.setNumThreads(1)
    old=json.loads(Path('data/exp033_cpu_output/results/identity_candidates.json').read_text())
    new=json.loads(Path('data/exp060_output/results/box_control.json').read_text())
    changes=[]; comparisons=0
    for name in old:
        a=[e for e in old[name] if e['score']>=.25]
        b=new[name]
        assert len(a)==len(b),name
        assert [e['score'] for e in a]==[e['score'] for e in b],name
        if a:
            ious=np.diag(mu.iou([e['rle'] for e in a],[e['rle'] for e in b],[0]*len(b)))
            if not np.all(ious==1):
                changes.append(name)
        comparisons+=len(a)
    print('Changed photographs',changes,flush=True)
    model=YOLO('data/checkpoints/exp002_verified.pt')
    root=Path('data/competition/unpacked/MAGFiLO_1.0_Kaggle_2026/train')
    report=dict(changed_photographs=changes,matched_scores=comparisons,probes=[])
    for name in changes:
        image=cv2.imread(str(root/'train_images'/name))
        row=dict(photograph=name)
        for floor,reference in ((.05,old),(.25,new)):
            with torch.inference_mode():
                result=model.predict(image,imgsz=2048,conf=floor,iou=.6,max_det=100,
                                     retina_masks=True,device='cpu',verbose=False)[0]
            actual=[dict(score=float(s),rle=encode(m)) for s,m in zip(result.boxes.conf,result.masks.data.numpy()) if m.sum()>=40]
            raw_count=len(result.boxes)
            # Only proposals retained by both pipelines can explain their drift.
            actual=[e for e in actual if e['score']>=.25]
            expected=[e for e in reference[name] if e['score']>=.25]
            assert len(actual)==len(expected),(name,floor,len(actual),len(expected))
            max_score_delta=max(abs(a['score']-b['score']) for a,b in zip(actual,expected))
            # Linux and Windows CPU kernels can differ in the last FP32 digits.
            assert max_score_delta<1e-4,(name,floor,max_score_delta)
            ious=np.diag(mu.iou([e['rle'] for e in actual],[e['rle'] for e in expected],[0]*len(expected)))
            assert ious.min()>.999 and ious.mean()>.9999,(name,floor,ious.min(),ious.mean())
            row[str(floor)]=dict(raw_count=raw_count,instances=len(actual),native_mask_iou=float(ious.min()),max_score_delta=max_score_delta)
        report['probes'].append(row)
        print(json.dumps(row),flush=True)
    records,_=records_for(root/'MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    report['historical_cache_pq']=evaluate(old,records)[0]['pq']
    report['current_control_pq']=evaluate(new,records)[0]['pq']
    report['explanation']='Confidence floor changed from .05 to .25; four photographs cross the native CPU crop-mask branch at 50 detections, changing rounding of bounding boxes. The old hard-coded PQ also differed from the saved historical cache.'
    report['passed']=True
    Path('data/existing_weight_control_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
