"""Generate authorized public probes using frozen, validated inference rules."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from pycocotools import mask as mu
from ultralytics import YOLO
from scripts.existing_weight_trials import BASE_SHA, infer, mask_nms, soft_fuse, interpolate, TRANSFORMS, dump
from shared.utils import paint_panoptic, encode_rle, check_no_overlap

RULES={'nms':dict(conf=.35,grow=-1,min_area=300),
       'weights':dict(conf=.385,grow=-1,min_area=300),
       'soft':dict(conf=.385,grow=-1,min_area=300)}

def read_image(path):
    gray=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if gray is not None and gray.ndim==3: gray=gray[:,:,0]
    assert gray is not None and gray.shape==(2048,2048),str(path)
    return np.repeat(gray[:,:,None],3,axis=2)

def predict(model,image,mode,device):
    if mode=='nms':
        return mask_nms(infer(model,image,iou=1.,max_det=1000,device=device),.3)
    if mode=='weights': return infer(model,image,device=device)
    entries=[]
    for transform in TRANSFORMS:
        entries.extend(infer(model,image,soft=True,transform=transform,device=device))
    return soft_fuse(entries,(.4,))['0.4']

def main():
    p=argparse.ArgumentParser()
    for name in ('weights','donor','images','test-images','reference','output'): p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--device',default='cpu')
    a=p.parse_args(); torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assert hashlib.sha256(a.weights.read_bytes()).hexdigest()==BASE_SHA
    assert hashlib.sha256(a.donor.read_bytes()).hexdigest()=='c08091f0df8c6b294baf1c4e9debc77c9ba022eff73ebd5cefb363e9558690aa'
    test=sorted(a.test_images.glob('*.jpeg'));assert len(test)==180
    a.output.mkdir(parents=True,exist_ok=True)
    manifest=dict(training=False,device=a.device,checkpoint_sha256=BASE_SHA,photographs=180,variants={})
    dump(a.output/'resolved.json',manifest);print('RESOLVED',json.dumps(manifest),flush=True)
    for mode in ('nms','weights','soft'):
        started=time.monotonic(); model=YOLO(a.weights)
        if mode=='weights':
            donor=YOLO(a.donor).model.float()
            state=interpolate(model.model.float().state_dict(),donor.state_dict(),.25)
            model.model.load_state_dict(state,strict=True);del donor,state
        model.model.eval().requires_grad_(False)
        reference=json.loads((a.reference/f'{mode}.json').read_text())
        parity=[]
        for name,expected in reference.items():
            actual=predict(model,read_image(a.images/name),mode,a.device)
            expected=[e for e in expected if e['score']>=.30]
            actual=[e for e in actual if e['score']>=.30]
            assert len(expected)==len(actual),(mode,name,'proposal count differs')
            if expected:
                iou=np.asarray(mu.iou([e['rle'] for e in expected],[e['rle'] for e in actual],[0]*len(actual)))
                matches=iou.argmax(1)
                assert len(set(matches))==len(matches),(mode,name,'ambiguous match')
                overlap=iou[np.arange(len(expected)),matches]
                delta=max(abs(e['score']-actual[j]['score']) for e,j in zip(expected,matches))
                assert overlap.min()>.95 and overlap.mean()>.99 and delta<.001,(mode,name,overlap.min(),overlap.mean(),delta)
                parity.append(dict(photograph=name,min_iou=float(overlap.min()),mean_iou=float(overlap.mean()),max_score_delta=delta))
        dump(a.output/f'{mode}_parity.json',parity)
        print('PARITY',mode,json.dumps(parity),flush=True)
        rule=RULES[mode];cache={};counts={}
        target=a.output/f'{mode}_submission.csv'
        with target.open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['filament_id','segmentation_rle'])
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            for i,path in enumerate(test,1):
                entries=predict(model,read_image(path),mode,a.device)
                cache[path.name]=entries
                candidates=[(e['score'],cv2.erode(mu.decode(e['rle']),kernel)) for e in entries if e['score']>=rule['conf']]
                painted=paint_panoptic(candidates,min_area=rule['min_area'])
                counts[path.stem]=len(painted)
                for j,(_,mask,_) in enumerate(painted,1): writer.writerow([f'{path.stem}_{j}',encode_rle(mask)])
                if i%10==0:
                    print('PROGRESS',mode,i,180,round(time.monotonic()-started,1),flush=True)
                    dump(a.output/f'{mode}_candidates.json',cache)
        check_no_overlap(str(target))
        dump(a.output/f'{mode}_candidates.json',cache)
        manifest['variants'][mode]=dict(rule=rule,counts=counts,rows=sum(counts.values()),
                                        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),seconds=time.monotonic()-started)
        dump(a.output/'submission_manifest.json',manifest)
        print('COMPLETE',mode,manifest['variants'][mode]['rows'],flush=True)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__=='__main__':main()
