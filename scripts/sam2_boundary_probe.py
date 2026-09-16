"""Controlled validation of pretrained SAM2 crop refinement; no test labels/training."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from pycocotools import mask as mu
import torch
from ultralytics.models.sam import SAM2Predictor
from experiments.exp_033_inference.src.sweep import encode, evaluate, records_for


def crop_prompt(mask):
    yy,xx=np.nonzero(mask)
    side=min(2048,max(128,int(max(np.ptp(xx),np.ptp(yy))*1.4)+64))
    x0=int(np.clip((int(xx.max())+int(xx.min())-side)//2,0,2048-side))
    y0=int(np.clip((int(yy.max())+int(yy.min())-side)//2,0,2048-side))
    seed=mask[y0:y0+side,x0:x0+side]
    y,x=np.nonzero(seed)
    coords=np.column_stack((x,y)).astype(float)
    axis=np.linalg.svd(coords-coords.mean(0),full_matrices=False)[2][0]
    position=(coords-coords.mean(0))@axis
    distance=cv2.distanceTransform(seed,cv2.DIST_L2,5)[y,x]
    points=[]
    for quantile in (.15,.5,.85):
        delta=np.abs(position-np.quantile(position,quantile))
        eligible=delta<=max(2,float(np.quantile(delta,.12)))
        index=np.argmax(np.where(eligible,distance,-1))
        points.append([int(x[index]),int(y[index])])
    box=[max(0,int(x.min())-4),max(0,int(y.min())-4),min(side,int(x.max())+5),min(side,int(y.max())+5)]
    return x0,y0,side,seed,box,points


def main():
    p=argparse.ArgumentParser()
    for key in ('cache','annotations','images','output'):p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--device',default='0');a=p.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a.output.mkdir(parents=True,exist_ok=True)
    records,split=records_for(a.annotations);source=json.loads(a.cache.read_text())
    assert set(records)==set(source) and len(records)==106
    names=sorted(records);screen=[names[i] for i in np.linspace(0,len(names)-1,24,dtype=int)]
    report=dict(model='sam2.1_b.pt',training=False,device=a.device,split=split.summary(),
                prompt_source='existing D4 predictions only',conf=.35,min_area=100,
                screen_photographs=screen,gate_absolute_pq_gain=.005,
                annotation_sha256=hashlib.sha256(a.annotations.read_bytes()).hexdigest(),
                guard=dict(min_seed_iou=.3,min_area_ratio=.25,max_area_ratio=4),results={})
    (a.output/'resolved.json').write_text(json.dumps(report,indent=2));print('RESOLVED',json.dumps(report),flush=True)
    predictor=SAM2Predictor(overrides=dict(model='sam2.1_b.pt',task='segment',mode='predict',
                                         imgsz=1024,device=a.device,verbose=False,save=False))
    cache={kind:{} for kind in ('box','box_points')};events=[]
    started=time.monotonic()
    def infer_names(selected):
        for index,name in enumerate(selected,1):
            gray=cv2.imread(str(a.images/name),cv2.IMREAD_GRAYSCALE)
            if gray.ndim==3:gray=gray[:,:,0]
            assert gray.shape==(2048,2048)
            entries=[e for e in source[name] if e['score']>=.35]
            for kind in cache:cache[kind][name]=[]
            for number,entry in enumerate(entries):
                mask=mu.decode(entry['rle']).astype(np.uint8)
                if not mask.any():continue
                x,y,side,seed,box,points=crop_prompt(mask)
                image=np.repeat(gray[y:y+side,x:x+side,None],3,axis=2)
                predictor.reset_image();predictor.set_image(image)
                if 'sam_weights_sha256' not in report:
                    report['sam_weights_sha256']=hashlib.sha256(Path('sam2.1_b.pt').read_bytes()).hexdigest()
                    report['torch']=torch.__version__
                    report['accelerator']=torch.cuda.get_device_name(0)
                    (a.output/'resolved.json').write_text(json.dumps(report,indent=2))
                for kind in cache:
                    kwargs=dict(bboxes=[box])
                    if kind=='box_points':kwargs.update(points=[points],labels=[[1,1,1]])
                    result=predictor(**kwargs)[0]
                    refined=mask.copy();accepted=False;overlap=0.;ratio=0.
                    if result.masks is not None and len(result.masks.data)==1:
                        crop=result.masks.data[0].cpu().numpy().astype(np.uint8)
                        assert crop.shape==seed.shape,(crop.shape,seed.shape)
                        intersection=np.count_nonzero(crop&seed);union=np.count_nonzero(crop|seed)
                        overlap=intersection/max(1,union);ratio=float(crop.sum())/max(1,int(seed.sum()))
                        accepted=overlap>=.3 and .25<=ratio<=4
                        if accepted:
                            refined=np.zeros_like(mask);refined[y:y+side,x:x+side]=crop
                    cache[kind][name].append(dict(score=entry['score'],rle=encode(refined)))
                    events.append(dict(photograph=name,instance=number,method=kind,accepted=accepted,seed_iou=overlap,area_ratio=ratio))
            for kind in cache:(a.output/f'{kind}_candidates.json').write_text(json.dumps(cache[kind]))
            (a.output/'refinement_events.json').write_text(json.dumps(events))
            print('PROGRESS',len(cache['box']),106,round(time.monotonic()-started,1),flush=True)
    def score(selected,label):
        subset={n:records[n] for n in selected}
        baseline=evaluate(source,subset,conf=.35,grow=-1,min_area=100)[0]
        rows=[]
        for kind in cache:
            for grow in (-1,0):
                row=evaluate(cache[kind],subset,conf=.35,grow=grow,min_area=100)[0]
                rows.append(dict(row,method=kind))
        result=dict(photographs=len(selected),baseline=baseline,variants=rows,best=max(rows,key=lambda r:r['pq']))
        report['results'][label]=result
        (a.output/'report.json').write_text(json.dumps(report,indent=2));print('RESULT',label,json.dumps(result),flush=True)
        return result
    infer_names(screen);result=score(screen,'screen')
    if result['best']['pq']>=result['baseline']['pq']+.005:
        infer_names([n for n in names if n not in screen]);score(names,'full_validation')
    report['complete']=True;report['seconds']=time.monotonic()-started
    report['full_validation']=len(cache['box'])==106
    (a.output/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
