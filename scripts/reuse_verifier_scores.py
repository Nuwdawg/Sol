"""Bounded CPU screen: reuse frozen crop scores with photo-held-out calibration.

The old verifier checkpoint was selected on canonical validation AP, so this is
cross-fitting of calibration only, not a new independent validation set.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mu
from scipy.optimize import minimize
from scipy.special import expit

from experiments.exp_025_verifier.src.model import Verifier
from experiments.exp_025_verifier.src.harvest import crop_of
from experiments.exp_033_inference.src.sweep import records_for, encode, pq_rle
from shared.utils import aggregate_pq, paint_panoptic


def calibrated_probabilities(train_x, train_y, test_x):
    mean=train_x.mean(axis=0);scale=np.maximum(train_x.std(axis=0),1e-8)
    design=np.column_stack([np.ones(len(train_x)),(train_x-mean)/scale])
    heldout=np.column_stack([np.ones(len(test_x)),(test_x-mean)/scale])
    def objective(w):
        logits=design@w
        penalty=w.copy();penalty[0]=0
        value=np.sum(np.logaddexp(0,logits)-train_y*logits)+.5*np.dot(penalty,penalty)
        gradient=design.T@(expit(logits)-train_y)+penalty
        return value,gradient
    result=minimize(objective,np.zeros(design.shape[1]),jac=True,method='L-BFGS-B')
    assert result.success,result.message
    return expit(heldout@result.x)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('weights','cache','annotations','images','output'):
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);cv2.setNumThreads(1)
    checkpoint=torch.load(a.weights,map_location='cpu',weights_only=False)
    model=Verifier(checkpoint.get('width',24));model.load_state_dict(checkpoint['state']);model.eval()
    cache=json.loads(a.cache.read_text());records,split=records_for(a.annotations)
    assert len(records)==106 and set(cache)==set(records)
    names=sorted(records);pool={};features=[];labels=[];photo_ids=[];entry_rows={}
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    for pi,name in enumerate(names):
        image=cv2.imread(str(a.images/name),cv2.IMREAD_GRAYSCALE)
        assert image is not None and image.shape==(2048,2048)
        # Freeze paint priority to the original detector. Calibration only
        # selects from this disjoint pool; it never invents mask boundaries.
        painted=paint_panoptic([(e['score'],cv2.erode(mu.decode(e['rle']),kernel))
                                for e in cache[name]],min_area=300)
        entries=[];patches=[]
        for score,mask,area in painted:
            patch,patch_mask=crop_of(image,mask)
            patches.append(np.stack([patch.astype(np.float32)/255,patch_mask.astype(np.float32)]))
            entries.append(dict(score=float(score),rle=encode(mask),area=area))
        with torch.inference_mode():
            logits=np.concatenate([model(torch.from_numpy(np.stack(patches[i:i+32]))).numpy()
                                   for i in range(0,len(patches),32)]) if patches else []
        for e,v in zip(entries,logits):e['verifier_logit']=float(v)
        pool[name]=entries
        entry_rows[name]=[]
        for iid,truth in records[name]:
            iou=np.asarray(mu.iou([e['rle'] for e in entries],truth,[0]*len(truth))) if entries and truth else np.zeros((len(entries),len(truth)))
            y=iou.max(axis=1)>.5 if truth else np.zeros(len(entries),bool)
            rows=[]
            for e,target in zip(entries,y):
                s=np.clip(e['score'],1e-5,1-1e-5)
                rows.append(len(features));features.append([np.log(s/(1-s)),np.clip(e['verifier_logit'],-12,12)])
                labels.append(int(target));photo_ids.append(pi)
            entry_rows[name].append(rows)
        if (pi+1)%20==0:print('SCORED',pi+1,flush=True)
    (a.output/'pool.json').write_text(json.dumps(pool))
    x=np.asarray(features);y=np.asarray(labels);groups=np.asarray(photo_ids)
    predicted={mode:np.full(len(y),np.nan) for mode in ('confidence','confidence_verifier')}
    folds=[]
    partitions=np.array_split(np.random.default_rng(2026).permutation(len(names)),5)
    for test in partitions:
        train=np.setdiff1d(np.arange(len(names)),test)
        fit=np.isin(groups,train);hold=np.isin(groups,test)
        assert not set(groups[fit])&set(groups[hold])
        for mode,nfeatures in [('confidence',1),('confidence_verifier',2)]:
            predicted[mode][hold]=calibrated_probabilities(x[fit,:nfeatures],y[fit],x[hold,:nfeatures])
        folds.append(dict(train=[names[i] for i in train],heldout=[names[i] for i in test]))
    for name in names:
        for mode in predicted:
            rows=entry_rows[name]
            probs=predicted[mode][rows[0]]
            for other in rows[1:]:np.testing.assert_allclose(probs,predicted[mode][other])
            for e,v in zip(pool[name],probs):e[mode]=float(v)

    def evaluate(mode,threshold,promote=False):
        scores=[]
        for name in names:
            emitted=[e['rle'] for e in pool[name] if e[mode]>=threshold or (promote and e['score']>=.35)]
            scores.extend(pq_rle(emitted,truth) for _,truth in records[name])
        return dict(aggregate_pq(scores),mode=mode,threshold=threshold,promotion_only=promote)

    baseline=evaluate('score',.35)
    assert abs(baseline['pq']-.455863866)<1e-6,baseline
    results=[]
    for mode in predicted:
        for threshold in (.30,.35,.40,.45,.50,.60,.70,.80,.90):
            for promote in (False,True):results.append(evaluate(mode,threshold,promote))
    report=dict(baseline=baseline,results=results,best=max(results,key=lambda r:r['pq']),
                split=split.summary(),folds=folds,
                checkpoint_sha256=hashlib.sha256(a.weights.read_bytes()).hexdigest(),
                limitation='Verifier was previously selected on validation AP. Only calibration is photo-held-out. Threshold sweep is exploratory.',
                status='complete')
    (a.output/'pool.json').write_text(json.dumps(pool))
    (a.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(baseline=baseline,best=report['best'])),flush=True)


if __name__=='__main__':main()
