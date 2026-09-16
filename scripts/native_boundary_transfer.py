"""Reuse native-model boundaries for matched existing detections; fixed emission rule."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from pycocotools import mask as mu
from experiments.exp_033_inference.src.sweep import encode,records_for,pq_rle
from shared.utils import paint_panoptic,aggregate_pq


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','donor','annotations','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1)
    base=json.loads(a.baseline.read_text());donor=json.loads(a.donor.read_text())
    records,_=records_for(a.annotations);assert set(base)==set(donor)==set(records)
    modes=('baseline','native','intersection','union')
    scores={m:[] for m in modes};caches={m:{} for m in modes};changes=[]
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    for n,name in enumerate(sorted(records)):
        b=[dict(score=e['score'],rle=encode(cv2.erode(mu.decode(e['rle']),kernel)))
           for e in base[name] if e['score']>=.35]
        d=[e for e in donor[name] if e['score']>=.25]
        ious=np.asarray(mu.iou([e['rle'] for e in b],[e['rle'] for e in d],[0]*len(d))) if b and d else np.zeros((len(b),len(d)))
        match={};used=set()
        for i,j in sorted(np.argwhere(ious>.6),key=lambda pair:-ious[tuple(pair)]):
            if i not in match and j not in used:
                match[int(i)]=int(j);used.add(int(j))
                changes.append(dict(photograph=name,base=int(i),donor=int(j),iou=float(ious[i,j])))
        # Decode each pool once per photograph. All modes retain base confidences.
        bm=[mu.decode(e['rle']) for e in b];dm=[mu.decode(e['rle']) for e in d]
        for mode in modes:
            entries=[]
            for i,e in enumerate(b):
                mask=bm[i]
                if i in match and mode!='baseline':
                    candidate=dm[match[i]]
                    mask=candidate if mode=='native' else (mask&candidate if mode=='intersection' else mask|candidate)
                entries.append((e['score'],mask))
            emitted=[encode(mask) for _,mask,_ in paint_panoptic(entries,min_area=300)]
            caches[mode][name]=emitted
            for iid,truth in records[name]:scores[mode].append(dict(pq_rle(emitted,truth),image_id=iid,photograph=name))
        if (n+1)%20==0:print('SCORED',n+1,flush=True)
    metrics={mode:aggregate_pq(scores[mode]) for mode in modes}
    assert abs(metrics['baseline']['pq']-.45586386872921264)<1e-8
    report=dict(status='complete',metrics=metrics,matched_instances=len(changes),
                settings=dict(base_conf=.35,donor_conf=.25,base_erosion=1,donor_erosion=0,min_area=300,match_iou=.6),
                best=max(metrics,key=lambda k:metrics[k]['pq']))
    for mode in modes:(a.output/f'{mode}.json').write_text(json.dumps(caches[mode]))
    (a.output/'records.json').write_text(json.dumps(scores))
    (a.output/'changes.json').write_text(json.dumps(changes))
    (a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
