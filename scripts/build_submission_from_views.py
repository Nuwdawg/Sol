"""Build a submission from cached test-view predictions."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np, csv
from pycocotools import mask as mu
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utils import paint_panoptic, encode_rle
from experiments.exp_033_inference.src.cached_tta import fuse

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--views', nargs='+', required=True); p.add_argument('--output', required=True)
    p.add_argument('--conf', type=float, default=.4); p.add_argument('--grow', type=int, default=-1); p.add_argument('--min-area', type=int, default=300)
    a=p.parse_args(); kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3));
    views={}
    for path in a.views:
        data=json.loads(Path(path).read_text())
        if set(data).intersection(views):
            raise ValueError('Test-view shards overlap')
        views.update(data)
    if len(views)!=180:
        raise ValueError(f'Expected all 180 test photographs, found {len(views)}')
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    target=Path(a.output)
    temporary=target.with_suffix(target.suffix+'.tmp')
    rows=0
    with temporary.open('w', newline='') as fh:
        out=csv.writer(fh); out.writerow(['filament_id','segmentation_rle'])
        for name, vals in views.items():
            cand=[]
            fused = fuse(vals, .5, .5)
            for entry in fused:
                score, rle = entry['score'], entry['rle']
                if float(score)<a.conf: continue
                m=mu.decode({'size':rle['size'],'counts':rle['counts'].encode('ascii')})
                if a.grow<0: m=cv2.erode(m,kernel,iterations=-a.grow)
                elif a.grow>0: m=cv2.dilate(m,kernel,iterations=a.grow)
                cand.append((float(score),m))
            for i, (_,m,_) in enumerate(paint_panoptic(cand,min_area=a.min_area), 1):
                out.writerow([f'{Path(name).stem}_{i}', encode_rle(m)])
                rows += 1
    temporary.replace(target)
    print(json.dumps({'rows':rows,'images':len(views),'conf':a.conf,'grow':a.grow,'min_area':a.min_area}))

if __name__=='__main__': main()
