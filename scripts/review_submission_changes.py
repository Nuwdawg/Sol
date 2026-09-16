"""Native-image review panels for changed instances; no test labels are used."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from pycocotools import mask as mu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(path):
    result=defaultdict(list)
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            stem=row['filament_id'].rsplit('_',1)[0]
            result[stem].append(dict(size=[2048,2048],counts=row['segmentation_rle']))
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ('baseline','candidate','images','output'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();cv2.setNumThreads(1);a.output.mkdir(parents=True,exist_ok=True)
    old,new=load(a.baseline),load(a.candidate)
    added=[];removed=[];changed=[]
    for stem in sorted(set(old)|set(new)):
        x,y=old[stem],new[stem]
        iou=np.asarray(mu.iou(y,x,[0]*len(x))) if x and y else np.zeros((len(y),len(x)))
        for i,rle in enumerate(y):
            overlap=float(iou[i].max()) if x else 0.
            row=dict(stem=stem,index=i,area=int(mu.area(rle)),best_other_iou=overlap,kind='added' if overlap<=.5 else 'boundary_changed')
            if overlap<=.5:added.append(row)
            elif overlap<1:changed.append(row)
        for i,rle in enumerate(x):
            overlap=float(iou[:,i].max()) if y else 0.
            if overlap<=.5:removed.append(dict(stem=stem,index=i,area=int(mu.area(rle)),best_other_iou=overlap,kind='removed'))
    added.sort(key=lambda r:-r['area']);removed.sort(key=lambda r:-r['area']);changed.sort(key=lambda r:-r['area'])
    selected=[]
    # Predetermined sampling across sizes, plus removals: avoids only showing wins.
    for pool in (added,removed):
        if pool:
            for i in sorted(set((0,len(pool)//2,len(pool)-1))):selected.append(pool[i])
    if changed:selected.append(changed[0])
    fig,axes=plt.subplots(len(selected),3,figsize=(12,4*len(selected)),squeeze=False)
    for row,(sample,axs) in enumerate(zip(selected,axes)):
        stem=sample['stem'];source=old if sample['kind']=='removed' else new
        chosen=mu.decode(source[stem][sample['index']])
        yy,xx=np.nonzero(chosen)
        side=int(min(2048,max(320,max(xx.max()-xx.min(),yy.max()-yy.min())+100)))
        x0=int(np.clip((int(xx.max())+int(xx.min())-side)//2,0,2048-side))
        y0=int(np.clip((int(yy.max())+int(yy.min())-side)//2,0,2048-side))
        image=cv2.imread(str(a.images/(stem+'.jpeg')),cv2.IMREAD_GRAYSCALE)
        assert image is not None and image.shape==(2048,2048)
        crop=image[y0:y0+side,x0:x0+side]
        lo,hi=np.percentile(crop,(1,99))
        for ax,title in zip(axs,('Observation','Previous submission','Candidate submission')):
            ax.imshow(crop,cmap='gray',vmin=lo,vmax=max(hi,lo+1),interpolation='nearest')
            ax.set_title(title,fontsize=10);ax.set_axis_off()
        for ax,entries in ((axs[1],old[stem]),(axs[2],new[stem])):
            for rle in entries:
                mask=mu.decode(rle)[y0:y0+side,x0:x0+side]
                if mask.any():ax.contour(mask,levels=[.5],colors=['#00ffff'],linewidths=.65)
        axs[0].text(0,-.04,f"{sample['kind']} | {stem} | area {sample['area']} px",transform=axs[0].transAxes,fontsize=8)
        sample['crop']=dict(x=x0,y=y0,size=side)
        # Save each strip separately for inspection without resizing a long sheet.
        fig.canvas.draw()
    fig.suptitle('Changed-mask review: identical display contrast per row; contours only, no test labels',fontsize=11)
    fig.tight_layout(rect=(0,.01,1,.98))
    fig.canvas.draw()
    fig.savefig(a.output/'contact_sheet.png',dpi=160)
    for i,axs in enumerate(axes):
        bbox=matplotlib.transforms.Bbox.union([ax.get_tightbbox(fig.canvas.get_renderer()) for ax in axs])
        bbox=bbox.transformed(fig.dpi_scale_trans.inverted()).expanded(1.02,1.05)
        fig.savefig(a.output/f'case_{i+1}.png',dpi=160,bbox_inches=bbox)
    plt.close(fig)
    report=dict(baseline=str(a.baseline),candidate=str(a.candidate),
                added_instances=len(added),removed_instances=len(removed),changed_boundaries=len(changed),
                added=added,removed=removed,boundary_changes=changed,selected=selected,
                note='IoU<=0.5 defines unmatched instances for sampling; visual observations are not test ground truth.')
    (a.output/'review_manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('added_instances','removed_instances','changed_boundaries','selected')}),flush=True)

if __name__=='__main__':main()
