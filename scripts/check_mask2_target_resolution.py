"""Measure information retained by exp046 targets on training records only."""
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from shared.data_split import make_split


def main():
    cv2.setNumThreads(1)
    source = Path('data/MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    coco = json.loads(source.read_text())
    split = make_split(str(source))
    chosen = set(np.random.default_rng(2026).choice(split.train_image_ids,32,replace=False).tolist())
    assert chosen.isdisjoint(split.val_image_ids)
    records = {r['id']:r for r in coco['images']}
    rows = []
    for annotation in coco['annotations']:
        if annotation['image_id'] not in chosen:
            continue
        record = records[annotation['image_id']]
        seg = annotation['segmentation']
        if isinstance(seg,list):
            seg = mu.merge(mu.frPyObjects(seg,record['height'],record['width']))
        elif isinstance(seg['counts'],list):
            seg = mu.frPyObjects(seg,record['height'],record['width'])
        truth = mu.decode(seg)
        target = cv2.resize(truth.astype(np.float32),(256,256),interpolation=cv2.INTER_AREA)
        restored = cv2.resize(target,(record['width'],record['height']),interpolation=cv2.INTER_LINEAR)
        scores = {}
        for threshold in (.2,.35,.4,.5,.6):
            binary = restored >= threshold
            intersection = np.count_nonzero(binary & truth.astype(bool))
            union = int(binary.sum())+int(truth.sum())-intersection
            scores[str(threshold)] = intersection/max(1,union)
        rows.append(dict(annotation_id=annotation['id'],image_id=annotation['image_id'],
                         native_area=int(truth.sum()),peak=float(restored.max()),iou=scores))
    report = dict(note='Ideal area-target reconstruction on 32 training records; not model accuracy',
                  records=sorted(chosen),instances=len(rows),target_size=256,
                  thresholds={str(t):dict(mean_iou=float(np.mean([r['iou'][str(t)] for r in rows])),
                     fraction_above_half=float(np.mean([r['iou'][str(t)]>.5 for r in rows])))
                     for t in (.2,.35,.4,.5,.6)},rows=rows)
    output = Path('data/exp051_target_resolution'); output.mkdir(exist_ok=True)
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','records')}),flush=True)


if __name__ == '__main__':
    main()
