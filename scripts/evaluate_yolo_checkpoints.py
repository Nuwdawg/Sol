"""Compare trained checkpoints on canonical per-annotator PQ, then D4 the best.

Keeps native RLE predictions for visual review. Does not submit or use test labels.
"""
import argparse
import gc
import hashlib
from pathlib import Path
import time

import cv2
import torch
from ultralytics import YOLO

from scripts.existing_weight_trials import dump, infer
from experiments.exp_007_tta.src.dihedral import TRANSFORMS
from experiments.exp_033_inference.src.cached_tta import fuse
from experiments.exp_033_inference.src.sweep import records_for, evaluate_thresholds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', nargs='+', required=True)
    p.add_argument('--annotations', required=True)
    p.add_argument('--images', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='0')
    p.add_argument('--floor',type=float,default=.25)
    p.add_argument('--force-d4',action='store_true')
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    records, split = records_for(args.annotations)
    assert len(records) == 106 and len(split.val_image_ids) == 180
    report = dict(split=split.summary(), checkpoints={}, status='running',
                  annotation_sha256=hashlib.sha256(Path(args.annotations).read_bytes()).hexdigest())
    started = time.monotonic()

    def harvest(weights, transforms):
        model = YOLO(weights)
        cache = {}
        for i, name in enumerate(sorted(records)):
            image = cv2.imread(str(Path(args.images)/name))
            assert image is not None and image.shape[:2] == (2048, 2048), name
            entries = []
            for transform in transforms:
                entries.extend(infer(model, image, transform=transform, device=args.device,
                                     conf=args.floor,max_det=1000 if args.floor<.25 else 100))
            cache[name] = (entries if len(transforms) == 1 else
                           fuse([(e['score'], e['rle']) for e in entries], .5, .5))
            if (i+1) % 20 == 0:
                print('VALIDATION',Path(weights).name,len(transforms),i+1,flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return cache

    def score(cache):
        rows = []
        for grow in (-1, 0):
            for area in (100, 300):
                thresholds=tuple(c for c in (.05,.075,.10,.15,.20,.25,.30,.35,.40,.45,.50,.60) if c>=args.floor)
                rows.extend(evaluate_thresholds(cache, records,thresholds, grow=grow, min_area=area))
        return dict(best=max(rows, key=lambda r:r['pq']), sweep=rows)

    seen = set()
    for index, weights in enumerate(args.weights):
        sha = hashlib.sha256(Path(weights).read_bytes()).hexdigest()
        if sha in seen:
            continue
        seen.add(sha)
        key = f'{index}_{Path(weights).stem}'
        cache = harvest(weights, [(0, False)])
        dump(root/f'{key}_identity.json',cache)
        result = score(cache)
        result.update(weights=str(weights), sha256=sha)
        report['checkpoints'][key] = result
        dump(root/'report.json',report)
        print('IDENTITY_RESULT',key,result['best'],flush=True)
    best = max(report['checkpoints'],key=lambda k:report['checkpoints'][k]['best']['pq'])
    report['best_checkpoint'] = best
    # D4 may rescue a small single-view regression. Larger regressions stop here.
    if args.force_d4 or report['checkpoints'][best]['best']['pq'] >= .435:
        cache = harvest(report['checkpoints'][best]['weights'],TRANSFORMS)
        dump(root/'best_d4.json',cache)
        report['d4'] = score(cache)
        print('D4_RESULT',report['d4']['best'],flush=True)
    report.update(status='complete',seconds=time.monotonic()-started)
    dump(root/'report.json',report)


if __name__ == '__main__':
    main()
