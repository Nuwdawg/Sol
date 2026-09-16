"""Screen D4 view-support scoring while preserving the existing fused masks."""
import json
from pathlib import Path
import time

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import evaluate_thresholds, records_for


def main():
    cv2.setNumThreads(1)
    output = Path('data/exp050_d4_support'); output.mkdir(exist_ok=True)
    cache = json.loads(Path('data/exp033_cached_tta/candidates_0.5_0.5.json').read_text())
    views = json.loads(Path('data/caches/exp007/outputs/views.json').read_text())
    records, split = records_for(Path('data/MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
    assert set(cache) == set(views) == set(records)
    # Reconstruct the deployed seed clustering, then verify every original score.
    # Existing masks remain byte-identical; only confidence ranking is screened.
    from experiments.exp_033_inference.src.cached_tta import vote_rle
    for name, entries in views.items():
        order = sorted(range(len(entries)), key=lambda i: -entries[i][0])
        overlaps = mu.iou([e[1] for e in entries], [e[1] for e in entries], [0]*len(entries))
        used, clustered = set(), []
        for seed in order:
            if seed in used:
                continue
            members = [seed]; used.add(seed)
            for i in order:
                if i not in used and overlaps[seed,i] >= .5:
                    members.append(i); used.add(i)
            mask = vote_rle([entries[i][1] for i in members], max(1,round(.5*len(members))))
            if mu.area(mask) < 40:
                continue
            mean = float(np.mean([entries[i][0] for i in members]))
            clustered.append((mean,len(members),mask))
        assert len(clustered) == len(cache[name]), name
        for entry, (mean,count,mask) in zip(cache[name],clustered):
            assert entry['rle'] == mask, name
            assert abs(entry['score']-mean*(.5+.5*count/8)) < 1e-10, name
            entry.update(mean=mean,support=count/8)
    started = time.perf_counter()
    results = []
    for policy in ('control','mean','linear_support'):
        path = output/f'{policy}.json'
        if path.exists():
            rows = json.loads(path.read_text())
        else:
            adjusted = {name:[dict(rle=e['rle'], score=(e['score'] if policy == 'control' else
                         e['mean']*(e['support'] if policy == 'linear_support' else 1)))
                         for e in entries] for name,entries in cache.items()}
            rows = evaluate_thresholds(adjusted, records, [.2,.25,.3,.35,.4,.45,.5,.55,.6],grow=-1,min_area=300)
            path.write_text(json.dumps(rows,indent=2))
        if policy == 'control':
            assert abs(next(r for r in rows if r['conf'] == .4)['pq']-.45707595553373687) < 1e-10
        results.extend(dict(row,policy=policy) for row in rows)
        print(json.dumps(dict(policy=policy,best=max(rows,key=lambda r:r['pq']),
                              elapsed=time.perf_counter()-started)),flush=True)
    (output/'summary.json').write_text(json.dumps(dict(split=split.summary(),sweep=results,
                                      best=max(results,key=lambda r:r['pq'])),indent=2))


if __name__ == '__main__':
    main()
