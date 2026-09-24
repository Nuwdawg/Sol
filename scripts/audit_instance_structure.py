"""Diagnostic pair-union opportunities; labels are used only for validation audit.

Oracle unions are not deployable predictions or a claimed attainable score.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from pycocotools import mask as mu
from experiments.exp_033_inference.src.sweep import records_for, pq_rle
from shared.utils import aggregate_pq


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', required=True)
    p.add_argument('--annotations', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    predictions = json.loads(Path(a.predictions).read_text())
    records, _ = records_for(a.annotations)
    assert set(predictions) == set(records)
    totals = dict(missed_without_overlap=0, missed_with_overlap=0,
                  recoverable_by_pair_union=0, possible_merged_predictions=0)
    metrics, examples = [], []
    for name, passes in records.items():
        pred = predictions[name]
        for iid, truth in passes:
            metrics.append(pq_rle(pred, truth))
            iou = np.asarray(mu.iou(pred, truth, [0]*len(truth))) if pred and truth else np.zeros((len(pred),len(truth)))
            matched_p, matched_g = set(), set()
            for i,j in sorted(np.argwhere(iou>.5), key=lambda x:-iou[tuple(x)]):
                if i not in matched_p and j not in matched_g:
                    matched_p.add(int(i)); matched_g.add(int(j))
            for j in range(len(truth)):
                if j in matched_g: continue
                candidates = [i for i in range(len(pred)) if iou[i,j]>.05 and i not in matched_p]
                totals['missed_with_overlap' if np.any(iou[:,j]>0) else 'missed_without_overlap'] += 1
                best, pair = 0., None
                for i,k in itertools.combinations(candidates,2):
                    score = float(mu.iou([mu.merge([pred[i],pred[k]])],[truth[j]],[0])[0,0])
                    if score>best: best,pair=score,(i,k)
                if best>.5:
                    totals['recoverable_by_pair_union'] += 1
                    examples.append(dict(photograph=name,record=iid,truth=j,pair=pair,union_iou=best))
            for i in range(len(pred)):
                if i in matched_p: continue
                candidates=[j for j in range(len(truth)) if j not in matched_g and iou[i,j]>.05]
                if any(float(mu.iou([pred[i]],[mu.merge([truth[j],truth[k]])],[0])[0,0])>.5
                       for j,k in itertools.combinations(candidates,2)):
                    totals['possible_merged_predictions'] += 1
    report=dict(baseline=aggregate_pq(metrics),counts=totals,examples=examples,
                interpretation='Label-assisted diagnostic only; pair opportunities can overlap and are not an achievable score forecast.')
    Path(a.output).write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='examples'}),flush=True)


if __name__=='__main__': main()
