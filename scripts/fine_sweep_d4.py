"""Fine local sweep of the existing D4 candidate cache."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.exp_033_inference.src.sweep import records_for, evaluate_thresholds

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True)
    p.add_argument('--annotations', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    cache = json.loads(Path(a.cache).read_text())
    records, _ = records_for(a.annotations)
    thresholds = [round(i / 1000, 3) for i in range(375, 396)]
    rows = []
    for min_area in (50, 75, 100, 150, 200, 250, 300, 400, 500):
        for grow in (-1,):
            rows.extend(evaluate_thresholds(cache, records, thresholds, grow=grow, min_area=min_area))
            for r in rows[-len(thresholds):]:
                r['min_area'] = min_area
        best = max((r for r in rows if r['min_area'] == min_area), key=lambda x: x['pq'])
        print(json.dumps(best), flush=True)
    rows.sort(key=lambda x: -x['pq'])
    Path(a.output).write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(rows[:20], indent=2))

if __name__ == '__main__':
    main()
