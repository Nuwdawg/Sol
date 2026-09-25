"""Apply the existing canonical fusion sweep to completed Mask2Former validation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from scripts.evaluate_public_fusion import yolo_predictions
from experiments.exp_033_inference.src.sweep import records_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mask2-results',required=True,type=Path)
    parser.add_argument('--annotations',required=True,type=Path)
    parser.add_argument('--yolo',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    report = json.loads((args.mask2_results/'validation.json').read_text())
    if report.get('complete') is not True or report.get('validation_photographs') != 106:
        raise ValueError('Fusion requires completed canonical validation')
    if report['annotation_sha256'] != hashlib.sha256(args.annotations.read_bytes()).hexdigest():
        raise ValueError('Annotation source differs from model validation')
    best = report['best']
    if best['grow'] not in (-1,0) or best['min_area'] != 300:
        raise ValueError('Unexpected standalone deployment rule')
    cache = json.loads((args.mask2_results/f"candidates_{best['mask_threshold']}.json").read_text())
    records, _ = records_for(args.annotations)
    if set(cache) != set(records):
        raise ValueError('Secondary predictions differ from canonical photographs')
    predictions = yolo_predictions(cache,conf=best['conf'],grow=best['grow'])
    args.output.mkdir(parents=True,exist_ok=True)
    payload = args.output/'mask2_painted.json'
    payload.write_text(json.dumps(dict(variants={'candidate':predictions},
          checkpoint_sha256=report['checkpoint_sha256'],standalone_rule=best)),encoding='utf-8')
    destination = args.output/'fusion_results.json'
    subprocess.run([sys.executable,'-m','scripts.evaluate_public_fusion','--public',str(payload),
                    '--yolo',str(args.yolo),'--annotations',str(args.annotations),
                    '--output',str(destination)],check=True)
    rows = json.loads(destination.read_text())
    control = next(row for row in rows if row['name'] == 'yolo_d4_control')
    if abs(control['pq']-.45707595553373687) > 1e-10:
        raise ValueError(f'D4 control changed: {control}')
    (args.output/'decision.json').write_text(json.dumps(dict(
        checkpoint_sha256=report['checkpoint_sha256'],standalone_pq=best['pq'],
        control=control,best=rows[0],improves_d4=rows[0]['pq']>control['pq']),indent=2))


if __name__ == '__main__':
    main()
