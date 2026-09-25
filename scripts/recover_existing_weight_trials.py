"""Recover completed reports only after an independent native-control audit."""
import json
from pathlib import Path

audit_path=Path('data/existing_weight_control_audit.json')
audit=json.loads(audit_path.read_text())
assert audit['passed'] and len(audit['probes'])==4
expected=audit['current_control_pq']
control=json.loads(Path('data/exp060_output/results/box_control.json').read_text())
other=json.loads(Path('data/exp061_recovery/results/alpha_0.0.json').read_text())
assert control==other,'Independent baseline runs differ'
summary={}
for number,root in ((60,Path('data/exp060_output/results')),(61,Path('data/exp061_recovery/results'))):
    report=json.loads((root/'report.json').read_text())
    assert report['status']=='control_mismatch'
    assert report['full_validation'] and report['photographs']==106
    assert abs(report['historical_control']['actual']-expected)<1e-12
    report['status']='complete_recovered'
    report['control_audit']=dict(path=str(audit_path),expected=expected,passed=True,
                               explanation=audit['explanation'])
    report['original_kernel_status']='ERROR: completed computations, failed stale historical control check'
    report['historical_control']['superseded_by']='control_audit; original failure retained for provenance'
    (root/'recovered_report.json').write_text(json.dumps(report,indent=2))
    standard=Path(f'data/exp{number:03d}_output/results')
    standard.mkdir(parents=True,exist_ok=True)
    (standard/'recovered_report.json').write_text(json.dumps(report,indent=2))
    scores={k:v['best']['pq'] for k,v in report['variants'].items()}
    summary[str(number)]=dict(scores=scores,control=report['control'],best_variant=report['best_variant'],
                             report=str(root/'recovered_report.json'))
Path('data/existing_weight_recovery.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
