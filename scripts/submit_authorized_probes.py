"""Finish explicitly authorized exp063 submissions; never launches compute."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
COMP='filament-segmentation-2026'
SLUG='cosmicwitness/sol-exp063-submission-probes'
OUT=ROOT/'data/exp063_v2_output'
STATE=ROOT/'data/exp063_submission_state_v2.json'
ENV=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
CLI=Path(sys.executable).with_name('kaggle.exe' if os.name=='nt' else 'kaggle')
SPECS=[('nms','probes/nms_submission.csv'),('weights','probes/weights_submission.csv'),
       ('soft','probes/soft_submission.csv'),('mask2','mask2/submission.csv')]

def save(state):
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(STATE)

def validate(source,kind):
    import cv2
    from shared.utils import check_no_overlap
    cv2.setNumThreads(1)
    with source.open(newline='') as f:
        reader=csv.DictReader(f)
        assert reader.fieldnames==['filament_id','segmentation_rle']
        rows=list(reader)
    names={p.stem for p in (ROOT/'data/competition/unpacked/MAGFiLO_1.0_Kaggle_2026/test/test_images').glob('*.jpeg')}
    assert len(names)==180 and rows
    assert len({r['filament_id'] for r in rows})==len(rows)
    assert all(r['filament_id'].rsplit('_',1)[0] in names and r['segmentation_rle'] for r in rows)
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    if kind=='mask2':
        report=json.loads((OUT/'mask2/submission_report.json').read_text())
        assert report['submission_sha256']==sha and report['photographs']==180
    else:
        manifest=json.loads((OUT/'probes/submission_manifest.json').read_text())
        report=manifest['variants'][kind]
        assert manifest['training'] is False and manifest['photographs']==180
        assert report['sha256']==sha and set(report['counts'])==names
        parity=json.loads((OUT/f'probes/{kind}_parity.json').read_text())
        assert len(parity)==4
    assert report['rows']==len(rows)
    check_no_overlap(str(source))
    return dict(sha256=sha,rows=len(rows))

def main():
    from kaggle import api
    from kagglesdk.competitions.types.competition_api_service import ApiGetSubmissionLimitsRequest
    state=json.loads(STATE.read_text()) if STATE.exists() else dict(slug=SLUG,downloaded=False,submissions={})
    state['pid']=os.getpid()
    while True:
        try:
            status=str(api.kernels_status(SLUG).status).upper()
            state['kernel_status']=status;state['checked_at']=time.strftime('%Y-%m-%d %H:%M:%S');save(state)
            if not any(s in status for s in ('COMPLETE','ERROR','CANCEL')):
                time.sleep(120);continue
            if not state['downloaded']:
                result=subprocess.run([sys.executable,'-u','-m','scripts.download_kernel',SLUG,str(OUT)],
                                      cwd=ROOT,env=ENV,capture_output=True,text=True,encoding='utf-8',timeout=1200)
                if result.returncode:raise RuntimeError('Output download failed: '+result.stderr[-500:])
                state['downloaded']=True;save(state)
            submissions=api.competition_submissions(COMP)
            for kind,relative in SPECS:
                description=f'exp063 authorized public probe: {kind}; existing weights; validated CSV'
                matches=[s for s in submissions if s.description==description]
                if matches:
                    s=matches[0]
                    state['submissions'][kind]=dict(ref=s.ref,status=str(s.status),public_score=s.public_score)
                    save(state);continue
                source=OUT/relative
                if not source.exists():
                    state['submissions'][kind]=dict(status='missing_output');save(state);continue
                # Ambiguous network responses require readback, never blind reposts.
                previous=state['submissions'].get(kind,{})
                if previous.get('status')=='submitting':continue
                with api.build_kaggle_client() as client:
                    request=ApiGetSubmissionLimitsRequest();request.competition_name=COMP
                    allowance=client.competitions.competition_api_client.get_submission_limits(request)
                    if allowance.num_allowed_now<1:
                        state['waiting_for_submission_slot']=True;save(state);break
                verified=validate(source,kind)
                target=ROOT/f'data/exp063_{kind}.csv';target.write_bytes(source.read_bytes())
                state['submissions'][kind]=dict(status='submitting',description=description,**verified);save(state)
                result=subprocess.run([str(CLI),'competitions','submit','-c',COMP,'-f',str(target),'-m',description],
                                      cwd=ROOT,env=ENV,capture_output=True,text=True,encoding='utf-8',timeout=300)
                state['submissions'][kind]['cli_returncode']=result.returncode
                print(kind,result.stdout.strip(),result.stderr[-500:],flush=True);save(state)
                submissions=api.competition_submissions(COMP)
            save(state)
            if len(state['submissions'])==4 and all('COMPLETE' in s.get('status','').upper() or s.get('status')=='missing_output' for s in state['submissions'].values()):
                print(json.dumps(state),flush=True);return
        except Exception as error:
            state['last_error']=f'{type(error).__name__}: {error}';save(state)
            print(state['last_error'],flush=True)
        time.sleep(120)

if __name__=='__main__':main()
