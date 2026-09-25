"""Observe already-launched jobs, download and summarize; never relaunch or submit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
CLI=Path(sys.executable).with_name('kaggle.exe' if os.name=='nt' else 'kaggle')
env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')

def main():
    jobs=json.loads((ROOT/'data/existing_weight_jobs.json').read_text())
    state_file=ROOT/'data/existing_weight_monitor.json'
    states=json.loads(state_file.read_text()).get('jobs',{}) if state_file.exists() else {}
    while True:
        for job in jobs:
            slug=job['slug']
            if states.get(slug,{}).get('downloaded'):
                recovered=Path(job['output'])/'results/recovered_report.json'
                if recovered.exists():
                    r=json.loads(recovered.read_text())
                    states[slug]['result']=dict(status=r['status'],control=r['control_audit'],
                                               best_variant=r['best_variant'],
                                               scores={k:v['best']['pq'] for k,v in r['variants'].items()})
                continue
            state=dict(slug=slug,checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),downloaded=False)
            try:
                result=subprocess.run([str(CLI),'kernels','status',slug],capture_output=True,text=True,encoding='utf-8',env=env,timeout=60)
                state['status']=result.stdout.strip()
                if result.returncode:
                    state['error']=result.stderr[-1500:]
                if any(f'{s}"' in state['status'].lower() for s in ['complete','error','cancelled']):
                    result=subprocess.run([sys.executable,'-u','-m','scripts.download_kernel',slug,job['output']],
                                          cwd=ROOT,capture_output=True,text=True,encoding='utf-8',env=env,timeout=900)
                    state['downloaded']=result.returncode==0
                    if result.returncode:
                        state['error']=result.stderr[-1500:]
                    report=Path(job['output'])/'results/report.json'
                    if report.exists():
                        r=json.loads(report.read_text())
                        state['result']=dict(status=r['status'],control=r.get('historical_control'),
                                             best_variant=r.get('best_variant'),
                                             scores={k:v['best']['pq'] for k,v in r['variants'].items()})
            except Exception as error:
                state['error']=f'{type(error).__name__}: {error}'
            states[slug]=state
            print(json.dumps(state),flush=True)
            target=ROOT/'data/existing_weight_monitor.json'
            temp=target.with_suffix('.tmp')
            temp.write_text(json.dumps(dict(pid=os.getpid(),jobs=states),indent=2))
            temp.replace(target)
        temp=state_file.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(pid=os.getpid(),jobs=states),indent=2))
        temp.replace(state_file)
        if all(s.get('downloaded') for s in states.values()) and len(states)==len(jobs):
            return
        time.sleep(180)

if __name__=='__main__':
    main()
