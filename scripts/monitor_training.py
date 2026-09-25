"""Bounded Kaggle polling and measured progression from probe to training.

No GPU kernels or submissions are launched here. A full training launch requires
the exact trainer to finish real-data TPU steps and checkpoint reload first.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
MONITOR = ROOT/'data/monitor'
LOG = MONITOR/'training-monitor.log'
STATE = MONITOR/'training-state.json'
BACKEND = 'tpu'
LOCK_NAME = 'training.lock'
SLUGS = {
    'diagnostic': 'cosmicwitness/sol-exp035-mask2-tpu-probe',
    'real_probe': 'cosmicwitness/sol-exp047-mask2-real-probe',
    'training': 'cosmicwitness/sol-exp046-mask2-train',
    'evaluation': 'cosmicwitness/sol-exp048-mask2-eval',
}
TERMINAL = {'COMPLETE', 'ERROR', 'CANCELLED', 'CANCELED'}
FOLDERS = dict(real_probe='kaggle/exp047_mask2_real_probe',
               training='kaggle/exp046_mask2_train',evaluation='kaggle/exp048_mask2_eval')
OUTPUTS = dict(real_probe='data/exp047_output_v2',training='data/exp046_output',
               evaluation='data/exp048_output')


def configure_cpu():
    global BACKEND, STATE, LOG, LOCK_NAME
    BACKEND = 'cpu'
    STATE = MONITOR/'cpu-training-state.json'
    LOG = MONITOR/'cpu-training-monitor.log'
    LOCK_NAME = 'cpu-training.lock'
    for key, number, suffix in [('real_probe',52,'probe'),('training',53,'train'),
                                ('evaluation',54,'eval')]:
        SLUGS[key] = f'cosmicwitness/sol-exp{number:03d}-mask2-cpu-{suffix}'
        FOLDERS[key] = f'kaggle/exp{number:03d}_mask2_cpu_{suffix}'
        OUTPUTS[key] = f'data/exp{number:03d}_output'


def log(message):
    with LOG.open('a', encoding='utf-8') as handle:
        handle.write(time.strftime('%Y-%m-%d %H:%M:%S')+' '+message+'\n')


def run_command(arguments, timeout=60):
    try:
        run = subprocess.run(arguments, cwd=ROOT, text=True,
                             encoding='utf-8', errors='replace', stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=timeout,
                             env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8'),
                             creationflags=subprocess.CREATE_NO_WINDOW)
        return run.returncode, run.stdout.strip()
    except subprocess.TimeoutExpired:
        return -1, 'Observation timed out; retry the same kernel'


def command(*arguments, timeout=60):
    return run_command(['kaggle',*arguments],timeout=timeout)


def status(slug):
    _, output = command('kernels', 'status', slug)
    match = re.search(r'KernelWorkerStatus\.([A-Z]+)', output)
    if match:
        return match[1]
    if '404' in output or 'not found' in output.lower():
        return 'MISSING'
    if 'Cannot access kernel' in output:
        return 'INACCESSIBLE'
    return 'UNKNOWN'


def download(slug, directory, required):
    marker = ROOT/directory/'.download-complete.json'
    target = ROOT/directory/required
    if marker.exists() and target.exists():
        verified = json.loads(marker.read_text())
        entry = verified.get('files',{}).get(required)
        if verified.get('slug') == slug and entry and target.stat().st_size == entry['bytes'] > 0:
            return True
    code, message = run_command(['python','-X','utf8','-m','scripts.download_kernel',slug,directory], timeout=600)
    log(message)
    return code == 0 and target.exists() and target.stat().st_size > 0


def finite_steps(report, count=3):
    seconds, losses = report.get('seconds', []), report.get('losses', [])
    return (report.get('status') == 'complete' and len(seconds) >= count and
            len(losses) >= count and all(math.isfinite(x) for x in losses) and
            all(math.isfinite(x) and x > 0 for x in seconds))


def real_probe_passes(report):
    digest = hashlib.sha256((ROOT/'kaggle/exp046_mask2_train/train.py').read_bytes()).hexdigest()
    return (finite_steps(report, count=4) and report.get('backend') == BACKEND and
            report.get('fp32_sampling') is True and
            report.get('bf16') is True and report.get('checkpointed_layers', 0) > 0 and
            report.get('size') == 1024 and report.get('smoke_model') is False and
            report.get('checkpoint_reload') is True and report.get('validation_forward') is True and
            report.get('code_sha256') == digest and
            report.get('loss') == 'Hungarian class+BCE+Dice' and
            max(report['seconds'][-2:]) < (25 if BACKEND == 'cpu' else 30))


def launch_once(key, folder, seconds, state):
    current = status(SLUGS[key])
    if current in {'QUEUED', 'RUNNING', 'COMPLETE'}:
        state[key+'_launched'] = True
        log(f'{key} already exists: {current}; preserving it')
        return True
    if current == 'INACCESSIBLE' and not state.get(key+'_launched'):
        # Kaggle reports permission denied for new private slugs. Verify absence
        # from the account's own search before creating a planned kernel.
        code, result = command('kernels', 'list', '--mine', '--search',
                               SLUGS[key].split('/')[-1], '--format', 'json')
        if code == 0:
            rows = [] if result.strip() == 'Not found' else json.loads(result)
            if not any(row.get('ref') == SLUGS[key] for row in rows):
                current = 'MISSING'
    if current != 'MISSING':
        log(f'{key} status {current}; no automatic replacement')
        return False
    code, result = command('kernels', 'push', '-p', folder, '-t', str(seconds), timeout=120)
    log(result)
    if code == 0 and 'successfully pushed' in result:
        state[key+'_launched'] = True
        return True
    return False


def tick(state):
    if not state.get('real_probe_launched'):
        if not state.get('diagnostic_reviewed'):
            state['stopped'] = True
            return 'diagnostic requires review before corrected probe launch'
        launch_once('real_probe', FOLDERS['real_probe'], 1800 if BACKEND == 'cpu' else 5400, state)
        return 'launching real-data probe'
    real_status = status(SLUGS['real_probe'])
    state['real_probe_status'] = real_status
    if real_status not in TERMINAL:
        return 'waiting for real-data probe'
    if not download(SLUGS['real_probe'], OUTPUTS['real_probe'], 'train_report.json'):
        if real_status != 'COMPLETE':
            state['stopped'] = True
            return 'real-data probe failed; logs downloaded for review'
        return 'waiting for real-data artifacts'
    report = json.loads((ROOT/OUTPUTS['real_probe']/'train_report.json').read_text())
    if not real_probe_passes(report):
        state['stopped'] = True
        return 'real-data probe failed throughput or lifecycle gate'
    if not state.get('training_launched'):
        launch_once('training', FOLDERS['training'], 39600, state)
        return 'launching measured training run'
    training_status = status(SLUGS['training'])
    state['training_status'] = training_status
    if training_status not in TERMINAL:
        return 'waiting for training'
    if not download(SLUGS['training'], OUTPUTS['training'], 'train_report.json'):
        if training_status != 'COMPLETE':
            state['stopped'] = True
            return 'training failed; logs downloaded for review'
        return 'waiting for training artifacts'
    if not state.get('evaluation_launched'):
        launch_once('evaluation', FOLDERS['evaluation'], 14400, state)
        return 'launching CPU validation and conditional CSV generation'
    evaluation_status = status(SLUGS['evaluation'])
    state['evaluation_status'] = evaluation_status
    if evaluation_status not in TERMINAL:
        return 'waiting for CPU evaluation'
    if not download(SLUGS['evaluation'], OUTPUTS['evaluation'], 'results/validation.json'):
        if evaluation_status != 'COMPLETE':
            state['stopped'] = True
            return 'CPU evaluation failed; inspect downloaded logs'
        return 'waiting for evaluation artifacts'
    report = json.loads((ROOT/OUTPUTS['evaluation']/'results/validation.json').read_text())
    state['validation_pq'] = report['best']['pq']
    fusion_dir = ROOT/OUTPUTS['evaluation']/'fusion'
    decision_path = fusion_dir/'decision.json'
    if not decision_path.exists():
        state['message'] = 'evaluating cached Mask2Former/D4 fusion'
        state['checked_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        STATE.write_text(json.dumps(state,indent=2),encoding='utf-8')
        code, message = run_command([str(ROOT/'.venv/Scripts/python.exe'),'-m',
            'scripts.evaluate_mask2_fusion','--mask2-results',
            str(ROOT/OUTPUTS['evaluation']/'results'),'--annotations',
            str(ROOT/'data/MAGFiLO_1.0_Annotations_kaggle2026_train.json'),'--yolo',
            str(ROOT/'data/exp033_cached_tta/candidates_0.5_0.5.json'),
            '--output',str(fusion_dir)],timeout=1800)
        log(message)
        if code != 0 or not decision_path.exists():
            state['stopped'] = True
            return 'model evaluation downloaded; cached fusion needs review'
    fusion = json.loads(decision_path.read_text())
    state['fusion_pq'] = fusion['best']['pq']
    state['stopped'] = True
    if fusion['improves_d4'] and fusion['best']['pq'] > report['best']['pq']:
        return 'fusion improves validation; build and verify its submission CSV next'
    if report.get('eligible_for_submission'):
        return 'validated CSV downloaded; submission review next'
    return 'model and fusion did not beat current validation best; new experiment needed'


def main():
    import msvcrt
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu',action='store_true')
    if parser.parse_args().cpu:
        configure_cpu()
    MONITOR.mkdir(parents=True, exist_ok=True)
    with (MONITOR/LOCK_NAME).open('a+b') as lock:
        lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return
        state = json.loads(STATE.read_text()) if STATE.exists() else {}
        if state.get('stopped'):
            log('Saved terminal state requires review; no automatic restart')
            return
        state['pid'] = os.getpid()
        log(f'monitor started pid={os.getpid()}')
        previous, heartbeat = None, 0.
        while not state.get('stopped'):
            try:
                message = tick(state)
            except Exception as exc:
                message = f'retry after {type(exc).__name__}: {exc}'
            state['message'] = message
            state['checked_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
            STATE.write_text(json.dumps(state, indent=2), encoding='utf-8')
            if message != previous or time.monotonic()-heartbeat >= 600:
                log(message); previous = message; heartbeat = time.monotonic()
            if not state.get('stopped'):
                time.sleep(60)


if __name__ == '__main__':
    main()
