import copy
import hashlib
import math
import json

from scripts import monitor_training as monitor


def valid_report():
    return dict(status='complete', steps=4, size=1024, backend='tpu', smoke_model=False,
                losses=[8., 7., 6., 5.], seconds=[400., 250., 2., 2.],
                bf16=True, checkpointed_layers=27, fp32_sampling=True,
                checkpoint_reload=True, validation_forward=True,
                code_sha256=hashlib.sha256((monitor.ROOT/'kaggle/exp046_mask2_train/train.py').read_bytes()).hexdigest(),
                loss='Hungarian class+BCE+Dice')


def test_launch_gate_rejects_wrong_code_bad_losses_and_slow_training():
    good = valid_report()
    assert monitor.real_probe_passes(good)
    for change in [dict(code_sha256='stale'), dict(losses=[8., math.nan, 6.]),
                   dict(seconds=[400., 250., 31., 2.]), dict(backend='cpu'),
                   dict(bf16=False), dict(checkpointed_layers=0),
                   dict(checkpoint_reload=False), dict(smoke_model=True), dict(size=256)]:
        bad = copy.deepcopy(good); bad.update(change)
        assert not monitor.real_probe_passes(bad)


def test_running_kernel_is_never_pushed_again(monkeypatch):
    monkeypatch.setattr(monitor, 'status', lambda slug: 'RUNNING')
    monkeypatch.setattr(monitor, 'log', lambda message: None)
    monkeypatch.setattr(monitor, 'command', lambda *a, **k: (_ for _ in ()).throw(AssertionError('duplicate push')))
    state = {}
    assert monitor.launch_once('training', 'unused', 1, state)
    assert state['training_launched']


def test_observation_failure_does_not_restart_job(monkeypatch):
    monkeypatch.setattr(monitor, 'status', lambda slug: 'UNKNOWN')
    monkeypatch.setattr(monitor, 'log', lambda message: None)
    monkeypatch.setattr(monitor, 'command', lambda *a, **k: (_ for _ in ()).throw(AssertionError('unverified push')))
    assert not monitor.launch_once('training', 'unused', 1, {})


def test_cpu_requires_cpu_report_and_faster_steps(monkeypatch):
    monkeypatch.setattr(monitor,'BACKEND','cpu')
    report = valid_report()
    assert not monitor.real_probe_passes(report)
    report['backend'] = 'cpu'
    assert monitor.real_probe_passes(report)
    report['seconds'][-1] = 25
    assert not monitor.real_probe_passes(report)


def test_cpu_pipeline_uses_distinct_jobs_state_and_lock(monkeypatch):
    for key in ('SLUGS','FOLDERS','OUTPUTS'):
        monkeypatch.setattr(monitor,key,getattr(monitor,key).copy())
    for key in ('BACKEND','STATE','LOG','LOCK_NAME'):
        monkeypatch.setattr(monitor,key,getattr(monitor,key))
    monitor.configure_cpu()
    assert monitor.SLUGS['training'].endswith('exp053-mask2-cpu-train')
    assert monitor.OUTPUTS['real_probe'] == 'data/exp052_output'
    assert monitor.FOLDERS['evaluation'] == 'kaggle/exp054_mask2_cpu_eval'
    assert monitor.STATE.name == 'cpu-training-state.json'
    assert monitor.LOCK_NAME == 'cpu-training.lock'


def test_empty_or_unverified_artifact_is_downloaded_again(monkeypatch,tmp_path):
    monkeypatch.setattr(monitor,'ROOT',tmp_path)
    monkeypatch.setattr(monitor,'log',lambda message:None)
    calls = []
    monkeypatch.setattr(monitor,'run_command',lambda args,**kw: (calls.append(args) or -1,'failed'))
    output = tmp_path/'output'; output.mkdir()
    (output/'checkpoint.pt').write_bytes(b'')
    assert not monitor.download('owner/kernel','output','checkpoint.pt')
    assert calls and calls[0][4] == 'scripts.download_kernel'
    (output/'.download-complete.json').write_text(json.dumps(dict(slug='owner/kernel',
          files={'checkpoint.pt':{'bytes':12}})))
    assert not monitor.download('owner/kernel','output','checkpoint.pt')
    (output/'checkpoint.pt').write_bytes(b'valid marker')
    calls.clear()
    assert monitor.download('owner/kernel','output','checkpoint.pt')
    assert not calls
