"""Package a bounded CPU fallback measurement from the exact pending trainer."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root/'kaggle/exp047_mask2_real_probe'
output = root/'kaggle/exp052_mask2_cpu_probe'
output.mkdir(exist_ok=True)
wrapper = (source/'kernel.py').read_text()
old = "['--probe-steps', '4', '--seconds', '5100', '--bf16', '--checkpoint-activations']"
new = "['--backend', 'cpu', '--probe-steps', '4', '--seconds', '1500', '--bf16', '--checkpoint-activations']"
assert wrapper.count(old) == 1
wrapper = wrapper.replace(old,new).replace("os.environ['PJRT_DEVICE']='TPU'", "os.environ.pop('PJRT_DEVICE',None)")
assert 'timeout=5280' in wrapper
wrapper = wrapper.replace('timeout=5280','timeout=1680')
(output/'kernel.py').write_text(wrapper,encoding='utf-8')
metadata = json.loads((source/'kernel-metadata.json').read_text())
metadata.update(id='cosmicwitness/sol-exp052-mask2-cpu-probe',title='Sol exp052 mask2 cpu probe',
                enable_gpu=False,enable_tpu=False)
metadata.pop('machine_shape',None)
(output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(output)
