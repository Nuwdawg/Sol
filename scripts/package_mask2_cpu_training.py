"""Prepare CPU training and evaluation from the measured, unchanged trainer."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root/'kaggle/exp056_mask2_cpu_no_checkpoint'
output = root/'kaggle/exp053_mask2_cpu_train'
output.mkdir(exist_ok=True)
wrapper = (source/'kernel.py').read_text()
old = "['--backend', 'cpu', '--probe-steps', '4', '--seconds', '1500']"
assert wrapper.count(old) == 1
wrapper = wrapper.replace(old,"['--backend', 'cpu', '--seconds', '41400']")
assert 'timeout=1680' in wrapper
wrapper = wrapper.replace('timeout=1680','timeout=43080')
(output/'kernel.py').write_text(wrapper,encoding='utf-8')
metadata = json.loads((source/'kernel-metadata.json').read_text())
metadata.update(id='cosmicwitness/sol-exp053-mask2-cpu-train',title='Sol exp053 mask2 cpu train',
                enable_gpu=False,enable_tpu=False)
metadata.pop('machine_shape',None)
(output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(output)

source = root/'kaggle/exp048_mask2_eval'
output = root/'kaggle/exp054_mask2_cpu_eval'
output.mkdir(exist_ok=True)
(output/'kernel.py').write_bytes((source/'kernel.py').read_bytes())
metadata = json.loads((source/'kernel-metadata.json').read_text())
metadata.update(id='cosmicwitness/sol-exp054-mask2-cpu-eval',title='Sol exp054 mask2 cpu eval',
                kernel_sources=['cosmicwitness/sol-exp053-mask2-cpu-train'])
(output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(output)
