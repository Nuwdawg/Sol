"""Measure CPU FP32 training without activation-recomputation overhead."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root/'kaggle/exp055_mask2_cpu_fp32'
output = root/'kaggle/exp056_mask2_cpu_no_checkpoint'
output.mkdir(exist_ok=True)
wrapper = (source/'kernel.py').read_text()
assert wrapper.count(", '--checkpoint-activations'") == 1
wrapper = wrapper.replace(", '--checkpoint-activations'",'')
(output/'kernel.py').write_text(wrapper,encoding='utf-8')
metadata = json.loads((source/'kernel-metadata.json').read_text())
metadata.update(id='cosmicwitness/sol-exp056-mask2-cpu-no-checkpoint',
                title='Sol exp056 mask2 cpu no checkpoint')
assert metadata['enable_gpu'] is False and metadata['enable_tpu'] is False
(output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(output)
