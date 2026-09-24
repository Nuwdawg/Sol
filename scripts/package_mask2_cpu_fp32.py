"""Prepare the FP32 counterpart of the existing CPU BF16 feasibility probe."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root/'kaggle/exp052_mask2_cpu_probe'
output = root/'kaggle/exp055_mask2_cpu_fp32'
output.mkdir(exist_ok=True)
wrapper = (source/'kernel.py').read_text()
assert wrapper.count("'--bf16', ") == 1
wrapper = wrapper.replace("'--bf16', ", '')
(output/'kernel.py').write_text(wrapper,encoding='utf-8')
metadata = json.loads((source/'kernel-metadata.json').read_text())
metadata.update(id='cosmicwitness/sol-exp055-mask2-cpu-fp32',title='Sol exp055 mask2 cpu fp32')
assert metadata['enable_gpu'] is False and metadata['enable_tpu'] is False
(output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(output)
