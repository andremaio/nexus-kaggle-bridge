#!/usr/bin/env python3
import json
from pathlib import Path

out = {}
try:
    import jax
    out['jax_version'] = jax.__version__
    out['jax_devices'] = [str(x) for x in jax.devices()]
    out['jax_platforms'] = sorted({getattr(x, 'platform', None) for x in jax.devices()})
    out['tpu_visible_jax'] = any(getattr(x, 'platform', None) == 'tpu' for x in jax.devices())
except Exception as exc:
    out['jax_error'] = repr(exc)
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    out['torch_xla_version'] = getattr(torch_xla, '__version__', None)
    out['xla_device'] = str(xm.xla_device())
except Exception as exc:
    out['torch_xla_error'] = repr(exc)
Path('/kaggle/working/tpu-probe.json').write_text(json.dumps(out, sort_keys=True, indent=2) + '\n')
print('NEXUS_TPU_PROBE=' + json.dumps(out, sort_keys=True))
