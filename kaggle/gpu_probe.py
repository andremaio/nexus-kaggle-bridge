#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess

out = {"cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES")}
try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"], text=True, capture_output=True, timeout=20)
    out["nvidia_smi_rc"] = smi.returncode
    out["nvidia_smi"] = smi.stdout.strip()
    out["nvidia_smi_err"] = smi.stderr.strip()[-1000:]
except Exception as exc:
    out["nvidia_smi_error"] = repr(exc)
try:
    import torch
    out["torch_version"] = torch.__version__
    out["torch_cuda_version"] = torch.version.cuda
    out["torch_cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        out["device_name"] = torch.cuda.get_device_name(0)
        out["device_capability"] = list(torch.cuda.get_device_capability(0))
        try:
            x = torch.tensor([1.0], device="cuda")
            out["cuda_op_ok"] = float((x + 1).item()) == 2.0
        except Exception as exc:
            out["cuda_op_ok"] = False
            out["cuda_op_error"] = repr(exc)
except Exception as exc:
    out["torch_error"] = repr(exc)
Path('/kaggle/working/gpu-probe.json').write_text(json.dumps(out, sort_keys=True, indent=2) + '\n')
print('NEXUS_GPU_PROBE=' + json.dumps(out, sort_keys=True))
