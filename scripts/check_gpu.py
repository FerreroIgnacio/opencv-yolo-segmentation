"""Verifica que PyTorch vea la GPU NVIDIA (CUDA)."""

import torch

print(f"torch {torch.__version__} | CUDA build: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA NO disponible. Si tenés una NVIDIA, reinstalá torch con CUDA:\n"
        "  pip uninstall -y torch torchvision\n"
        "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130"
    )
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"cuda:{i} -> {props.name} | {props.total_memory / 1024**3:.1f} GB | compute {props.major}.{props.minor}")
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
print("Test FP16 matmul OK:", (x @ x).shape)
