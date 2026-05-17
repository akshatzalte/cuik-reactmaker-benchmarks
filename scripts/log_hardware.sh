#!/bin/bash
# Log hardware specs for reproducibility.
# Usage: bash scripts/log_hardware.sh > results/hardware.txt

echo "=== Date ==="
date

echo ""
echo "=== CPU ==="
lscpu | grep -E "Model name|Socket|Core|Thread|CPU MHz|L[23] cache"

echo ""
echo "=== Memory ==="
free -h | head -2

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
    --format=csv,noheader 2>/dev/null || echo "No NVIDIA GPU found"

echo ""
echo "=== CUDA driver ==="
nvidia-smi | grep "Driver Version" 2>/dev/null || echo "nvidia-smi not available"

echo ""
echo "=== Key package versions ==="
python -c "
import torch; print('torch:', torch.__version__, '| CUDA:', torch.version.cuda)
import chemprop; print('chemprop:', chemprop.__version__)
import cuik_molmaker; print('cuik_molmaker: installed at', cuik_molmaker.__file__)
from rdkit import rdBase; print('rdkit:', rdBase.rdkitVersion)
"
