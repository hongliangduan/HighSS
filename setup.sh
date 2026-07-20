#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${HIGHSS_ENV_NAME:-highss}"
PYTHON_VERSION="${HIGHSS_PYTHON_VERSION:-3.10}"
TORCH_INDEX="${HIGHSS_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

cd "$ROOT"

command -v conda >/dev/null 2>&1 || {
    echo "ERROR: conda is not installed or is not in PATH." >&2
    exit 1
}

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "Using existing conda environment: $ENV_NAME"
else
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
fi

conda activate "$ENV_NAME"

python -m pip install --upgrade "pip<27" "setuptools<82" wheel

# Install the validated CUDA 12.8 PyTorch build first. The local projects
# declare torch>=2.2, so this installed build satisfies their dependency and
# will not be replaced by a CPU or incompatible CUDA build.
python -m pip install \
    "torch==2.11.0" \
    "torchvision==0.26.0" \
    "torchaudio==2.11.0" \
    --index-url "$TORCH_INDEX"

# IMPORTANT: do not use --no-deps here. HighSS and the vendored Boltz runtime
# both declare required Python packages in their pyproject.toml files.
python -m pip install -e "$ROOT/boltz" -e "$ROOT"

python -m pip check

python - <<'PY'
import Bio
import click
import einops
import gemmi
import hydra
import matplotlib
import modelcif
import numpy
import pandas
import pytorch_lightning
import requests
import scipy
import seaborn
import torch
import tree
import wandb
import yaml

import boltz
import boltz.data.msa
import highss
from highss import design_engine

print("HighSS:", getattr(highss, "__version__", "unknown"))
print("HighSS source:", highss.__file__)
print("Boltz source:", boltz.__file__)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU 0:", torch.cuda.get_device_name(0))
print("PyYAML:", yaml.__version__)
print("NumPy:", numpy.__version__)
print("gemmi:", gemmi.__version__)
print("Boltz MSA import: PASS")
print("HighSS design engine import: PASS")
PY

highss --help >/dev/null
boltz --help >/dev/null

cd "$ROOT"
python -m unittest -v tests.test_cli_paths

echo
echo "HighSS installation: PASS"
echo "Activate with: conda activate $ENV_NAME"
