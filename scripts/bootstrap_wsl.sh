#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_root="${HOME}/miniconda3"
environment_name="filament-tracer"

if [[ ! -f "${conda_root}/etc/profile.d/conda.sh" ]]; then
    echo "Miniconda was not found at ${conda_root}." >&2
    exit 1
fi

source "${conda_root}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "${environment_name}"; then
    conda create \
        --yes \
        --name "${environment_name}" \
        --override-channels \
        --channel conda-forge \
        python=3.12 \
        pip
fi

conda activate "${environment_name}"
python -m pip install --upgrade pip
python -m pip install --editable "${project_root}[dev]"

python - <<'PY'
import napari
import numpy
import scipy

print(f"napari={napari.__version__}")
print(f"numpy={numpy.__version__}")
print(f"scipy={scipy.__version__}")
PY

echo "Environment '${environment_name}' is ready."

