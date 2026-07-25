#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate filament-tracer
cd "${project_root}"

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"

python -m pytest --cov=filament_tracer --cov-report=term-missing
python -m ruff check .

