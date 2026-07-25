#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate filament-tracer
cd "${project_root}"

# WSLg's Zink path can render Image layers while silently dropping napari's
# 3D Points visuals. Software rendering reliably draws seeds and skeletons.
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"

exec napari --with filament-tracer
