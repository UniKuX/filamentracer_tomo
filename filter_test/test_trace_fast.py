"""Fast tracing smoke test for VS Code.

Edit the settings in the CONFIG block, then click "Run Python File" in VS Code.
This script keeps everything local to the chosen ROI so iteration stays quick.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from filament_trace_tomo.inputs import load_tracing_inputs_from_roi_bounds, parse_roi_bounds
from filament_trace_tomo.preprocessing import PreprocessingOptions, preprocess_tracing_inputs
from filament_trace_tomo.tracing import TracingOptions, trace_filament_from_preprocessed


# ---------------------------------------------------------------------------
# CONFIG: edit these values and press Run in VS Code.
# ---------------------------------------------------------------------------
VOLUME_PATH = "filter_test/Position_58_corrected.mrc"
ROI_BOUNDS = "85:368,209:483,194:200"
SEED_POINT = "94,439,197"

USE_HESSIAN = True
HESSIAN_BLACK_RIDGES = True
HESSIAN_SIGMAS = (1.0, 2.0)
NORMALIZATION = "zscore"

TRACE_MIN_INTENSITY_PERCENTILE = 80.0
TRACE_MAX_STEPS_PER_DIRECTION = 80
TRACE_STEP_RADIUS = 1
TRACE_NEIGHBORHOOD_RADIUS = 2
TRACE_MIN_DIRECTION_ALIGNMENT = -0.2

WRITE_TRACE_CSV = True
TRACE_CSV_PATH = "filter_test/trace_fast_points.csv"
# ---------------------------------------------------------------------------


def crop_bounds(volume: np.ndarray, roi_bounds: str) -> tuple[np.ndarray, tuple[int, int, int]]:
    (x_start, x_stop), (y_start, y_stop), (z_start, z_stop) = parse_roi_bounds(roi_bounds)
    cropped = volume[z_start:z_stop, y_start:y_stop, x_start:x_stop]
    return cropped.astype(np.float32), (x_start, y_start, z_start)


def write_trace_csv(path: str | Path, points_xyz: np.ndarray, scores: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_vox", "y_vox", "z_vox", "score"])
        for point, score in zip(points_xyz, scores):
            writer.writerow([float(point[0]), float(point[1]), float(point[2]), float(score)])
    return path


def main() -> None:
    t0 = time.perf_counter()
    inputs = load_tracing_inputs_from_roi_bounds(VOLUME_PATH, ROI_BOUNDS, SEED_POINT)
    t1 = time.perf_counter()

    preprocessing = PreprocessingOptions(
        normalization=NORMALIZATION,
        filter_method="hessian" if USE_HESSIAN else "none",
        hessian_sigmas=HESSIAN_SIGMAS,
        hessian_black_ridges=HESSIAN_BLACK_RIDGES,
        fill_outside_roi=0.0,
    )
    preprocessed = preprocess_tracing_inputs(inputs, preprocessing)
    t2 = time.perf_counter()

    roi_values = preprocessed.volume[preprocessed.roi_mask]
    intensity_threshold = float(np.percentile(roi_values, TRACE_MIN_INTENSITY_PERCENTILE))

    trace = trace_filament_from_preprocessed(
        preprocessed,
        inputs.seed_point,
        options=TracingOptions(
            min_intensity=intensity_threshold,
            max_steps_per_direction=TRACE_MAX_STEPS_PER_DIRECTION,
            step_radius=TRACE_STEP_RADIUS,
            neighborhood_radius=TRACE_NEIGHBORHOOD_RADIUS,
            min_direction_alignment=TRACE_MIN_DIRECTION_ALIGNMENT,
        ),
    )
    t3 = time.perf_counter()

    print(f"volume_path: {Path(VOLUME_PATH).resolve()}")
    print(f"roi_bounds: {ROI_BOUNDS}")
    print(f"seed_point: {SEED_POINT}")
    print(f"loaded shape z,y,x: {inputs.shape_zyx}")
    print(f"roi voxels: {int(inputs.roi_mask.sum())}")
    print(f"preprocessing: {preprocessing}")
    print(f"trace threshold (percentile {TRACE_MIN_INTENSITY_PERCENTILE}): {intensity_threshold:.6f}")
    print(f"trace points: {trace.num_points}")
    print(f"trace length voxels: {float(trace.cumulative_lengths_voxels()[-1]):.3f}")
    print(f"first point xyz: {trace.points_xyz[0].tolist()}")
    print(f"seed-centered point xyz: {trace.seed_point_xyz.tolist()}")
    print(f"last point xyz: {trace.points_xyz[-1].tolist()}")
    print(f"timing load/preprocess/trace (s): {t1 - t0:.2f} / {t2 - t1:.2f} / {t3 - t2:.2f}")

    if WRITE_TRACE_CSV:
        csv_path = write_trace_csv(TRACE_CSV_PATH, trace.points_xyz, trace.scores)
        print(f"wrote trace csv: {csv_path.resolve()}")

    roi_crop, roi_origin = crop_bounds(preprocessed.volume, ROI_BOUNDS)
    z_start = roi_origin[2]
    seed_z = int(round(inputs.seed_point[2]))
    local_seed_z = seed_z - z_start
    if 0 <= local_seed_z < roi_crop.shape[0]:
        seed_slice = roi_crop[local_seed_z]
        print(
            "seed slice min/max/mean:",
            f"{float(seed_slice.min()):.6f}",
            f"{float(seed_slice.max()):.6f}",
            f"{float(seed_slice.mean()):.6f}",
        )


if __name__ == "__main__":
    main()
