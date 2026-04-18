"""Generate a cylindrical ROI mask, apply it to a tomogram, and preprocess it.

Edit the CONFIG block, then run this file from VS Code with "Run Python File".
Outputs:
    - cropped ROI mask MRC
    - cropped masked raw tomogram MRC
    - cropped masked preprocessed tomogram MRC
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import mrcfile
import numpy as np

from filament_trace_tomo.inputs import load_mrc_data, load_mrc_voxel_size_angst, write_roi_mask_mrc
from filament_trace_tomo.preprocessing import (
    PreprocessingOptions,
    PreprocessedVolume,
    preprocess_volume,
    write_preprocessed_mrc,
)


# ---------------------------------------------------------------------------
# CONFIG: edit these values and press Run in VS Code.
# ---------------------------------------------------------------------------
VOLUME_PATH = "filter_test/Position_58_corrected.mrc"
START_POINT_XYZ = (60.0, 268.0, 187.0)
END_POINT_XYZ = (480.0, 458.0, 187.0)

# Coordinate adjustment switches for matching Fiji/viewer conventions.
FLIP_Y = True
FLIP_X = False

DIAMETER_VALUE = 50.0
DIAMETER_UNIT = "voxels"  # "voxels" or "angstrom"
PADDING_VALUE = 2.0
PADDING_UNIT = "voxels"  # "voxels" or "angstrom"

# Preprocessing settings.
NORMALIZATION = "zscore"  # "none", "zscore", "robust"
CLIP_PERCENTILES = None   # e.g. (1.0, 99.0)
USE_HESSIAN = True
HESSIAN_BLACK_RIDGES = True
HESSIAN_SIGMAS = (1.0, 2.0, 3.0)
USE_GAUSSIAN = False
GAUSSIAN_SIGMA = 1.0
FILL_OUTSIDE_ROI = 0.0

OUTPUT_MASK_PATH = "filter_test/cylindrical_roi_mask_cropped.mrc"
OUTPUT_MASKED_RAW_PATH = "filter_test/roi_masked_raw_tomogram_cropped.mrc"
OUTPUT_PREPROCESSED_PATH = "filter_test/roi_masked_preprocessed_tomogram_cropped.mrc"
# ---------------------------------------------------------------------------


def to_voxels(length_value: float, unit: str, voxel_size_angst: float) -> float:
    unit_normalized = unit.strip().lower()
    if unit_normalized in {"voxel", "voxels", "px", "pixel", "pixels"}:
        return float(length_value)
    if unit_normalized in {"a", "angstrom", "angstroms"}:
        return float(length_value) / voxel_size_angst
    raise ValueError(f"Unsupported unit '{unit}'. Use 'voxels' or 'angstrom'.")


def convert_input_point_to_array_xyz(
    point_xyz: tuple[float, float, float],
    volume_shape_zyx: tuple[int, int, int],
    *,
    flip_x: bool,
    flip_y: bool,
) -> tuple[float, float, float]:
    x, y, z = point_xyz
    _, y_size, x_size = volume_shape_zyx
    if flip_x:
        x = (x_size - 1) - x
    if flip_y:
        y = (y_size - 1) - y
    return float(x), float(y), float(z)


def cylinder_bounds_from_segment(
    volume_shape_zyx: tuple[int, int, int],
    start_xyz: tuple[float, float, float],
    end_xyz: tuple[float, float, float],
    radius_vox: float,
) -> tuple[slice, slice, slice]:
    if radius_vox <= 0:
        raise ValueError("Cylinder radius must be positive")

    start = np.asarray(start_xyz, dtype=np.float32)
    end = np.asarray(end_xyz, dtype=np.float32)
    z_size, y_size, x_size = volume_shape_zyx
    min_xyz = np.floor(np.minimum(start, end) - radius_vox).astype(int)
    max_xyz = np.ceil(np.maximum(start, end) + radius_vox).astype(int)

    x0 = max(0, int(min_xyz[0]))
    y0 = max(0, int(min_xyz[1]))
    z0 = max(0, int(min_xyz[2]))
    x1 = min(x_size - 1, int(max_xyz[0]))
    y1 = min(y_size - 1, int(max_xyz[1]))
    z1 = min(z_size - 1, int(max_xyz[2]))
    return slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1)


def cylindrical_mask_from_segment(
    volume_shape_zyx: tuple[int, int, int],
    start_xyz: tuple[float, float, float],
    end_xyz: tuple[float, float, float],
    radius_vox: float,
) -> tuple[np.ndarray, tuple[slice, slice, slice]]:
    if radius_vox <= 0:
        raise ValueError("Cylinder radius must be positive")

    start = np.asarray(start_xyz, dtype=np.float32)
    end = np.asarray(end_xyz, dtype=np.float32)
    segment = end - start
    segment_length_sq = float(np.dot(segment, segment))
    if segment_length_sq == 0:
        raise ValueError("Start and end points must be different")

    crop_slices = cylinder_bounds_from_segment(volume_shape_zyx, start_xyz, end_xyz, radius_vox)
    z_slice, y_slice, x_slice = crop_slices
    zz, yy, xx = np.mgrid[z_slice, y_slice, x_slice]
    points_xyz = np.stack((xx, yy, zz), axis=-1).astype(np.float32)

    relative = points_xyz - start
    projection = np.sum(relative * segment, axis=-1) / segment_length_sq
    projection_clipped = np.clip(projection, 0.0, 1.0)[..., None]
    closest_points = start + projection_clipped * segment
    distances_sq = np.sum((points_xyz - closest_points) ** 2, axis=-1)

    local_mask = distances_sq <= radius_vox**2
    return local_mask, crop_slices


def build_preprocessing_options() -> PreprocessingOptions:
    filter_method = "none"
    if USE_HESSIAN:
        filter_method = "hessian"
    elif USE_GAUSSIAN:
        filter_method = "gaussian"

    return PreprocessingOptions(
        normalization=NORMALIZATION,
        clip_percentiles=CLIP_PERCENTILES,
        filter_method=filter_method,
        gaussian_sigma=GAUSSIAN_SIGMA,
        hessian_sigmas=HESSIAN_SIGMAS,
        hessian_black_ridges=HESSIAN_BLACK_RIDGES,
        fill_outside_roi=FILL_OUTSIDE_ROI,
    )


def main() -> None:
    t0 = time.perf_counter()
    voxel_size_angst = load_mrc_voxel_size_angst(VOLUME_PATH)
    with mrcfile.mmap(VOLUME_PATH, permissive=True) as mrc:
        volume_shape = tuple(int(v) for v in mrc.data.shape)
    t1 = time.perf_counter()

    diameter_vox = to_voxels(DIAMETER_VALUE, DIAMETER_UNIT, voxel_size_angst)
    padding_vox = to_voxels(PADDING_VALUE, PADDING_UNIT, voxel_size_angst)
    radius_vox = 0.5 * diameter_vox + padding_vox
    start_xyz = convert_input_point_to_array_xyz(
        START_POINT_XYZ,
        volume_shape,
        flip_x=FLIP_X,
        flip_y=FLIP_Y,
    )
    end_xyz = convert_input_point_to_array_xyz(
        END_POINT_XYZ,
        volume_shape,
        flip_x=FLIP_X,
        flip_y=FLIP_Y,
    )

    roi_mask, crop_slices = cylindrical_mask_from_segment(volume_shape, start_xyz, end_xyz, radius_vox)
    t2 = time.perf_counter()

    with mrcfile.mmap(VOLUME_PATH, permissive=True) as mrc:
        cropped_volume = np.asarray(mrc.data[crop_slices], dtype=np.float32).copy()

    mask_path = write_roi_mask_mrc(OUTPUT_MASK_PATH, roi_mask, voxel_size_angst=voxel_size_angst)

    masked_raw = np.where(roi_mask, cropped_volume, np.float32(FILL_OUTSIDE_ROI)).astype(np.float32)
    masked_raw_volume = PreprocessedVolume(
        volume=masked_raw,
        roi_mask=roi_mask,
        options=PreprocessingOptions(fill_outside_roi=FILL_OUTSIDE_ROI),
    )
    masked_raw_path = write_preprocessed_mrc(
        OUTPUT_MASKED_RAW_PATH,
        masked_raw_volume,
        reference_path=VOLUME_PATH,
    )

    preprocessing = build_preprocessing_options()
    preprocessed = preprocess_volume(cropped_volume, roi_mask, options=preprocessing)
    preprocessed_path = write_preprocessed_mrc(
        OUTPUT_PREPROCESSED_PATH,
        preprocessed,
        voxel_size_angst=voxel_size_angst,
    )
    t3 = time.perf_counter()

    roi_values_raw = cropped_volume[roi_mask]
    roi_values_pre = preprocessed.volume[roi_mask]
    start = np.asarray(start_xyz, dtype=np.float32)
    end = np.asarray(end_xyz, dtype=np.float32)
    z_slice, y_slice, x_slice = crop_slices

    print(f"volume_path: {Path(VOLUME_PATH).resolve()}")
    print(f"full volume shape z,y,x: {volume_shape}")
    print(
        "crop bounds z,y,x:",
        f"{z_slice.start}:{z_slice.stop}, {y_slice.start}:{y_slice.stop}, {x_slice.start}:{x_slice.stop}",
    )
    print(f"cropped volume shape z,y,x: {cropped_volume.shape}")
    print(f"voxel size A: {voxel_size_angst:.6f}")
    print(f"input start xyz: {START_POINT_XYZ}")
    print(f"input end xyz: {END_POINT_XYZ}")
    print(f"flip_x: {FLIP_X}")
    print(f"flip_y: {FLIP_Y}")
    print(f"array start xyz: {tuple(start_xyz)}")
    print(f"array end xyz: {tuple(end_xyz)}")
    print(f"axis length voxels: {float(np.linalg.norm(end - start)):.3f}")
    print(f"diameter input: {DIAMETER_VALUE} {DIAMETER_UNIT}")
    print(f"padding input: {PADDING_VALUE} {PADDING_UNIT}")
    print(f"effective radius voxels: {radius_vox:.3f}")
    print(f"roi voxels: {int(roi_mask.sum())}")
    print(f"raw ROI min/max/mean/std: {float(roi_values_raw.min()):.6f} / {float(roi_values_raw.max()):.6f} / {float(roi_values_raw.mean()):.6f} / {float(roi_values_raw.std()):.6f}")
    print(f"pre ROI min/max/mean/std: {float(roi_values_pre.min()):.6f} / {float(roi_values_pre.max()):.6f} / {float(roi_values_pre.mean()):.6f} / {float(roi_values_pre.std()):.6f}")
    print(f"preprocessing: {preprocessing}")
    print(f"mask output: {Path(mask_path).resolve()}")
    print(f"masked raw output: {Path(masked_raw_path).resolve()}")
    print(f"preprocessed output: {Path(preprocessed_path).resolve()}")
    print(f"timing load/mask/filter+write (s): {t1 - t0:.2f} / {t2 - t1:.2f} / {t3 - t2:.2f}")


if __name__ == "__main__":
    main()
