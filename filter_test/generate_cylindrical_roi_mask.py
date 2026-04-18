"""Generate a cylindrical ROI mask around a user-defined filament axis.

Edit the CONFIG block, then run this file from VS Code with "Run Python File".
The mask is written as an MRC aligned to the source tomogram.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from filament_trace_tomo.inputs import load_mrc_data, load_mrc_voxel_size_angst, write_roi_mask_mrc


# ---------------------------------------------------------------------------
# CONFIG: edit these values and press Run in VS Code.
# ---------------------------------------------------------------------------
VOLUME_PATH = "filter_test/Position_58_corrected.mrc"
START_POINT_XYZ = (60.0, 268.0, 187.0)
END_POINT_XYZ = (480.0, 458.0, 187.0)

# Coordinate adjustment switches for matching Fiji/viewer conventions.
# If the mask appears mirrored vertically, set FLIP_Y = True.
FLIP_Y = True
FLIP_X = False

# Diameter can be given in voxels or Angstrom.
DIAMETER_VALUE = 50.0
DIAMETER_UNIT = "voxels"  # "voxels" or "angstrom"

# Extra padding beyond the cylinder radius, useful if you want a looser ROI.
PADDING_VALUE = 2.0
PADDING_UNIT = "voxels"  # "voxels" or "angstrom"

OUTPUT_MASK_PATH = "filter_test/cylindrical_roi_mask.mrc"
WRITE_ENDPOINT_MARKERS = True
START_MARKER_PATH = "filter_test/cylindrical_roi_start_marker.mrc"
END_MARKER_PATH = "filter_test/cylindrical_roi_end_marker.mrc"
# ---------------------------------------------------------------------------


def to_voxels(length_value: float, unit: str, voxel_size_angst: float) -> float:
    unit_normalized = unit.strip().lower()
    if unit_normalized in {"voxel", "voxels", "px", "pixel", "pixels"}:
        return float(length_value)
    if unit_normalized in {"a", "angstrom", "angstroms"}:
        return float(length_value) / voxel_size_angst
    raise ValueError(f"Unsupported unit '{unit}'. Use 'voxels' or 'angstrom'.")


def cylindrical_mask_from_segment(
    volume_shape_zyx: tuple[int, int, int],
    start_xyz: tuple[float, float, float],
    end_xyz: tuple[float, float, float],
    radius_vox: float,
) -> np.ndarray:
    """Create a finite cylinder mask around the line segment from start to end."""

    if radius_vox <= 0:
        raise ValueError("Cylinder radius must be positive")

    start = np.asarray(start_xyz, dtype=np.float32)
    end = np.asarray(end_xyz, dtype=np.float32)
    segment = end - start
    segment_length_sq = float(np.dot(segment, segment))
    if segment_length_sq == 0:
        raise ValueError("Start and end points must be different")

    z_size, y_size, x_size = volume_shape_zyx
    min_xyz = np.floor(np.minimum(start, end) - radius_vox).astype(int)
    max_xyz = np.ceil(np.maximum(start, end) + radius_vox).astype(int)

    x0 = max(0, int(min_xyz[0]))
    y0 = max(0, int(min_xyz[1]))
    z0 = max(0, int(min_xyz[2]))
    x1 = min(x_size - 1, int(max_xyz[0]))
    y1 = min(y_size - 1, int(max_xyz[1]))
    z1 = min(z_size - 1, int(max_xyz[2]))

    mask = np.zeros(volume_shape_zyx, dtype=bool)
    zz, yy, xx = np.mgrid[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
    points_xyz = np.stack((xx, yy, zz), axis=-1).astype(np.float32)

    relative = points_xyz - start
    projection = np.sum(relative * segment, axis=-1) / segment_length_sq
    projection_clipped = np.clip(projection, 0.0, 1.0)[..., None]
    closest_points = start + projection_clipped * segment
    distances_sq = np.sum((points_xyz - closest_points) ** 2, axis=-1)

    local_mask = distances_sq <= radius_vox**2
    mask[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] = local_mask
    return mask


def convert_input_point_to_array_xyz(
    point_xyz: tuple[float, float, float],
    volume_shape_zyx: tuple[int, int, int],
    *,
    flip_x: bool,
    flip_y: bool,
) -> tuple[float, float, float]:
    """Convert user input coordinates into array-space x,y,z."""

    x, y, z = point_xyz
    _, y_size, x_size = volume_shape_zyx
    if flip_x:
        x = (x_size - 1) - x
    if flip_y:
        y = (y_size - 1) - y
    return float(x), float(y), float(z)


def marker_mask_from_point(
    volume_shape_zyx: tuple[int, int, int],
    point_xyz: tuple[float, float, float],
) -> np.ndarray:
    """Create a single-voxel marker mask for debugging coordinates."""

    x, y, z = point_xyz
    xi, yi, zi = int(round(x)), int(round(y)), int(round(z))
    z_size, y_size, x_size = volume_shape_zyx
    if not (0 <= xi < x_size and 0 <= yi < y_size and 0 <= zi < z_size):
        raise ValueError(f"Marker point {point_xyz} is outside volume bounds")
    mask = np.zeros(volume_shape_zyx, dtype=bool)
    mask[zi, yi, xi] = True
    return mask


def main() -> None:
    t0 = time.perf_counter()
    volume = load_mrc_data(VOLUME_PATH)
    voxel_size_angst = load_mrc_voxel_size_angst(VOLUME_PATH)
    t1 = time.perf_counter()

    diameter_vox = to_voxels(DIAMETER_VALUE, DIAMETER_UNIT, voxel_size_angst)
    padding_vox = to_voxels(PADDING_VALUE, PADDING_UNIT, voxel_size_angst)
    radius_vox = 0.5 * diameter_vox + padding_vox
    start_xyz = convert_input_point_to_array_xyz(
        START_POINT_XYZ,
        volume.shape,
        flip_x=FLIP_X,
        flip_y=FLIP_Y,
    )
    end_xyz = convert_input_point_to_array_xyz(
        END_POINT_XYZ,
        volume.shape,
        flip_x=FLIP_X,
        flip_y=FLIP_Y,
    )

    mask = cylindrical_mask_from_segment(
        volume.shape,
        start_xyz,
        end_xyz,
        radius_vox,
    )
    t2 = time.perf_counter()

    output_path = write_roi_mask_mrc(
        OUTPUT_MASK_PATH,
        mask,
        reference_path=VOLUME_PATH,
    )
    start_marker_path = None
    end_marker_path = None
    if WRITE_ENDPOINT_MARKERS:
        start_marker = marker_mask_from_point(volume.shape, start_xyz)
        end_marker = marker_mask_from_point(volume.shape, end_xyz)
        start_marker_path = write_roi_mask_mrc(
            START_MARKER_PATH,
            start_marker,
            reference_path=VOLUME_PATH,
        )
        end_marker_path = write_roi_mask_mrc(
            END_MARKER_PATH,
            end_marker,
            reference_path=VOLUME_PATH,
        )
    t3 = time.perf_counter()

    start = np.asarray(start_xyz, dtype=np.float32)
    end = np.asarray(end_xyz, dtype=np.float32)
    axis_length_vox = float(np.linalg.norm(end - start))

    print(f"volume_path: {Path(VOLUME_PATH).resolve()}")
    print(f"volume shape z,y,x: {volume.shape}")
    print(f"voxel size A: {voxel_size_angst:.6f}")
    print(f"input start xyz: {START_POINT_XYZ}")
    print(f"input end xyz: {END_POINT_XYZ}")
    print(f"flip_x: {FLIP_X}")
    print(f"flip_y: {FLIP_Y}")
    print(f"array start xyz: {start_xyz}")
    print(f"array end xyz: {end_xyz}")
    print(f"axis length voxels: {axis_length_vox:.3f}")
    print(f"diameter input: {DIAMETER_VALUE} {DIAMETER_UNIT}")
    print(f"padding input: {PADDING_VALUE} {PADDING_UNIT}")
    print(f"effective radius voxels: {radius_vox:.3f}")
    print(f"mask voxels: {int(mask.sum())}")
    print(f"mask output: {Path(output_path).resolve()}")
    if start_marker_path is not None and end_marker_path is not None:
        print(f"start marker output: {Path(start_marker_path).resolve()}")
        print(f"end marker output: {Path(end_marker_path).resolve()}")
    print(f"timing load/mask/write (s): {t1 - t0:.2f} / {t2 - t1:.2f} / {t3 - t2:.2f}")


if __name__ == "__main__":
    main()
