"""Cylindrical ROI toolbox for testing multiple filter techniques in one run.

Edit the CONFIG block, then run from VS Code with "Run Python File".

This script:
1. Converts start/end coordinates into array-space with optional flips.
2. Builds a cylindrical ROI mask around the filament axis.
3. Crops the tomogram to a tight bounding box around that cylinder.
4. Writes the cropped ROI mask and optional endpoint markers.
5. Applies multiple filtering variants to the cropped ROI.
6. Writes one MRC per variant for Fiji inspection.
7. Optionally saves a PNG slice comparison.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
from skimage.filters import frangi, hessian as hessian_filter, meijering, sato

from filament_trace_tomo.inputs import load_mrc_voxel_size_angst, write_roi_mask_mrc
from filament_trace_tomo.preprocessing import (
    PreprocessedVolume,
    PreprocessingOptions,
    preprocess_volume,
    write_preprocessed_mrc,
)


# ---------------------------------------------------------------------------
# CONFIG: edit these values and press Run in VS Code.
# ---------------------------------------------------------------------------
VOLUME_PATH = "filter_test/rec_Position_28_half1.mrc"
START_POINT_XYZ = (44.0, 210.0, 245.0)
END_POINT_XYZ = (470.0, 440.0, 245.0)
FLIP_Y = True
FLIP_X = False

DIAMETER_VALUE = 55.0
DIAMETER_UNIT = "voxels"  # "voxels" or "angstrom"
PADDING_VALUE = 2.0
PADDING_UNIT = "voxels"   # "voxels" or "angstrom"

# Filtering and normalization settings shared by the variants below.
NORMALIZATION = "robust"  # "none", "zscore", "robust"
CLIP_PERCENTILES = None   # e.g. (1.0, 99.0)
FILL_OUTSIDE_ROI = 0.0
FILTER_SIGMAS = (1.0, 2.0, 3.0, 4.0)
RIDGES_ARE_DARK = False
GAUSSIAN_SIGMA = 1.0

# Choose which variants to generate.
RUN_RAW = True
RUN_GAUSSIAN = False
RUN_HESSIAN = False
RUN_SATO = False
RUN_FRANGI = False
RUN_MEIJERING = True

WRITE_ENDPOINT_MARKERS = False
WRITE_SLICE_COMPARISON = True
SLICE_COMPARISON_PATH = "filter_test/cylindrical_roi_filter_comparison.png"

OUTPUT_DIR = "filter_test/filter_outputs"
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
    return distances_sq <= radius_vox**2, crop_slices


def marker_mask_from_point(
    volume_shape_zyx: tuple[int, int, int],
    point_xyz: tuple[float, float, float],
) -> np.ndarray:
    x, y, z = point_xyz
    xi, yi, zi = int(round(x)), int(round(y)), int(round(z))
    z_size, y_size, x_size = volume_shape_zyx
    if not (0 <= xi < x_size and 0 <= yi < y_size and 0 <= zi < z_size):
        raise ValueError(f"Marker point {point_xyz} is outside volume bounds")
    mask = np.zeros(volume_shape_zyx, dtype=bool)
    mask[zi, yi, xi] = True
    return mask


def normalize_for_display(image: np.ndarray, percentiles: tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    low, high = np.percentile(image, percentiles)
    if high == low:
        return np.zeros_like(image, dtype=np.float32)
    clipped = np.clip(image, low, high)
    return ((clipped - low) / (high - low)).astype(np.float32)


def write_volume(path: Path, volume: np.ndarray, roi_mask: np.ndarray, voxel_size_angst: float) -> Path:
    wrapped = PreprocessedVolume(
        volume=np.asarray(volume, dtype=np.float32),
        roi_mask=roi_mask,
        options=PreprocessingOptions(fill_outside_roi=FILL_OUTSIDE_ROI),
    )
    return write_preprocessed_mrc(path, wrapped, voxel_size_angst=voxel_size_angst)


def preprocess_variant(
    name: str,
    cropped_volume: np.ndarray,
    roi_mask: np.ndarray,
) -> np.ndarray:
    if name == "raw":
        return np.where(roi_mask, cropped_volume, np.float32(FILL_OUTSIDE_ROI)).astype(np.float32)
    if name == "gaussian":
        return preprocess_volume(
            cropped_volume,
            roi_mask,
            options=PreprocessingOptions(
                normalization=NORMALIZATION,
                clip_percentiles=CLIP_PERCENTILES,
                filter_method="gaussian",
                gaussian_sigma=GAUSSIAN_SIGMA,
                fill_outside_roi=FILL_OUTSIDE_ROI,
            ),
        ).volume
    if name == "hessian":
        return preprocess_volume(
            cropped_volume,
            roi_mask,
            options=PreprocessingOptions(
                normalization=NORMALIZATION,
                clip_percentiles=CLIP_PERCENTILES,
                filter_method="hessian",
                hessian_sigmas=FILTER_SIGMAS,
                hessian_black_ridges=RIDGES_ARE_DARK,
                fill_outside_roi=FILL_OUTSIDE_ROI,
            ),
        ).volume

    # For the skimage ridge filters below, apply the same normalization first,
    # then run the ridge operator directly on the cropped volume.
    normalized = preprocess_volume(
        cropped_volume,
        roi_mask,
        options=PreprocessingOptions(
            normalization=NORMALIZATION,
            clip_percentiles=CLIP_PERCENTILES,
            filter_method="none",
            fill_outside_roi=FILL_OUTSIDE_ROI,
        ),
    ).volume
    black_ridges = RIDGES_ARE_DARK

    if name == "sato":
        result = sato(normalized, sigmas=FILTER_SIGMAS, black_ridges=black_ridges)
    elif name == "frangi":
        result = frangi(normalized, sigmas=FILTER_SIGMAS, black_ridges=black_ridges)
    elif name == "meijering":
        result = meijering(normalized, sigmas=FILTER_SIGMAS, black_ridges=black_ridges)
    else:
        raise ValueError(f"Unsupported filter variant: {name}")

    masked = np.asarray(result, dtype=np.float32).copy()
    masked[~roi_mask] = np.float32(FILL_OUTSIDE_ROI)
    return masked


def selected_variants() -> list[str]:
    variants: list[str] = []
    if RUN_RAW:
        variants.append("raw")
    if RUN_GAUSSIAN:
        variants.append("gaussian")
    if RUN_HESSIAN:
        variants.append("hessian")
    if RUN_SATO:
        variants.append("sato")
    if RUN_FRANGI:
        variants.append("frangi")
    if RUN_MEIJERING:
        variants.append("meijering")
    if not variants:
        raise ValueError("Select at least one filter variant to run")
    return variants


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
    z_slice, y_slice, x_slice = crop_slices

    with mrcfile.mmap(VOLUME_PATH, permissive=True) as mrc:
        cropped_volume = np.asarray(mrc.data[crop_slices], dtype=np.float32).copy()
    t2 = time.perf_counter()

    output_dir = (PROJECT_ROOT / OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_path = write_roi_mask_mrc(output_dir / "cylindrical_roi_mask.mrc", roi_mask, voxel_size_angst=voxel_size_angst)
    if WRITE_ENDPOINT_MARKERS:
        start_marker = marker_mask_from_point(volume_shape, start_xyz)[crop_slices]
        end_marker = marker_mask_from_point(volume_shape, end_xyz)[crop_slices]
        start_marker_path = write_roi_mask_mrc(output_dir / "cylindrical_roi_start_marker.mrc", start_marker, voxel_size_angst=voxel_size_angst)
        end_marker_path = write_roi_mask_mrc(output_dir / "cylindrical_roi_end_marker.mrc", end_marker, voxel_size_angst=voxel_size_angst)
    else:
        start_marker_path = None
        end_marker_path = None

    variants = selected_variants()
    results: dict[str, np.ndarray] = {}
    output_paths: dict[str, Path] = {}
    for variant in variants:
        result = preprocess_variant(variant, cropped_volume, roi_mask)
        results[variant] = result
        output_paths[variant] = write_volume(output_dir / f"{variant}.mrc", result, roi_mask, voxel_size_angst)
    t3 = time.perf_counter()

    if WRITE_SLICE_COMPARISON:
        slice_z_local = cropped_volume.shape[0] // 2
        fig, axes = plt.subplots(1, len(variants), figsize=(4 * len(variants), 4))
        if len(variants) == 1:
            axes = [axes]
        for axis, variant in zip(axes, variants):
            axis.imshow(normalize_for_display(results[variant][slice_z_local]), cmap=plt.cm.gray)
            axis.set_title(variant)
            axis.set_xticks([])
            axis.set_yticks([])
        fig.suptitle(
            f"{Path(VOLUME_PATH).name} | crop z={z_slice.start}:{z_slice.stop}, "
            f"y={y_slice.start}:{y_slice.stop}, x={x_slice.start}:{x_slice.stop}"
        )
        fig.tight_layout()
        figure_path = (PROJECT_ROOT / SLICE_COMPARISON_PATH).resolve()
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        figure_path = None
    t4 = time.perf_counter()

    start = np.asarray(start_xyz, dtype=np.float32)
    end = np.asarray(end_xyz, dtype=np.float32)
    print(f"volume_path: {Path(VOLUME_PATH).resolve()}")
    print(f"full volume shape z,y,x: {volume_shape}")
    print(f"crop bounds z,y,x: {z_slice.start}:{z_slice.stop}, {y_slice.start}:{y_slice.stop}, {x_slice.start}:{x_slice.stop}")
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
    print(f"variants: {variants}")
    print(f"mask output: {mask_path}")
    if start_marker_path is not None and end_marker_path is not None:
        print(f"start marker output: {start_marker_path}")
        print(f"end marker output: {end_marker_path}")
    for variant in variants:
        roi_values = results[variant][roi_mask]
        print(
            f"{variant} ROI min/max/mean/std: "
            f"{float(roi_values.min()):.6f} / {float(roi_values.max()):.6f} / "
            f"{float(roi_values.mean()):.6f} / {float(roi_values.std()):.6f}"
        )
        print(f"{variant} output: {output_paths[variant]}")
    if figure_path is not None:
        print(f"slice comparison figure: {figure_path}")
    print(
        f"timing load-shape/crop-read/filter/write/figure (s): "
        f"{t1 - t0:.2f} / {t2 - t1:.2f} / {t3 - t2:.2f} / {t4 - t3:.2f}"
    )


if __name__ == "__main__":
    main()
