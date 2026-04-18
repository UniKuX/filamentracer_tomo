"""Input loading and validation for tomogram filament tracing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mrcfile
import numpy as np
from numpy.typing import NDArray


SeedPoint = tuple[float, float, float]
RoiBounds = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class TracingInputs:
    """Validated inputs needed before filament tracing starts."""

    volume: NDArray[np.float32]
    roi_mask: NDArray[np.bool_]
    seed_point: SeedPoint
    voxel_size_angst: float
    tomo_name: str
    volume_path: Path
    mask_path: Path | None

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return self.volume.shape


def parse_seed_point(seed: str | Sequence[float]) -> SeedPoint:
    """Parse a seed point supplied as ``x,y,z`` text or a 3-number sequence."""

    if isinstance(seed, str):
        parts = [part.strip() for part in seed.split(",")]
        if len(parts) != 3:
            raise ValueError("Seed point must contain exactly three comma-separated values: x,y,z")
        try:
            values = tuple(float(part) for part in parts)
        except ValueError as exc:
            raise ValueError("Seed point values must be numeric") from exc
    else:
        if len(seed) != 3:
            raise ValueError("Seed point must contain exactly three values: x,y,z")
        values = tuple(float(value) for value in seed)

    return values  # type: ignore[return-value]


def parse_roi_bounds(bounds: str | Sequence[Sequence[int]]) -> RoiBounds:
    """Parse ROI bounds in x, y, z order.

    String bounds use Python-style half-open ranges:
    ``x_start:x_stop,y_start:y_stop,z_start:z_stop``.
    """

    if isinstance(bounds, str):
        range_parts = [part.strip() for part in bounds.split(",")]
        if len(range_parts) != 3:
            raise ValueError(
                "ROI bounds must contain three comma-separated ranges: "
                "x_start:x_stop,y_start:y_stop,z_start:z_stop"
            )
        parsed_bounds = []
        for range_part in range_parts:
            limits = [limit.strip() for limit in range_part.split(":")]
            if len(limits) != 2:
                raise ValueError("Each ROI range must use start:stop syntax")
            try:
                parsed_bounds.append((int(limits[0]), int(limits[1])))
            except ValueError as exc:
                raise ValueError("ROI bounds must be integer voxel coordinates") from exc
        values = tuple(parsed_bounds)
    else:
        if len(bounds) != 3:
            raise ValueError("ROI bounds must contain exactly three ranges in x, y, z order")
        values = tuple((int(axis_bounds[0]), int(axis_bounds[1])) for axis_bounds in bounds)

    for start, stop in values:
        if stop <= start:
            raise ValueError("Each ROI bound must have stop greater than start")

    return values  # type: ignore[return-value]


def load_mrc_data(path: str | Path) -> NDArray[np.float32]:
    """Load an MRC/CCP4 volume as a z, y, x float32 NumPy array."""

    path = Path(path)
    with mrcfile.open(path, permissive=True) as mrc:
        return np.asarray(mrc.data, dtype=np.float32).copy()


def load_mrc_voxel_size_angst(path: str | Path) -> float:
    """Read isotropic voxel size in Angstrom from an MRC file."""

    path = Path(path)
    with mrcfile.open(path, permissive=True) as mrc:
        voxel_size = mrc.voxel_size
        values = np.array([voxel_size.x, voxel_size.y, voxel_size.z], dtype=float)

    if np.any(values <= 0):
        raise ValueError(f"MRC voxel size must be positive for {path}")
    if not np.allclose(values, values[0], rtol=1e-4, atol=1e-6):
        raise ValueError(
            f"Anisotropic voxel sizes are not supported yet for {path}: "
            f"x={values[0]}, y={values[1]}, z={values[2]}"
        )
    return float(values[0])


def create_roi_mask_from_bounds(
    volume_shape: tuple[int, int, int],
    bounds: str | Sequence[Sequence[int]],
    *,
    clip: bool = False,
) -> NDArray[np.bool_]:
    """Create a box ROI mask from x, y, z half-open coordinate bounds."""

    (x_start, x_stop), (y_start, y_stop), (z_start, z_stop) = parse_roi_bounds(bounds)
    z_size, y_size, x_size = volume_shape

    if clip:
        x_start, x_stop = max(0, x_start), min(x_size, x_stop)
        y_start, y_stop = max(0, y_start), min(y_size, y_stop)
        z_start, z_stop = max(0, z_start), min(z_size, z_stop)
    elif not (
        0 <= x_start < x_stop <= x_size
        and 0 <= y_start < y_stop <= y_size
        and 0 <= z_start < z_stop <= z_size
    ):
        raise ValueError(
            f"ROI bounds are outside volume bounds "
            f"x=[0,{x_size}], y=[0,{y_size}], z=[0,{z_size}]"
        )

    if not (x_start < x_stop and y_start < y_stop and z_start < z_stop):
        raise ValueError("Clipped ROI bounds are empty")

    roi_mask = np.zeros(volume_shape, dtype=bool)
    roi_mask[z_start:z_stop, y_start:y_stop, x_start:x_stop] = True
    return roi_mask


def write_roi_mask_mrc(
    path: str | Path,
    roi_mask: NDArray[np.bool_],
    *,
    reference_path: str | Path | None = None,
    voxel_size_angst: float | None = None,
    overwrite: bool = True,
) -> Path:
    """Write a boolean ROI mask as an MRC file with 0/1 integer values."""

    path = Path(path)
    if reference_path is not None:
        voxel_size_angst = load_mrc_voxel_size_angst(reference_path)
    if voxel_size_angst is None:
        raise ValueError("A reference MRC or voxel size is required to write an ROI mask")

    with mrcfile.new(path, overwrite=overwrite) as mrc:
        mrc.set_data(roi_mask.astype(np.int8))
        mrc.voxel_size = voxel_size_angst

    return path


def validate_roi_mask(mask: NDArray[np.float32], volume_shape: tuple[int, int, int]) -> NDArray[np.bool_]:
    """Validate and binarize an ROI mask volume."""

    if mask.shape != volume_shape:
        raise ValueError(f"Mask shape {mask.shape} does not match volume shape {volume_shape}")

    roi_mask = mask > 0
    if not np.any(roi_mask):
        raise ValueError("ROI mask is empty")
    return roi_mask


def validate_seed_in_mask(seed_point: SeedPoint, roi_mask: NDArray[np.bool_]) -> None:
    """Validate that an x, y, z seed lies inside the volume and ROI mask."""

    x, y, z = seed_point
    zi, yi, xi = (int(round(z)), int(round(y)), int(round(x)))
    z_size, y_size, x_size = roi_mask.shape

    if not (0 <= xi < x_size and 0 <= yi < y_size and 0 <= zi < z_size):
        raise ValueError(
            f"Seed point {seed_point} is outside volume bounds "
            f"x=[0,{x_size - 1}], y=[0,{y_size - 1}], z=[0,{z_size - 1}]"
        )
    if not roi_mask[zi, yi, xi]:
        raise ValueError(f"Seed point {seed_point} is outside the ROI mask")


def load_tracing_inputs(
    volume_path: str | Path,
    mask_path: str | Path,
    seed_point: str | Sequence[float],
    *,
    voxel_size_angst: float | None = None,
    tomo_name: str | None = None,
) -> TracingInputs:
    """Load and validate tomogram, ROI mask, seed point, and voxel size."""

    volume_path = Path(volume_path)
    mask_path = Path(mask_path)

    volume = load_mrc_data(volume_path)
    mask = load_mrc_data(mask_path)
    roi_mask = validate_roi_mask(mask, volume.shape)
    parsed_seed = parse_seed_point(seed_point)
    validate_seed_in_mask(parsed_seed, roi_mask)

    if voxel_size_angst is None:
        voxel_size_angst = load_mrc_voxel_size_angst(volume_path)
    if voxel_size_angst <= 0:
        raise ValueError("Voxel size must be positive")

    return TracingInputs(
        volume=volume,
        roi_mask=roi_mask,
        seed_point=parsed_seed,
        voxel_size_angst=float(voxel_size_angst),
        tomo_name=tomo_name or volume_path.stem,
        volume_path=volume_path,
        mask_path=mask_path,
    )


def load_tracing_inputs_from_roi_bounds(
    volume_path: str | Path,
    roi_bounds: str | Sequence[Sequence[int]],
    seed_point: str | Sequence[float],
    *,
    voxel_size_angst: float | None = None,
    tomo_name: str | None = None,
    mask_output_path: str | Path | None = None,
    clip_roi_bounds: bool = False,
) -> TracingInputs:
    """Load a tomogram and create an ROI mask from user-provided x, y, z bounds."""

    volume_path = Path(volume_path)
    volume = load_mrc_data(volume_path)
    roi_mask = create_roi_mask_from_bounds(volume.shape, roi_bounds, clip=clip_roi_bounds)
    parsed_seed = parse_seed_point(seed_point)
    validate_seed_in_mask(parsed_seed, roi_mask)

    if voxel_size_angst is None:
        voxel_size_angst = load_mrc_voxel_size_angst(volume_path)
    if voxel_size_angst <= 0:
        raise ValueError("Voxel size must be positive")

    written_mask_path = None
    if mask_output_path is not None:
        written_mask_path = write_roi_mask_mrc(
            mask_output_path,
            roi_mask,
            voxel_size_angst=voxel_size_angst,
        )

    return TracingInputs(
        volume=volume,
        roi_mask=roi_mask,
        seed_point=parsed_seed,
        voxel_size_angst=float(voxel_size_angst),
        tomo_name=tomo_name or volume_path.stem,
        volume_path=volume_path,
        mask_path=written_mask_path,
    )
