"""Lightweight preprocessing for voxel-based filament tracing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mrcfile
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage.filters import hessian as hessian_filter

from filament_trace_tomo.inputs import TracingInputs, load_mrc_voxel_size_angst


NormalizationMethod = Literal["none", "zscore", "robust"]
FilterMethod = Literal["none", "gaussian", "median", "hessian"]


@dataclass(frozen=True)
class PreprocessingOptions:
    """Configurable preprocessing kept conservative for already-denoised tomograms."""

    invert_density: bool = False
    normalization: NormalizationMethod = "none"
    clip_percentiles: tuple[float, float] | None = None
    filter_method: FilterMethod = "none"
    gaussian_sigma: float = 1.0
    median_size: int = 3
    hessian_sigmas: tuple[float, ...] = (1.0, 2.0, 3.0)
    hessian_alpha: float = 0.5
    hessian_beta: float = 0.5
    hessian_gamma: float = 15.0
    hessian_black_ridges: bool = True
    fill_outside_roi: float | None = 0.0


@dataclass(frozen=True)
class PreprocessedVolume:
    """Preprocessed tomogram data plus the options used to produce it."""

    volume: NDArray[np.float32]
    roi_mask: NDArray[np.bool_]
    options: PreprocessingOptions


def preprocess_tracing_inputs(
    tracing_inputs: TracingInputs,
    options: PreprocessingOptions | None = None,
) -> PreprocessedVolume:
    """Preprocess a loaded tomogram using statistics estimated inside the ROI."""

    return preprocess_volume(
        tracing_inputs.volume,
        tracing_inputs.roi_mask,
        options=options,
    )


def write_preprocessed_mrc(
    path: str | Path,
    preprocessed: PreprocessedVolume,
    *,
    reference_path: str | Path | None = None,
    voxel_size_angst: float | None = None,
    overwrite: bool = True,
) -> Path:
    """Write a preprocessed tomogram as an MRC file."""

    path = Path(path)
    if reference_path is not None:
        voxel_size_angst = load_mrc_voxel_size_angst(reference_path)
    if voxel_size_angst is None:
        raise ValueError("A reference MRC or voxel size is required to write a preprocessed tomogram")

    with mrcfile.new(path, overwrite=overwrite) as mrc:
        mrc.set_data(np.asarray(preprocessed.volume, dtype=np.float32))
        mrc.voxel_size = voxel_size_angst

    return path


def preprocess_volume(
    volume: NDArray[np.float32],
    roi_mask: NDArray[np.bool_],
    *,
    options: PreprocessingOptions | None = None,
) -> PreprocessedVolume:
    """Apply optional inversion, normalization, clipping, and filtering."""

    options = options or PreprocessingOptions()
    _validate_preprocessing_inputs(volume, roi_mask, options)

    processed = np.asarray(volume, dtype=np.float32).copy()
    if options.invert_density:
        processed *= -1.0

    if options.clip_percentiles is not None:
        processed = _clip_by_roi_percentiles(processed, roi_mask, options.clip_percentiles)

    if options.normalization == "zscore":
        processed = _zscore_normalize_roi(processed, roi_mask)
    elif options.normalization == "robust":
        processed = _robust_normalize_roi(processed, roi_mask)

    processed = _filter_volume(processed, options)

    if options.fill_outside_roi is not None:
        processed = processed.copy()
        processed[~roi_mask] = np.float32(options.fill_outside_roi)

    return PreprocessedVolume(
        volume=np.asarray(processed, dtype=np.float32),
        roi_mask=roi_mask,
        options=options,
    )


def _validate_preprocessing_inputs(
    volume: NDArray[np.float32],
    roi_mask: NDArray[np.bool_],
    options: PreprocessingOptions,
) -> None:
    if volume.shape != roi_mask.shape:
        raise ValueError(f"ROI mask shape {roi_mask.shape} does not match volume shape {volume.shape}")
    if not np.any(roi_mask):
        raise ValueError("ROI mask is empty")
    if not np.isfinite(volume[roi_mask]).all():
        raise ValueError("Volume contains non-finite values inside the ROI")
    if options.clip_percentiles is not None:
        low, high = options.clip_percentiles
        if not (0 <= low < high <= 100):
            raise ValueError("Clip percentiles must satisfy 0 <= low < high <= 100")
    if options.gaussian_sigma <= 0:
        raise ValueError("Gaussian sigma must be positive")
    if options.median_size <= 0:
        raise ValueError("Median size must be positive")
    if options.median_size % 2 == 0:
        raise ValueError("Median size must be odd")
    if not options.hessian_sigmas:
        raise ValueError("Hessian sigmas must contain at least one scale")
    if any(sigma <= 0 for sigma in options.hessian_sigmas):
        raise ValueError("Hessian sigmas must all be positive")
    if options.hessian_alpha <= 0:
        raise ValueError("Hessian alpha must be positive")
    if options.hessian_beta <= 0:
        raise ValueError("Hessian beta must be positive")
    if options.hessian_gamma <= 0:
        raise ValueError("Hessian gamma must be positive")


def _clip_by_roi_percentiles(
    volume: NDArray[np.float32],
    roi_mask: NDArray[np.bool_],
    percentiles: tuple[float, float],
) -> NDArray[np.float32]:
    low, high = np.percentile(volume[roi_mask], percentiles)
    return np.clip(volume, low, high).astype(np.float32)


def _zscore_normalize_roi(
    volume: NDArray[np.float32],
    roi_mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    roi_values = volume[roi_mask]
    mean = float(np.mean(roi_values))
    std = float(np.std(roi_values))
    if std == 0:
        raise ValueError("Cannot z-score normalize an ROI with zero standard deviation")
    return ((volume - mean) / std).astype(np.float32)


def _robust_normalize_roi(
    volume: NDArray[np.float32],
    roi_mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    roi_values = volume[roi_mask]
    median = float(np.median(roi_values))
    q1, q3 = np.percentile(roi_values, [25, 75])
    iqr = float(q3 - q1)
    if iqr == 0:
        raise ValueError("Cannot robust-normalize an ROI with zero interquartile range")
    return ((volume - median) / iqr).astype(np.float32)


def _filter_volume(
    volume: NDArray[np.float32],
    options: PreprocessingOptions,
) -> NDArray[np.float32]:
    if options.filter_method == "none":
        return volume
    if options.filter_method == "gaussian":
        return ndimage.gaussian_filter(volume, sigma=options.gaussian_sigma).astype(np.float32)
    if options.filter_method == "median":
        return ndimage.median_filter(volume, size=options.median_size).astype(np.float32)
    if options.filter_method == "hessian":
        return hessian_filter(
            volume,
            sigmas=options.hessian_sigmas,
            alpha=options.hessian_alpha,
            beta=options.hessian_beta,
            gamma=options.hessian_gamma,
            black_ridges=options.hessian_black_ridges,
        ).astype(np.float32)
    raise ValueError(f"Unsupported filter method: {options.filter_method}")
