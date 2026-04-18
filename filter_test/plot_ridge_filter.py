"""Compare skimage ridge filters on a user-selected CryoET tomogram ROI.

Run from VS Code or the terminal, for example:

    python filter_test/plot_ridge_filter.py

or with custom settings:

    python filter_test/plot_ridge_filter.py \
        --volume rec_Position_1_3.mrc \
        --roi 200:300,320:400,60:90 \
        --slice-z 75 \
        --output filter_test/ridge_comparison.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import frangi, hessian, meijering, sato

from filament_trace_tomo.inputs import create_roi_mask_from_bounds, load_mrc_data, parse_roi_bounds


def original(image: np.ndarray, **_: object) -> np.ndarray:
    """Return the original image, ignoring any extra keyword arguments."""

    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--volume",
        type=Path,
        default=PROJECT_ROOT / "filter_test/Position_58_corrected.mrc",
        help="Path to the input MRC tomogram.",
    )
    parser.add_argument(
        "--roi",
        default="85:209,368:484,197:198",
        help="ROI bounds in x_start:x_stop,y_start:y_stop,z_start:z_stop voxel coordinates.",
    )
    parser.add_argument(
        "--slice-z",
        type=int,
        default=None,
        help="Global z index to display. Defaults to the ROI midpoint.",
    )
    parser.add_argument(
        "--black-ridges",
        action="store_true",
        help="Detect dark ridges instead of bright ridges.",
    )
    parser.add_argument(
        "--sigmas",
        default="1,2,3",
        help="Comma-separated sigma values for the ridge filters.",
    )
    parser.add_argument(
        "--clip-percentiles",
        default="1,99",
        help="Percentiles for display normalization of the plotted slice.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path. If omitted, the plot is shown interactively.",
    )
    return parser.parse_args()


def parse_sigmas(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parse_percentiles(text: str) -> tuple[float, float]:
    low_text, high_text = [part.strip() for part in text.split(",")]
    low, high = float(low_text), float(high_text)
    if not (0 <= low < high <= 100):
        raise ValueError("Display clip percentiles must satisfy 0 <= low < high <= 100")
    return low, high


def crop_roi(volume: np.ndarray, roi_bounds: str) -> tuple[np.ndarray, tuple[int, int, int]]:
    (x_start, x_stop), (y_start, y_stop), (z_start, z_stop) = parse_roi_bounds(roi_bounds)
    roi_mask = create_roi_mask_from_bounds(volume.shape, roi_bounds)
    cropped = volume[z_start:z_stop, y_start:y_stop, x_start:x_stop]
    if not np.any(roi_mask[z_start:z_stop, y_start:y_stop, x_start:x_stop]):
        raise ValueError("Requested ROI crop is empty")
    return cropped.astype(np.float32), (x_start, y_start, z_start)


def normalize_for_display(image: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    low, high = np.percentile(image, percentiles)
    if high == low:
        return np.zeros_like(image, dtype=np.float32)
    clipped = np.clip(image, low, high)
    return ((clipped - low) / (high - low)).astype(np.float32)


def main() -> None:
    args = parse_args()
    volume_path = args.volume if args.volume.is_absolute() else PROJECT_ROOT / args.volume
    volume = load_mrc_data(volume_path)
    roi_volume, (x_start, y_start, z_start) = crop_roi(volume, args.roi)

    sigmas = parse_sigmas(args.sigmas)
    display_percentiles = parse_percentiles(args.clip_percentiles)

    if args.slice_z is None:
        slice_z_global = z_start + roi_volume.shape[0] // 2
    else:
        slice_z_global = args.slice_z

    slice_z_local = slice_z_global - z_start
    if not (0 <= slice_z_local < roi_volume.shape[0]):
        raise ValueError(
            f"slice-z {slice_z_global} is outside ROI z range [{z_start}, {z_start + roi_volume.shape[0] - 1}]"
        )

    slice_2d = roi_volume[slice_z_local]
    cmap = plt.cm.gray
    filter_specs = [
        ("original", original),
        ("meijering", meijering),
        ("sato", sato),
        ("frangi", frangi),
        ("hessian", hessian),
    ]

    fig, axes = plt.subplots(1, len(filter_specs), figsize=(4 * len(filter_specs), 4))
    if len(filter_specs) == 1:
        axes = [axes]

    for axis, (name, func) in zip(axes, filter_specs):
        if func is original:
            result = func(slice_2d)
        else:
            result = func(slice_2d, sigmas=sigmas, black_ridges=args.black_ridges)
        axis.imshow(normalize_for_display(np.asarray(result, dtype=np.float32), display_percentiles), cmap=cmap)
        axis.set_title(name)
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle(
        f"{volume_path.name} | ROI x={x_start}:{x_start + roi_volume.shape[2]}, "
        f"y={y_start}:{y_start + roi_volume.shape[1]}, z={z_start}:{z_start + roi_volume.shape[0]} | "
        f"slice z={slice_z_global} | black_ridges={args.black_ridges}"
    )
    fig.tight_layout()

    if args.output is not None:
        output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved ridge comparison figure to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
