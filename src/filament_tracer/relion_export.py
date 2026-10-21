"""Standalone skeleton resampling and RELION coordinate-import STAR export."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import starfile

from filament_tracer.models import SkeletonProject


def resample_path(points, voxel_size, distance):
    """Return physical ZYX samples and arclengths, starting at the first vertex.

    Consecutive duplicate vertices are ignored. The last endpoint is included
    only if it falls at a spacing multiple (no shortened final interval).
    """
    points = np.asarray(points, dtype=float)
    voxel = np.asarray(voxel_size, dtype=float)
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("Inter-box distance must be finite and positive")
    if voxel.shape != (3,) or not np.isfinite(voxel).all() or np.any(voxel <= 0):
        raise ValueError("Voxel spacing must be finite and positive")
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Skeleton coordinates must be finite N x 3 values")
    if len(points) < 2:
        raise ValueError("Each filament needs at least two distinct vertices")
    physical = points * voxel
    keep = np.r_[True, np.linalg.norm(np.diff(physical, axis=0), axis=1) > 1e-8]
    physical = physical[keep]
    if len(physical) < 2:
        raise ValueError("Each filament needs at least two distinct vertices")
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(physical, axis=0), axis=1))]
    count = int(np.floor(arc[-1] / distance + 1e-10)) + 1
    if count > 10_000_000:
        raise ValueError("Too many particles; increase inter-box distance")
    sample_arc = np.minimum(np.arange(count) * distance, arc[-1])
    samples = np.column_stack(
        [np.interp(sample_arc, arc, physical[:, axis]) for axis in range(3)]
    )
    return samples, sample_arc


def helical_priors(points, voxel_size, sample_arc, window):
    """Local physical tangents -> RELION 3D helical tilt/psi, in degrees.

    The reference helix lies along Z. Its rotated axis is
    (-cos(psi)*sin(tilt), sin(psi)*sin(tilt), cos(tilt)) in tomogram XYZ.
    """
    if not np.isfinite(window) or window <= 0:
        raise ValueError("Tangent window must be finite and positive")
    physical = np.asarray(points, dtype=float) * np.asarray(voxel_size)
    keep = np.r_[True, np.linalg.norm(np.diff(physical, axis=0), axis=1) > 1e-8]
    physical = physical[keep]
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(physical, axis=0), axis=1))]
    low = np.maximum(0, sample_arc - window / 2)
    high = np.minimum(arc[-1], sample_arc + window / 2)
    tangent = np.column_stack(
        [
            np.interp(high, arc, physical[:, axis])
            - np.interp(low, arc, physical[:, axis])
            for axis in (2, 1, 0)
        ]
    )
    lengths = np.linalg.norm(tangent, axis=1)
    if np.any(lengths < 1e-8):
        raise ValueError(
            "Undefined local tangent; check path reversals or tangent window"
        )
    tangent /= lengths[:, None]
    tilt = np.degrees(np.arccos(np.clip(tangent[:, 2], -1, 1)))
    psi = np.degrees(np.arctan2(tangent[:, 1], -tangent[:, 0]))
    # At the poles azimuth is undetermined: use a deterministic zero.
    psi[np.linalg.norm(tangent[:, :2], axis=1) < 1e-12] = 0
    return tilt, psi


def coordinate_table(
    project,
    inter_box_distance,
    relion_pixel_size,
    tomo_name,
    *,
    tangent_window=None,
):
    """Uncentered XYZ coordinates in unbinned RELION pixels, with tube IDs."""
    if not np.isfinite(relion_pixel_size) or relion_pixel_size <= 0:
        raise ValueError("RELION pixel size must be finite and positive")
    window = inter_box_distance if tangent_window is None else tangent_window
    if not tomo_name.strip() or any(c.isspace() for c in tomo_name):
        raise ValueError("Tomogram name must be nonempty and contain no whitespace")
    if not project.filaments:
        raise ValueError("Skeleton contains no filaments")
    ids = [f.filament_id for f in project.filaments]
    if len(set(ids)) != len(ids) or any(i < 1 for i in ids):
        raise ValueError("Filament IDs must be unique positive integers")
    shape = np.asarray(project.volume.shape_zyx)
    if np.any(shape <= 0):
        raise ValueError("Invalid tomogram shape")
    tables = []
    for filament in project.filaments:
        points = np.asarray([p.position_zyx for p in filament.points], dtype=float)
        try:
            samples, arc = resample_path(
                points, project.volume.voxel_size_zyx, inter_box_distance
            )
            if np.any(points < 0) or np.any(points > shape - 1):
                raise ValueError("Skeleton vertices lie outside the tomogram")
            tilt, psi = helical_priors(
                points, project.volume.voxel_size_zyx, arc, window
            )
        except ValueError as error:
            raise ValueError(f"Filament {filament.filament_id}: {error}") from error
        xyz = samples[:, ::-1] / relion_pixel_size
        tables.append(
            pd.DataFrame(
                {
                    "rlnTomoName": tomo_name,
                    "rlnCoordinateX": xyz[:, 0],
                    "rlnCoordinateY": xyz[:, 1],
                    "rlnCoordinateZ": xyz[:, 2],
                    "rlnHelicalTubeID": filament.filament_id,
                    "rlnHelicalTrackLengthAngst": arc,
                    "rlnAnglePsiPrior": psi,
                    "rlnAngleTiltPrior": tilt,
                    "rlnAnglePsi": psi,
                    "rlnAngleTilt": tilt,
                }
            )
        )
    return pd.concat(tables, ignore_index=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skeleton", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--inter-box-distance",
        required=True,
        type=float,
        help="Spacing along each skeleton in angstroms",
    )
    parser.add_argument(
        "--relion-pixel-size",
        required=True,
        type=float,
        help="Unbinned motion-corrected tilt-series pixel size in Å",
    )
    parser.add_argument(
        "--tomo-name",
        required=True,
        help="Exact rlnTomoName in RELION's tomograms.star",
    )
    parser.add_argument(
        "--tangent-window",
        type=float,
        help="Full local tangent window in Å (default: inter-box distance)",
    )
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("Output exists; choose a new filename")
        project = SkeletonProject.load(args.skeleton)
        table = coordinate_table(
            project,
            args.inter_box_distance,
            args.relion_pixel_size,
            args.tomo_name,
            tangent_window=args.tangent_window,
        )
        starfile.write({"particles": table}, args.output, overwrite=False)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(
        f"Exported {len(table)} coordinates from {len(project.filaments)} filaments "
        f"to {args.output}. Import this coordinate STAR into RELION first."
    )
    print(
        "Tilt/Psi follow skeleton point order; biological polarity is unknown. "
        "Axial rotation angles and priors are not exported."
    )


if __name__ == "__main__":
    main()
