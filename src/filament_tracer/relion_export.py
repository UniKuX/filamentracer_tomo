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


def coordinate_table(project, inter_box_distance, relion_pixel_size, tomo_name):
    """Uncentered XYZ coordinates in unbinned RELION pixels, with tube IDs."""
    if not np.isfinite(relion_pixel_size) or relion_pixel_size <= 0:
        raise ValueError("RELION pixel size must be finite and positive")
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
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("Output exists; choose a new filename")
        project = SkeletonProject.load(args.skeleton)
        table = coordinate_table(
            project, args.inter_box_distance, args.relion_pixel_size, args.tomo_name
        )
        starfile.write({"particles": table}, args.output, overwrite=False)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(
        f"Exported {len(table)} coordinates from {len(project.filaments)} filaments "
        f"to {args.output}. Import this coordinate STAR into RELION first."
    )


if __name__ == "__main__":
    main()
