"""Export saved filament coordinates for WarpTools ts_export_particles."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import starfile

from filament_tracer.models import SkeletonProject
from filament_tracer.relion_export import coordinate_table


def warp_coordinate_table(
    project, inter_box_distance, coords_angpix, tomo_name, offset_xyz_angst=(0, 0, 0)
):
    """Uncentered XYZ pixels; offset maps the picked volume into Warp's volume.

    Pass the same coords_angpix to WarpTools. No normalization, centering,
    axis flips, or one-pixel shifts are applied.
    """
    if not np.isfinite(coords_angpix) or coords_angpix <= 0:
        raise ValueError("Coordinate pixel size must be finite and positive")
    offset = np.asarray(offset_xyz_angst, dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("Offset must contain three finite XYZ values in angstroms")
    table = coordinate_table(project, inter_box_distance, coords_angpix, tomo_name)
    columns = [f"rlnCoordinate{axis}" for axis in "XYZ"]
    xyz = table[columns].to_numpy() + offset / coords_angpix
    if not np.isfinite(xyz).all() or np.any(xyz < 0):
        raise ValueError("Transformed Warp coordinates must be finite and nonnegative")
    table[columns] = xyz
    return table.rename(columns={"rlnTomoName": "rlnMicrographName"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skeleton", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--inter-box-distance", required=True, type=float,
        help="Spacing along the filament in angstroms",
    )
    parser.add_argument(
        "--coords-angpix", required=True, type=float,
        help="Coordinate pixel size in angstroms; pass as WarpTools --coords_angpix",
    )
    parser.add_argument(
        "--tomo-name", required=True,
        help="Exact Warp tilt-series name, as used in its particle STAR files",
    )
    parser.add_argument(
        "--offset-xyz-angst", nargs=3, type=float, default=(0, 0, 0),
        metavar=("X", "Y", "Z"),
        help="Translation from picked-volume origin to Warp-volume origin in angstroms",
    )
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("Output exists; choose a new filename")
        project = SkeletonProject.load(args.skeleton)
        table = warp_coordinate_table(
            project, args.inter_box_distance, args.coords_angpix,
            args.tomo_name, args.offset_xyz_angst,
        )
        starfile.write({"particles": table}, args.output, overwrite=False)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(
        f"Exported {len(table)} coordinates to {args.output}. "
        "Use WarpTools ts_export_particles with "
        f"--coords_angpix {args.coords_angpix:g} "
        "and --2d (not --normalized_coords)."
    )


if __name__ == "__main__":
    main()
