#!/usr/bin/env python3
"""Export filament-tracer skeleton JSON as a WarpTools RELION 3.0 STAR file."""
import argparse
import json
from pathlib import Path
import warnings

import numpy as np


def build_rows(project, tomo_name, spacing, coords_angpix, polarity_known=False):
    """Sample physical XYZ poses, with the filament along the particle Z axis."""
    from morphosamplers import Path as MorphPath, path_samplers
    from scipy.spatial.transform import Rotation

    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('Spacing must be finite and positive')
    if not np.isfinite(coords_angpix) or coords_angpix <= 0:
        raise ValueError('Coordinate pixel size must be finite and positive')
    # Warp groups on the literal name and its STAR reader splits on whitespace.
    if (not isinstance(tomo_name, str) or not tomo_name.endswith('.tomostar')
            or any(c.isspace() or c in "'\"/\\#" for c in tomo_name)):
        raise ValueError('Tomogram name must be a .tomostar filename without paths, quotes, or whitespace')
    voxel = np.asarray(project['volume']['voxel_size_zyx'], dtype=float)
    shape = np.asarray(project['volume']['shape_zyx'], dtype=float)
    if (voxel.shape != (3,) or shape.shape != (3,)
            or not np.isfinite(np.r_[voxel, shape]).all()
            or np.any(voxel <= 0) or np.any(shape < 1)):
        raise ValueError('Invalid skeleton voxel sizes or dimensions')

    rows, ids = [], set()
    for filament in project['filaments']:
        tube = filament['filament_id']
        if type(tube) is not int or tube < 1 or tube in ids:
            raise ValueError('Filament IDs must be unique positive integers')
        ids.add(tube)
        points = np.asarray([p['position_zyx'] for p in filament['points']], dtype=float)
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
                or not np.isfinite(points).all() or np.any(points < 0)
                or np.any(points > shape - 1)):
            raise ValueError(f'Invalid or out-of-bounds points in filament {tube}')
        xyz = (points * voxel)[:, ::-1]
        xyz = xyz[np.r_[True, np.linalg.norm(np.diff(xyz, axis=0), axis=1) >= 1e-6]]
        if len(xyz) < 2:
            raise ValueError(f'Zero-length filament {tube}')
        path = MorphPath(control_points=xyz)
        poses = path_samplers.HelicalPoseSampler(spacing=spacing, twist=0).sample(path)
        if len(poses) == 0:
            raise ValueError(f'Filament {tube} produces no poses at this spacing')
        # Full tomogram-to-particle rotation. RELION 5's extra Y basis rotation
        # belongs to its split subtomogram/particle representation, not this file.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Gimbal lock detected', category=UserWarning)
            angles = Rotation.from_matrix(poses.orientations).inv().as_euler('ZYZ', degrees=True)
        fine_points = path_samplers.PointSampler(spacing=0.1).sample(path)
        length = np.linalg.norm(np.diff(fine_points, axis=0), axis=1).sum()
        distances = np.linspace(0, length, len(poses))
        for position, euler, distance in zip(poses.positions, angles, distances):
            row = dict(zip(['rlnCoordinate' + a for a in 'XYZ'], position / coords_angpix))
            row.update(zip(['rlnAngle' + a for a in ('Rot', 'Tilt', 'Psi')], euler))
            row.update(rlnMicrographName=tomo_name, rlnHelicalTubeID=tube,
                       rlnHelicalTrackLengthAngst=float(distance),
                       rlnAngleTiltPrior=float(euler[1]), rlnAnglePsiPrior=float(euler[2]))
            if not polarity_known:
                row['rlnAnglePsiFlipRatio'] = 0.5
            rows.append(row)
    if not rows:
        raise ValueError('Skeleton contains no filaments')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skeleton', required=True, type=Path)
    parser.add_argument('--tomo-name', required=True, help='Exact Warp filename, e.g. TS_01.tomostar')
    parser.add_argument('--coords-angpix', required=True, type=float,
                        help='Coordinate pixel size in Å; pass the same value to Warp --coords_angpix')
    parser.add_argument('--inter-box-distance', required=True, type=float, help='Physical spacing in Å')
    parser.add_argument('--polarity-known', action='store_true', help='Omit the unknown-polarity psi flip ratio')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        import pandas as pd
        import starfile
        if args.output.exists():
            raise ValueError('Output exists; choose a new filename')
        project = json.loads(args.skeleton.read_text())
        rows = build_rows(project, args.tomo_name, args.inter_box_distance,
                          args.coords_angpix, args.polarity_known)
        starfile.write({'': pd.DataFrame(rows)}, args.output, overwrite=False)
        print(f'Wrote {len(rows)} particles to {args.output}. '
              f'Use WarpTools --coords_angpix {args.coords_angpix:g}.')
    except (ValueError, TypeError, OSError, KeyError, ImportError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
