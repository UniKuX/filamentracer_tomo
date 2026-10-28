#!/usr/bin/env python3
"""Export filament-tracer skeleton JSON as RELION tomography particles STAR.

Dependencies: numpy, pandas, starfile, scipy, morphosamplers. No filament-tracer installation needed.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def build_rows(project, metadata, spacing, center='relion-picker', polarity_known=False):
    """Use RELION afdd816's filament pose generation and STAR conventions."""
    from morphosamplers import Path as MorphPath, path_samplers
    from scipy.spatial.transform import Rotation as R

    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('Spacing must be finite and positive')
    voxel = np.asarray(project['volume']['voxel_size_zyx'], dtype=float)
    shape = np.asarray(project['volume']['shape_zyx'], dtype=float)
    size = np.array([metadata['rlnTomoSize'+a] for a in 'XYZ'], dtype=float)
    pixel = float(metadata['rlnTomoTiltSeriesPixelSize'])
    if any(v.shape != (3,) for v in (voxel, shape, size)) or not np.isfinite(np.r_[voxel, shape, size, pixel]).all() or np.any(np.r_[voxel, shape, size, pixel] <= 0):
        raise ValueError('Invalid voxel size, tomogram dimensions, or pixel size')
    if not np.allclose((shape*voxel)[::-1], size*pixel, rtol=0.01, atol=2*max(voxel)):
        raise ValueError('Skeleton volume extent differs from RELION tomogram. Register cropped/resized volumes before export.')
    offset = (size/2 - (1 if center == 'relion-picker' else 0))*pixel
    rotated_basis = R.from_euler("y", angles=90, degrees=True).as_matrix()
    rot_prior, tilt_prior, psi_prior = R.from_matrix(rotated_basis).as_euler("ZYZ", degrees=True)
    rows, ids = [], set()
    for filament in project['filaments']:
        tube = filament['filament_id']
        if not isinstance(tube, int) or tube < 1 or tube in ids:
            raise ValueError('Filament IDs must be unique positive integers')
        ids.add(tube)
        p = np.asarray([x['position_zyx'] for x in filament['points']], dtype=float)
        if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2 or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > shape-1):
            raise ValueError(f'Invalid or out-of-bounds points in filament {tube}')
        xyz = (p*voxel)[:, ::-1] - offset
        xyz = xyz[np.r_[True, np.linalg.norm(np.diff(xyz, axis=0), axis=1) >= 1e-6]]
        if len(xyz) < 2:
            raise ValueError(f'Zero-length filament {tube}')
        path = MorphPath(control_points=xyz)
        poses = path_samplers.HelicalPoseSampler(spacing=spacing, twist=0).sample(path)
        if len(poses) == 0:
            raise ValueError(f'Filament {tube} produces no poses at this spacing')
        rotated_orientations = poses.orientations @ rotated_basis
        eulers = R.from_matrix(rotated_orientations).inv().as_euler('ZYZ', degrees=True)
        points = path_samplers.PointSampler(spacing=0.1).sample(path)
        total_length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        # Intentionally reproduce upstream, including its pixel-size division
        # despite the column name ending in Angst.
        distances = np.linspace(0, 1, num=len(poses)) * (total_length / pixel)
        for position, angles, distance in zip(poses.positions, eulers, distances):
            row = {'rlnTomoName': str(metadata['rlnTomoName']), 'rlnHelicalTubeID': tube,
                   'rlnHelicalTrackLengthAngst': float(distance)}
            row.update(zip(['rlnCenteredCoordinate'+a+'Angst' for a in 'XYZ'], position))
            row.update(zip(['rlnTomoSubtomogram'+a for a in ('Rot', 'Tilt', 'Psi')], angles))
            row.update(rlnAngleRot=rot_prior, rlnAngleTilt=tilt_prior, rlnAnglePsi=psi_prior, rlnAngleTiltPrior=tilt_prior, rlnAnglePsiPrior=psi_prior)
            if not polarity_known:
                row['rlnAnglePsiFlipRatio'] = 0.5
            rows.append(row)
    if not rows:
        raise ValueError('Skeleton contains no filaments')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skeleton', required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--tomograms', type=Path, help='RELION global tomogram/tilt-series STAR containing sizes and pixel size')
    source.add_argument('--relion-pixel-size', type=float, help='Unbinned tilt-series pixel size in Å; derive volume extent from skeleton metadata')
    parser.add_argument('--tomo-name', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--inter-box-distance', required=True, type=float, help='Physical spacing in Å')
    parser.add_argument('--center-convention', choices=['relion-picker', 'relion-cpp'], default='relion-picker', help='Default matches Python filament picker: (size/2-1)*pixel; cpp uses size/2*pixel')
    parser.add_argument('--polarity-known', action='store_true', help='Omit 0.5 psi flip ratio; use only for known directed skeletons')
    args = parser.parse_args()
    try:
        import pandas as pd
        import starfile
        if args.output.exists():
            raise ValueError('Output exists; choose a new filename')
        project = json.loads(args.skeleton.read_text())
        if args.relion_pixel_size is not None:
            pixel = args.relion_pixel_size
            if not np.isfinite(pixel) or pixel <= 0:
                raise ValueError('RELION pixel size must be finite and positive')
            shape = np.asarray(project['volume']['shape_zyx'], dtype=float)
            voxel = np.asarray(project['volume']['voxel_size_zyx'], dtype=float)
            if shape.shape != (3,) or voxel.shape != (3,):
                raise ValueError('Skeleton dimensions and voxel sizes must have three ZYX components')
            size = (shape*voxel)[::-1]/pixel
            metadata = dict(rlnTomoName=args.tomo_name, rlnTomoTiltSeriesPixelSize=pixel)
            metadata.update(zip(['rlnTomoSize'+a for a in 'XYZ'], size))
        else:
            blocks = starfile.read(args.tomograms, always_dict=True, parse_as_string=['rlnTomoName'])
            required = {'rlnTomoName', 'rlnTomoTiltSeriesPixelSize', 'rlnTomoSizeX', 'rlnTomoSizeY', 'rlnTomoSizeZ'}
            matches = []
            for table in blocks.values():
                if isinstance(table, pd.DataFrame) and required.issubset(table.columns):
                    matches.extend(table.loc[table.rlnTomoName == args.tomo_name].to_dict('records'))
            if len(matches) != 1:
                raise ValueError('Expected exactly one matching tomogram row with dimensions and tilt-series pixel size')
            metadata = matches[0]
        rows = build_rows(project, metadata, args.inter_box_distance, args.center_convention, args.polarity_known)
        starfile.write({'particles': pd.DataFrame(rows)}, args.output, overwrite=False)
        print(f'Wrote {len(rows)} particles to {args.output}. Center convention: {args.center_convention}.')
    except (ValueError, OSError, KeyError, ImportError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
