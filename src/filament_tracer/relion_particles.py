"""Read-only RELION particle-position conversion for tomogram overlays."""

import numpy as np
import pandas as pd
import starfile


def read_particles(path):
    blocks = starfile.read(path, always_dict=True, parse_as_string=["rlnTomoName"])
    table = blocks.get("particles")
    if not isinstance(table, pd.DataFrame):
        candidates = [
            v
            for v in blocks.values()
            if isinstance(v, pd.DataFrame) and "rlnTomoName" in v
        ]
        if len(candidates) != 1:
            raise ValueError("Expected one particles table with rlnTomoName")
        table = candidates[0]
    if table.empty or "rlnTomoName" not in table:
        raise ValueError("Particle table is empty or lacks rlnTomoName")
    if table.rlnTomoName.isna().any():
        raise ValueError("Missing tomogram names")
    return table, blocks.get("optics", pd.DataFrame())


def numeric_triplet(table, columns, optional=False):
    present = [c in table for c in columns]
    if not any(present) and optional:
        return np.zeros((len(table), 3))
    if not all(present):
        raise ValueError(f"Missing coordinate/angle fields: {', '.join(columns)}")
    values = table[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite particle coordinates, shifts, or angles")
    return values


def subtomogram_matrices(angles):
    """RELION Euler::anglesToMatrix3, mapping subtomogram to tomogram XYZ."""
    phi, theta, psi = np.radians(angles).T
    cp, ct, cc = np.cos(phi), np.cos(theta), np.cos(psi)
    sp, st, sc = np.sin(phi), np.sin(theta), np.sin(psi)
    return np.array(
        [
            [cc * ct * cp - sc * sp, cc * ct * sp + sc * cp, -cc * st],
            [-sc * ct * cp - cc * sp, -sc * ct * sp + cc * cp, sc * st],
            [st * cp, st * sp, ct],
        ]
    ).transpose(2, 0, 1)


def particle_positions(table, optics, voxel_zyx, center_zyx, apply_shifts=True):
    """Convert a single-tomogram table to zero-based ZYX data coordinates.

    Center is explicit in displayed-volume voxels (needed for cropped volumes).
    Refined position follows RELION ParticleSet::getPosition: coordinate minus
    subtomogram_matrix @ origin_shift. rlnAngleRot/Tilt/Psi do not rotate shifts.
    """
    voxel = np.asarray(voxel_zyx, dtype=float)
    center = np.asarray(center_zyx, dtype=float)
    if (
        voxel.shape != (3,)
        or not np.isfinite(voxel).all()
        or np.any(voxel <= 0)
        or center.shape != (3,)
        or not np.isfinite(center).all()
    ):
        raise ValueError("Invalid tomogram voxel size or center")
    columns = [f"rlnCenteredCoordinate{a}Angst" for a in "XYZ"]
    centered = any(c in table for c in columns)
    if centered:
        xyz = numeric_triplet(table, columns)
    else:
        xyz = numeric_triplet(table, [f"rlnCoordinate{a}" for a in "XYZ"])
        label = "rlnTomoTiltSeriesPixelSize"
        if label in table:
            pixel = table[label].to_numpy(dtype=float)
        elif (
            isinstance(optics, pd.DataFrame)
            and label in optics
            and "rlnOpticsGroup" in optics
            and "rlnOpticsGroup" in table
        ):
            if optics.rlnOpticsGroup.duplicated().any():
                raise ValueError("Duplicate optics group IDs")
            pixel = table.rlnOpticsGroup.map(
                optics.set_index("rlnOpticsGroup")[label]
            ).to_numpy(dtype=float)
        else:
            raise ValueError(
                "Pixel coordinates require rlnTomoTiltSeriesPixelSize "
                "and matching optics groups"
            )
        if not np.isfinite(pixel).all() or np.any(pixel <= 0):
            raise ValueError("Missing or invalid tilt-series pixel size")
        xyz = xyz * pixel[:, None]
    if apply_shifts:
        if any(f"rlnOrigin{a}" in table for a in "XYZ") and not any(
            f"rlnOrigin{a}Angst" in table for a in "XYZ"
        ):
            raise ValueError(
                "Legacy pixel origin shifts are unsupported; "
                "disable shifts or convert them to angstroms"
            )
        shifts = numeric_triplet(table, [f"rlnOrigin{a}Angst" for a in "XYZ"], True)
        angles = numeric_triplet(
            table, [f"rlnTomoSubtomogram{a}" for a in ("Rot", "Tilt", "Psi")], True
        )
        xyz = xyz - np.einsum("nij,nj->ni", subtomogram_matrices(angles), shifts)
    zyx = xyz[:, ::-1] / voxel
    if centered:
        zyx += center
    return zyx
