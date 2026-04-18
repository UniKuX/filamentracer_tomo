"""RELION 5 filament particle STAR export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import starfile


RELION5_FILAMENT_COLUMNS = [
    "rlnTomoName",
    "rlnHelicalTubeID",
    "rlnHelicalTrackLengthAngst",
    "rlnCenteredCoordinateXAngst",
    "rlnCenteredCoordinateYAngst",
    "rlnCenteredCoordinateZAngst",
    "rlnTomoSubtomogramRot",
    "rlnTomoSubtomogramTilt",
    "rlnTomoSubtomogramPsi",
    "rlnAngleRot",
    "rlnAngleTilt",
    "rlnAnglePsi",
    "rlnAngleTiltPrior",
    "rlnAnglePsiPrior",
]

RELION5_FLOAT_COLUMNS = [
    column
    for column in RELION5_FILAMENT_COLUMNS
    if column not in {"rlnTomoName", "rlnHelicalTubeID"}
]


@dataclass(frozen=True)
class Relion5FilamentParticle:
    """One sampled filament coordinate row for RELION 5 subtomogram workflows."""

    tomo_name: str
    helical_tube_id: int
    helical_track_length_angst: float
    centered_coordinate_x_angst: float
    centered_coordinate_y_angst: float
    centered_coordinate_z_angst: float
    tomo_subtomogram_rot: float
    tomo_subtomogram_tilt: float
    tomo_subtomogram_psi: float
    angle_rot: float
    angle_tilt: float
    angle_psi: float
    angle_tilt_prior: float
    angle_psi_prior: float

    def to_star_row(self) -> dict[str, object]:
        return {
            "rlnTomoName": self.tomo_name,
            "rlnHelicalTubeID": self.helical_tube_id,
            "rlnHelicalTrackLengthAngst": self.helical_track_length_angst,
            "rlnCenteredCoordinateXAngst": self.centered_coordinate_x_angst,
            "rlnCenteredCoordinateYAngst": self.centered_coordinate_y_angst,
            "rlnCenteredCoordinateZAngst": self.centered_coordinate_z_angst,
            "rlnTomoSubtomogramRot": self.tomo_subtomogram_rot,
            "rlnTomoSubtomogramTilt": self.tomo_subtomogram_tilt,
            "rlnTomoSubtomogramPsi": self.tomo_subtomogram_psi,
            "rlnAngleRot": self.angle_rot,
            "rlnAngleTilt": self.angle_tilt,
            "rlnAnglePsi": self.angle_psi,
            "rlnAngleTiltPrior": self.angle_tilt_prior,
            "rlnAnglePsiPrior": self.angle_psi_prior,
        }


def particles_to_dataframe(
    particles: Iterable[Relion5FilamentParticle],
) -> pd.DataFrame:
    """Convert sampled filament particles to the exact RELION 5 column order."""

    dataframe = pd.DataFrame([particle.to_star_row() for particle in particles])
    dataframe = dataframe.reindex(columns=RELION5_FILAMENT_COLUMNS)
    if not dataframe.empty:
        dataframe["rlnHelicalTubeID"] = dataframe["rlnHelicalTubeID"].astype(int)
        dataframe[RELION5_FLOAT_COLUMNS] = dataframe[RELION5_FLOAT_COLUMNS].astype(float)
    return dataframe


def write_relion5_filament_star(
    path: str | Path,
    particles: Iterable[Relion5FilamentParticle],
    *,
    overwrite: bool = True,
    float_format: str = "%.6f",
) -> None:
    """Write RELION 5 filament particles as a ``data_particles`` STAR block.

    `starfile` expects DataFrame columns without leading underscores; it writes
    labels such as ``_rlnTomoName #1`` in the STAR loop.
    """

    dataframe = particles_to_dataframe(particles)
    starfile.write(
        {"particles": dataframe},
        path,
        overwrite=overwrite,
        float_format=float_format,
    )
