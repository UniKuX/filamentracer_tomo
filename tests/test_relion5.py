from pathlib import Path

import pandas as pd

from filament_trace_tomo.relion5 import (
    RELION5_FILAMENT_COLUMNS,
    Relion5FilamentParticle,
    particles_to_dataframe,
    write_relion5_filament_star,
)


def _example_particles():
    return [
        Relion5FilamentParticle(
            tomo_name="Position_51_2",
            helical_tube_id=1,
            helical_track_length_angst=0.0,
            centered_coordinate_x_angst=-3974.394070,
            centered_coordinate_y_angst=4168.292866,
            centered_coordinate_z_angst=96.189681,
            tomo_subtomogram_rot=90.0,
            tomo_subtomogram_tilt=123.880889,
            tomo_subtomogram_psi=142.172287,
            angle_rot=0.0,
            angle_tilt=90.0,
            angle_psi=0.0,
            angle_tilt_prior=90.0,
            angle_psi_prior=0.0,
        ),
        Relion5FilamentParticle(
            tomo_name="Position_51_2",
            helical_tube_id=1,
            helical_track_length_angst=26.225340,
            centered_coordinate_x_angst=-3941.064049,
            centered_coordinate_y_angst=4125.357794,
            centered_coordinate_z_angst=96.189681,
            tomo_subtomogram_rot=90.0,
            tomo_subtomogram_tilt=123.880889,
            tomo_subtomogram_psi=142.184580,
            angle_rot=0.0,
            angle_tilt=90.0,
            angle_psi=0.0,
            angle_tilt_prior=90.0,
            angle_psi_prior=0.0,
        ),
    ]


def test_particles_to_dataframe_uses_relion5_column_order():
    dataframe = particles_to_dataframe(_example_particles())

    assert list(dataframe.columns) == RELION5_FILAMENT_COLUMNS
    assert dataframe.loc[0, "rlnTomoName"] == "Position_51_2"
    assert dataframe.loc[1, "rlnHelicalTrackLengthAngst"] == 26.225340
    assert dataframe["rlnHelicalTubeID"].dtype == int
    assert dataframe["rlnAngleRot"].dtype == float


def test_write_relion5_filament_star_uses_particles_block(monkeypatch, tmp_path):
    calls = []

    def fake_write(data, path, *, overwrite, float_format):
        calls.append((data, path, overwrite, float_format))

    monkeypatch.setattr("filament_trace_tomo.relion5.starfile.write", fake_write)

    output_path = tmp_path / "particles.star"
    write_relion5_filament_star(output_path, _example_particles(), overwrite=False)

    data, path, overwrite, float_format = calls[0]
    assert path == output_path
    assert overwrite is False
    assert float_format == "%.6f"
    assert list(data) == ["particles"]
    assert isinstance(data["particles"], pd.DataFrame)
    assert list(data["particles"].columns) == RELION5_FILAMENT_COLUMNS


def test_write_relion5_filament_star_writes_six_decimal_float_values(tmp_path):
    output_path = tmp_path / "particles.star"

    write_relion5_filament_star(output_path, _example_particles())

    text = output_path.read_text()
    assert "data_particles" in text
    assert "_rlnAnglePsiPrior #14" in text
    assert "0.000000" in text
    assert "90.000000" in text
