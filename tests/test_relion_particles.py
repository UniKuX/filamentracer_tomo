import numpy as np
import pandas as pd
import pytest
import starfile

from filament_tracer.relion_particles import (
    particle_positions,
    read_particles,
    subtomogram_matrices,
)


def particle_table():
    return pd.DataFrame(
        {
            "rlnTomoName": ["Position_1", "Position_2"],
            "rlnCenteredCoordinateXAngst": [10.0, 20.0],
            "rlnCenteredCoordinateYAngst": [20.0, 30.0],
            "rlnCenteredCoordinateZAngst": [30.0, 40.0],
            "rlnOriginXAngst": [2.0, 0.0],
            "rlnOriginYAngst": [4.0, 0.0],
            "rlnOriginZAngst": [6.0, 0.0],
            "rlnTomoSubtomogramRot": [0.0, 0.0],
            "rlnTomoSubtomogramTilt": [0.0, 0.0],
            "rlnTomoSubtomogramPsi": [90.0, 0.0],
            "rlnHelicalTubeID": [1, 2],
            "rlnClassNumber": [1, 2],
        }
    )


def test_center_anisotropic_voxels_rotated_refinement_offsets():
    table = particle_table().iloc[:1]
    # RELION Psi=90 sends (x,y,z) to (y,-x,z), subtract (4,-2,6).
    result = particle_positions(table, pd.DataFrame(), [3, 2, 1], [10, 20, 30])
    np.testing.assert_allclose(result, [[18, 31, 36]])
    original = particle_positions(table, pd.DataFrame(), [3, 2, 1], [10, 20, 30], False)
    np.testing.assert_allclose(original, [[20, 30, 40]])
    matrices = subtomogram_matrices(np.array([[78.56, 116.25, 164.93]]))
    np.testing.assert_allclose(matrices[0] @ matrices[0].T, np.eye(3), atol=1e-14)
    assert np.linalg.det(matrices[0]) == pytest.approx(1)


def test_legacy_coordinates_optics_mapping_and_bad_fields():
    table = pd.DataFrame(
        {
            "rlnCoordinateX": [8],
            "rlnCoordinateY": [16],
            "rlnCoordinateZ": [24],
            "rlnOpticsGroup": [3],
        }
    )
    optics = pd.DataFrame({"rlnOpticsGroup": [3], "rlnTomoTiltSeriesPixelSize": [2]})
    np.testing.assert_allclose(
        particle_positions(table, optics, [4] * 3, [50] * 3), [[12, 8, 4]]
    )
    with pytest.raises(ValueError, match="pixel size"):
        particle_positions(table.assign(rlnOpticsGroup=4), optics, [4] * 3, [50] * 3)
    with pytest.raises(ValueError, match="Missing"):
        particle_positions(
            particle_table().drop(columns="rlnOriginYAngst"), optics, [4] * 3, [50] * 3
        )


def test_star_display_filters_transforms_and_is_read_only(make_napari_viewer, tmp_path):
    from filament_tracer._particle_widget import PARTICLE_LAYER
    from filament_tracer._widget import FilamentTracerWidget

    path = tmp_path / "particles.star"
    starfile.write({"particles": particle_table()}, path)
    viewer = make_napari_viewer()
    image = viewer.add_image(
        np.zeros((40, 60, 80)), scale=(3, 2, 1), translate=(2, 3, 4)
    )
    widget = FilamentTracerWidget(viewer)
    widget._set_image_layer(image)
    display = widget._particle_display
    display.load_star(path)
    assert display.tomogram.count() == 2
    display.display()
    layer = viewer.layers[PARTICLE_LAYER]
    assert len(layer.data) == 1
    assert not layer.editable
    np.testing.assert_allclose(layer.data, [[28, 41, 46]])
    np.testing.assert_allclose(layer.scale, image.scale)
    np.testing.assert_allclose(layer.translate, image.translate)
    assert widget._project is None
    display.shifts.setChecked(False)
    display.display()
    np.testing.assert_allclose(viewer.layers[PARTICLE_LAYER].data, [[30, 40, 50]])
    display.tomogram.setCurrentText("Position_2")
    display.classes.setCurrentIndex(1)
    display.display()
    assert viewer.layers[PARTICLE_LAYER].features.rlnClassNumber.tolist() == [2]
    widget._set_image_layer(image)
    assert PARTICLE_LAYER not in viewer.layers


def test_example_star_if_present():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "run_data.star"
    if not path.exists():
        pytest.skip("User-provided example not distributed with package")
    table, optics = read_particles(path)
    rows = table[table.rlnTomoName == "Position_1"]
    assert len(rows) > 0
    result = particle_positions(rows, optics, [13.36] * 3, [112.5, 360, 250])
    assert result.shape == (len(rows), 3)
    assert np.isfinite(result).all()
