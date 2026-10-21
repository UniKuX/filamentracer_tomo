import numpy as np
import pytest
import starfile

from filament_tracer.models import (
    PlaneDefinition,
    SkeletonProject,
    TracingParameters,
    VolumeMetadata,
)
from filament_tracer.relion_export import (
    coordinate_table,
    helical_priors,
    main,
    resample_path,
)
from filament_tracer.tracing import initialize_skeletons


@pytest.mark.parametrize(
    "direction",
    [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
        [1, 2, 3],
    ],
)
def test_helical_prior_axis_matches_physical_tangent(direction):
    from filament_tracer.relion_particles import subtomogram_matrices

    points = np.array([[10, 10, 10], np.array([10, 10, 10]) + direction])
    voxel = [4, 2, 1]
    _, arc = resample_path(points, voxel, 1)
    tilt, psi = helical_priors(points, voxel, arc, 3)
    matrices = subtomogram_matrices(np.column_stack([np.full(len(arc), 37), tilt, psi]))
    expected = (np.array(direction) * voxel)[::-1]
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(
        matrices[:, :, 2], np.tile(expected, (len(arc), 1)), atol=1e-12
    )


def test_prior_bend_duplicates_single_sample_and_validation():
    points = [[0, 0, 0], [0, 0, 0], [0, 0, 10], [0, 10, 10]]
    tilt, psi = helical_priors(points, [1] * 3, np.array([0, 10, 20]), 4)
    np.testing.assert_allclose(tilt, [90, 90, 90])
    np.testing.assert_allclose(psi, [180, 135, 90])
    short = coordinate_table(make_project(), 1000, 1, "TS")
    assert len(short) == 2
    assert "rlnAngleRot" not in short
    assert "rlnAngleRotPrior" not in short
    np.testing.assert_allclose(short.rlnAngleTiltPrior, 0)
    for kwargs in ({"tangent_window": float("nan")}, {"tangent_window": 0}):
        with pytest.raises(ValueError):
            coordinate_table(make_project(), 10, 1, "TS", **kwargs)
    with pytest.raises(ValueError, match="Undefined"):
        helical_priors([[0, 0, 0], [0, 0, 10], [0, 0, 0]], [1] * 3, np.array([10]), 4)




def make_project():
    plane = PlaneDefinition(position_zyx=(0, 0, 0), normal_zyx=(1, 0, 0))
    return SkeletonProject(
        volume=VolumeMetadata(
            name="test", shape_zyx=(32, 32, 32), voxel_size_zyx=(4, 2, 1)
        ),
        plane_a=plane,
        plane_b=plane,
        seed_matches=[],
        tracing_parameters=TracingParameters(),
        filaments=initialize_skeletons(
            np.array([[1, 2, 3], [1, 5, 6]]),
            np.array([[6, 2, 3], [6, 5, 6]]),
            [(0, 0), (1, 1)],
            3,
        ),
    )


def test_resampling_bends_duplicates_and_endpoints():
    points = [[0, 0, 0], [0, 0, 0], [0, 0, 10], [0, 10, 10]]
    samples, arc = resample_path(points, [1, 1, 1], 6)
    np.testing.assert_allclose(arc, [0, 6, 12, 18])
    np.testing.assert_allclose(samples, [[0, 0, 0], [0, 0, 6], [0, 2, 10], [0, 8, 10]])
    samples, arc = resample_path(points, [1, 1, 1], 10)
    np.testing.assert_array_equal(samples[-1], points[-1])
    assert len(resample_path(points, [1, 1, 1], 100)[0]) == 1
    with pytest.raises(ValueError):
        resample_path([[0, 0, 0], [0, 0, 0]], [1, 1, 1], 10)


def test_xyz_physical_scaling_and_filament_ids():
    table = coordinate_table(make_project(), 10, 0.5, "TS_01")
    assert len(table) == 6
    np.testing.assert_allclose(table.rlnCoordinateX[:3], [6, 6, 6])
    np.testing.assert_allclose(table.rlnCoordinateY[:3], [8, 8, 8])
    np.testing.assert_allclose(table.rlnCoordinateZ[:3], [8, 28, 48])
    assert table.rlnHelicalTubeID.tolist() == [1, 1, 1, 2, 2, 2]
    assert table.rlnHelicalTrackLengthAngst.tolist() == [0, 10, 20, 0, 10, 20]


@pytest.mark.parametrize(
    "distance,pixel", [(0, 1), (-1, 1), (float("nan"), 1), (10, 0), (10, float("inf"))]
)
def test_invalid_export_parameters(distance, pixel):
    with pytest.raises(ValueError):
        coordinate_table(make_project(), distance, pixel, "TS_01")


def test_export_cli_roundtrip_and_no_overwrite(tmp_path):
    source = tmp_path / "test.ftskeleton.json"
    output = tmp_path / "test.star"
    make_project().save(source)
    args = [
        "--skeleton",
        str(source),
        "--output",
        str(output),
        "--inter-box-distance",
        "10",
        "--relion-pixel-size",
        "0.5",
        "--tomo-name",
        "TS_01",
        "--tangent-window",
        "15",
    ]
    main(args)
    table = starfile.read(output)
    assert len(table) == 6
    assert table.rlnTomoName.tolist() == ["TS_01"] * 6
    assert "rlnAngleRot" not in table
    assert "rlnAngleRotPrior" not in table
    for angle, value in (("Tilt", 0), ("Psi", 0)):
        np.testing.assert_allclose(table[f"rlnAngle{angle}Prior"], value)
        np.testing.assert_allclose(table[f"rlnAngle{angle}"], value)
    original = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert output.read_bytes() == original


def test_bad_skeleton_is_not_silently_exported():
    project = make_project()
    project.filaments[0].points[0].position_zyx = (-1, 2, 3)
    with pytest.raises(ValueError, match="outside"):
        coordinate_table(project, 10, 1, "TS_01")
    project = make_project()
    project.filaments[0].points = project.filaments[0].points[:1]
    with pytest.raises(ValueError, match="Filament 1"):
        coordinate_table(project, 10, 1, "TS_01")
