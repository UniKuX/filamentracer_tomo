from pathlib import Path

import numpy as np
import pytest

from filament_tracer.detection import detect_cross_section, extract_seed_template
from filament_tracer.geometry import match_seed_points
from filament_tracer.models import (
    PlaneDefinition,
    SeedMatch,
    SkeletonProject,
    TracingParameters,
    VolumeMetadata,
)
from filament_tracer.tracing import (
    TraceDiagnostic,
    build_seed_templates,
    extend_skeletons,
    initialize_skeletons,
)


def _bright_cylinder() -> np.ndarray:
    volume = np.zeros((32, 64, 64), dtype=np.float32)
    yy, xx = np.mgrid[:64, :64]
    for z_index in range(volume.shape[0]):
        center_x = 31.0 + 0.08 * z_index
        radius_squared = (yy - 32.0) ** 2 + (xx - center_x) ** 2
        volume[z_index] = np.exp(-radius_squared / (2.0 * 1.8**2))
    return volume


def test_seed_matching_recovers_bundle_translation() -> None:
    points_a = np.array(
        [[10.0, 10.0, 10.0], [10.0, 20.0, 10.0], [10.0, 10.0, 20.0]]
    )
    translation = np.array([5.0, 1.0, -2.0])
    points_b = points_a[[2, 0, 1]] + translation

    result = match_seed_points(
        points_a,
        points_b,
        voxel_size_zyx=np.ones(3),
        max_residual_angstrom=0.1,
    )

    assert {(a, b) for a, b, _ in result.pairs} == {(0, 1), (1, 2), (2, 0)}
    assert not result.unmatched_a
    assert not result.unmatched_b


def test_screen_relative_slab_rotation_stays_orthonormal() -> None:
    from filament_tracer._widget import FilamentTracerWidget

    normal, up = FilamentTracerWidget._rotate_screen_relative(
        normal_world=np.array((1.0, 0.0, 0.0)),
        view_world=np.array((-1.0, 0.0, 0.0)),
        up_world=np.array((0.0, -1.0, 0.0)),
        delta_x=40.0,
        delta_y=25.0,
        sensitivity=0.35,
    )

    assert np.linalg.norm(normal) == pytest.approx(1.0)
    assert np.linalg.norm(up) == pytest.approx(1.0)
    assert np.dot(normal, up) == pytest.approx(0.0, abs=1e-12)
    assert not np.allclose(normal, (1.0, 0.0, 0.0))


def test_detector_finds_cross_section_center() -> None:
    volume = _bright_cylinder()
    parameters = TracingParameters(
        diameter_angstrom=50.0,
        polarity="bright",
        confidence_threshold=0.1,
        search_radius_voxels=4.0,
    )

    result = detect_cross_section(
        volume,
        predicted_data_zyx=np.array([16.0, 34.0, 29.0]),
        tangent_data_zyx=np.array([1.0, 0.0, 0.08]),
        voxel_size_zyx=np.array([10.0, 10.0, 10.0]),
        parameters=parameters,
    )

    expected = np.array([16.0, 32.0, 31.0 + 0.08 * 16.0])
    assert np.linalg.norm(result.position_zyx - expected) < 1.0
    assert result.confidence > 0.2
    assert result.diagnostic is not None
    assert result.diagnostic.patch.shape == result.diagnostic.response.shape
    assert result.diagnostic.template.size > 0
    assert result.diagnostic.search_mask.dtype == bool


def test_detector_uses_real_seed_crop() -> None:
    volume = _bright_cylinder()
    parameters = TracingParameters(
        diameter_angstrom=50.0,
        template_kind="seed_crop",
        search_radius_voxels=4.0,
    )
    voxel_size = np.array([10.0, 10.0, 10.0])
    tangent = np.array([1.0, 0.0, 0.08])
    template = extract_seed_template(
        volume,
        seed_data_zyx=np.array([8.0, 32.0, 31.64]),
        tangent_data_zyx=tangent,
        voxel_size_zyx=voxel_size,
        parameters=parameters,
    )

    assert template is not None
    result = detect_cross_section(
        volume,
        predicted_data_zyx=np.array([16.0, 34.0, 29.0]),
        tangent_data_zyx=tangent,
        voxel_size_zyx=voxel_size,
        parameters=parameters,
        empirical_template=template,
    )

    expected = np.array([16.0, 32.0, 31.0 + 0.08 * 16.0])
    assert np.linalg.norm(result.position_zyx - expected) < 1.0
    assert result.diagnostic is not None
    assert result.diagnostic.template_source == "seed crop"
    assert result.updated_template is not None
    assert result.updated_template.shape == template.shape
    assert np.mean(result.updated_template) == pytest.approx(0.0, abs=1e-6)
    assert np.std(result.updated_template) == pytest.approx(1.0, abs=1e-6)

    adapted = detect_cross_section(
        volume,
        predicted_data_zyx=np.array([18.0, 34.0, 29.0]),
        tangent_data_zyx=tangent,
        voxel_size_zyx=voxel_size,
        parameters=parameters,
        empirical_template=result.updated_template,
        empirical_template_source="adaptive step 1",
    )
    assert adapted.diagnostic is not None
    assert adapted.diagnostic.template_source == "adaptive step 1"


def test_tracer_extends_a_seeded_filament() -> None:
    volume = _bright_cylinder()
    parameters = TracingParameters(
        diameter_angstrom=50.0,
        step_voxels=2.0,
        max_steps=3,
        polarity="bright",
        confidence_threshold=0.1,
        search_radius_voxels=3.0,
    )
    points_a = np.array([[5.0, 32.0, 31.4]])
    points_b = np.array([[8.0, 32.0, 31.64]])
    filaments = initialize_skeletons(points_a, points_b, [(0, 0)], 25.0)
    templates = build_seed_templates(
        volume,
        filaments,
        (10.0, 10.0, 10.0),
        parameters,
        "forward",
    )
    diagnostics: list[TraceDiagnostic] = []

    extend_skeletons(
        volume,
        filaments,
        voxel_size_zyx=(10.0, 10.0, 10.0),
        parameters=parameters,
        direction="forward",
        diagnostics=diagnostics,
        seed_templates=templates,
    )

    assert len(filaments[0].points) == 5
    assert filaments[0].forward_stop_reason is None
    assert filaments[0].points[-1].position_zyx[0] > 13.0
    assert len(diagnostics) == 3
    assert all(record.accepted for record in diagnostics)
    assert all(record.detector is not None for record in diagnostics)
    assert [
        record.detector.template_source
        for record in diagnostics
        if record.detector is not None
    ] == ["seed crop", "adaptive step 1", "adaptive step 2"]


def test_skeleton_project_round_trip(tmp_path: Path) -> None:
    filaments = initialize_skeletons(
        np.array([[1.0, 2.0, 3.0]]),
        np.array([[4.0, 5.0, 6.0]]),
        [(0, 0)],
        35.0,
    )
    project = SkeletonProject(
        volume=VolumeMetadata(
            name="synthetic",
            shape_zyx=(10, 20, 30),
            voxel_size_zyx=(5.0, 5.0, 5.0),
        ),
        plane_a=PlaneDefinition(
            position_zyx=(1.0, 2.0, 3.0),
            normal_zyx=(1.0, 0.0, 0.0),
        ),
        plane_b=PlaneDefinition(
            position_zyx=(4.0, 5.0, 6.0),
            normal_zyx=(1.0, 0.0, 0.0),
        ),
        seed_matches=[SeedMatch(a_index=0, b_index=0, residual_angstrom=0.0)],
        tracing_parameters=TracingParameters(),
        filaments=filaments,
    )
    output = tmp_path / "trace.ftskeleton.json"

    project.save(output)
    restored = SkeletonProject.load(output)

    assert restored == project
