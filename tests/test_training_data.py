from types import SimpleNamespace

import mrcfile
import numpy as np
import pytest

from filament_tracer.models import (
    PlaneDefinition,
    SkeletonProject,
    TracingParameters,
    VolumeMetadata,
)
from filament_tracer.tracing import initialize_skeletons
from filament_tracer.training_data import (
    export,
    generate_patches,
    intersection_labels,
    prepare,
    read_manifest,
    write_manifest,
)


def project():
    plane = PlaneDefinition(position_zyx=(10, 32, 32), normal_zyx=(1, 0, 0))
    return SkeletonProject(
        volume=VolumeMetadata(
            name="synthetic", shape_zyx=(64, 64, 64), voxel_size_zyx=(2, 1, 1)
        ),
        plane_a=plane,
        plane_b=plane,
        seed_matches=[],
        tracing_parameters=TracingParameters(),
        filaments=initialize_skeletons(
            np.array([[10, 32, 32]]), np.array([[50, 32, 32]]), [(0, 0)], 3
        ),
    )


def test_prepare_from_mrc_requires_review(tmp_path):
    skeleton = tmp_path / "manual.ftskeleton.json"
    project().save(skeleton)
    tomogram = tmp_path / "volume.mrc"
    with mrcfile.new(tomogram) as handle:
        handle.set_data(np.zeros((64, 64, 64), dtype=np.float32))
    output = tmp_path / "review"
    prepare(
        SimpleNamespace(
            skeleton=str(skeleton),
            tomogram=str(tomogram),
            output=str(output),
            patch_size=33,
            pixel_spacing=1,
            step=20,
            tangent_window=20,
            diameter=6,
            variants=2,
            offset_fraction=0.35,
            seed=1,
            allow_corrected_automatic=False,
        )
    )
    manifest = read_manifest(output)
    assert manifest["patches"]
    assert all(p["status"] == "pending" for p in manifest["patches"])
    with np.load(output / manifest["patches"][0]["file"], allow_pickle=False) as patch:
        assert patch["image"].shape == (33, 33)


def test_random_offset_labels_recover_physical_skeleton():
    p = project()
    volume = np.indices(p.volume.shape_zyx)[1].astype(np.float32)
    kwargs = dict(
        size=33, spacing=1, step=20, tangent_window=20, diameter=6, variants=3, seed=3
    )
    records, rejected = generate_patches(volume, p, **kwargs)
    repeated, _ = generate_patches(volume, p, **kwargs)
    assert len(records) > 3 and not rejected
    positions = []
    for (image, centers, g), (other_image, _, _) in zip(records, repeated, strict=True):
        np.testing.assert_array_equal(image, other_image)
        physical = (
            np.array(g["origin_physical_zyx"])
            + (centers[0, 0] - 16) * np.array(g["basis_v_zyx"])
            + (centers[0, 1] - 16) * np.array(g["basis_u_zyx"])
        )
        np.testing.assert_allclose(physical[1:], [32, 32], atol=1e-6)
        positions.append(centers[0])
    assert np.std(positions, axis=0).min() > 1


def test_neighbor_intersections_or_ambiguity_rejection():
    paths = {
        1: np.array([[-10, 0, 0], [10, 0, 0]]),
        2: np.array([[-10, 8, 0], [10, 8, 0]]),
    }
    args = (
        np.zeros(3),
        np.array([0, 0, 1]),
        np.array([0, -1, 0]),
        np.array([1, 0, 0]),
        33,
        1,
        3,
    )
    labels, ids, reason = intersection_labels(paths, *args)
    assert reason is None and ids == [1, 2] and len(labels) == 2
    paths[2] = np.array([[-10, 15, 0], [10, 15, 0]])
    assert "boundary" in intersection_labels(paths, *args)[2]
    paths[2] = np.array([[0, -10, 5], [0, 10, 5]])
    assert "parallel" in intersection_labels(paths, *args)[2]


def test_rotated_plane_and_vertex_intersection_not_duplicated():
    normal = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    u = np.array([0.0, 0.0, 1.0])
    v = np.cross(normal, u)
    paths = {1: np.array([-10 * normal, np.zeros(3), 10 * normal])}
    labels, ids, reason = intersection_labels(
        paths, np.zeros(3), u, v, normal, 33, 1, 3
    )
    assert reason is None and ids == [1]
    np.testing.assert_allclose(labels, [[16, 16]])


def test_export_excludes_pending_and_rejected_patches(tmp_path):
    entries = []
    for index, state in enumerate(("pending", "accepted", "rejected")):
        name = f"{index}.npz"
        np.savez_compressed(tmp_path / name, image=np.full((33, 33), index))
        entries.append(
            dict(
                file=name,
                status=state,
                centers_rc=[[10, 15], [20, 15]],
                geometry=dict(
                    patch_size=33, diameter_angstrom=6, pixel_spacing_angstrom=1
                ),
            )
        )
    write_manifest(tmp_path, dict(version=1, patches=entries, rejected=[]))
    output = tmp_path / "training.npz"
    export(SimpleNamespace(dataset=str(tmp_path), output=str(output)))
    with np.load(output, allow_pickle=False) as data:
        assert data["images"].shape == (1, 33, 33)
        assert np.all(data["images"] == 1)
        np.testing.assert_array_equal(data["center_offsets"], [0, 2])
    manifest = read_manifest(tmp_path)
    manifest["patches"][1]["status"] = "pending"
    write_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="explicitly accepted"):
        export(
            SimpleNamespace(dataset=str(tmp_path), output=str(tmp_path / "none.npz"))
        )
