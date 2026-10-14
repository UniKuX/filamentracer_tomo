"""Single-microtubule patch preparation and explicit label review, without torch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mrcfile
import numpy as np

from filament_tracer.detection import extract_oriented_patch
from filament_tracer.geometry import normalize, transport_plane_basis
from filament_tracer.models import SkeletonProject


def interpolate_path(points, distances):
    lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if lengths[-1] <= 0 or np.any(np.diff(lengths) <= 1e-8):
        raise ValueError(
            "Skeleton contains duplicate consecutive points or zero length"
        )
    return np.column_stack(
        [np.interp(distances, lengths, points[:, axis]) for axis in range(3)]
    ), lengths[-1]


def segment_hits_box(a, b, half_extent):
    """Conservative segment versus expanded patch/tube bounding box."""
    lo, hi = 0.0, 1.0
    for x, delta, extent in zip(a, b - a, half_extent, strict=True):
        if abs(delta) < 1e-10:
            if abs(x) > extent:
                return False
        else:
            t0, t1 = sorted(((-extent - x) / delta, (extent - x) / delta))
            lo, hi = max(lo, t0), min(hi, t1)
            if lo > hi:
                return False
    return True


def intersection_labels(
    paths, origin, u, v, normal, size, spacing, radius, max_oblique=30
):
    """Label complete cross-sections; exclude known ambiguous tube geometry."""
    half = (size - 1) * spacing / 2
    labels, ids = [], []
    for filament_id, points in paths.items():
        local = np.column_stack(
            ((points - origin) @ u, (points - origin) @ v, (points - origin) @ normal)
        )
        crossings = []
        touches = False
        for a, b in zip(local[:-1], local[1:], strict=True):
            delta = b - a
            if not segment_hits_box(
                a, b, np.array([half + radius, half + radius, radius])
            ):
                continue
            touches = True
            cosine = abs(delta[2]) / np.linalg.norm(delta)
            if cosine < np.cos(np.radians(max_oblique)):
                return [], [], "oblique or near-parallel microtubule"
            if a[2] * b[2] <= 0 and abs(delta[2]) > 1e-10:
                point = a - a[2] / delta[2] * delta
                if not any(np.linalg.norm(point - q) < 1e-5 for q in crossings):
                    crossings.append(point)
        if touches and not crossings:
            return [], [], "nearby microtubule without a reliable plane intersection"
        for point in crossings:
            if np.any(
                np.abs(point[:2]) + radius / np.cos(np.radians(max_oblique))
                > half - spacing
            ):
                return [], [], "partial microtubule at crop boundary"
            labels.append(
                [
                    point[1] / spacing + (size - 1) / 2,
                    point[0] / spacing + (size - 1) / 2,
                ]
            )
            ids.append(filament_id)
    return labels, ids, None


def generate_patches(
    volume,
    project,
    *,
    size=97,
    spacing=5.0,
    step=100.0,
    tangent_window=200.0,
    diameter=250.0,
    variants=3,
    offset_fraction=0.35,
    seed=0,
    allow_automatic=False,
):
    if size < 9 or size % 2 != 1:
        raise ValueError("Patch size must be odd and at least 9")
    if not all(
        np.isfinite(x) and x > 0 for x in (spacing, step, tangent_window, diameter)
    ):
        raise ValueError("Physical sizes and spacings must be finite and positive")
    if variants < 1 or not 0 < offset_fraction < 1:
        raise ValueError("Variants must be positive; offset fraction must be in (0, 1)")
    if tuple(volume.shape) != project.volume.shape_zyx:
        raise ValueError("Tomogram shape differs from skeleton project")
    if not allow_automatic and any(
        p.provenance == "automatic" for f in project.filaments for p in f.points
    ):
        raise ValueError(
            "Automatic vertices found; manually correct them first, then "
            "use --allow-corrected-automatic to confirm review"
        )
    voxel = np.array(project.volume.voxel_size_zyx)
    paths = {
        f.filament_id: np.array([p.position_zyx for p in f.points]) * voxel
        for f in project.filaments
        if len(f.points) >= 2
    }
    if not paths:
        raise ValueError("No skeletons with at least two points")
    if len(paths) != len(project.filaments):
        raise ValueError("Every supplied microtubule needs at least two vertices")
    rng = np.random.default_rng(seed)
    half = (size - 1) / 2
    radius = diameter / 2
    max_offset = min(
        half * offset_fraction, half - radius / spacing / np.cos(np.radians(30)) - 2
    )
    if max_offset < 1:
        raise ValueError(
            "Increase patch field of view to allow off-center complete tubes"
        )
    records, rejected = [], []
    for filament_id, points in paths.items():
        if not np.all(np.isfinite(points)):
            raise ValueError("Non-finite skeleton coordinates")
        _, length = interpolate_path(points, [0])
        previous_u = previous_v = None
        # Avoid endpoints, where local direction and completeness are less reliable.
        margin = min(tangent_window / 2, length / 4)
        for arc in np.arange(margin, length - margin + 1e-6, step):
            xyz, _ = interpolate_path(
                points,
                [
                    max(0, arc - tangent_window / 2),
                    arc,
                    min(length, arc + tangent_window / 2),
                ],
            )
            normal = normalize(xyz[2] - xyz[0])
            if previous_u is None:
                from filament_tracer.geometry import orthonormal_plane_basis

                u, v = orthonormal_plane_basis(normal)
            else:
                u, v = transport_plane_basis(normal, previous_u, previous_v)
            previous_u, previous_v = u, v
            for variant in range(variants):
                offset = rng.uniform(-max_offset, max_offset, 2)
                origin = xyz[1] + spacing * (offset[0] * v + offset[1] * u)
                labels, ids, reason = intersection_labels(
                    paths, origin, u, v, normal, size, spacing, radius
                )
                metadata = dict(
                    filament_id=filament_id,
                    arclength_angstrom=float(arc),
                    variant=variant,
                    origin_physical_zyx=origin.tolist(),
                    basis_u_zyx=u.tolist(),
                    basis_v_zyx=v.tolist(),
                    normal_zyx=normal.tolist(),
                    offset_rc_pixels=offset.tolist(),
                    voxel_size_zyx=voxel.tolist(),
                    pixel_spacing_angstrom=spacing,
                    patch_size=size,
                    diameter_angstrom=diameter,
                )
                if reason or filament_id not in ids:
                    rejected.append(
                        {**metadata, "reason": reason or "missing primary intersection"}
                    )
                    continue
                patch, valid, _, _ = extract_oriented_patch(
                    volume, origin / voxel, normal, voxel, half * spacing, spacing, u, v
                )
                # Floating-point ceil in the shared extractor can add a border.
                trim = (patch.shape[0] - size) // 2
                patch = patch[trim : trim + size, trim : trim + size]
                valid = valid[trim : trim + size, trim : trim + size]
                if not valid.all() or not np.isfinite(patch).all():
                    rejected.append(
                        {**metadata, "reason": "invalid or out-of-volume pixels"}
                    )
                    continue
                records.append(
                    (
                        patch,
                        np.array(labels, dtype=np.float32),
                        {**metadata, "intersection_filament_ids": ids},
                    )
                )
    return records, rejected


def prepare(args):
    project = SkeletonProject.load(args.skeleton)
    output = Path(args.output)
    if output.exists():
        raise ValueError("Output already exists; use a new directory")
    with mrcfile.mmap(args.tomogram, mode="r") as handle:
        records, rejected = generate_patches(
            handle.data,
            project,
            size=args.patch_size,
            spacing=args.pixel_spacing,
            step=args.step,
            tangent_window=args.tangent_window,
            diameter=args.diameter,
            variants=args.variants,
            offset_fraction=args.offset_fraction,
            seed=args.seed,
            allow_automatic=args.allow_corrected_automatic,
        )
    output.mkdir(parents=True)
    entries = []
    for index, (patch, centers, geometry) in enumerate(records):
        name = f"patch_{index:06d}.npz"
        np.savez_compressed(output / name, image=patch, centers_rc=centers)
        entries.append(
            dict(
                file=name,
                status="pending",
                centers_rc=centers.tolist(),
                geometry=geometry,
            )
        )
    manifest = dict(
        version=1,
        source_tomogram=str(Path(args.tomogram).resolve()),
        source_skeleton=str(Path(args.skeleton).resolve()),
        settings=vars(args),
        normalization="raw interpolated density",
        coordinate_convention="zero-based row,column; physical ZYX in angstroms",
        patches=entries,
        rejected=rejected,
    )
    write_manifest(output, manifest)
    print(
        f"Prepared {len(entries)} patches for review; excluded {len(rejected)}. "
        "No patches are training-approved until reviewed."
    )


def write_manifest(directory, manifest):
    path = Path(directory) / "manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_manifest(directory):
    manifest = json.loads(
        (Path(directory) / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest["version"] != 1:
        raise ValueError("Unsupported dataset version")
    return manifest


def validate_centers(centers, size, margin):
    points = np.asarray(centers, dtype=float).reshape(-1, 2)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("Every accepted patch needs at least one labeled microtubule")
    if np.any(points < margin) or np.any(points > size - 1 - margin):
        raise ValueError("A microtubule is cut off by the crop; reject this patch")
    return points


def review(args):
    import napari
    from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

    directory = Path(args.dataset)
    manifest = read_manifest(directory)
    entries = manifest["patches"]
    if not entries:
        raise ValueError(
            "No reviewable patches; inspect rejected reasons in manifest.json"
        )
    viewer = napari.Viewer(ndisplay=2, title="Microtubule training patch review")
    panel = QWidget()
    layout = QVBoxLayout(panel)
    instructions = QLabel(
        "Label ALL complete microtubule centers. Add missing centers, "
        "move/delete incorrect ones using Points tools. Reject if "
        "another tube is uncertain, oblique, or cut off. Accept is "
        "confirmation that no visible microtubule remains unlabeled."
    )
    instructions.setWordWrap(True)
    layout.addWidget(instructions)
    status = QLabel()
    status.setWordWrap(True)
    layout.addWidget(status)
    state = dict(
        index=next((i for i, e in enumerate(entries) if e["status"] == "pending"), 0)
    )

    def show():
        entry = entries[state["index"]]
        with np.load(directory / entry["file"], allow_pickle=False) as data:
            image = data["image"].copy()
        viewer.layers.clear()
        viewer.add_image(image, name="Raw perpendicular slice")
        state["points"] = viewer.add_points(
            np.array(entry["centers_rc"]).reshape(-1, 2),
            ndim=2,
            size=6,
            face_color="transparent",
            border_color="cyan",
            name="All tube centers",
        )
        state["points"].mode = "add"
        viewer.reset_view()
        status.setText(f"{state['index'] + 1}/{len(entries)} — {entry['status']}")

    def advance(decision=None, backwards=False):
        entry = entries[state["index"]]
        if decision:
            points = np.asarray(state["points"].data)
            if decision == "accepted":
                g = entry["geometry"]
                try:
                    validate_centers(
                        points,
                        g["patch_size"],
                        g["diameter_angstrom"] / 2 / g["pixel_spacing_angstrom"] + 1,
                    )
                except ValueError as error:
                    status.setText(str(error))
                    return
            entry.update(status=decision, centers_rc=points.tolist())
            write_manifest(directory, manifest)
        state["index"] = max(
            0, min(len(entries) - 1, state["index"] + (-1 if backwards else 1))
        )
        show()

    for text, callback in (
        ("Accept all labels and next", lambda: advance("accepted")),
        ("Reject ambiguous patch and next", lambda: advance("rejected")),
        ("Skip for later", lambda: advance()),
        ("Previous / revise decision", lambda: advance(backwards=True)),
        ("Finish (decisions already saved)", viewer.close),
    ):
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
    viewer.window.add_dock_widget(panel, area="right")
    show()
    napari.run()


def export(args):
    directory = Path(args.dataset)
    manifest = read_manifest(directory)
    accepted = [e for e in manifest["patches"] if e["status"] == "accepted"]
    if not accepted:
        raise ValueError("No explicitly accepted patches; run review first")
    output = Path(args.output)
    if output.exists():
        raise ValueError("Output exists; choose a new filename")
    images, centers, offsets = [], [], [0]
    for entry in accepted:
        g = entry["geometry"]
        rc = validate_centers(
            entry["centers_rc"],
            g["patch_size"],
            g["diameter_angstrom"] / 2 / g["pixel_spacing_angstrom"] + 1,
        )
        with np.load(directory / entry["file"], allow_pickle=False) as data:
            images.append(data["image"].copy())
        centers.extend(rc)
        offsets.append(len(centers))
    with output.open("xb") as stream:
        np.savez_compressed(
            stream,
            images=np.stack(images),
            centers_rc=np.asarray(centers, dtype=np.float32),
            center_offsets=np.asarray(offsets, dtype=np.int64),
            metadata=np.array(
                json.dumps({**manifest, "patches": accepted, "rejected": []})
            ),
        )
    print(f"Exported {len(accepted)} reviewed patches to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--tomogram", required=True)
    p.add_argument("--skeleton", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--patch-size", type=int, default=97)
    p.add_argument("--pixel-spacing", type=float, default=5)
    p.add_argument("--step", type=float, default=100)
    p.add_argument("--tangent-window", type=float, default=200)
    p.add_argument("--diameter", type=float, default=250)
    p.add_argument("--variants", type=int, default=3)
    p.add_argument("--offset-fraction", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-corrected-automatic", action="store_true")
    r = commands.add_parser("review")
    r.add_argument("dataset")
    e = commands.add_parser("export")
    e.add_argument("dataset")
    e.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        {"prepare": prepare, "review": review, "export": export}[args.command](args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
