"""Bundle-aware filament skeleton initialization and extension."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from filament_tracer.detection import (
    DetectionDiagnostic,
    OrientedTemplate,
    detect_cross_section,
    extract_seed_oriented_template,
)
from filament_tracer.geometry import data_vector_to_physical, normalize
from filament_tracer.models import (
    FilamentSkeleton,
    SkeletonPoint,
    TracingParameters,
)


@dataclass
class TraceDiagnostic:
    """One explainable prediction/detection decision during tracing."""

    filament_id: int
    direction: str
    step_number: int
    predicted_position_zyx: tuple[float, float, float]
    detected_position_zyx: tuple[float, float, float]
    confidence: float
    radius_angstrom: float
    valid_fraction: float
    accepted: bool
    reason: str
    detector: DetectionDiagnostic | None


def initialize_skeletons(
    points_a_zyx: NDArray[np.floating],
    points_b_zyx: NDArray[np.floating],
    pairs: list[tuple[int, int]],
    radius_angstrom: float,
) -> list[FilamentSkeleton]:
    """Create ordered two-point skeletons from accepted seed matches."""

    filaments: list[FilamentSkeleton] = []
    for filament_id, (a_index, b_index) in enumerate(pairs, start=1):
        filaments.append(
            FilamentSkeleton(
                filament_id=filament_id,
                points=[
                    SkeletonPoint(
                        position_zyx=tuple(float(v) for v in points_a_zyx[a_index]),
                        radius_angstrom=radius_angstrom,
                        confidence=1.0,
                        provenance="seed",
                    ),
                    SkeletonPoint(
                        position_zyx=tuple(float(v) for v in points_b_zyx[b_index]),
                        radius_angstrom=radius_angstrom,
                        confidence=1.0,
                        provenance="seed",
                    ),
                ],
            )
        )
    return filaments


def build_seed_templates(
    volume: ArrayLike,
    filaments: list[FilamentSkeleton],
    voxel_size_zyx: tuple[float, float, float],
    parameters: TracingParameters,
    direction: str,
) -> dict[int, OrientedTemplate]:
    """Generate one real, direction-specific seed crop per filament."""

    if direction not in {"forward", "backward"}:
        raise ValueError("direction must be 'forward' or 'backward'")
    if parameters.template_kind != "seed_crop":
        return {}

    templates: dict[int, OrientedTemplate] = {}
    voxel_size = np.asarray(voxel_size_zyx, dtype=float)
    for filament in filaments:
        seeds = [
            np.asarray(point.position_zyx, dtype=float)
            for point in filament.points
            if point.provenance == "seed"
        ]
        if len(seeds) < 2:
            continue
        plane_a_seed, plane_b_seed = seeds[0], seeds[1]
        if direction == "forward":
            center = plane_b_seed
            tangent = plane_b_seed - plane_a_seed
        else:
            center = plane_a_seed
            tangent = plane_a_seed - plane_b_seed
        try:
            template = extract_seed_oriented_template(
                volume,
                center,
                tangent,
                voxel_size,
                parameters,
            )
        except ValueError:
            template = None
        if template is not None:
            templates[filament.filament_id] = template
    return templates


def _inside_volume(
    position_zyx: NDArray[np.floating],
    shape_zyx: tuple[int, ...],
    margin: float = 1.0,
) -> bool:
    return bool(
        np.all(position_zyx >= margin)
        and np.all(position_zyx <= np.asarray(shape_zyx, dtype=float) - 1.0 - margin)
    )


def _endpoint_state(
    filament: FilamentSkeleton,
    direction: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if len(filament.points) < 2:
        raise ValueError("a filament needs at least two points before tracing")
    if direction == "forward":
        previous = np.asarray(filament.points[-2].position_zyx, dtype=float)
        current = np.asarray(filament.points[-1].position_zyx, dtype=float)
    else:
        previous = np.asarray(filament.points[1].position_zyx, dtype=float)
        current = np.asarray(filament.points[0].position_zyx, dtype=float)
    return previous, current


def extend_skeletons(
    volume: ArrayLike,
    filaments: list[FilamentSkeleton],
    voxel_size_zyx: tuple[float, float, float],
    parameters: TracingParameters,
    direction: str,
    diagnostics: list[TraceDiagnostic] | None = None,
    max_diagnostics: int = 250,
    seed_templates: dict[
        int,
        NDArray[np.float32] | OrientedTemplate,
    ]
    | None = None,
) -> list[FilamentSkeleton]:
    """Extend a bundle while adapting each template after accepted steps."""

    if direction not in {"forward", "backward"}:
        raise ValueError("direction must be 'forward' or 'backward'")
    if not filaments:
        return filaments

    voxel_size = np.asarray(voxel_size_zyx, dtype=float)
    mean_voxel = float(np.mean(voxel_size))
    physical_step = parameters.step_voxels * mean_voxel
    adaptive_templates = dict(seed_templates or {})
    template_sources = {
        filament_id: "seed crop" for filament_id in adaptive_templates
    }
    active = {
        filament.filament_id
        for filament in filaments
        if (
            filament.forward_stop_reason
            if direction == "forward"
            else filament.backward_stop_reason
        )
        is None
    }

    for step_number in range(1, parameters.max_steps + 1):
        if not active:
            break

        tangents_physical: list[NDArray[np.float64]] = []
        endpoint_by_id: dict[
            int, tuple[NDArray[np.float64], NDArray[np.float64]]
        ] = {}
        for filament in filaments:
            if filament.filament_id not in active:
                continue
            previous, current = _endpoint_state(filament, direction)
            endpoint_by_id[filament.filament_id] = (previous, current)
            tangents_physical.append(
                data_vector_to_physical(current - previous, voxel_size)
            )

        bundle_tangent = normalize(np.median(tangents_physical, axis=0))
        accepted_positions: list[tuple[int, NDArray[np.float64]]] = []
        candidate_template_updates: dict[int, OrientedTemplate] = {}

        for filament in filaments:
            if filament.filament_id not in active:
                continue
            previous, current = endpoint_by_id[filament.filament_id]
            individual = data_vector_to_physical(current - previous, voxel_size)
            blended_physical = normalize(0.75 * individual + 0.25 * bundle_tangent)
            predicted = current + blended_physical * physical_step / voxel_size

            if not _inside_volume(predicted, np.asanyarray(volume).shape):
                _record_diagnostic(
                    diagnostics,
                    max_diagnostics,
                    TraceDiagnostic(
                        filament_id=filament.filament_id,
                        direction=direction,
                        step_number=step_number,
                        predicted_position_zyx=_position_tuple(predicted),
                        detected_position_zyx=_position_tuple(predicted),
                        confidence=0.0,
                        radius_angstrom=parameters.diameter_angstrom / 2.0,
                        valid_fraction=0.0,
                        accepted=False,
                        reason="volume boundary",
                        detector=None,
                    ),
                )
                _set_stop_reason(filament, direction, "volume boundary")
                active.remove(filament.filament_id)
                continue

            detection = detect_cross_section(
                volume,
                predicted,
                blended_physical / voxel_size,
                voxel_size,
                parameters,
                empirical_template=(
                    adaptive_templates.get(filament.filament_id)
                ),
                empirical_template_source=template_sources.get(
                    filament.filament_id,
                    "seed crop",
                ),
            )
            record = TraceDiagnostic(
                filament_id=filament.filament_id,
                direction=direction,
                step_number=step_number,
                predicted_position_zyx=_position_tuple(predicted),
                detected_position_zyx=_position_tuple(detection.position_zyx),
                confidence=detection.confidence,
                radius_angstrom=detection.radius_angstrom,
                valid_fraction=detection.valid_fraction,
                accepted=False,
                reason="evaluating",
                detector=detection.diagnostic,
            )
            _record_diagnostic(diagnostics, max_diagnostics, record)
            threshold = (
                0.05
                if parameters.mode == "uninterrupted"
                else parameters.confidence_threshold
            )
            if detection.confidence < threshold:
                _set_stop_reason(
                    filament,
                    direction,
                    f"low confidence ({detection.confidence:.3f})",
                )
                record.reason = (
                    f"low confidence: {detection.confidence:.3f} < {threshold:.3f}"
                )
                active.remove(filament.filament_id)
                continue

            new_direction = detection.position_zyx - current
            bend = _physical_angle(individual, new_direction, voxel_size)
            if bend > parameters.max_bend_degrees:
                _set_stop_reason(
                    filament,
                    direction,
                    f"excessive bend ({bend:.1f}°)",
                )
                record.reason = (
                    f"excessive bend: {bend:.1f}° > "
                    f"{parameters.max_bend_degrees:.1f}°"
                )
                active.remove(filament.filament_id)
                continue

            accepted_positions.append(
                (filament.filament_id, detection.position_zyx)
            )
            point = SkeletonPoint(
                position_zyx=tuple(float(v) for v in detection.position_zyx),
                radius_angstrom=detection.radius_angstrom,
                confidence=detection.confidence,
                provenance="automatic",
            )
            if direction == "forward":
                filament.points.append(point)
            else:
                filament.points.insert(0, point)
            record.accepted = True
            record.reason = "accepted"
            if detection.updated_template is not None:
                candidate_template_updates[filament.filament_id] = (
                    detection.updated_template
                )

        collided = _stop_collisions(
            filaments,
            accepted_positions,
            voxel_size,
            minimum_distance=parameters.diameter_angstrom * 0.45,
            direction=direction,
            active=active,
        )
        if diagnostics is not None and collided:
            for record in diagnostics:
                if (
                    record.step_number == step_number
                    and record.direction == direction
                    and record.filament_id in collided
                ):
                    record.accepted = False
                    record.reason = "bundle assignment collision"
        for filament_id, template in candidate_template_updates.items():
            if filament_id in collided:
                continue
            adaptive_templates[filament_id] = template
            template_sources[filament_id] = f"adaptive step {step_number}"

    return filaments


def _physical_angle(
    first_physical: NDArray[np.floating],
    second_data: NDArray[np.floating],
    voxel_size: NDArray[np.floating],
) -> float:
    first = normalize(np.asarray(first_physical, dtype=float))
    second = data_vector_to_physical(second_data, voxel_size)
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _set_stop_reason(
    filament: FilamentSkeleton,
    direction: str,
    reason: str,
) -> None:
    if direction == "forward":
        filament.forward_stop_reason = reason
    else:
        filament.backward_stop_reason = reason


def _stop_collisions(
    filaments: list[FilamentSkeleton],
    accepted_positions: list[tuple[int, NDArray[np.float64]]],
    voxel_size: NDArray[np.float64],
    minimum_distance: float,
    direction: str,
    active: set[int],
) -> set[int]:
    """Flag newly accepted centers that collapse onto the same density."""

    by_id = {filament.filament_id: filament for filament in filaments}
    collided: set[int] = set()
    for first_index in range(len(accepted_positions)):
        first_id, first_position = accepted_positions[first_index]
        for second_id, second_position in accepted_positions[first_index + 1 :]:
            distance = float(
                np.linalg.norm((first_position - second_position) * voxel_size)
            )
            if distance >= minimum_distance:
                continue
            for filament_id in (first_id, second_id):
                if filament_id in collided:
                    continue
                filament = by_id[filament_id]
                if direction == "forward" and filament.points:
                    filament.points.pop()
                elif direction == "backward" and filament.points:
                    filament.points.pop(0)
                _set_stop_reason(filament, direction, "bundle assignment collision")
                active.discard(filament_id)
                collided.add(filament_id)
    return collided


def _record_diagnostic(
    diagnostics: list[TraceDiagnostic] | None,
    maximum: int,
    record: TraceDiagnostic,
) -> None:
    if diagnostics is None:
        return
    diagnostics.append(record)
    overflow = len(diagnostics) - maximum
    if overflow > 0:
        del diagnostics[:overflow]


def _position_tuple(
    position: NDArray[np.floating],
) -> tuple[float, float, float]:
    return tuple(float(value) for value in position)
