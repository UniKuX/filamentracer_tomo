"""Geometry helpers for seed matching and oriented cross-sections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class MatchResult:
    """Result of matching seed point indices between two planes."""

    pairs: tuple[tuple[int, int, float], ...]
    unmatched_a: tuple[int, ...]
    unmatched_b: tuple[int, ...]


def normalize(vector: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return a unit vector and reject degenerate input."""

    array = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(array))
    if not np.isfinite(length) or length <= 1e-12:
        raise ValueError("cannot normalize a zero or non-finite vector")
    return array / length


def data_vector_to_physical(
    vector_zyx: NDArray[np.floating],
    voxel_size_zyx: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Convert a direction from data coordinates to a physical unit vector."""

    return normalize(np.asarray(vector_zyx, dtype=float) * voxel_size_zyx)


def physical_vector_to_data(
    vector_zyx: NDArray[np.floating],
    voxel_size_zyx: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Convert a physical direction to a data-coordinate unit vector."""

    return normalize(np.asarray(vector_zyx, dtype=float) / voxel_size_zyx)


def orthonormal_plane_basis(
    normal_physical_zyx: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create a deterministic orthonormal basis perpendicular to ``normal``."""

    normal = normalize(normal_physical_zyx)
    reference = np.zeros(3, dtype=float)
    reference[int(np.argmin(np.abs(normal)))] = 1.0
    first = normalize(np.cross(normal, reference))
    second = normalize(np.cross(normal, first))
    return first, second


def transport_plane_basis(
    normal_physical_zyx: NDArray[np.floating],
    previous_first_physical_zyx: NDArray[np.floating],
    previous_second_physical_zyx: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Transport an existing in-plane frame to a nearby perpendicular plane.

    Projecting the prior first axis avoids the discontinuous global-reference
    axis changes produced by :func:`orthonormal_plane_basis`.
    """

    normal = normalize(normal_physical_zyx)
    previous_first = normalize(previous_first_physical_zyx)
    projected = previous_first - np.dot(previous_first, normal) * normal
    if (
        np.linalg.norm(projected) <= 1e-8
        and previous_second_physical_zyx is not None
    ):
        previous_second = normalize(previous_second_physical_zyx)
        projected = previous_second - np.dot(previous_second, normal) * normal
    if np.linalg.norm(projected) <= 1e-8:
        return orthonormal_plane_basis(normal)

    first = normalize(projected)
    if np.dot(first, previous_first) < 0.0:
        first = -first
    second = normalize(np.cross(normal, first))
    if previous_second_physical_zyx is not None:
        previous_second = normalize(previous_second_physical_zyx)
        if np.dot(second, previous_second) < 0.0:
            first = -first
            second = -second
    return first, second


def physical_angle_degrees(
    first_data_zyx: NDArray[np.floating],
    second_data_zyx: NDArray[np.floating],
    voxel_size_zyx: NDArray[np.floating],
) -> float:
    """Return the physical angle between two data-coordinate vectors."""

    first = data_vector_to_physical(first_data_zyx, voxel_size_zyx)
    second = data_vector_to_physical(second_data_zyx, voxel_size_zyx)
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def match_seed_points(
    points_a_zyx: NDArray[np.floating],
    points_b_zyx: NDArray[np.floating],
    voxel_size_zyx: NDArray[np.floating],
    max_residual_angstrom: float,
) -> MatchResult:
    """Match two unordered seed sets after estimating their shared translation.

    A centroid shift provides an initial bundle displacement. A first Hungarian
    assignment refines this displacement with a robust median, followed by the
    final gated assignment.
    """

    points_a = np.asarray(points_a_zyx, dtype=float)
    points_b = np.asarray(points_b_zyx, dtype=float)
    voxel_size = np.asarray(voxel_size_zyx, dtype=float)

    if points_a.ndim != 2 or points_a.shape[1:] != (3,):
        raise ValueError("plane A seeds must have shape (N, 3)")
    if points_b.ndim != 2 or points_b.shape[1:] != (3,):
        raise ValueError("plane B seeds must have shape (N, 3)")
    if len(points_a) == 0 or len(points_b) == 0:
        return MatchResult(
            pairs=(),
            unmatched_a=tuple(range(len(points_a))),
            unmatched_b=tuple(range(len(points_b))),
        )
    if max_residual_angstrom <= 0:
        raise ValueError("maximum residual must be positive")

    physical_a = points_a * voxel_size
    physical_b = points_b * voxel_size
    displacement = np.median(physical_b, axis=0) - np.median(physical_a, axis=0)

    provisional_cost = np.linalg.norm(
        physical_a[:, None, :] + displacement - physical_b[None, :, :],
        axis=2,
    )
    row_indices, column_indices = linear_sum_assignment(provisional_cost)
    pair_displacements = (
        physical_b[column_indices] - physical_a[row_indices]
    )
    displacement = np.median(pair_displacements, axis=0)

    cost = np.linalg.norm(
        physical_a[:, None, :] + displacement - physical_b[None, :, :],
        axis=2,
    )
    row_indices, column_indices = linear_sum_assignment(cost)

    pairs: list[tuple[int, int, float]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    for a_index, b_index in zip(row_indices, column_indices, strict=True):
        residual = float(cost[a_index, b_index])
        if residual <= max_residual_angstrom:
            pairs.append((int(a_index), int(b_index), residual))
            used_a.add(int(a_index))
            used_b.add(int(b_index))

    pairs.sort(key=lambda pair: pair[0])
    return MatchResult(
        pairs=tuple(pairs),
        unmatched_a=tuple(
            index for index in range(len(points_a)) if index not in used_a
        ),
        unmatched_b=tuple(
            index for index in range(len(points_b)) if index not in used_b
        ),
    )
