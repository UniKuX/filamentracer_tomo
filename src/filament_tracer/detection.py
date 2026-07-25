"""Train-free dot and ring detection on arbitrary tomogram cross-sections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.feature import match_template

from filament_tracer.geometry import (
    data_vector_to_physical,
    orthonormal_plane_basis,
    transport_plane_basis,
)
from filament_tracer.models import TracingParameters


@dataclass(frozen=True)
class DetectionDiagnostic:
    """Intermediate arrays used to explain one detector decision."""

    patch: NDArray[np.float32]
    template: NDArray[np.float32]
    response: NDArray[np.float32]
    search_mask: NDArray[np.bool_]
    sample_spacing_angstrom: float
    predicted_rc: tuple[float, float]
    detected_rc: tuple[float, float]
    template_source: str


@dataclass(frozen=True)
class OrientedTemplate:
    """A 2D template together with its physical in-plane coordinate frame."""

    pixels: NDArray[np.float32]
    basis_u_physical_zyx: NDArray[np.float64]
    basis_v_physical_zyx: NDArray[np.float64]

    @property
    def shape(self) -> tuple[int, int]:
        return self.pixels.shape


@dataclass(frozen=True)
class DetectionResult:
    """Best cross-section detection around a predicted center."""

    position_zyx: NDArray[np.float64]
    radius_angstrom: float
    confidence: float
    valid_fraction: float
    diagnostic: DetectionDiagnostic | None = None
    updated_template: OrientedTemplate | None = None


def _odd_size(value: float, minimum: int = 9) -> int:
    size = max(minimum, int(np.ceil(value)))
    return size if size % 2 == 1 else size + 1


def extract_oriented_patch(
    volume: ArrayLike,
    center_data_zyx: NDArray[np.floating],
    tangent_physical_zyx: NDArray[np.floating],
    voxel_size_zyx: NDArray[np.floating],
    radius_angstrom: float,
    sample_spacing_angstrom: float,
    previous_basis_u_physical_zyx: NDArray[np.floating] | None = None,
    previous_basis_v_physical_zyx: NDArray[np.floating] | None = None,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.bool_],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Interpolate a square patch perpendicular to a physical tangent."""

    if radius_angstrom <= 0 or sample_spacing_angstrom <= 0:
        raise ValueError("patch radius and sample spacing must be positive")

    center_data = np.asarray(center_data_zyx, dtype=float)
    voxel_size = np.asarray(voxel_size_zyx, dtype=float)
    center_physical = center_data * voxel_size
    if previous_basis_u_physical_zyx is None:
        basis_u, basis_v = orthonormal_plane_basis(tangent_physical_zyx)
    else:
        basis_u, basis_v = transport_plane_basis(
            tangent_physical_zyx,
            previous_basis_u_physical_zyx,
            previous_basis_v_physical_zyx,
        )

    size = _odd_size((2.0 * radius_angstrom) / sample_spacing_angstrom + 1)
    half = size // 2
    offsets = (np.arange(size, dtype=float) - half) * sample_spacing_angstrom
    grid_v, grid_u = np.meshgrid(offsets, offsets, indexing="ij")
    positions_physical = (
        center_physical[:, None, None]
        + basis_u[:, None, None] * grid_u
        + basis_v[:, None, None] * grid_v
    )
    coordinates_data = positions_physical / voxel_size[:, None, None]

    array = np.asanyarray(volume)
    patch = map_coordinates(
        array,
        coordinates_data,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32, copy=False)
    shape = np.asarray(array.shape, dtype=float)[:, None, None]
    valid = np.all(
        (coordinates_data >= 0.0) & (coordinates_data <= shape - 1.0),
        axis=0,
    )
    return patch, valid, basis_u, basis_v


def _cross_section_template(
    radius_pixels: float,
    template_kind: str,
) -> NDArray[np.float32]:
    extent = max(3.0, radius_pixels * 1.8)
    size = _odd_size(2.0 * extent + 1)
    center = size // 2
    yy, xx = np.mgrid[:size, :size]
    distance = np.hypot(yy - center, xx - center)

    if template_kind == "ring":
        width = max(0.8, radius_pixels * 0.22)
        template = np.exp(-0.5 * ((distance - radius_pixels) / width) ** 2)
    else:
        sigma = max(0.8, radius_pixels * 0.65)
        template = np.exp(-0.5 * (distance / sigma) ** 2)

    template -= float(template.mean())
    norm = float(np.linalg.norm(template))
    if norm > 0:
        template /= norm
    return template.astype(np.float32)


def extract_seed_oriented_template(
    volume: ArrayLike,
    seed_data_zyx: NDArray[np.floating],
    tangent_data_zyx: NDArray[np.floating],
    voxel_size_zyx: NDArray[np.floating],
    parameters: TracingParameters,
) -> OrientedTemplate | None:
    """Crop a seed cross-section and preserve its in-plane orientation."""

    voxel_size = np.asarray(voxel_size_zyx, dtype=float)
    tangent_physical = data_vector_to_physical(tangent_data_zyx, voxel_size)
    sample_spacing = float(np.min(voxel_size) / 2.0)
    crop_radius = max(
        parameters.diameter_angstrom,
        3.0 * float(np.max(voxel_size)),
    )
    patch, valid, basis_u, basis_v = extract_oriented_patch(
        volume,
        np.asarray(seed_data_zyx, dtype=float),
        tangent_physical,
        voxel_size,
        crop_radius,
        sample_spacing,
    )
    if float(valid.mean()) < 0.9 or np.count_nonzero(valid) < 16:
        return None
    valid_values = patch[valid]
    fill_value = float(np.median(valid_values))
    template = np.where(valid, patch, fill_value)
    template = gaussian_filter(template, sigma=0.55)
    template -= float(np.mean(template))
    deviation = float(np.std(template))
    if deviation <= 1e-8:
        return None
    template /= deviation
    return OrientedTemplate(
        pixels=template.astype(np.float32, copy=False),
        basis_u_physical_zyx=basis_u,
        basis_v_physical_zyx=basis_v,
    )


def extract_seed_template(
    volume: ArrayLike,
    seed_data_zyx: NDArray[np.floating],
    tangent_data_zyx: NDArray[np.floating],
    voxel_size_zyx: NDArray[np.floating],
    parameters: TracingParameters,
) -> NDArray[np.float32] | None:
    """Crop and normalize a real cross-section centered on a manual seed."""

    oriented = extract_seed_oriented_template(
        volume,
        seed_data_zyx,
        tangent_data_zyx,
        voxel_size_zyx,
        parameters,
    )
    return oriented.pixels if oriented is not None else None


def _subpixel_peak(
    response: NDArray[np.floating],
    row: int,
    column: int,
) -> tuple[float, float]:
    """Refine a discrete maximum with independent quadratic fits."""

    def offset(before: float, center: float, after: float) -> float:
        denominator = before - 2.0 * center + after
        if abs(denominator) <= 1e-12:
            return 0.0
        return float(np.clip(0.5 * (before - after) / denominator, -1.0, 1.0))

    row_offset = 0.0
    column_offset = 0.0
    if 0 < row < response.shape[0] - 1:
        row_offset = offset(
            float(response[row - 1, column]),
            float(response[row, column]),
            float(response[row + 1, column]),
        )
    if 0 < column < response.shape[1] - 1:
        column_offset = offset(
            float(response[row, column - 1]),
            float(response[row, column]),
            float(response[row, column + 1]),
        )
    return row_offset, column_offset


def _template_at_detection(
    patch: NDArray[np.floating],
    center_rc: tuple[float, float],
    shape: tuple[int, int],
) -> NDArray[np.float32] | None:
    """Crop and normalize the latest cross-section at a subpixel center."""

    height, width = shape
    if height <= 0 or width <= 0:
        return None
    row_offsets = np.arange(height, dtype=float) - (height - 1.0) / 2.0
    column_offsets = np.arange(width, dtype=float) - (width - 1.0) / 2.0
    rows, columns = np.meshgrid(
        center_rc[0] + row_offsets,
        center_rc[1] + column_offsets,
        indexing="ij",
    )
    if (
        float(np.min(rows)) < 0.0
        or float(np.max(rows)) > patch.shape[0] - 1.0
        or float(np.min(columns)) < 0.0
        or float(np.max(columns)) > patch.shape[1] - 1.0
    ):
        return None
    template = map_coordinates(
        np.asarray(patch, dtype=np.float32),
        (rows, columns),
        order=1,
        mode="nearest",
        prefilter=False,
    ).astype(np.float32, copy=False)
    template -= float(np.mean(template))
    deviation = float(np.std(template))
    if deviation <= 1e-8:
        return None
    template /= deviation
    return template


def detect_cross_section(
    volume: ArrayLike,
    predicted_data_zyx: NDArray[np.floating],
    tangent_data_zyx: NDArray[np.floating],
    voxel_size_zyx: NDArray[np.floating],
    parameters: TracingParameters,
    empirical_template: NDArray[np.floating] | OrientedTemplate | None = None,
    empirical_template_source: str = "seed crop",
) -> DetectionResult:
    """Find the best dot or ring near a predicted cross-section center."""

    voxel_size = np.asarray(voxel_size_zyx, dtype=float)
    tangent_physical = data_vector_to_physical(tangent_data_zyx, voxel_size)
    sample_spacing = float(np.min(voxel_size) / 2.0)
    template_radius = parameters.diameter_angstrom / 2.0
    patch_radius = max(
        parameters.diameter_angstrom * 1.75,
        parameters.search_radius_voxels * float(np.max(voxel_size))
        + parameters.diameter_angstrom,
    )

    oriented_empirical = (
        empirical_template
        if isinstance(empirical_template, OrientedTemplate)
        else None
    )
    patch, valid, basis_u, basis_v = extract_oriented_patch(
        volume,
        np.asarray(predicted_data_zyx, dtype=float),
        tangent_physical,
        voxel_size,
        patch_radius,
        sample_spacing,
        previous_basis_u_physical_zyx=(
            oriented_empirical.basis_u_physical_zyx
            if oriented_empirical is not None
            else None
        ),
        previous_basis_v_physical_zyx=(
            oriented_empirical.basis_v_physical_zyx
            if oriented_empirical is not None
            else None
        ),
    )
    valid_fraction = float(valid.mean())
    valid_values = patch[valid]
    if valid_fraction < 0.65 or valid_values.size < 16:
        return DetectionResult(
            position_zyx=np.asarray(predicted_data_zyx, dtype=float),
            radius_angstrom=template_radius,
            confidence=0.0,
            valid_fraction=valid_fraction,
            diagnostic=DetectionDiagnostic(
                patch=patch,
                template=np.empty((0, 0), dtype=np.float32),
                response=np.zeros_like(patch),
                search_mask=np.zeros_like(valid),
                sample_spacing_angstrom=sample_spacing,
                predicted_rc=(
                    float(patch.shape[0] / 2.0 - 0.5),
                    float(patch.shape[1] / 2.0 - 0.5),
                ),
                detected_rc=(
                    float(patch.shape[0] / 2.0 - 0.5),
                    float(patch.shape[1] / 2.0 - 0.5),
                ),
                template_source="unavailable",
            ),
        )

    fill_value = float(np.median(valid_values))
    working = np.where(valid, patch, fill_value)
    working = gaussian_filter(working, sigma=0.55)
    working -= float(np.mean(working))
    deviation = float(np.std(working))
    if deviation <= 1e-8:
        return DetectionResult(
            position_zyx=np.asarray(predicted_data_zyx, dtype=float),
            radius_angstrom=template_radius,
            confidence=0.0,
            valid_fraction=valid_fraction,
            diagnostic=DetectionDiagnostic(
                patch=working.astype(np.float32, copy=False),
                template=np.empty((0, 0), dtype=np.float32),
                response=np.zeros_like(working, dtype=np.float32),
                search_mask=np.zeros_like(valid),
                sample_spacing_angstrom=sample_spacing,
                predicted_rc=(
                    float(working.shape[0] / 2.0 - 0.5),
                    float(working.shape[1] / 2.0 - 0.5),
                ),
                detected_rc=(
                    float(working.shape[0] / 2.0 - 0.5),
                    float(working.shape[1] / 2.0 - 0.5),
                ),
                template_source="unavailable",
            ),
        )
    working /= deviation

    patch_center = np.asarray(working.shape, dtype=float) / 2.0 - 0.5
    yy, xx = np.indices(working.shape)
    search_radius_pixels = (
        parameters.search_radius_voxels * float(np.max(voxel_size)) / sample_spacing
    )
    search_mask = (
        (yy - patch_center[0]) ** 2 + (xx - patch_center[1]) ** 2
        <= search_radius_pixels**2
    )

    best_score = -np.inf
    best_response: NDArray[np.float32] | None = None
    best_template: NDArray[np.float32] | None = None
    best_index = (int(patch_center[0]), int(patch_center[1]))
    best_radius = template_radius
    using_empirical = empirical_template is not None
    if using_empirical:
        empirical_pixels = (
            empirical_template.pixels
            if isinstance(empirical_template, OrientedTemplate)
            else empirical_template
        )
        candidate_templates = [
            (
                template_radius,
                np.asarray(empirical_pixels, dtype=np.float32),
            )
        ]
        template_source = empirical_template_source
    else:
        fallback_kind = parameters.template_kind
        if fallback_kind == "seed_crop":
            fallback_kind = (
                "ring"
                if parameters.filament_kind == "microtubule"
                else "solid"
            )
        candidate_templates = [
            (
                template_radius * scale,
                _cross_section_template(
                    template_radius * scale / sample_spacing,
                    fallback_kind,
                ),
            )
            for scale in (0.8, 1.0, 1.2)
        ]
        template_source = f"ideal {fallback_kind}"

    for radius, template in candidate_templates:
        if (
            template.ndim != 2
            or template.size == 0
            or template.shape[0] > working.shape[0]
            or template.shape[1] > working.shape[1]
        ):
            continue
        response = match_template(
            working,
            template,
            pad_input=True,
            mode="constant",
            constant_values=0,
        ).astype(np.float32, copy=False)
        if using_empirical:
            scored = response
        elif parameters.polarity == "dark":
            scored = -response
        elif parameters.polarity == "bright":
            scored = response
        else:
            scored = np.abs(response)

        masked = np.where(search_mask, scored, -np.inf)
        flat_index = int(np.argmax(masked))
        index = tuple(
            int(value)
            for value in np.unravel_index(flat_index, masked.shape)
        )
        score = float(masked[index])
        if score > best_score:
            best_score = score
            best_response = scored
            best_template = template
            best_index = index
            best_radius = radius

    if best_response is None or best_template is None:
        return DetectionResult(
            position_zyx=np.asarray(predicted_data_zyx, dtype=float),
            radius_angstrom=template_radius,
            confidence=0.0,
            valid_fraction=valid_fraction,
            diagnostic=DetectionDiagnostic(
                patch=working.astype(np.float32, copy=False),
                template=np.empty((0, 0), dtype=np.float32),
                response=np.zeros_like(working, dtype=np.float32),
                search_mask=search_mask,
                sample_spacing_angstrom=sample_spacing,
                predicted_rc=(float(patch_center[0]), float(patch_center[1])),
                detected_rc=(float(patch_center[0]), float(patch_center[1])),
                template_source="invalid template",
            ),
        )
    row_offset, column_offset = _subpixel_peak(
        best_response,
        best_index[0],
        best_index[1],
    )
    detected_center = (
        float(best_index[0] + row_offset),
        float(best_index[1] + column_offset),
    )
    updated_pixels = _template_at_detection(
        working,
        detected_center,
        best_template.shape,
    )
    updated_template = (
        OrientedTemplate(
            pixels=updated_pixels,
            basis_u_physical_zyx=basis_u,
            basis_v_physical_zyx=basis_v,
        )
        if updated_pixels is not None
        else None
    )
    displacement_v = (
        best_index[0] + row_offset - patch_center[0]
    ) * sample_spacing
    displacement_u = (
        best_index[1] + column_offset - patch_center[1]
    ) * sample_spacing
    displacement_physical = (
        basis_u * displacement_u + basis_v * displacement_v
    )
    position_data = (
        np.asarray(predicted_data_zyx, dtype=float)
        + displacement_physical / voxel_size
    )

    return DetectionResult(
        position_zyx=position_data,
        radius_angstrom=float(best_radius),
        confidence=float(np.clip(best_score, 0.0, 1.0)),
        valid_fraction=valid_fraction,
        updated_template=updated_template,
        diagnostic=DetectionDiagnostic(
            patch=working.astype(np.float32, copy=False),
            template=best_template,
            response=best_response,
            search_mask=search_mask,
            sample_spacing_angstrom=sample_spacing,
            predicted_rc=(float(patch_center[0]), float(patch_center[1])),
            detected_rc=detected_center,
            template_source=template_source,
        ),
    )
