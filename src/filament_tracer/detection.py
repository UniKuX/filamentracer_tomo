"""Train-free dot and ring detection on arbitrary tomogram cross-sections."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.feature import match_template

from filament_tracer.geometry import (
    data_vector_to_physical,
    orthonormal_plane_basis,
    physical_vector_to_data,
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
    slab_slices_used: int = 1
    slab_spacing_angstrom: float = 0.0
    orientation_candidates: int = 1
    selected_orientation_offset_degrees: float = 0.0
    circularity: float = 0.0
    combined_score: float = 0.0
    selected_tangent_physical_zyx: tuple[float, float, float] | None = None


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


def extract_averaged_oriented_slab(
    volume: ArrayLike,
    center_data_zyx: NDArray[np.floating],
    tangent_physical_zyx: NDArray[np.floating],
    voxel_size_zyx: NDArray[np.floating],
    radius_angstrom: float,
    sample_spacing_angstrom: float,
    slab_slices: int,
    slab_spacing_angstrom: float,
    previous_basis_u_physical_zyx: NDArray[np.floating] | None = None,
    previous_basis_v_physical_zyx: NDArray[np.floating] | None = None,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.bool_],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Average parallel cross-sections with symmetric distance weights."""

    if slab_slices < 1 or slab_slices % 2 == 0:
        raise ValueError("slab_slices must be a positive odd integer")
    if slab_spacing_angstrom <= 0:
        raise ValueError("slab spacing must be positive")
    if slab_slices == 1:
        return extract_oriented_patch(
            volume,
            center_data_zyx,
            tangent_physical_zyx,
            voxel_size_zyx,
            radius_angstrom,
            sample_spacing_angstrom,
            previous_basis_u_physical_zyx,
            previous_basis_v_physical_zyx,
        )

    tangent = np.asarray(tangent_physical_zyx, dtype=float)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-12:
        raise ValueError("tangent must be non-zero")
    tangent /= tangent_norm
    voxel_size = np.asarray(voxel_size_zyx, dtype=float)
    center_physical = np.asarray(center_data_zyx, dtype=float) * voxel_size
    half = slab_slices // 2
    indices = np.arange(-half, half + 1, dtype=float)
    weights = np.power(0.5, np.abs(indices))

    numerator: NDArray[np.float64] | None = None
    valid_weight: NDArray[np.float64] | None = None
    basis_u: NDArray[np.float64] | None = None
    basis_v: NDArray[np.float64] | None = None
    for index, weight in zip(indices, weights, strict=True):
        offset_center = (center_physical + tangent * index * slab_spacing_angstrom)
        patch, valid, current_u, current_v = extract_oriented_patch(
            volume,
            offset_center / voxel_size,
            tangent,
            voxel_size,
            radius_angstrom,
            sample_spacing_angstrom,
            previous_basis_u_physical_zyx=(
                previous_basis_u_physical_zyx if basis_u is None else basis_u
            ),
            previous_basis_v_physical_zyx=(
                previous_basis_v_physical_zyx if basis_v is None else basis_v
            ),
        )
        if numerator is None:
            numerator = np.zeros_like(patch, dtype=np.float64)
            valid_weight = np.zeros_like(patch, dtype=np.float64)
            basis_u, basis_v = current_u, current_v
        numerator += weight * patch * valid
        valid_weight += weight * valid

    assert numerator is not None and valid_weight is not None
    assert basis_u is not None and basis_v is not None
    averaged = np.divide(
        numerator,
        valid_weight,
        out=np.zeros_like(numerator),
        where=valid_weight > 0,
    )
    valid = valid_weight >= 0.5 * float(np.sum(weights))
    return averaged.astype(np.float32), valid, basis_u, basis_v


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
    if parameters.use_slab_averaging and parameters.slab_slices > 1:
        slab_spacing = (
            parameters.slab_spacing_angstrom
            if parameters.slab_spacing_angstrom is not None
            else float(np.min(voxel_size))
        )
        patch, valid, basis_u, basis_v = extract_averaged_oriented_slab(
            volume,
            np.asarray(seed_data_zyx, dtype=float),
            tangent_physical,
            voxel_size,
            crop_radius,
            sample_spacing,
            parameters.slab_slices,
            slab_spacing,
        )
    else:
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


def _local_circularity(
    image: NDArray[np.floating],
    center_rc: tuple[float, float],
    radius_pixels: float,
) -> float:
    """Estimate local isotropy from intensity-weighted second moments."""

    radius = max(2.0, 1.5 * radius_pixels)
    yy, xx = np.indices(image.shape, dtype=float)
    local = (
        (yy - center_rc[0]) ** 2 + (xx - center_rc[1]) ** 2
        <= radius**2
    )
    if np.count_nonzero(local) < 6:
        return 0.0
    values = np.asarray(image, dtype=float)[local]
    weights = np.abs(values - float(np.median(values)))
    total = float(np.sum(weights))
    if total <= 1e-12:
        return 0.0
    coordinates = np.column_stack(
        (yy[local] - center_rc[0], xx[local] - center_rc[1])
    )
    centroid = np.sum(coordinates * weights[:, None], axis=0) / total
    centered = coordinates - centroid
    covariance = (centered * weights[:, None]).T @ centered / total
    eigenvalues = np.linalg.eigvalsh(covariance)
    largest = float(eigenvalues[-1])
    if largest <= 1e-12:
        return 0.0
    return float(np.clip(eigenvalues[0] / largest, 0.0, 1.0))


def _detect_cross_section_candidate(
    volume: ArrayLike,
    predicted_data_zyx: NDArray[np.floating],
    tangent_data_zyx: NDArray[np.floating],
    voxel_size_zyx: NDArray[np.floating],
    parameters: TracingParameters,
    empirical_template: NDArray[np.floating] | OrientedTemplate | None = None,
    empirical_template_source: str = "seed crop",
) -> DetectionResult:
    """Score one candidate tangent with the existing train-free detector."""

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
    previous_u = (
        oriented_empirical.basis_u_physical_zyx
        if oriented_empirical is not None
        else None
    )
    previous_v = (
        oriented_empirical.basis_v_physical_zyx
        if oriented_empirical is not None
        else None
    )
    slab_slices_used = (
        parameters.slab_slices
        if parameters.use_slab_averaging and parameters.slab_slices > 1
        else 1
    )
    slab_spacing = (
        parameters.slab_spacing_angstrom
        if parameters.slab_spacing_angstrom is not None
        else float(np.min(voxel_size))
    )
    if slab_slices_used > 1:
        patch, valid, basis_u, basis_v = extract_averaged_oriented_slab(
            volume,
            np.asarray(predicted_data_zyx, dtype=float),
            tangent_physical,
            voxel_size,
            patch_radius,
            sample_spacing,
            slab_slices_used,
            slab_spacing,
            previous_basis_u_physical_zyx=previous_u,
            previous_basis_v_physical_zyx=previous_v,
        )
    else:
        patch, valid, basis_u, basis_v = extract_oriented_patch(
            volume,
            np.asarray(predicted_data_zyx, dtype=float),
            tangent_physical,
            voxel_size,
            patch_radius,
            sample_spacing,
            previous_basis_u_physical_zyx=previous_u,
            previous_basis_v_physical_zyx=previous_v,
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
    confidence = float(np.clip(best_score, 0.0, 1.0))
    circularity = (
        _local_circularity(
            working,
            detected_center,
            best_radius / sample_spacing,
        )
        if parameters.orientation_search or parameters.circularity_weight > 0.0
        else 0.0
    )

    return DetectionResult(
        position_zyx=position_data,
        radius_angstrom=float(best_radius),
        confidence=confidence,
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
            slab_slices_used=slab_slices_used,
            slab_spacing_angstrom=(
                slab_spacing if slab_slices_used > 1 else 0.0
            ),
            circularity=circularity,
            combined_score=(
                confidence + parameters.circularity_weight * circularity
            ),
            selected_tangent_physical_zyx=tuple(
                float(value) for value in tangent_physical
            ),
        ),
    )


def _orientation_candidates(
    predicted_tangent_physical_zyx: NDArray[np.floating],
    parameters: TracingParameters,
    oriented_template: OrientedTemplate | None,
) -> list[tuple[NDArray[np.float64], float]]:
    """Generate deterministic tangent candidates in a small physical cone."""

    tangent = np.asarray(predicted_tangent_physical_zyx, dtype=float)
    tangent /= float(np.linalg.norm(tangent))
    candidates = [(tangent, 0.0)]
    if not parameters.orientation_search:
        return candidates

    if oriented_template is not None:
        basis_u, basis_v = transport_plane_basis(
            tangent,
            oriented_template.basis_u_physical_zyx,
            oriented_template.basis_v_physical_zyx,
        )
    else:
        basis_u, basis_v = orthonormal_plane_basis(tangent)
    azimuths = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    for step in range(1, parameters.orientation_search_steps + 1):
        angle_degrees = (
            parameters.orientation_search_degrees
            * step
            / parameters.orientation_search_steps
        )
        angle = np.radians(angle_degrees)
        for azimuth in azimuths:
            radial = np.cos(azimuth) * basis_u + np.sin(azimuth) * basis_v
            candidate = np.cos(angle) * tangent + np.sin(angle) * radial
            candidate /= float(np.linalg.norm(candidate))
            candidates.append((candidate, float(angle_degrees)))
    return candidates


def detect_cross_section(
    volume: ArrayLike,
    predicted_data_zyx: NDArray[np.floating],
    tangent_data_zyx: NDArray[np.floating],
    voxel_size_zyx: NDArray[np.floating],
    parameters: TracingParameters,
    empirical_template: NDArray[np.floating] | OrientedTemplate | None = None,
    empirical_template_source: str = "seed crop",
) -> DetectionResult:
    """Find a cross-section, optionally using a slab and orientation search."""

    if (
        not parameters.use_slab_averaging
        and not parameters.orientation_search
        and parameters.circularity_weight == 0.0
    ):
        return _detect_cross_section_candidate(
            volume,
            predicted_data_zyx,
            tangent_data_zyx,
            voxel_size_zyx,
            parameters,
            empirical_template,
            empirical_template_source,
        )

    voxel_size = np.asarray(voxel_size_zyx, dtype=float)
    predicted_tangent = data_vector_to_physical(
        tangent_data_zyx,
        voxel_size,
    )
    oriented_template = (
        empirical_template
        if isinstance(empirical_template, OrientedTemplate)
        else None
    )
    candidates = _orientation_candidates(
        predicted_tangent,
        parameters,
        oriented_template,
    )
    best_result: DetectionResult | None = None
    best_combined = -np.inf
    best_angle = 0.0
    best_tangent = predicted_tangent
    for candidate_tangent, angle_degrees in candidates:
        candidate_data = physical_vector_to_data(
            candidate_tangent,
            voxel_size,
        )
        result = _detect_cross_section_candidate(
            volume,
            predicted_data_zyx,
            candidate_data,
            voxel_size,
            parameters,
            empirical_template,
            empirical_template_source,
        )
        circularity = (
            result.diagnostic.circularity
            if result.diagnostic is not None
            else 0.0
        )
        bend_penalty = 0.01 * (
            angle_degrees / parameters.orientation_search_degrees
        ) ** 2
        combined = (
            result.confidence
            + parameters.circularity_weight * circularity
            - bend_penalty
        )
        if combined > best_combined:
            best_result = result
            best_combined = combined
            best_angle = angle_degrees
            best_tangent = candidate_tangent

    assert best_result is not None
    if best_result.diagnostic is None:
        return best_result
    slab_enabled = (
        parameters.use_slab_averaging and parameters.slab_slices > 1
    )
    slab_spacing_used = (
        parameters.slab_spacing_angstrom
        if parameters.slab_spacing_angstrom is not None
        else float(np.min(voxel_size))
    )
    diagnostic = replace(
        best_result.diagnostic,
        slab_slices_used=parameters.slab_slices if slab_enabled else 1,
        slab_spacing_angstrom=(
            slab_spacing_used if slab_enabled else 0.0
        ),
        orientation_candidates=len(candidates),
        selected_orientation_offset_degrees=best_angle,
        combined_score=float(best_combined),
        selected_tangent_physical_zyx=tuple(
            float(value) for value in best_tangent
        ),
    )
    return replace(best_result, diagnostic=diagnostic)
