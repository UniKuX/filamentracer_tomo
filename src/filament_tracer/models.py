"""Serializable data models shared by the tracing UI and core algorithms."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Vector3 = tuple[float, float, float]
FilamentKind = Literal["f_actin", "intermediate", "microtubule", "custom"]
Polarity = Literal["auto", "dark", "bright"]
TraceMode = Literal["guided", "step_by_step", "uninterrupted"]


class VolumeMetadata(BaseModel):
    """Metadata needed to interpret skeleton coordinates."""

    source_path: str | None = None
    name: str
    shape_zyx: tuple[int, int, int]
    voxel_size_zyx: Vector3

    @field_validator("voxel_size_zyx")
    @classmethod
    def validate_voxel_size(cls, value: Vector3) -> Vector3:
        if any(component <= 0 for component in value):
            raise ValueError("voxel sizes must all be positive")
        return value


class PlaneDefinition(BaseModel):
    """A plane in tomogram data coordinates."""

    position_zyx: Vector3
    normal_zyx: Vector3
    thickness: float = 1.0


class SeedMatch(BaseModel):
    """A one-to-one link between seed points on planes A and B."""

    a_index: int
    b_index: int
    residual_angstrom: float


class SkeletonPoint(BaseModel):
    """One ordered vertex in a filament skeleton."""

    position_zyx: Vector3
    radius_angstrom: float
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: Literal["seed", "automatic", "manual"] = "automatic"


class FilamentSkeleton(BaseModel):
    """An ordered, non-branching filament polyline."""

    filament_id: int
    points: list[SkeletonPoint]
    backward_stop_reason: str | None = None
    forward_stop_reason: str | None = None


class TracingParameters(BaseModel):
    """Parameters for classical cross-section detection and path following."""

    filament_kind: FilamentKind = "f_actin"
    template_kind: Literal["seed_crop", "solid", "ring"] = "seed_crop"
    diameter_angstrom: float = Field(default=70.0, gt=0)
    step_voxels: float = Field(default=2.0, gt=0)
    search_radius_voxels: float = Field(default=3.0, gt=0)
    max_steps: int = Field(default=100, ge=1, le=10_000)
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    max_bend_degrees: float = Field(default=35.0, gt=0.0, le=90.0)
    polarity: Polarity = "auto"
    mode: TraceMode = "guided"
    use_slab_averaging: bool = False
    slab_slices: int = Field(default=3, ge=1, le=11)
    slab_spacing_angstrom: float | None = Field(default=None, gt=0.0)
    orientation_search: bool = False
    orientation_search_degrees: float = Field(default=15.0, gt=0.0, le=45.0)
    orientation_search_steps: int = Field(default=1, ge=1, le=3)
    circularity_weight: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("slab_slices")
    @classmethod
    def validate_slab_slices(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("slab_slices must be odd")
        return value


class SkeletonProject(BaseModel):
    """Versioned output of Part 1 and input contract for Part 2."""

    schema_version: Literal["1.0"] = "1.0"
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    volume: VolumeMetadata
    plane_a: PlaneDefinition
    plane_b: PlaneDefinition
    seed_matches: list[SeedMatch]
    tracing_parameters: TracingParameters
    filaments: list[FilamentSkeleton]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SkeletonProject:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
