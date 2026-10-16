"""napari dock widget for Part 1: filament tracing and skeleton generation."""

from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Any

import mrcfile
import numpy as np
from napari import Viewer
from napari.layers import Image, Points, Shapes, Vectors
from napari.qt.threading import thread_worker
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from filament_tracer.geometry import (
    MatchResult,
    match_seed_points,
    normalize,
    smooth_skeleton,
)
from filament_tracer.models import (
    FilamentSkeleton,
    PlaneDefinition,
    SeedMatch,
    SkeletonPoint,
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

SEED_A_LAYER = "FT seed plane A"
SEED_B_LAYER = "FT seed plane B"
MATCH_LAYER = "FT seed matches"
SKELETON_PATH_LAYER = "FT skeleton paths"
SKELETON_POINT_LAYER = "FT skeleton vertices"
MANUAL_SEED_LAYER = "FT manual current plane"
PIVOT_LAYER = "FT slab rotation pivot"
DIAGNOSTIC_PATH_LAYER = "FT diagnostic filament"
DIAGNOSTIC_POINT_LAYER = "FT diagnostic step"
LIVE_TRACE_LAYER = "FT live accepted points"
LIVE_PREDICTION_LAYER = "FT live prediction"


class _TraceProgressSignals(QObject):
    updated = Signal(object)


@thread_worker
def _trace_in_worker(
    volume: Any,
    filaments: list,
    voxel_size_zyx: tuple[float, float, float],
    parameters: TracingParameters,
    directions: tuple[str, ...],
    progress: Any = None,
) -> tuple[list, list[TraceDiagnostic]]:
    """Extend copied skeleton models without touching Qt or napari state."""

    diagnostics: list[TraceDiagnostic] = []
    last_prediction = last_step = float("-inf")

    def predicted(filament_id: int, step: int, position: tuple) -> None:
        nonlocal last_prediction
        now = monotonic()
        if progress is not None and now - last_prediction >= 0.1:
            last_prediction = now
            progress.emit(("prediction", position, filament_id, direction, step))

    def completed(current: list, step: int) -> None:
        nonlocal last_step
        now = monotonic()
        if progress is not None and now - last_step >= 0.1:
            last_step = now
            # Copy values before crossing threads; worker models keep changing.
            positions = [p.position_zyx for f in current for p in f.points]
            ids = [f.filament_id for f in current for _p in f.points]
            progress.emit(("accepted", positions, ids, direction, step))

    for direction in directions:
        seed_templates = build_seed_templates(
            volume,
            filaments,
            voxel_size_zyx,
            parameters,
            direction,
        )
        for filament in filaments:
            if direction == "forward":
                filament.forward_stop_reason = None
            else:
                filament.backward_stop_reason = None
        extend_skeletons(
            volume,
            filaments,
            voxel_size_zyx,
            parameters,
            direction,
            diagnostics=diagnostics,
            seed_templates=seed_templates,
            on_prediction=predicted if progress is not None else None,
            on_step=completed if progress is not None else None,
        )
    return filaments, diagnostics


class FilamentTracerWidget(QWidget):
    """Tomogram tracing, model training, and refined particle display."""

    def __init__(self, napari_viewer: Viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self._image_layer: Image | None = None
        self._mrc_handles: list[mrcfile.mrcmemmap.MrcMemmap] = []
        self._plane_a: PlaneDefinition | None = None
        self._plane_b: PlaneDefinition | None = None
        self._match_result: MatchResult | None = None
        self._project: SkeletonProject | None = None
        self._smoothing_undo = None
        self._trace_worker = None
        self._trace_progress_signals = _TraceProgressSignals(self)
        self._trace_progress_signals.updated.connect(self._show_trace_progress)
        self._diagnostics: list[TraceDiagnostic] = []
        self._manual_session_active = False
        self._individual_id: int | None = None
        self._individual_history: list[SkeletonProject | None] = []
        self._manual_commits: list[
            tuple[PlaneDefinition, np.ndarray, SkeletonProject | None]
        ] = []
        self._manual_last_message = ""
        if self._seed_mouse_callback not in self.viewer.mouse_drag_callbacks:
            self.viewer.mouse_drag_callbacks.append(self._seed_mouse_callback)

        self._tabs = QTabWidget()
        self._tabs.setObjectName("workflow_tabs")
        self._tabs.setUsesScrollButtons(True)
        self._tabs.addTab(
            self._build_scroll_tab(
                self._build_volume_group(), self._build_seed_group()
            ),
            "1. Tomogram & seeds",
        )
        self._tabs.addTab(
            self._build_scroll_tab(self._build_manual_group()),
            "2. Manual tracing",
        )
        self._tabs.addTab(
            self._build_scroll_tab(
                self._build_tracing_group(), self._build_diagnostics_group()
            ),
            "3. Assisted tracing",
        )
        self._tabs.addTab(self._build_training_tab(), "4. Model training")
        from filament_tracer._particle_widget import ParticleDisplayWidget

        self._particle_display = ParticleDisplayWidget(self)
        self._tabs.addTab(
            self._build_scroll_tab(self._particle_display), "5. Particle STAR display"
        )

        self._skeleton_controls = self._build_skeleton_group()
        self._skeleton_controls.setVisible(False)
        self._tabs.currentChanged.connect(
            lambda index: self._skeleton_controls.setVisible(index in (1, 2))
        )

        layout = QVBoxLayout()
        layout.addWidget(self._tabs)
        layout.addWidget(self._skeleton_controls)
        self.setLayout(layout)

    # ------------------------------------------------------------------
    # UI construction

    @staticmethod
    def _build_scroll_tab(*groups: QWidget) -> QWidget:
        page = QWidget()
        content = QWidget()
        content_layout = QVBoxLayout()

        for group in groups:
            content_layout.addWidget(group)
        content_layout.addStretch()
        content.setLayout(content_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        page_layout = QVBoxLayout()
        page_layout.addWidget(scroll)
        page.setLayout(page_layout)
        return page

    def _build_volume_group(self) -> QGroupBox:
        group = QGroupBox("Tomogram")
        layout = QVBoxLayout()

        buttons = QHBoxLayout()
        open_button = QPushButton("Open MRC/REC")
        open_button.clicked.connect(self._open_mrc_dialog)
        active_button = QPushButton("Use active image")
        active_button.clicked.connect(self._use_active_image)
        buttons.addWidget(open_button)
        buttons.addWidget(active_button)
        layout.addLayout(buttons)

        self._volume_label = QLabel("No 3D image selected")
        self._volume_label.setWordWrap(True)
        layout.addWidget(self._volume_label)

        voxel_form = QFormLayout()
        voxel_row = QHBoxLayout()
        self._voxel_spins: list[QDoubleSpinBox] = []
        for axis in "ZYX":
            spin = self._double_spin(0.001, 100_000.0, 1.0, 4)
            spin.setSuffix(f" Å {axis}")
            self._voxel_spins.append(spin)
            voxel_row.addWidget(spin)
        voxel_form.addRow("Voxel size:", voxel_row)
        layout.addLayout(voxel_form)

        plane_button = QPushButton("Enable 3D plane view")
        plane_button.clicked.connect(self._enable_plane_view)
        layout.addWidget(plane_button)
        group.setLayout(layout)
        return group

    def _build_seed_group(self) -> QGroupBox:
        group = QGroupBox("Parallel planes and bundle seeds")
        layout = QVBoxLayout()

        instruction = QLabel(
            "Orient the napari image plane, capture A, then Ctrl+click each "
            "filament cross-section. Move the plane, capture B, and mark the "
            "same bundle again. Right-drag rotates around the crosshair. "
            "Middle-drag pans without tilting. Ctrl+right-click sets the pivot on the "
            "central plane. Use selected point sets its exact 3D position. "
            "Shift+left-drag translates it along its normal."
        )
        instruction.setWordWrap(True)
        layout.addWidget(instruction)

        capture_row = QHBoxLayout()
        capture_a = QPushButton("Capture plane A")
        capture_a.clicked.connect(lambda: self._capture_plane("A"))
        capture_b = QPushButton("Capture plane B")
        capture_b.clicked.connect(lambda: self._capture_plane("B"))
        capture_row.addWidget(capture_a)
        capture_row.addWidget(capture_b)
        layout.addLayout(capture_row)

        move_row = QHBoxLayout()
        self._plane_move_spin = self._double_spin(0.1, 1000.0, 5.0, 1)
        self._plane_move_spin.setSuffix(" voxels")
        move_back = QPushButton("Move −")
        move_back.clicked.connect(lambda: self._move_plane(-1.0))
        move_forward = QPushButton("Move +")
        move_forward.clicked.connect(lambda: self._move_plane(1.0))
        move_row.addWidget(QLabel("Plane move:"))
        move_row.addWidget(self._plane_move_spin)
        move_row.addWidget(move_back)
        move_row.addWidget(move_forward)
        layout.addLayout(move_row)

        slab_form = QFormLayout()
        slab_row = QHBoxLayout()
        self._slab_thickness = self._double_spin(0.25, 100.0, 2.0, 2)
        self._slab_thickness.setSuffix(" voxels")
        self._slab_thickness.valueChanged.connect(self._apply_slab_settings)
        self._slab_projection = QComboBox()
        self._slab_projection.addItem("Average", "average")
        self._slab_projection.addItem("Maximum", "mip")
        self._slab_projection.addItem("Minimum", "minip")
        self._slab_projection.currentIndexChanged.connect(self._apply_slab_settings)
        slab_row.addWidget(self._slab_thickness)
        slab_row.addWidget(self._slab_projection)
        slab_form.addRow("Slab:", slab_row)

        rotation_row = QHBoxLayout()
        self._rotation_sensitivity = self._double_spin(0.01, 2.0, 0.35, 2)
        self._rotation_sensitivity.setSuffix("°/pixel")
        self._lock_camera_to_slab = QCheckBox("Keep slab face-on")
        self._lock_camera_to_slab.setChecked(True)
        rotation_row.addWidget(self._rotation_sensitivity)
        rotation_row.addWidget(self._lock_camera_to_slab)
        slab_form.addRow("Right-drag rotation:", rotation_row)

        pivot_row = QHBoxLayout()
        selected_pivot = QPushButton("Use selected point as pivot")
        selected_pivot.clicked.connect(self._use_selected_pivot)
        reset_pivot = QPushButton("Reset pivot to slice centre")
        reset_pivot.clicked.connect(self._reset_slab_pivot)
        pivot_row.addWidget(selected_pivot)
        pivot_row.addWidget(reset_pivot)
        slab_form.addRow(pivot_row)

        face_camera = QPushButton("Face slab toward camera")
        face_camera.clicked.connect(self._face_plane_to_camera)
        slab_form.addRow(face_camera)
        layout.addLayout(slab_form)

        target_row = QHBoxLayout()
        self._seed_target = QComboBox()
        self._seed_target.addItem("Off", None)
        self._seed_target.addItem("Mark plane A", "A")
        self._seed_target.addItem("Mark plane B", "B")
        clear_a = QPushButton("Clear A")
        clear_a.clicked.connect(lambda: self._clear_layer(SEED_A_LAYER))
        clear_b = QPushButton("Clear B")
        clear_b.clicked.connect(lambda: self._clear_layer(SEED_B_LAYER))
        target_row.addWidget(QLabel("Ctrl+click target:"))
        target_row.addWidget(self._seed_target)
        target_row.addWidget(clear_a)
        target_row.addWidget(clear_b)
        layout.addLayout(target_row)

        match_form = QFormLayout()
        self._match_tolerance = self._double_spin(1.0, 10_000.0, 100.0, 1)
        self._match_tolerance.setSuffix(" Å")
        match_form.addRow("Match residual limit:", self._match_tolerance)
        layout.addLayout(match_form)

        match_button = QPushButton("Match seeds and initialize skeleton")
        match_button.clicked.connect(self._match_and_initialize)
        layout.addWidget(match_button)

        self._seed_status = QLabel("Plane A: 0 seeds · Plane B: 0 seeds")
        self._seed_status.setWordWrap(True)
        layout.addWidget(self._seed_status)
        group.setLayout(layout)
        return group

    def _build_manual_group(self) -> QGroupBox:
        outer = QGroupBox("Manual tracing")
        outer_layout = QVBoxLayout(outer)
        self._manual_workflow = QComboBox()
        self._manual_workflow.addItems(
            ["Individual filament", "Multi-filament plane matching"]
        )
        outer_layout.addWidget(self._manual_workflow)
        self._individual_panel = QWidget()
        individual_layout = QVBoxLayout(self._individual_panel)
        help_label = QLabel(
            "New filament → left-click its cross-section → move the plane → "
            "click again → Finish filament. Each click is recorded immediately. "
            "Existing filaments are kept; no automatic matching is used. "
            "Use Save skeleton to save all filaments to disk."
        )
        help_label.setWordWrap(True)
        individual_layout.addWidget(help_label)
        row = QHBoxLayout()
        new_button = QPushButton("New filament")
        new_button.clicked.connect(self._new_individual_filament)
        self._individual_finish = QPushButton("Finish filament")
        self._individual_finish.clicked.connect(self._finish_individual_filament)
        self._individual_undo = QPushButton("Undo last point")
        self._individual_undo.clicked.connect(self._undo_individual_point)
        for button in (new_button, self._individual_finish, self._individual_undo):
            row.addWidget(button)
        individual_layout.addLayout(row)
        self._individual_status = QLabel(
            "No active filament. Click New filament to begin."
        )
        self._individual_status.setWordWrap(True)
        individual_layout.addWidget(self._individual_status)
        outer_layout.addWidget(self._individual_panel)
        group = QGroupBox("Full manual multi-plane tracing")
        self._multi_manual_panel = group
        outer_layout.addWidget(group)
        layout = QVBoxLayout()

        instruction = QLabel(
            "Start a manual trace, left-click every visible filament on the "
            "current slab, and commit the plane. Move the slab and repeat. "
            "Points are matched without relying on click order and connected "
            "to the existing skeleton paths."
        )
        instruction.setWordWrap(True)
        layout.addWidget(instruction)

        session_row = QHBoxLayout()
        start_button = QPushButton("Start new manual skeleton")
        start_button.clicked.connect(self._start_manual_trace)
        continue_button = QPushButton("Continue current skeleton")
        continue_button.clicked.connect(self._continue_manual_trace)
        finish_button = QPushButton("Finish manual mode")
        finish_button.clicked.connect(self._finish_manual_trace)
        session_row.addWidget(start_button)
        session_row.addWidget(continue_button)
        session_row.addWidget(finish_button)
        layout.addLayout(session_row)

        self._manual_click_mode = QCheckBox(
            "Manual marking active: left-click adds a point"
        )
        self._manual_click_mode.setEnabled(False)
        layout.addWidget(self._manual_click_mode)

        mark_row = QHBoxLayout()
        undo_mark = QPushButton("Undo last mark")
        undo_mark.clicked.connect(self._undo_manual_mark)
        clear_marks = QPushButton("Clear current marks")
        clear_marks.clicked.connect(self._clear_manual_marks)
        mark_row.addWidget(undo_mark)
        mark_row.addWidget(clear_marks)
        layout.addLayout(mark_row)

        commit_row = QHBoxLayout()
        commit_button = QPushButton("Commit marked plane")
        commit_button.clicked.connect(self._commit_manual_plane)
        commit_move_button = QPushButton("Commit and move +")
        commit_move_button.clicked.connect(
            lambda: self._commit_manual_plane(move_after=True)
        )
        undo_plane = QPushButton("Undo last committed plane")
        undo_plane.clicked.connect(self._undo_manual_plane)
        commit_row.addWidget(commit_button)
        commit_row.addWidget(commit_move_button)
        commit_row.addWidget(undo_plane)
        layout.addLayout(commit_row)

        self._manual_status = QLabel(
            "Manual mode is off. Start new or continue a skeleton."
        )
        self._manual_status.setWordWrap(True)
        layout.addWidget(self._manual_status)
        group.setLayout(layout)
        self._multi_manual_panel.hide()
        self._individual_finish.setEnabled(False)
        self._individual_undo.setEnabled(False)
        self._manual_workflow.currentIndexChanged.connect(self._change_manual_workflow)
        return outer

    def _change_manual_workflow(self, index: int) -> None:
        self._finish_individual_filament()
        self._finish_manual_trace()
        self._individual_panel.setVisible(index == 0)
        self._multi_manual_panel.setVisible(index == 1)

    def _new_individual_filament(self, *_args: Any) -> None:
        if self._require_image() is None:
            return
        self._sync_vertex_edits(render=False)
        self._individual_id = 1 + max(
            (f.filament_id for f in self._project.filaments)
            if self._project is not None
            else (),
            default=0,
        )
        self._individual_history.clear()
        self._seed_target.setCurrentIndex(0)
        self._refresh_individual_status()

    def _finish_individual_filament(self, *_args: Any) -> None:
        self._individual_id = None
        self._individual_history.clear()
        self._refresh_individual_status()

    def _refresh_individual_status(self) -> None:
        active = self._individual_id is not None
        self._individual_finish.setEnabled(active)
        self._individual_undo.setEnabled(active and bool(self._individual_history))
        count = (
            next(
                (
                    len(f.points)
                    for f in self._project.filaments
                    if f.filament_id == self._individual_id
                ),
                0,
            )
            if self._project is not None
            else 0
        )
        self._individual_status.setText(
            f"Active filament {self._individual_id}: {count} points. "
            "Move the plane and click its next cross-section; "
            "Finish filament when done."
            if active
            else "No active filament. Existing paths are kept. "
            "Click New filament to begin."
        )

    def _append_individual_point(self, point: np.ndarray) -> None:
        if self._individual_id is None:
            return
        plane = self._current_plane_definition()
        if plane is None:
            return
        self._sync_vertex_edits(render=False)
        self._individual_history.append(
            self._project.model_copy(deep=True) if self._project is not None else None
        )
        if self._project is None:
            self._project = SkeletonProject(
                volume=self._volume_metadata(self._image_layer),
                plane_a=plane,
                plane_b=plane,
                seed_matches=[],
                tracing_parameters=self._parameters(),
                filaments=[],
            )
        filament = next(
            (
                f
                for f in self._project.filaments
                if f.filament_id == self._individual_id
            ),
            None,
        )
        if filament is None:
            filament = FilamentSkeleton(filament_id=self._individual_id, points=[])
            self._project.filaments.append(filament)
        filament.points.append(
            SkeletonPoint(
                position_zyx=tuple(float(value) for value in point),
                radius_angstrom=self._diameter.value() / 2,
                confidence=1.0,
                provenance="manual",
            )
        )
        self._project.plane_b = plane
        self._render_skeleton()
        self._refresh_individual_status()

    def _undo_individual_point(self, *_args: Any) -> None:
        if self._individual_id is None or not self._individual_history:
            return
        self._project = self._individual_history.pop()
        for name in (SKELETON_PATH_LAYER, SKELETON_POINT_LAYER):
            self._remove_layer(name)
        self._render_skeleton()
        self._refresh_individual_status()

    def _build_tracing_group(self) -> QGroupBox:
        group = QGroupBox("Assisted bundle tracing")
        layout = QVBoxLayout()
        self._detector = QComboBox()
        self._detector.addItem("Template matching", "template")
        self._detector.addItem("Neural network", "neural")
        selector = QFormLayout()
        selector.addRow("Detector:", self._detector)
        layout.addLayout(selector)
        self._template_settings = QGroupBox("Template matching settings")
        form = QFormLayout()

        self._preset = QComboBox()
        self._preset.addItem("F-actin", ("f_actin", "seed_crop", 70.0))
        self._preset.addItem(
            "Intermediate filament",
            ("intermediate", "seed_crop", 100.0),
        )
        self._preset.addItem(
            "Microtubule",
            ("microtubule", "seed_crop", 250.0),
        )
        self._preset.addItem("Custom", ("custom", "seed_crop", None))
        self._preset.currentIndexChanged.connect(self._apply_preset)
        form.addRow("Filament preset:", self._preset)

        self._diameter = self._double_spin(5.0, 2000.0, 70.0, 1)
        self._diameter.setSuffix(" Å")
        form.addRow("Expected diameter:", self._diameter)

        self._template = QComboBox()
        self._template.addItem("Crop from manual seed", "seed_crop")
        self._template.addItem("Ideal solid dot (fallback)", "solid")
        self._template.addItem("Ideal ring (fallback)", "ring")
        form.addRow("Matching template:", self._template)

        self._polarity = QComboBox()
        self._polarity.addItem("Automatic", "auto")
        self._polarity.addItem("Dark density", "dark")
        self._polarity.addItem("Bright density", "bright")
        form.addRow("Density polarity:", self._polarity)

        self._use_slab_averaging = QCheckBox("Average parallel cross-sections")
        form.addRow("Robust slab:", self._use_slab_averaging)

        self._slab_slices_detector = QComboBox()
        for slices in (1, 3, 5, 7, 9, 11):
            self._slab_slices_detector.addItem(str(slices), slices)
        self._slab_slices_detector.setCurrentIndex(1)
        form.addRow("Detector slab slices:", self._slab_slices_detector)

        self._slab_spacing_detector = self._double_spin(
            0.0,
            1000.0,
            0.0,
            1,
        )
        self._slab_spacing_detector.setSpecialValueText("Auto")
        self._slab_spacing_detector.setSuffix(" Å")
        form.addRow("Detector slab spacing:", self._slab_spacing_detector)

        self._orientation_search = QCheckBox("Search nearby perpendicular planes")
        form.addRow("Robust orientation:", self._orientation_search)

        self._orientation_search_degrees = self._double_spin(
            1.0,
            45.0,
            15.0,
            1,
        )
        self._orientation_search_degrees.setSuffix("°")
        form.addRow(
            "Orientation cone half-angle:",
            self._orientation_search_degrees,
        )

        self._orientation_search_steps = QSpinBox()
        self._orientation_search_steps.setRange(1, 3)
        self._orientation_search_steps.setValue(1)
        form.addRow("Orientation cone rings:", self._orientation_search_steps)

        self._circularity_weight = self._double_spin(0.0, 1.0, 0.0, 2)
        self._circularity_weight.setSingleStep(0.05)
        form.addRow("Circularity score weight:", self._circularity_weight)

        self._template_settings.setLayout(form)
        layout.addWidget(self._template_settings)
        self._neural_settings = QGroupBox("Neural detector settings")
        neural_form = QFormLayout()
        self._neural_model = QLineEdit()
        self._neural_model.setReadOnly(True)
        self._neural_model.setPlaceholderText("Select a saved model")
        model_row = QHBoxLayout()
        model_row.addWidget(self._neural_model)
        browse_model = QPushButton("Browse…")
        browse_model.clicked.connect(self._select_neural_model)
        model_row.addWidget(browse_model)
        neural_form.addRow("Model:", model_row)
        self._neural_device = QComboBox()
        self._neural_device.addItems(["Auto", "CPU", "CUDA"])
        neural_form.addRow("Device:", self._neural_device)
        notice = QLabel(
            "Neural inference is not implemented yet. Model selection does "
            "not load or validate the file; tracing is unavailable for this detector."
        )
        notice.setWordWrap(True)
        neural_form.addRow(notice)
        self._neural_settings.setLayout(neural_form)
        self._neural_settings.hide()
        layout.addWidget(self._neural_settings)
        shared = QGroupBox("Shared tracing controls")
        form = QFormLayout()

        self._step_size = self._double_spin(0.25, 100.0, 2.0, 2)
        self._step_size.setSuffix(" voxels")
        form.addRow("Trace step:", self._step_size)

        self._search_radius = self._double_spin(0.5, 100.0, 3.0, 1)
        self._search_radius.setSuffix(" voxels")
        form.addRow("Center search radius:", self._search_radius)

        self._max_steps = QSpinBox()
        self._max_steps.setRange(1, 10_000)
        self._max_steps.setValue(100)
        form.addRow("Maximum steps:", self._max_steps)

        self._confidence = self._double_spin(0.0, 1.0, 0.25, 2)
        self._confidence.setSingleStep(0.05)
        form.addRow("Confidence threshold:", self._confidence)

        self._max_bend = self._double_spin(1.0, 90.0, 35.0, 1)
        self._max_bend.setSuffix("°")
        form.addRow("Maximum bend per step:", self._max_bend)

        self._trace_mode = QComboBox()
        self._trace_mode.addItem("Guided (stop at uncertainty)", "guided")
        self._trace_mode.addItem("Step-by-step", "step_by_step")
        self._trace_mode.addItem("Uninterrupted", "uninterrupted")
        form.addRow("Mode:", self._trace_mode)

        self._direction = QComboBox()
        self._direction.addItem("Both directions", ("backward", "forward"))
        self._direction.addItem("Forward from plane B", ("forward",))
        self._direction.addItem("Backward from plane A", ("backward",))
        form.addRow("Direction:", self._direction)

        self._trace_button = QPushButton("Trace / extend skeleton")
        self._trace_button.clicked.connect(self._start_trace)
        form.addRow(self._trace_button)

        self._trace_status = QLabel("Match seed planes before tracing.")
        self._trace_status.setWordWrap(True)
        form.addRow(self._trace_status)
        shared.setLayout(form)
        layout.addWidget(shared)
        group.setLayout(layout)
        self._detector.currentIndexChanged.connect(self._update_detector_controls)
        return group

    def _select_neural_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select saved model", "", "Model files (*);;All files (*)"
        )
        if path:
            self._neural_model.setText(path)

    def _update_detector_controls(self) -> None:
        neural = self._detector.currentData() == "neural"
        self._template_settings.setVisible(not neural)
        self._neural_settings.setVisible(neural)
        self._trace_button.setEnabled(not neural)
        self._show_selected_diagnostic()

    def _build_diagnostics_group(self) -> QGroupBox:
        group = QGroupBox("Tracing diagnostics")
        layout = QVBoxLayout()

        explanation = QLabel(
            "Steps are grouped by filament, then direction. The selected filament "
            "and predicted step are highlighted and labelled in the viewer. "
            "The patch shows the detector's single slice or averaged slab, "
            "using the settings saved for that step (not the viewing slab). "
            "Template detection shows its template; neural detection shows "
            "a predicted-center overlay. Saved results retain their detector type."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        self._diagnostic_combo = QComboBox()
        self._diagnostic_combo.setObjectName("diagnostic_selector")
        self._diagnostic_combo.currentIndexChanged.connect(
            self._show_selected_diagnostic
        )
        layout.addWidget(self._diagnostic_combo)

        navigation = QHBoxLayout()
        previous = QPushButton("Previous step")
        previous.clicked.connect(lambda: self._advance_diagnostic(-1))
        following = QPushButton("Next step")
        following.clicked.connect(lambda: self._advance_diagnostic(1))
        centre = QPushButton("Centre on step")
        centre.clicked.connect(self._centre_diagnostic)
        navigation.addWidget(previous)
        navigation.addWidget(following)
        navigation.addWidget(centre)
        layout.addLayout(navigation)

        grid = QGridLayout()
        self._diagnostic_patch = self._diagnostic_image_label()
        self._diagnostic_template = self._diagnostic_image_label()
        self._diagnostic_response = self._diagnostic_image_label()
        self._diagnostic_patch_title = QLabel("Perpendicular patch")
        self._diagnostic_patch_title.setWordWrap(True)
        grid.addWidget(self._diagnostic_patch_title, 0, 0)
        self._diagnostic_detail_title = QLabel("Selected template")
        grid.addWidget(self._diagnostic_detail_title, 0, 1)
        grid.addWidget(self._diagnostic_patch, 1, 0)
        grid.addWidget(self._diagnostic_template, 1, 1)
        grid.addWidget(QLabel("Response map"), 2, 0, 1, 2)
        grid.addWidget(self._diagnostic_response, 3, 0, 1, 2)
        layout.addLayout(grid)

        self._diagnostic_summary = QLabel(
            "Run tracing to populate detector diagnostics."
        )
        self._diagnostic_summary.setWordWrap(True)
        self._diagnostic_summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._diagnostic_summary)
        group.setLayout(layout)
        return group

    @staticmethod
    def _diagnostic_image_label() -> QLabel:
        label = QLabel("No data")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumSize(150, 150)
        label.setStyleSheet(
            "QLabel { background: #111; color: #aaa; border: 1px solid #555; }"
        )
        return label

    def _build_skeleton_group(self) -> QGroupBox:
        group = QGroupBox("Editable skeleton project")
        layout = QVBoxLayout()
        instruction = QLabel(
            "Edit points in the “FT skeleton vertices” layer, then synchronize "
            "before continuing or saving."
        )
        instruction.setWordWrap(True)
        layout.addWidget(instruction)

        row = QHBoxLayout()
        sync_button = QPushButton("Sync vertex edits")
        sync_button.clicked.connect(self._sync_vertex_edits)
        save_button = QPushButton("Save skeleton")
        save_button.clicked.connect(self._save_project_dialog)
        load_button = QPushButton("Load skeleton")
        load_button.clicked.connect(self._load_project_dialog)
        row.addWidget(sync_button)
        row.addWidget(save_button)
        row.addWidget(load_button)
        layout.addLayout(row)

        self._project_status = QLabel("No skeleton project")
        smooth_form = QFormLayout()
        self._smoothing_scale = self._double_spin(0.0, 10000.0, 100.0, 1)
        self._smoothing_scale.setSuffix(" Å")
        self._smoothing_scale.setToolTip(
            "Physical smoothing length. Larger values smooth more; "
            "excessive smoothing can straighten real bends. Endpoints stay fixed."
        )
        smooth_form.addRow("Smoothing scale", self._smoothing_scale)
        self._smoothing_id = QSpinBox()
        self._smoothing_id.setRange(0, 1000000)
        self._smoothing_id.setSpecialValueText("All filaments")
        smooth_form.addRow("Filament ID", self._smoothing_id)
        layout.addLayout(smooth_form)
        smooth_row = QHBoxLayout()
        smooth_button = QPushButton("Smooth skeleton")
        smooth_button.clicked.connect(self._smooth_project)
        undo_smooth = QPushButton("Undo smoothing")
        undo_smooth.clicked.connect(self._undo_smoothing)
        smooth_row.addWidget(smooth_button)
        smooth_row.addWidget(undo_smooth)
        layout.addLayout(smooth_row)
        self._smoothing_status = QLabel(
            "Finish tracing before smoothing; endpoints stay fixed."
        )
        self._smoothing_status.setWordWrap(True)
        layout.addWidget(self._smoothing_status)
        self._project_status.setWordWrap(True)
        layout.addWidget(self._project_status)
        group.setLayout(layout)
        return group

    @staticmethod
    def _build_training_tab() -> QWidget:
        views = QTabWidget()
        views.setObjectName("training_views")
        stages = (
            (
                "Data",
                "Training data preparation will use tomograms and annotated "
                "skeletons. Dataset creation and splitting are not implemented yet.",
                ("Prepare dataset",),
            ),
            (
                "Train",
                "Model training and saving are not implemented yet. "
                "After a trained model is saved, Use for tracing will select it "
                "in Assisted tracing.",
                ("Train model", "Save model", "Use for tracing"),
            ),
            (
                "Evaluate",
                "Evaluation on held-out annotations and prediction "
                "previews will be available when the neural backend is implemented.",
                ("Evaluate model",),
            ),
        )
        for title, description, actions in stages:
            page = QWidget()
            layout = QVBoxLayout()
            label = QLabel(description)
            label.setWordWrap(True)
            layout.addWidget(label)
            for action in actions:
                button = QPushButton(action)
                button.setEnabled(False)
                button.setToolTip("Requires the future model training backend.")
                layout.addWidget(button)
            layout.addStretch()
            page.setLayout(layout)
            views.addTab(page, title)
        return views

    @staticmethod
    def _double_spin(
        minimum: float,
        maximum: float,
        value: float,
        decimals: int,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setDecimals(decimals)
        return spin

    # ------------------------------------------------------------------
    # Volume and plane interaction

    def _open_mrc_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open cryo-ET tomogram",
            "",
            "MRC tomograms (*.mrc *.rec *.map);;All files (*)",
        )
        if path:
            self.open_mrc(path)

    def open_mrc(self, path: str | Path) -> Image:
        """Memory-map an MRC-family volume and add it to napari."""

        handle = mrcfile.mmap(str(path), mode="r", permissive=True)
        if handle.data.ndim != 3:
            handle.close()
            raise ValueError("filament tracing requires a 3D tomogram")
        voxel_size = (
            float(handle.voxel_size.z),
            float(handle.voxel_size.y),
            float(handle.voxel_size.x),
        )
        if any(not np.isfinite(value) or value <= 0 for value in voxel_size):
            voxel_size = (1.0, 1.0, 1.0)

        self._mrc_handles.append(handle)
        layer = self.viewer.add_image(
            handle.data,
            name=Path(path).stem,
            scale=voxel_size,
            colormap="gray",
            metadata={
                "filament_tracer_source_path": str(Path(path).resolve()),
                "filament_tracer_voxel_size_zyx": voxel_size,
            },
        )
        self._set_image_layer(layer)
        return layer

    def _use_active_image(self) -> None:
        layer = self.viewer.layers.selection.active
        if not isinstance(layer, Image):
            self._show_error("Select a 3D Image layer in napari first.")
            return
        self._set_image_layer(layer)

    def _set_image_layer(self, layer: Image) -> None:
        if layer.ndim != 3 or np.asanyarray(layer.data).ndim != 3:
            self._show_error("Part 1 currently supports one 3D tomogram at a time.")
            return

        if self._image_layer is not None:
            self._image_layer.plane.events.position.disconnect(self._update_pivot_marker)
        self._image_layer = layer
        layer.plane.events.position.connect(self._update_pivot_marker)

        metadata_voxel = layer.metadata.get("filament_tracer_voxel_size_zyx")
        if metadata_voxel is not None:
            voxel_size = tuple(float(value) for value in metadata_voxel)
        else:
            voxel_size = tuple(abs(float(value)) for value in layer.scale[-3:])
        for spin, value in zip(self._voxel_spins, voxel_size, strict=True):
            spin.setValue(value if value > 0 else 1.0)

        self._volume_label.setText(
            f"{layer.name}: shape {tuple(np.asanyarray(layer.data).shape)}"
        )
        if layer.depiction != "plane":
            layer.plane.position = (
                np.asarray(np.asanyarray(layer.data).shape, dtype=float) - 1.0
            ) / 2.0
            layer.plane.normal = (1.0, 0.0, 0.0)
        self._enable_plane_view()
        self._update_pivot_marker()

        self._particle_display.image_changed()

    def _enable_plane_view(self) -> None:
        layer = self._require_image()
        if layer is None:
            return
        self.viewer.dims.ndisplay = 3
        layer.depiction = "plane"
        if not np.all(np.isfinite(layer.plane.position)):
            layer.plane.position = (
                np.asarray(np.asanyarray(layer.data).shape, dtype=float) - 1.0
            ) / 2.0
        self._apply_slab_settings()
        self.viewer.layers.selection.active = layer

    def _apply_slab_settings(self, *_args: Any) -> None:
        """Apply the dock controls to napari's oblique image slab."""

        layer = self._image_layer
        if layer is None:
            return
        layer.plane.thickness = float(self._slab_thickness.value())
        layer.rendering = str(self._slab_projection.currentData())

    @staticmethod
    def _data_direction_to_world(
        layer: Image,
        direction_data: np.ndarray,
    ) -> np.ndarray:
        """Transform a data-coordinate direction without translating it."""

        origin_data = np.asarray(layer.plane.position, dtype=float)
        origin_world = np.asarray(layer.data_to_world(origin_data), dtype=float)
        endpoint_world = np.asarray(
            layer.data_to_world(origin_data + normalize(direction_data)),
            dtype=float,
        )
        return normalize(endpoint_world - origin_world)

    @staticmethod
    def _world_direction_to_data(
        layer: Image,
        direction_world: np.ndarray,
    ) -> np.ndarray:
        """Transform a world-coordinate direction without translating it."""

        origin_data = np.asarray(layer.plane.position, dtype=float)
        origin_world = np.asarray(layer.data_to_world(origin_data), dtype=float)
        data_origin = np.asarray(layer.world_to_data(origin_world), dtype=float)
        data_endpoint = np.asarray(
            layer.world_to_data(origin_world + normalize(direction_world)),
            dtype=float,
        )
        return normalize(data_endpoint - data_origin)

    @staticmethod
    def _rotate_vector(
        vector: np.ndarray,
        axis: np.ndarray,
        angle_degrees: float,
    ) -> np.ndarray:
        """Rotate a direction around an axis using Rodrigues' formula."""

        direction = normalize(vector)
        rotation_axis = normalize(axis)
        angle = np.deg2rad(float(angle_degrees))
        rotated = (
            direction * np.cos(angle)
            + np.cross(rotation_axis, direction) * np.sin(angle)
            + rotation_axis * np.dot(rotation_axis, direction) * (1.0 - np.cos(angle))
        )
        return normalize(rotated)

    @classmethod
    def _rotate_screen_relative(
        cls,
        normal_world: np.ndarray,
        view_world: np.ndarray,
        up_world: np.ndarray,
        delta_x: float,
        delta_y: float,
        sensitivity: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rotate a slab normal around the camera's screen-space axes."""

        normal = normalize(normal_world)
        view = normalize(view_world)
        up = normalize(up_world)
        right = normalize(np.cross(view, up))

        rotated_normal = cls._rotate_vector(
            normal,
            up,
            -float(delta_x) * sensitivity,
        )
        rotated_normal = cls._rotate_vector(
            rotated_normal,
            right,
            float(delta_y) * sensitivity,
        )
        rotated_up = cls._rotate_vector(
            up,
            right,
            float(delta_y) * sensitivity,
        )
        rotated_up = rotated_up - np.dot(rotated_up, rotated_normal) * rotated_normal
        if np.linalg.norm(rotated_up) <= 1e-12:
            rotated_up = np.cross(rotated_normal, right)
        return normalize(rotated_normal), normalize(rotated_up)

    def _face_plane_to_camera(self, *_args: Any) -> None:
        """Set the slab normal from the current camera view direction."""

        layer = self._require_image()
        if layer is None:
            return
        normal_world = -normalize(
            np.asarray(self.viewer.camera.view_direction, dtype=float)
        )
        layer.plane.normal = self._world_direction_to_data(layer, normal_world)
        self.viewer.layers.selection.active = layer

    def _capture_plane(self, label: str) -> None:
        layer = self._require_image()
        if layer is None:
            return
        plane = PlaneDefinition(
            position_zyx=tuple(float(value) for value in layer.plane.position),
            normal_zyx=tuple(float(value) for value in normalize(layer.plane.normal)),
            thickness=float(layer.plane.thickness),
        )
        if label == "A":
            self._plane_a = plane
            self._seed_target.setCurrentIndex(1)
        else:
            self._plane_b = plane
            self._seed_target.setCurrentIndex(2)
        self._refresh_seed_status()

    def _move_plane(self, sign: float) -> None:
        layer = self._require_image()
        if layer is None:
            return
        distance = sign * self._plane_move_spin.value()
        layer.plane.position = (
            np.asarray(layer.plane.position, dtype=float)
            + normalize(layer.plane.normal) * distance
        )

    def _update_pivot_marker(self, *_args: Any) -> None:
        layer = self._image_layer
        if layer is None:
            return
        point = np.asarray([layer.data_to_world(layer.plane.position)])
        marker = self._layer_by_name(PIVOT_LAYER)
        if isinstance(marker, Points):
            marker.data = point
        else:
            selected = list(self.viewer.layers.selection)
            active = self.viewer.layers.selection.active
            marker = self.viewer.add_points(
                point, name=PIVOT_LAYER, symbol="cross", size=8,
                face_color="magenta", border_color="white",
                blending="translucent_no_depth", out_of_slice_display=True,
                canvas_size_limits=(14, 14),
            )
            marker.editable = False
            self.viewer.layers.selection.clear()
            self.viewer.layers.selection.update(selected)
            self.viewer.layers.selection.active = active

    def _use_selected_pivot(self, *_args: Any) -> None:
        layer = self._image_layer
        points = self.viewer.layers.selection.active
        if layer is None:
            return
        if (
            not isinstance(points, Points)
            or points.name == PIVOT_LAYER
            or len(points.selected_data) != 1
        ):
            self._show_error("Select exactly one tracing or seed point first.")
            return
        index = next(iter(points.selected_data))
        layer.plane.position = layer.world_to_data(
            points.data_to_world(points.data[index])
        )

    def _reset_slab_pivot(self, *_args: Any) -> None:
        layer = self._image_layer
        if layer is None:
            return
        centre = (np.asarray(layer.data.shape, dtype=float) - 1) / 2
        normal = normalize(np.asarray(layer.plane.normal, dtype=float))
        anchor = np.asarray(layer.plane.position, dtype=float)
        # Project the volume centre onto this slice without changing its depth.
        layer.plane.position = centre - normal * np.dot(centre - anchor, normal)

    def _seed_mouse_callback(self, _viewer: Viewer, event: Any) -> Any:
        layer = self._image_layer
        if layer is None or event.type != "mouse_press":
            return

        if event.button == 3 and self.viewer.dims.ndisplay == 3:
            start_mouse = np.asarray(event.pos, dtype=float)
            start_center = np.asarray(self.viewer.camera.center, dtype=float)
            up = normalize(np.asarray(self.viewer.camera.up_direction))
            view = normalize(np.asarray(self.viewer.camera.view_direction))
            right = normalize(np.cross(view, up))
            zoom = float(self.viewer.camera.zoom)
            original_pan = self.viewer.camera.mouse_pan
            self.viewer.camera.mouse_pan = False
            event.handled = True
            try:
                yield
                while event.type == "mouse_move":
                    delta = np.asarray(event.pos, dtype=float) - start_mouse
                    self.viewer.camera.center = tuple(
                        start_center + (up * delta[1] - right * delta[0]) / zoom
                    )
                    event.handled = True
                    yield
            finally:
                self.viewer.camera.mouse_pan = original_pan
            return

        if (
            event.button == 2
            and self._has_control_modifier(event)
            and self.viewer.dims.ndisplay == 3
        ):
            position, direction = layer.click_plane_from_click_data(
                event.position, event.view_direction, list(self.viewer.dims.displayed)
            )
            point = layer.plane.intersect_with_line(position, direction)
            if point is not None and np.all(np.isfinite(point)):
                layer.plane.position = point
            event.handled = True
            return

        if event.button == 2 and self.viewer.dims.ndisplay == 3:
            start_mouse = np.asarray(event.pos, dtype=float)
            pivot_data = np.asarray(layer.plane.position, dtype=float).copy()
            pivot_world = np.asarray(
                layer.data_to_world(pivot_data),
                dtype=float,
            )
            camera_offset = np.asarray(self.viewer.camera.center) - pivot_world
            start_normal_world = self._data_direction_to_world(
                layer,
                np.asarray(layer.plane.normal, dtype=float),
            )
            start_view_world = normalize(
                np.asarray(self.viewer.camera.view_direction, dtype=float)
            )
            start_up_world = normalize(
                np.asarray(self.viewer.camera.up_direction, dtype=float)
            )
            event.handled = True
            yield

            while event.type == "mouse_move":
                mouse = np.asarray(event.pos, dtype=float)
                delta = mouse - start_mouse
                normal_world, up_world = self._rotate_screen_relative(
                    start_normal_world,
                    start_view_world,
                    start_up_world,
                    delta_x=float(delta[0]),
                    delta_y=float(delta[1]),
                    sensitivity=float(self._rotation_sensitivity.value()),
                )
                # The plane anchor is the persistent, visible rotation pivot.
                layer.plane.position = pivot_data
                layer.plane.normal = self._world_direction_to_data(
                    layer,
                    normal_world,
                )
                if self._lock_camera_to_slab.isChecked():
                    offset_length = np.linalg.norm(camera_offset)
                    if offset_length > 1e-12:
                        offset, _ = self._rotate_screen_relative(
                            camera_offset, start_view_world, start_up_world,
                            float(delta[0]), float(delta[1]),
                            float(self._rotation_sensitivity.value()),
                        )
                        self.viewer.camera.center = tuple(
                            pivot_world + offset * offset_length
                        )
                    self.viewer.camera.set_view_direction(
                        view_direction=-normal_world,
                        up_direction=up_world,
                    )
                self.viewer.layers.selection.active = layer
                event.handled = True
                yield
            return

        if (
            event.button == 1
            and self._has_shift_modifier(event)
            and self.viewer.dims.ndisplay == 3
        ):
            start_mouse = np.asarray(event.pos, dtype=float)
            start_position = np.asarray(
                layer.plane.position,
                dtype=float,
            ).copy()
            normal = normalize(np.asarray(layer.plane.normal, dtype=float))
            dragged = False
            original_mouse_pan = layer.mouse_pan
            layer.mouse_pan = False
            event.handled = True
            try:
                yield

                while event.type == "mouse_move":
                    mouse = np.asarray(event.pos, dtype=float)
                    delta = mouse - start_mouse
                    if not dragged and float(np.linalg.norm(delta)) <= 6.0:
                        event.handled = True
                        yield
                        continue
                    dragged = True
                    distance = (
                        float(start_mouse[1] - mouse[1])
                        * self._plane_move_spin.value()
                        / 40.0
                    )
                    layer.plane.position = start_position + normal * distance
                    if self._lock_camera_to_slab.isChecked():
                        self.viewer.camera.center = tuple(
                            float(value)
                            for value in layer.data_to_world(layer.plane.position)
                        )
                    self.viewer.layers.selection.active = layer
                    event.handled = True
                    yield
            finally:
                layer.mouse_pan = original_mouse_pan
            return

        manual_marking = (
            self._manual_session_active
            and self._manual_click_mode.isEnabled()
            and self._manual_click_mode.isChecked()
            and not self._has_any_modifier(event)
        )
        individual_marking = (
            self._manual_workflow.currentIndex() == 0
            and self._individual_id is not None
            and not self._has_any_modifier(event)
        )
        target = self._seed_target.currentData()
        if event.button != 1:
            return
        if (
            not manual_marking
            and not individual_marking
            and (target not in {"A", "B"} or not self._has_control_modifier(event))
        ):
            return
        displayed = list(self.viewer.dims.displayed)
        if len(displayed) != 3:
            return

        line_position, line_direction = layer.click_plane_from_click_data(
            event.position,
            event.view_direction,
            displayed,
        )
        intersection = layer.plane.intersect_with_line(
            line_position,
            line_direction,
        )
        if (
            intersection is None
            or not np.all(np.isfinite(intersection))
            or not np.all(intersection >= 0)
            or not np.all(
                intersection
                <= np.asarray(np.asanyarray(layer.data).shape, dtype=float) - 1.0
            )
        ):
            return

        if individual_marking:
            self._append_individual_point(np.asarray(intersection, dtype=float))
            self.viewer.layers.selection.active = layer
            event.handled = True
            return

        seed_layer = (
            self._manual_seed_layer() if manual_marking else self._seed_layer(target)
        )
        seed_layer.add(np.asarray(intersection, dtype=float))
        seed_layer.refresh()
        self.viewer.layers.selection.active = layer
        event.handled = True
        if manual_marking:
            self._refresh_manual_status()
        else:
            self._refresh_seed_status()

    @staticmethod
    def _has_control_modifier(event: Any) -> bool:
        return any(
            "control" in str(modifier).lower() or str(modifier).lower() == "ctrl"
            for modifier in getattr(event, "modifiers", ())
        )

    @staticmethod
    def _has_shift_modifier(event: Any) -> bool:
        return any(
            "shift" in str(modifier).lower()
            for modifier in getattr(event, "modifiers", ())
        )

    @staticmethod
    def _has_any_modifier(event: Any) -> bool:
        return bool(tuple(getattr(event, "modifiers", ())))

    # ------------------------------------------------------------------
    # Seeds, project initialization, and rendering

    def _seed_layer(self, label: str) -> Points:
        name = SEED_A_LAYER if label == "A" else SEED_B_LAYER
        color = "#00d7ff" if label == "A" else "#ffb000"
        existing = self._layer_by_name(name)
        if isinstance(existing, Points):
            self._style_seed_layer(existing, color)
            return existing
        transform = self._overlay_transform()
        layer = self.viewer.add_points(
            np.empty((0, 3), dtype=float),
            ndim=3,
            name=name,
            face_color=color,
            border_color="white",
            border_width=0.15,
            border_width_is_relative=True,
            size=6.0,
            symbol="disc",
            opacity=1.0,
            blending="translucent_no_depth",
            canvas_size_limits=(8.0, 64.0),
            out_of_slice_display=True,
            **transform,
        )
        self._style_seed_layer(layer, color)
        return layer

    def _manual_seed_layer(self) -> Points:
        existing = self._layer_by_name(MANUAL_SEED_LAYER)
        if isinstance(existing, Points):
            self._style_seed_layer(existing, "#fff04a")
            existing.border_color = "#ff4d6d"
            return existing
        layer = self.viewer.add_points(
            np.empty((0, 3), dtype=float),
            ndim=3,
            name=MANUAL_SEED_LAYER,
            face_color="#fff04a",
            border_color="#ff4d6d",
            border_width=0.18,
            border_width_is_relative=True,
            size=6.0,
            symbol="disc",
            opacity=1.0,
            blending="translucent_no_depth",
            canvas_size_limits=(8.0, 64.0),
            out_of_slice_display=True,
            **self._overlay_transform(),
        )
        self._style_seed_layer(layer, "#fff04a")
        layer.border_color = "#ff4d6d"
        return layer

    @staticmethod
    def _style_seed_layer(layer: Points, color: str) -> None:
        """Keep coplanar seeds visible over the rendered image plane."""

        layer.visible = True
        layer.opacity = 1.0
        layer.blending = "translucent_no_depth"
        layer.out_of_slice_display = True
        layer.canvas_size_limits = (8.0, 64.0)
        layer.size = 6.0
        layer.symbol = "disc"
        layer.face_color = color
        layer.border_color = "white"
        layer.border_width = 0.15
        layer.border_width_is_relative = True

    def _clear_layer(self, name: str) -> None:
        layer = self._layer_by_name(name)
        if isinstance(layer, Points):
            layer.data = np.empty((0, 3), dtype=float)
        self._match_result = None
        self._refresh_seed_status()

    def _start_manual_trace(self, *_args: Any) -> None:
        image = self._require_image()
        if image is None:
            return
        if self._project is not None:
            response = QMessageBox.question(
                self,
                "Start new manual skeleton",
                "Replace the current in-memory skeleton with a new manual "
                "trace? Save it first if it should be kept.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if response != QMessageBox.StandardButton.Yes:
                return

        self._manual_workflow.setCurrentIndex(1)
        self._project = None
        self._plane_a = None
        self._plane_b = None
        self._match_result = None
        self._manual_commits.clear()
        self._manual_last_message = "Ready for the first plane."
        self._manual_session_active = True
        self._manual_click_mode.setEnabled(True)
        self._manual_click_mode.setChecked(True)
        self._seed_target.setCurrentIndex(0)
        self._clear_manual_marks()
        for name in (MATCH_LAYER, SKELETON_PATH_LAYER, SKELETON_POINT_LAYER):
            self._remove_layer(name)
        self._refresh_seed_status()
        self._refresh_manual_status()

    def _continue_manual_trace(self, *_args: Any) -> None:
        if self._project is None or not any(
            filament.points for filament in self._project.filaments
        ):
            self._show_error("Create or load a skeleton before continuing it.")
            return
        self._manual_workflow.setCurrentIndex(1)
        self._sync_vertex_edits(render=False)
        self._manual_commits.clear()
        self._manual_last_message = "Continuing the current skeleton."
        self._manual_session_active = True
        self._manual_click_mode.setEnabled(True)
        self._manual_click_mode.setChecked(True)
        self._seed_target.setCurrentIndex(0)
        self._clear_manual_marks()
        self._refresh_manual_status()

    def _finish_manual_trace(self, *_args: Any) -> None:
        self._manual_session_active = False
        self._manual_click_mode.setChecked(False)
        self._manual_click_mode.setEnabled(False)
        self._manual_last_message = "Manual mode finished."
        self._refresh_manual_status()

    def _clear_manual_marks(self, *_args: Any) -> None:
        layer = self._layer_by_name(MANUAL_SEED_LAYER)
        if isinstance(layer, Points):
            layer.data = np.empty((0, 3), dtype=float)
        self._refresh_manual_status()

    def _undo_manual_mark(self, *_args: Any) -> None:
        layer = self._layer_by_name(MANUAL_SEED_LAYER)
        if not isinstance(layer, Points) or len(layer.data) == 0:
            return
        layer.data = np.asarray(layer.data, dtype=float)[:-1]
        self._refresh_manual_status()

    def _current_plane_definition(self) -> PlaneDefinition | None:
        layer = self._require_image()
        if layer is None:
            return None
        return PlaneDefinition(
            position_zyx=tuple(float(value) for value in layer.plane.position),
            normal_zyx=tuple(float(value) for value in normalize(layer.plane.normal)),
            thickness=float(layer.plane.thickness),
        )

    def _commit_manual_plane(
        self,
        move_after: bool = False,
    ) -> None:
        if not self._manual_session_active:
            self._show_error("Start or continue manual mode first.")
            return
        image = self._require_image()
        marker_layer = self._layer_by_name(MANUAL_SEED_LAYER)
        if image is None:
            return
        if not isinstance(marker_layer, Points) or len(marker_layer.data) == 0:
            self._show_error("Mark at least one filament on the current plane.")
            return

        self._sync_vertex_edits(render=False)
        points = np.asarray(marker_layer.data, dtype=float).copy()
        plane = self._current_plane_definition()
        if plane is None:
            return
        project_before = (
            self._project.model_copy(deep=True) if self._project is not None else None
        )
        radius = self._diameter.value() / 2.0

        if self._project is None:
            filaments = [
                FilamentSkeleton(
                    filament_id=index,
                    points=[
                        SkeletonPoint(
                            position_zyx=tuple(float(value) for value in point),
                            radius_angstrom=radius,
                            confidence=1.0,
                            provenance="manual",
                        )
                    ],
                )
                for index, point in enumerate(points, start=1)
            ]
            self._project = SkeletonProject(
                volume=self._volume_metadata(image),
                plane_a=plane,
                plane_b=plane,
                seed_matches=[],
                tracing_parameters=self._parameters(),
                filaments=filaments,
            )
            self._plane_a = plane
            self._plane_b = plane
            message = f"Started {len(filaments)} manual filament paths."
        else:
            active_filaments = [
                filament for filament in self._project.filaments if filament.points
            ]
            endpoints = np.asarray(
                [filament.points[-1].position_zyx for filament in active_filaments],
                dtype=float,
            ).reshape((-1, 3))
            result = match_seed_points(
                endpoints,
                points,
                np.asarray(self._voxel_size(), dtype=float),
                self._match_tolerance.value(),
            )
            if active_filaments and not result.pairs:
                self._show_error(
                    "No current marks could be connected to the previous "
                    "plane. Increase the match residual limit or correct the "
                    "marks."
                )
                return

            for endpoint_index, point_index, _ in result.pairs:
                active_filaments[endpoint_index].points.append(
                    SkeletonPoint(
                        position_zyx=tuple(
                            float(value) for value in points[point_index]
                        ),
                        radius_angstrom=radius,
                        confidence=1.0,
                        provenance="manual",
                    )
                )

            next_id = (
                max(
                    (filament.filament_id for filament in self._project.filaments),
                    default=0,
                )
                + 1
            )
            for point_index in result.unmatched_b:
                self._project.filaments.append(
                    FilamentSkeleton(
                        filament_id=next_id,
                        points=[
                            SkeletonPoint(
                                position_zyx=tuple(
                                    float(value) for value in points[point_index]
                                ),
                                radius_angstrom=radius,
                                confidence=1.0,
                                provenance="manual",
                            )
                        ],
                    )
                )
                next_id += 1

            self._project.plane_b = plane
            self._project.tracing_parameters = self._parameters()
            self._plane_b = plane
            message = (
                f"Connected {len(result.pairs)} paths"
                f"; {len(result.unmatched_a)} previous paths missing"
                f"; {len(result.unmatched_b)} new paths started."
            )

        self._manual_commits.append((plane, points, project_before))
        marker_layer.data = np.empty((0, 3), dtype=float)
        self._manual_last_message = message
        self._render_skeleton()
        self._refresh_manual_status()
        if move_after:
            self._move_plane(1.0)

    def _undo_manual_plane(self, *_args: Any) -> None:
        if not self._manual_commits:
            self._manual_last_message = "No plane committed in this session."
            self._refresh_manual_status()
            return
        plane, points, project_before = self._manual_commits.pop()
        self._project = project_before
        image = self._require_image()
        if image is not None:
            image.plane.position = plane.position_zyx
            image.plane.normal = plane.normal_zyx
            image.plane.thickness = plane.thickness
        marker_layer = self._manual_seed_layer()
        marker_layer.data = points
        if self._project is None:
            self._plane_a = None
            self._plane_b = None
            for name in (SKELETON_PATH_LAYER, SKELETON_POINT_LAYER):
                self._remove_layer(name)
        else:
            self._plane_a = self._project.plane_a
            self._plane_b = self._project.plane_b
            self._render_skeleton()
        self._manual_last_message = "Last plane restored as editable current marks."
        self._refresh_manual_status()

    def _refresh_manual_status(self) -> None:
        layer = self._layer_by_name(MANUAL_SEED_LAYER)
        marked = len(layer.data) if isinstance(layer, Points) else 0
        filaments = len(self._project.filaments) if self._project else 0
        vertices = (
            sum(len(filament.points) for filament in self._project.filaments)
            if self._project
            else 0
        )
        state = "active" if self._manual_session_active else "off"
        suffix = f" {self._manual_last_message}" if self._manual_last_message else ""
        self._manual_status.setText(
            f"Manual mode {state}: {marked} current marks; "
            f"{len(self._manual_commits)} planes committed this session; "
            f"{filaments} paths / {vertices} vertices.{suffix}"
        )

    def _match_and_initialize(self) -> None:
        if self._plane_a is None or self._plane_b is None:
            self._show_error("Capture both plane A and plane B before matching.")
            return
        layer_a = self._layer_by_name(SEED_A_LAYER)
        layer_b = self._layer_by_name(SEED_B_LAYER)
        if not isinstance(layer_a, Points) or not isinstance(layer_b, Points):
            self._show_error("Mark seeds on both planes before matching.")
            return

        points_a = np.asarray(layer_a.data, dtype=float)
        points_b = np.asarray(layer_b.data, dtype=float)
        result = match_seed_points(
            points_a,
            points_b,
            np.asarray(self._voxel_size(), dtype=float),
            self._match_tolerance.value(),
        )
        if not result.pairs:
            self._show_error(
                "No seed pairs passed the residual limit. Increase the limit "
                "or correct the marked points."
            )
            return

        self._match_result = result
        self._render_matches(points_a, points_b, result)
        parameters = self._parameters()
        pairs = [(a_index, b_index) for a_index, b_index, _ in result.pairs]
        filaments = initialize_skeletons(
            points_a,
            points_b,
            pairs,
            parameters.diameter_angstrom / 2.0,
        )
        image = self._require_image()
        if image is None:
            return
        self._project = SkeletonProject(
            volume=self._volume_metadata(image),
            plane_a=self._plane_a,
            plane_b=self._plane_b,
            seed_matches=[
                SeedMatch(
                    a_index=a_index,
                    b_index=b_index,
                    residual_angstrom=residual,
                )
                for a_index, b_index, residual in result.pairs
            ],
            tracing_parameters=parameters,
            filaments=filaments,
        )
        self._render_skeleton()
        self._refresh_seed_status()
        self._trace_status.setText(f"Initialized {len(filaments)} filament skeletons.")

    def _render_matches(
        self,
        points_a: np.ndarray,
        points_b: np.ndarray,
        result: MatchResult,
    ) -> None:
        vectors = np.asarray(
            [
                [points_a[a_index], points_b[b_index] - points_a[a_index]]
                for a_index, b_index, _ in result.pairs
            ],
            dtype=float,
        )
        existing = self._layer_by_name(MATCH_LAYER)
        if isinstance(existing, Vectors):
            existing.data = vectors
            return
        self.viewer.add_vectors(
            vectors,
            name=MATCH_LAYER,
            edge_color="#7cff6b",
            edge_width=2.0,
            length=1.0,
            **self._overlay_transform(),
        )

    def _render_skeleton(self) -> None:
        if self._project is None:
            return
        paths = [
            np.asarray(
                [point.position_zyx for point in filament.points],
                dtype=float,
            )
            for filament in self._project.filaments
            if len(filament.points) >= 2
        ]
        colors = [
            self._filament_color(filament.filament_id)
            for filament in self._project.filaments
            if len(filament.points) >= 2
        ]

        path_layer = self._layer_by_name(SKELETON_PATH_LAYER)
        if isinstance(path_layer, Shapes):
            path_layer.data = paths
            path_layer.edge_color = colors
        elif paths:
            path_layer = self.viewer.add_shapes(
                paths,
                shape_type=["path"] * len(paths),
                ndim=3,
                name=SKELETON_PATH_LAYER,
                edge_color=colors,
                edge_width=1.2,
                face_color="transparent",
                **self._overlay_transform(),
            )
            path_layer.editable = False

        positions: list[tuple[float, float, float]] = []
        filament_ids: list[int] = []
        sequences: list[int] = []
        radii: list[float] = []
        confidences: list[float] = []
        provenances: list[str] = []
        vertex_colors: list[str] = []
        for filament in self._project.filaments:
            for sequence, point in enumerate(filament.points):
                positions.append(point.position_zyx)
                filament_ids.append(filament.filament_id)
                sequences.append(sequence)
                radii.append(point.radius_angstrom)
                confidences.append(point.confidence)
                provenances.append(point.provenance)
                vertex_colors.append(self._filament_color(filament.filament_id))

        features = {
            "filament_id": np.asarray(filament_ids, dtype=int),
            "sequence": np.asarray(sequences, dtype=int),
            "radius_angstrom": np.asarray(radii, dtype=float),
            "confidence": np.asarray(confidences, dtype=float),
            "provenance": np.asarray(provenances, dtype=str),
        }
        vertex_layer = self._layer_by_name(SKELETON_POINT_LAYER)
        data = np.asarray(positions, dtype=float).reshape((-1, 3))
        if isinstance(vertex_layer, Points):
            vertex_layer.data = data
            vertex_layer.features = features
            vertex_layer.face_color = vertex_colors
        else:
            self.viewer.add_points(
                data,
                ndim=3,
                name=SKELETON_POINT_LAYER,
                features=features,
                face_color=vertex_colors,
                border_color="white",
                size=1.0,
                **self._overlay_transform(),
            )

        total_points = sum(len(filament.points) for filament in self._project.filaments)
        self._project_status.setText(
            f"{len(self._project.filaments)} filaments · "
            f"{total_points} skeleton vertices"
        )

    @staticmethod
    def _filament_color(filament_id: int) -> str:
        palette = (
            "#00d7ff",
            "#ffb000",
            "#7cff6b",
            "#ff6bce",
            "#b28dff",
            "#ff6b5f",
            "#6ba6ff",
            "#e7ff5f",
        )
        return palette[(filament_id - 1) % len(palette)]

    # ------------------------------------------------------------------
    # Tracing and editing

    def _start_trace(self) -> None:
        if self._trace_worker is not None and self._trace_worker.is_running:
            return
        if self._detector.currentData() == "neural":
            self._trace_status.setText("Neural inference is not implemented yet.")
            return
        if self._project is None:
            self._match_and_initialize()
        if self._project is None:
            return
        image = self._require_image()
        if image is None:
            return

        self._sync_vertex_edits(render=False)
        parameters = self._parameters()
        if parameters.mode == "step_by_step":
            parameters.max_steps = 1
        self._project.tracing_parameters = parameters
        filaments = [
            filament.model_copy(deep=True) for filament in self._project.filaments
        ]
        directions = tuple(self._direction.currentData())

        self._trace_button.setEnabled(False)
        self._detector.setEnabled(False)
        self._trace_status.setText("Tracing local cross-sections…")
        self._clear_trace_progress()
        self._accept_trace_progress = True
        worker = _trace_in_worker(
            image.data,
            filaments,
            self._voxel_size(),
            parameters,
            directions,
            self._trace_progress_signals.updated,
        )
        worker.returned.connect(self._trace_finished)
        worker.errored.connect(self._trace_failed)
        worker.finished.connect(lambda: self._trace_button.setEnabled(True))
        worker.finished.connect(lambda: self._detector.setEnabled(True))
        self._trace_worker = worker
        worker.start()

    def _trace_finished(
        self,
        result: tuple[list, list[TraceDiagnostic]],
    ) -> None:
        self._clear_trace_progress()
        if self._project is None:
            return
        filaments, diagnostics = result
        self._project.filaments = filaments
        self._diagnostics = diagnostics
        self._render_skeleton()
        self._refresh_diagnostics()
        stops = []
        for filament in filaments:
            if filament.backward_stop_reason:
                stops.append(
                    f"F{filament.filament_id} back: {filament.backward_stop_reason}"
                )
            if filament.forward_stop_reason:
                stops.append(
                    f"F{filament.filament_id} forward: {filament.forward_stop_reason}"
                )
        if stops:
            self._trace_status.setText(
                "Tracing paused/stopped. " + " · ".join(stops[:5])
            )
        else:
            self._trace_status.setText("Tracing step completed.")

    def _clear_trace_progress(self) -> None:
        self._accept_trace_progress = False
        self._remove_layer(LIVE_TRACE_LAYER)
        self._remove_layer(LIVE_PREDICTION_LAYER)

    def _show_trace_progress(self, update: tuple) -> None:
        if not getattr(self, "_accept_trace_progress", False):
            return
        kind, positions, ids, direction, step = update
        prediction = kind == "prediction"
        name = LIVE_PREDICTION_LAYER if prediction else LIVE_TRACE_LAYER
        data = np.asarray([positions] if prediction else positions).reshape((-1, 3))
        colors = "#ffd43b" if prediction else [self._filament_color(i) for i in ids]
        layer = self._layer_by_name(name)
        if isinstance(layer, Points):
            layer.data = data
            layer.face_color = colors
        else:
            selected = list(self.viewer.layers.selection)
            active = self.viewer.layers.selection.active
            layer = self.viewer.add_points(
                data, name=name, face_color=colors, border_color="white",
                symbol="cross" if prediction else "disc", size=6,
                canvas_size_limits=(14, 14) if prediction else (6, 12),
                blending="translucent_no_depth", out_of_slice_display=True,
                **self._overlay_transform(),
            )
            layer.editable = False
            self.viewer.layers.selection.clear()
            self.viewer.layers.selection.update(selected)
            self.viewer.layers.selection.active = active
        layer.visible = True
        if prediction:
            self._trace_status.setText(
                f"Tracing filament {ids} · {direction} step {step} · "
                "yellow cross = evaluating"
            )
        else:
            marker = self._layer_by_name(LIVE_PREDICTION_LAYER)
            if marker is not None:
                marker.visible = False
            self._trace_status.setText(
                f"Tracing · {direction} step {step} completed · "
                f"{len(data)} accepted/seed points"
            )

    def _refresh_diagnostics(self) -> None:
        self._diagnostic_combo.blockSignals(True)
        self._diagnostic_combo.clear()
        ordered = sorted(
            enumerate(self._diagnostics),
            key=lambda item: (
                item[1].filament_id,
                0 if item[1].direction == "forward" else 1,
                item[1].step_number,
            ),
        )
        for index, record in ordered:
            state = "accepted" if record.accepted else "stopped"
            self._diagnostic_combo.addItem(
                f"Filament {record.filament_id} · {record.direction} "
                f"step {record.step_number} · "
                f"{record.confidence:.3f} · {state}",
                index,
            )
        self._diagnostic_combo.blockSignals(False)
        if self._diagnostics:
            self._diagnostic_combo.setCurrentIndex(0)
            self._show_selected_diagnostic()
        else:
            self._remove_layer(DIAGNOSTIC_PATH_LAYER)
            self._remove_layer(DIAGNOSTIC_POINT_LAYER)
            self._clear_diagnostic_images("No detector evaluation was produced.")

    def _advance_diagnostic(self, offset: int) -> None:
        index = self._diagnostic_combo.currentIndex() + offset
        if 0 <= index < self._diagnostic_combo.count():
            self._diagnostic_combo.setCurrentIndex(index)

    def _centre_diagnostic(self, *_args: Any) -> None:
        index = self._diagnostic_combo.currentData()
        if index is None or self._image_layer is None:
            return
        record = self._diagnostics[int(index)]
        self.viewer.camera.center = self._image_layer.data_to_world(
            record.predicted_position_zyx
        )

    def _highlight_diagnostic(self, record: TraceDiagnostic) -> None:
        """Identify the actual filament without moving the camera or slab."""
        if self._image_layer is None:
            return
        selected = list(self.viewer.layers.selection)
        active = self.viewer.layers.selection.active
        self._remove_layer(DIAGNOSTIC_PATH_LAYER)
        self._remove_layer(DIAGNOSTIC_POINT_LAYER)
        color = self._filament_color(record.filament_id)
        if self._project is not None:
            filament = next(
                (f for f in self._project.filaments
                 if f.filament_id == record.filament_id), None
            )
            if filament is not None and len(filament.points) >= 2:
                path = self.viewer.add_shapes(
                    [np.asarray([p.position_zyx for p in filament.points])],
                    shape_type="path", name=DIAGNOSTIC_PATH_LAYER,
                    edge_color=color, edge_width=4, face_color="transparent",
                    blending="translucent_no_depth", **self._overlay_transform(),
                )
                path.editable = False
        point = self.viewer.add_points(
            [record.predicted_position_zyx], name=DIAGNOSTIC_POINT_LAYER,
            face_color=color, border_color="white", size=8,
            symbol="ring", canvas_size_limits=(16, 16),
            blending="translucent_no_depth", out_of_slice_display=True,
            features={"label": [
                f"Filament {record.filament_id} · {record.direction} "
                f"step {record.step_number} (predicted)"
            ]},
            text={"string": "{label}", "color": "white", "size": 12,
                  "translation": [0, 12, 12]},
            **self._overlay_transform(),
        )
        point.editable = False
        self.viewer.layers.selection.clear()
        self.viewer.layers.selection.update(
            layer for layer in selected if layer in self.viewer.layers
        )
        if active in self.viewer.layers:
            self.viewer.layers.selection.active = active

    def _show_selected_diagnostic(self) -> None:
        index = self._diagnostic_combo.currentData()
        if index is None or not 0 <= int(index) < len(self._diagnostics):
            self._diagnostic_detail_title.setText(
                "Predicted-center overlay"
                if self._detector.currentData() == "neural"
                else "Selected template"
            )
            return
        record = self._diagnostics[int(index)]
        self._highlight_diagnostic(record)
        detector = record.detector
        if detector is None:
            self._clear_diagnostic_images(record.reason)
            self._diagnostic_summary.setText(self._diagnostic_text(record))
            return

        if detector.slab_slices_used > 1:
            self._diagnostic_patch_title.setText(
                f"Perpendicular slab · average of {detector.slab_slices_used} slices"
                f"\nSpacing: {detector.slab_spacing_angstrom:g} Å"
            )
        else:
            self._diagnostic_patch_title.setText("Perpendicular patch · single slice")

        patch_pixmap = self._diagnostic_pixmap(
            detector.patch,
            predicted_rc=detector.predicted_rc,
            detected_rc=detector.detected_rc,
            accepted=record.accepted,
        )
        neural = detector.detector_kind == "neural"
        self._diagnostic_detail_title.setText(
            "Predicted-center overlay" if neural else "Selected template"
        )
        template_pixmap = (
            patch_pixmap if neural else self._diagnostic_pixmap(detector.template)
        )
        scored_response = np.where(
            detector.search_mask,
            detector.response,
            np.nan,
        )
        response_pixmap = self._diagnostic_pixmap(
            scored_response,
            predicted_rc=detector.predicted_rc,
            detected_rc=detector.detected_rc,
            accepted=record.accepted,
        )
        self._diagnostic_patch.setPixmap(patch_pixmap)
        self._diagnostic_template.setPixmap(template_pixmap)
        self._diagnostic_response.setPixmap(response_pixmap)
        self._diagnostic_summary.setText(self._diagnostic_text(record))

    def _clear_diagnostic_images(self, message: str) -> None:
        self._diagnostic_patch_title.setText("Perpendicular patch · no data")
        for label in (
            self._diagnostic_patch,
            self._diagnostic_template,
            self._diagnostic_response,
        ):
            label.clear()
            label.setText("No data")
        self._diagnostic_summary.setText(message)

    @staticmethod
    def _diagnostic_text(record: TraceDiagnostic) -> str:
        predicted = ", ".join(f"{value:.2f}" for value in record.predicted_position_zyx)
        detected = ", ".join(f"{value:.2f}" for value in record.detected_position_zyx)
        state = "accepted" if record.accepted else "stopped"
        robust = ""
        if record.detector is not None:
            detector = record.detector
            robust = (
                f"\nRobust sampling: {detector.slab_slices_used} slice(s)"
                f" · spacing: {detector.slab_spacing_angstrom:g} Å"
                f" · {detector.orientation_candidates} orientation(s)"
                f" · selected offset "
                f"{detector.selected_orientation_offset_degrees:.1f}°"
                f"\nCircularity: {detector.circularity:.3f} · "
                f"combined score: {detector.combined_score:.3f}"
            )
        return (
            f"Filament {record.filament_id} · {record.direction} "
            f"step {record.step_number} · {state}\n"
            f"Predicted ZYX: ({predicted})\n"
            f"Detected ZYX: ({detected})\n"
            f"Confidence: {record.confidence:.3f} · "
            f"radius: {record.radius_angstrom:.1f} Å · "
            f"valid patch: {record.valid_fraction:.1%}\n"
            f"Template: "
            f"{record.detector.template_source if record.detector else 'none'}"
            f"{robust}\n"
            f"Decision: {record.reason}"
        )

    @staticmethod
    def _diagnostic_pixmap(
        array: np.ndarray,
        predicted_rc: tuple[float, float] | None = None,
        detected_rc: tuple[float, float] | None = None,
        accepted: bool = True,
    ) -> QPixmap:
        values = np.asarray(array, dtype=float)
        if values.size == 0:
            return QPixmap()
        finite = np.isfinite(values)
        if not np.any(finite):
            gray = np.zeros(values.shape, dtype=np.uint8)
        else:
            low, high = np.percentile(values[finite], (1.0, 99.0))
            if high <= low:
                high = low + 1.0
            normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
            normalized[~finite] = 0.0
            gray = np.asarray(normalized * 255.0, dtype=np.uint8)

        height, width = gray.shape
        image = QImage(
            gray.data,
            width,
            height,
            int(gray.strides[0]),
            QImage.Format.Format_Grayscale8,
        ).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            220,
            220,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        if predicted_rc is None and detected_rc is None:
            return pixmap

        scale_x = pixmap.width() / width
        scale_y = pixmap.height() / height
        painter = QPainter(pixmap)
        if predicted_rc is not None:
            row, column = predicted_rc
            x_coord = column * scale_x
            y_coord = row * scale_y
            painter.setPen(QPen(QColor("#ffd43b"), 2))
            painter.drawLine(
                int(x_coord - 6),
                int(y_coord),
                int(x_coord + 6),
                int(y_coord),
            )
            painter.drawLine(
                int(x_coord),
                int(y_coord - 6),
                int(x_coord),
                int(y_coord + 6),
            )
        if detected_rc is not None:
            row, column = detected_rc
            x_coord = column * scale_x
            y_coord = row * scale_y
            color = QColor("#00e5ff" if accepted else "#ff4d6d")
            painter.setPen(QPen(color, 2))
            painter.drawEllipse(
                int(x_coord - 6),
                int(y_coord - 6),
                12,
                12,
            )
        painter.end()
        return pixmap

    def _trace_failed(self, error: Exception) -> None:
        self._clear_trace_progress()
        self._trace_status.setText(f"Tracing failed: {error}")
        self._show_error(f"Tracing failed:\n{error}")

    def _sync_vertex_edits(self, render: bool = True) -> None:
        if self._project is None:
            return
        layer = self._layer_by_name(SKELETON_POINT_LAYER)
        if not isinstance(layer, Points):
            return
        features = layer.features
        required = {
            "filament_id",
            "sequence",
            "radius_angstrom",
            "confidence",
            "provenance",
        }
        if not required.issubset(features.columns):
            self._show_error("Skeleton vertex features are incomplete.")
            return

        data = np.asarray(layer.data, dtype=float)
        by_id = {filament.filament_id: filament for filament in self._project.filaments}
        for filament_id, filament in by_id.items():
            indices = np.flatnonzero(
                np.asarray(features["filament_id"], dtype=int) == filament_id
            )
            ordered = sorted(
                indices,
                key=lambda index: int(features.iloc[index]["sequence"]),
            )
            if len(ordered) < 1:
                continue
            points: list[SkeletonPoint] = []
            for index in ordered:
                row = features.iloc[index]
                points.append(
                    SkeletonPoint(
                        position_zyx=tuple(float(value) for value in data[index]),
                        radius_angstrom=float(row["radius_angstrom"]),
                        confidence=float(row["confidence"]),
                        provenance=str(row["provenance"]),
                    )
                )
            filament.points = points
        if render:
            self._render_skeleton()
            self._trace_status.setText("Manual skeleton vertex edits synchronized.")

    # ------------------------------------------------------------------
    # Persistence

    def _smoothing_ready(self) -> bool:
        if self._individual_id is not None or self._manual_session_active:
            self._smoothing_status.setText(
                "Finish filament / Finish manual mode first."
            )
            return False
        if self._trace_worker is not None and self._trace_worker.is_running:
            self._smoothing_status.setText("Wait for assisted tracing to finish.")
            return False
        if self._project is None:
            self._smoothing_status.setText("Create or load a skeleton first.")
            return False
        self._sync_vertex_edits(render=False)
        return True

    def _smooth_project(self, *_args: Any) -> None:
        if not self._smoothing_ready():
            return
        before = self._project.model_copy(deep=True)
        target = self._smoothing_id.value()
        changed = 0
        for filament in self._project.filaments:
            if target and filament.filament_id != target:
                continue
            if len(filament.points) < 3:
                continue
            points = np.array([p.position_zyx for p in filament.points])
            smoothed = smooth_skeleton(
                points,
                self._project.volume.voxel_size_zyx,
                self._smoothing_scale.value(),
            )
            if np.allclose(points, smoothed, rtol=0, atol=1e-10):
                continue
            for point, position in zip(filament.points, smoothed, strict=True):
                point.position_zyx = tuple(float(v) for v in position)
            changed += 1
        if not changed:
            self._smoothing_status.setText(
                "No changes: check ID, scale, and that paths have at least 3 points."
            )
            return
        self._smoothing_undo = (before, self._project.model_dump_json())
        self._manual_commits.clear()
        self._individual_history.clear()
        self._render_skeleton()
        self._smoothing_status.setText(
            f"Smoothed {changed} filaments. Inspect before saving; "
            "Undo smoothing available."
        )

    def _undo_smoothing(self, *_args: Any) -> None:
        if not self._smoothing_ready():
            return
        if self._smoothing_undo is None:
            self._smoothing_status.setText("No smoothing operation to undo.")
            return
        before, after = self._smoothing_undo
        if self._project.model_dump_json() != after:
            self._smoothing_status.setText(
                "Cannot undo: skeleton changed since smoothing. "
                "Later edits are preserved."
            )
            return
        self._project = before
        self._smoothing_undo = None
        self._render_skeleton()
        self._smoothing_status.setText("Restored the skeleton before smoothing.")

    def _save_project_dialog(self) -> None:
        if self._project is None:
            self._show_error("Create or load a skeleton before saving.")
            return
        self._sync_vertex_edits(render=False)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save filament skeleton",
            "trace.ftskeleton.json",
            "Filament skeleton (*.ftskeleton.json);;JSON (*.json)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".ftskeleton.json"
        self._project.save(path)
        self._project_status.setText(f"Saved {path}")

    def _load_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load filament skeleton",
            "",
            "Filament skeleton (*.ftskeleton.json *.json);;All files (*)",
        )
        if path:
            self.load_project(path)

    def load_project(self, path: str | Path) -> SkeletonProject:
        project = SkeletonProject.load(path)
        image = self._image_layer
        if image is not None:
            image_shape = tuple(int(value) for value in np.asanyarray(image.data).shape)
            if image_shape != project.volume.shape_zyx:
                raise ValueError(
                    "skeleton volume shape does not match the active tomogram"
                )
        self._project = project
        self._smoothing_undo = None
        self._finish_individual_filament()
        self._manual_session_active = False
        self._manual_commits.clear()
        self._manual_last_message = "Loaded skeleton; choose Continue to append."
        self._manual_click_mode.setChecked(False)
        self._manual_click_mode.setEnabled(False)
        self._clear_manual_marks()
        self._plane_a = project.plane_a
        self._plane_b = project.plane_b
        for spin, value in zip(
            self._voxel_spins,
            project.volume.voxel_size_zyx,
            strict=True,
        ):
            spin.setValue(value)
        self._apply_parameters(project.tracing_parameters)
        self._render_skeleton()
        self._project_status.setText(f"Loaded {path}")
        self._refresh_manual_status()
        return project

    # ------------------------------------------------------------------
    # Small helpers

    def _parameters(self) -> TracingParameters:
        preset_kind, _, _ = self._preset.currentData()
        return TracingParameters(
            filament_kind=preset_kind,
            template_kind=self._template.currentData(),
            diameter_angstrom=self._diameter.value(),
            step_voxels=self._step_size.value(),
            search_radius_voxels=self._search_radius.value(),
            max_steps=self._max_steps.value(),
            confidence_threshold=self._confidence.value(),
            max_bend_degrees=self._max_bend.value(),
            polarity=self._polarity.currentData(),
            mode=self._trace_mode.currentData(),
            use_slab_averaging=self._use_slab_averaging.isChecked(),
            slab_slices=self._slab_slices_detector.currentData(),
            slab_spacing_angstrom=(
                self._slab_spacing_detector.value()
                if self._slab_spacing_detector.value() > 0.0
                else None
            ),
            orientation_search=self._orientation_search.isChecked(),
            orientation_search_degrees=(self._orientation_search_degrees.value()),
            orientation_search_steps=self._orientation_search_steps.value(),
            circularity_weight=self._circularity_weight.value(),
        )

    def _apply_parameters(self, parameters: TracingParameters) -> None:
        kind_to_index = {
            "f_actin": 0,
            "intermediate": 1,
            "microtubule": 2,
            "custom": 3,
        }
        self._preset.setCurrentIndex(kind_to_index[parameters.filament_kind])
        self._diameter.setValue(parameters.diameter_angstrom)
        self._set_combo_data(self._template, parameters.template_kind)
        self._set_combo_data(self._polarity, parameters.polarity)
        self._step_size.setValue(parameters.step_voxels)
        self._search_radius.setValue(parameters.search_radius_voxels)
        self._max_steps.setValue(parameters.max_steps)
        self._confidence.setValue(parameters.confidence_threshold)
        self._max_bend.setValue(parameters.max_bend_degrees)
        self._set_combo_data(self._trace_mode, parameters.mode)
        self._use_slab_averaging.setChecked(parameters.use_slab_averaging)
        self._set_combo_data(
            self._slab_slices_detector,
            parameters.slab_slices,
        )
        self._slab_spacing_detector.setValue(parameters.slab_spacing_angstrom or 0.0)
        self._orientation_search.setChecked(parameters.orientation_search)
        self._orientation_search_degrees.setValue(parameters.orientation_search_degrees)
        self._orientation_search_steps.setValue(parameters.orientation_search_steps)
        self._circularity_weight.setValue(parameters.circularity_weight)

    def _apply_preset(self) -> None:
        _, template_kind, diameter = self._preset.currentData()
        self._set_combo_data(self._template, template_kind)
        if diameter is not None:
            self._diameter.setValue(diameter)

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: Any) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _voxel_size(self) -> tuple[float, float, float]:
        return tuple(spin.value() for spin in self._voxel_spins)

    def _volume_metadata(self, image: Image) -> VolumeMetadata:
        return VolumeMetadata(
            source_path=image.metadata.get("filament_tracer_source_path"),
            name=image.name,
            shape_zyx=tuple(int(value) for value in np.asanyarray(image.data).shape),
            voxel_size_zyx=self._voxel_size(),
        )

    def _overlay_transform(self) -> dict[str, Any]:
        image = self._image_layer
        if image is None:
            return {}
        return {
            "scale": tuple(float(value) for value in image.scale),
            "translate": tuple(float(value) for value in image.translate),
            "rotate": np.asarray(image.rotate),
            "shear": np.asarray(image.shear),
            "affine": image.affine,
        }

    def _layer_by_name(self, name: str) -> Any | None:
        try:
            return self.viewer.layers[name]
        except KeyError:
            return None

    def _remove_layer(self, name: str) -> None:
        layer = self._layer_by_name(name)
        if layer is not None:
            self.viewer.layers.remove(layer)

    def _require_image(self) -> Image | None:
        if self._image_layer is None:
            self._show_error("Open or select a 3D tomogram first.")
            return None
        return self._image_layer

    def _refresh_seed_status(self) -> None:
        layer_a = self._layer_by_name(SEED_A_LAYER)
        layer_b = self._layer_by_name(SEED_B_LAYER)
        count_a = len(layer_a.data) if isinstance(layer_a, Points) else 0
        count_b = len(layer_b.data) if isinstance(layer_b, Points) else 0
        plane_a = "captured" if self._plane_a else "not captured"
        plane_b = "captured" if self._plane_b else "not captured"
        suffix = ""
        if self._match_result is not None:
            suffix = (
                f" · {len(self._match_result.pairs)} matches"
                f" · {len(self._match_result.unmatched_a)} unmatched A"
                f" · {len(self._match_result.unmatched_b)} unmatched B"
            )
        self._seed_status.setText(
            f"Plane A ({plane_a}): {count_a} seeds · "
            f"Plane B ({plane_b}): {count_b} seeds{suffix}"
        )

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Filament Tracer", message)

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        self._clear_trace_progress()
        if self._image_layer is not None:
            self._image_layer.plane.events.position.disconnect(self._update_pivot_marker)
        try:
            self.viewer.mouse_drag_callbacks.remove(self._seed_mouse_callback)
        except ValueError:
            pass
        for handle in self._mrc_handles:
            handle.close()
        self._mrc_handles.clear()
        super().closeEvent(event)
