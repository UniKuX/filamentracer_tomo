from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mrcfile
import numpy as np
import pytest
from napari.utils.interactions import mouse_press_callbacks
from qtpy.QtWidgets import QTabWidget

from filament_tracer import __version__
from filament_tracer.detection import DetectionDiagnostic
from filament_tracer.models import PlaneDefinition
from filament_tracer.tracing import TraceDiagnostic


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_live_trace_updates_before_worker_finishes(
    make_napari_viewer, monkeypatch, qtbot,
) -> None:
    from threading import Event

    from filament_tracer import _widget
    from filament_tracer.models import TracingParameters
    from filament_tracer.tracing import initialize_skeletons

    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((20, 32, 32)), scale=(2, 3, 4))
    widget = _widget.FilamentTracerWidget(viewer)
    widget._set_image_layer(image)
    filaments = initialize_skeletons(
        np.array([[5, 12, 12]]), np.array([[8, 12, 12]]), [(0, 0)], 10,
    )
    release_prediction = Event()
    release_result = Event()
    monkeypatch.setattr(_widget, "build_seed_templates", lambda *args: {})

    def extend(*args, on_prediction, on_step, **kwargs):
        on_prediction(1, 1, (10, 12, 12))
        assert release_prediction.wait(10)
        on_step(filaments, 1)
        assert release_result.wait(10)

    monkeypatch.setattr(_widget, "extend_skeletons", extend)
    widget._accept_trace_progress = True
    worker = _widget._trace_in_worker(
        image.data, filaments, (2, 3, 4), TracingParameters(), ("forward",),
        widget._trace_progress_signals.updated,
    )
    worker.returned.connect(widget._trace_finished)
    errors = []
    worker.errored.connect(errors.append)
    centre = viewer.camera.center
    plane = np.asarray(image.plane.position).copy()
    worker.start()
    try:
        qtbot.waitUntil(lambda: _widget.LIVE_PREDICTION_LAYER in viewer.layers)
        marker = viewer.layers[_widget.LIVE_PREDICTION_LAYER]
        np.testing.assert_allclose(marker.data_to_world(marker.data[0]), (20, 36, 48))
        assert worker.is_running
        release_prediction.set()
        qtbot.waitUntil(lambda: _widget.LIVE_TRACE_LAYER in viewer.layers)
        assert len(viewer.layers[_widget.LIVE_TRACE_LAYER].data) == 2
        assert not marker.visible
        assert viewer.layers.selection.active is image
        np.testing.assert_allclose(viewer.camera.center, centre)
        np.testing.assert_allclose(image.plane.position, plane)
    finally:
        release_prediction.set()
        release_result.set()
        qtbot.waitUntil(lambda: not worker.is_running)
    qtbot.waitUntil(lambda: _widget.LIVE_TRACE_LAYER not in viewer.layers)
    assert _widget.LIVE_PREDICTION_LAYER not in viewer.layers
    assert not errors


def test_workflow_widget(make_napari_viewer) -> None:
    from filament_tracer._widget import FilamentTracerWidget

    viewer = make_napari_viewer()
    widget = FilamentTracerWidget(viewer)

    tabs = widget.findChild(QTabWidget, "workflow_tabs")
    assert tabs.count() == 5
    assert [tabs.tabText(index) for index in range(5)] == [
        "1. Tomogram & seeds",
        "2. Manual tracing",
        "3. Assisted tracing",
        "4. Model training",
        "5. Particle STAR display",
    ]
    assert tabs.widget(0).isAncestorOf(widget._volume_label)
    assert tabs.widget(0).isAncestorOf(widget._seed_target)
    assert tabs.widget(1).isAncestorOf(widget._manual_click_mode)
    assert tabs.widget(2).isAncestorOf(widget._trace_button)
    assert tabs.widget(2).isAncestorOf(widget._diagnostic_combo)
    for index in range(5):
        tabs.setCurrentIndex(index)
        assert widget._skeleton_controls.isHidden() == (index not in (1, 2))
    tabs.setCurrentIndex(0)

    training = widget.findChild(QTabWidget, "training_views")
    assert [training.tabText(i) for i in range(3)] == [
        "Data",
        "Train",
        "Evaluate",
    ]
    widget._detector.setCurrentIndex(1)
    assert widget._template_settings.isHidden()
    assert not widget._neural_settings.isHidden()
    assert not widget._trace_button.isEnabled()
    widget._start_trace()
    assert widget._trace_worker is None
    widget._detector.setCurrentIndex(0)
    assert not widget._template_settings.isHidden()
    assert widget._neural_settings.isHidden()
    assert widget._trace_button.isEnabled()

    detector = DetectionDiagnostic(
        patch=np.arange(81, dtype=np.float32).reshape(9, 9),
        template=np.ones((5, 5), dtype=np.float32),
        response=np.eye(9, dtype=np.float32),
        search_mask=np.ones((9, 9), dtype=bool),
        sample_spacing_angstrom=5.0,
        predicted_rc=(4.0, 4.0),
        detected_rc=(5.0, 3.0),
        template_source="seed crop",
    )
    widget._diagnostics = [
        TraceDiagnostic(
            filament_id=1,
            direction="forward",
            step_number=1,
            predicted_position_zyx=(1.0, 2.0, 3.0),
            detected_position_zyx=(1.0, 2.5, 3.0),
            confidence=0.42,
            radius_angstrom=35.0,
            valid_fraction=1.0,
            accepted=True,
            reason="accepted",
            detector=detector,
        )
    ]
    widget._refresh_diagnostics()

    assert widget._diagnostic_combo.count() == 1
    assert widget._diagnostic_patch.pixmap() is not None
    assert not widget._diagnostic_patch.pixmap().isNull()
    assert "Confidence: 0.420" in widget._diagnostic_summary.text()
    widget._detector.setCurrentIndex(1)
    assert widget._diagnostic_detail_title.text() == "Selected template"
    widget._diagnostics[0].detector = replace(detector, detector_kind="neural")
    widget._show_selected_diagnostic()
    assert widget._diagnostic_detail_title.text() == "Predicted-center overlay"


def test_widget_initializes_skeleton_from_seed_layers(make_napari_viewer) -> None:
    from filament_tracer._widget import FilamentTracerWidget

    viewer = make_napari_viewer()
    image = viewer.add_image(
        np.zeros((20, 32, 32), dtype=np.float32),
        scale=(10.0, 10.0, 10.0),
    )
    widget = FilamentTracerWidget(viewer)
    widget._set_image_layer(image)
    widget._plane_a = PlaneDefinition(
        position_zyx=(5.0, 16.0, 16.0),
        normal_zyx=(1.0, 0.0, 0.0),
    )
    widget._plane_b = PlaneDefinition(
        position_zyx=(8.0, 16.0, 16.0),
        normal_zyx=(1.0, 0.0, 0.0),
    )
    widget._seed_layer("A").data = np.array([[5.0, 12.0, 12.0], [5.0, 20.0, 20.0]])
    widget._seed_layer("B").data = np.array([[8.0, 12.0, 12.0], [8.0, 20.0, 20.0]])

    widget._match_and_initialize()

    assert widget._project is not None
    assert len(widget._project.filaments) == 2
    assert all(len(filament.points) == 2 for filament in widget._project.filaments)
    assert "2 filaments" in widget._project_status.text()

    vertex_layer = viewer.layers["FT skeleton vertices"]
    edited = np.asarray(vertex_layer.data).copy()
    edited[0, 1] += 1.5
    vertex_layer.data = edited
    widget._sync_vertex_edits()
    assert widget._project.filaments[0].points[0].position_zyx[1] == 13.5


def test_diagnostics_group_steps_and_identify_filament(make_napari_viewer) -> None:
    from filament_tracer._widget import (
        DIAGNOSTIC_PATH_LAYER,
        DIAGNOSTIC_POINT_LAYER,
        FilamentTracerWidget,
    )
    from filament_tracer.tracing import initialize_skeletons

    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((20, 32, 32)), scale=(2, 3, 4))
    widget = FilamentTracerWidget(viewer)
    widget._set_image_layer(image)
    filaments = initialize_skeletons(
        np.array([[5, 12, 12], [5, 20, 20]]),
        np.array([[8, 12, 12], [8, 20, 20]]), [(0, 0), (1, 1)], 10,
    )
    widget._project = SimpleNamespace(filaments=filaments)
    detector = DetectionDiagnostic(
        patch=np.ones((9, 9), dtype=np.float32),
        template=np.ones((3, 3), dtype=np.float32),
        response=np.ones((9, 9), dtype=np.float32),
        search_mask=np.ones((9, 9), dtype=bool),
        sample_spacing_angstrom=2, predicted_rc=(4, 4), detected_rc=(4, 4),
        template_source="seed crop", slab_slices_used=5, slab_spacing_angstrom=3,
    )
    record = TraceDiagnostic(
        filament_id=1, direction="forward", step_number=1,
        predicted_position_zyx=(9, 12, 12), detected_position_zyx=(9, 12, 12),
        confidence=0.8, radius_angstrom=10, valid_fraction=1,
        accepted=True, reason="accepted", detector=detector,
    )
    widget._diagnostics = [
        record, replace(record, filament_id=2),
        replace(record, step_number=2),
        replace(record, filament_id=2, step_number=2),
        replace(record, direction="backward"),
    ]
    centre = viewer.camera.center
    plane = np.asarray(image.plane.position).copy()
    widget._refresh_diagnostics()
    assert [widget._diagnostic_combo.itemData(i) for i in range(5)] == [0, 2, 4, 1, 3]
    assert "average of 5 slices" in widget._diagnostic_patch_title.text()
    assert "Spacing: 3 Å" in widget._diagnostic_patch_title.text()
    # Changing live settings must not relabel saved detector results.
    widget._use_slab_averaging.setChecked(False)
    widget._show_selected_diagnostic()
    assert "average of 5 slices" in widget._diagnostic_patch_title.text()
    np.testing.assert_allclose(
        viewer.layers[DIAGNOSTIC_PATH_LAYER].data[0],
        [p.position_zyx for p in filaments[0].points],
    )
    marker = viewer.layers[DIAGNOSTIC_POINT_LAYER]
    assert "Filament 1" in marker.features["label"][0]
    np.testing.assert_allclose(
        marker.data_to_world(marker.data[0]), image.data_to_world((9, 12, 12))
    )
    assert viewer.layers.selection.active is image
    np.testing.assert_allclose(viewer.camera.center, centre)
    widget._centre_diagnostic()
    np.testing.assert_allclose(viewer.camera.center, image.data_to_world((9, 12, 12)))
    np.testing.assert_allclose(image.plane.position, plane)
    widget._advance_diagnostic(1)
    assert widget._diagnostic_combo.currentData() == 2
    widget._diagnostic_combo.setCurrentIndex(3)
    assert "Filament 2" in viewer.layers[DIAGNOSTIC_POINT_LAYER].features["label"][0]
    widget._diagnostics[1].detector = replace(detector, slab_slices_used=1)
    widget._show_selected_diagnostic()
    assert "single slice" in widget._diagnostic_patch_title.text()
    widget._diagnostics = []
    widget._refresh_diagnostics()
    assert DIAGNOSTIC_PATH_LAYER not in viewer.layers
    assert DIAGNOSTIC_POINT_LAYER not in viewer.layers


def test_widget_memory_maps_mrc(
    make_napari_viewer,
    tmp_path: Path,
) -> None:
    from filament_tracer._widget import FilamentTracerWidget

    path = tmp_path / "small.mrc"
    with mrcfile.new(path) as handle:
        handle.set_data(np.zeros((8, 16, 12), dtype=np.float32))
        handle.voxel_size = 12.5

    viewer = make_napari_viewer()
    widget = FilamentTracerWidget(viewer)
    layer = widget.open_mrc(path)

    assert layer.data.shape == (8, 16, 12)
    assert isinstance(layer.data, np.memmap)
    assert widget._voxel_size() == (12.5, 12.5, 12.5)
    assert layer.depiction == "plane"


def test_ctrl_click_adds_visible_seeds_from_viewer_callback(
    make_napari_viewer,
) -> None:
    from filament_tracer._widget import FilamentTracerWidget

    viewer = make_napari_viewer()
    image = viewer.add_image(
        np.zeros((20, 32, 32), dtype=np.float32),
        scale=(10.0, 10.0, 10.0),
    )
    widget = FilamentTracerWidget(viewer)
    widget._set_image_layer(image)
    widget._seed_target.setCurrentIndex(1)
    image.plane.position = (5.0, 16.0, 16.0)
    image.plane.normal = (1.0, 0.0, 0.0)
    event = SimpleNamespace(
        type="mouse_press",
        button=1,
        modifiers=("Control",),
        position=image.data_to_world((5.0, 14.0, 15.0)),
        view_direction=np.array((1.0, 0.0, 0.0)),
        handled=False,
    )

    mouse_press_callbacks(viewer, event)

    seeds = viewer.layers["FT seed plane A"]
    assert len(seeds.data) == 1
    np.testing.assert_allclose(seeds.data[0], (5.0, 14.0, 15.0))
    assert seeds.visible
    assert np.all(np.asarray(seeds.size) >= 6.0)
    assert seeds.blending == "translucent_no_depth"
    assert seeds.canvas_size_limits == (8.0, 64.0)
    np.testing.assert_allclose(
        seeds.data_to_world(seeds.data[0]),
        image.data_to_world(seeds.data[0]),
    )
    assert viewer.layers.selection.active is image
    assert event.handled


def test_right_drag_rotates_oblique_slab(make_napari_viewer) -> None:
    from filament_tracer._widget import FilamentTracerWidget

    viewer = make_napari_viewer()
    image = viewer.add_image(
        np.zeros((20, 32, 32), dtype=np.float32),
        scale=(2.0, 3.0, 4.0),
    )
    widget = FilamentTracerWidget(viewer)
    widget._set_image_layer(image)
    image.plane.position = (6.0, 14.0, 15.0)
    viewer.camera.set_view_direction(
        view_direction=(-1.0, 0.0, 0.0),
        up_direction=(0.0, -1.0, 0.0),
    )
    start_position = np.asarray(image.plane.position, dtype=float).copy()
    start_normal = np.asarray(image.plane.normal, dtype=float).copy()
    start_center = np.asarray(viewer.camera.center).copy()
    event = SimpleNamespace(
        type="mouse_press",
        button=2,
        pos=(100.0, 100.0),
        handled=False,
    )

    drag = widget._seed_mouse_callback(viewer, event)
    next(drag)
    assert event.handled
    np.testing.assert_allclose(
        viewer.camera.center,
        start_center,
    )

    event.type = "mouse_move"
    event.pos = (150.0, 120.0)
    next(drag)

    rotated_normal = np.asarray(image.plane.normal, dtype=float)
    assert np.all(np.isfinite(rotated_normal))
    assert not np.allclose(rotated_normal, start_normal)
    np.testing.assert_allclose(image.plane.position, start_position)
    rotated_normal_world = widget._data_direction_to_world(
        image,
        rotated_normal,
    )
    np.testing.assert_allclose(
        viewer.camera.view_direction,
        -rotated_normal_world,
        atol=1e-6,
    )

    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(drag)

    next_position = np.array((9.0, 12.0, 10.0))
    image.plane.position = next_position
    next_center = np.asarray(viewer.camera.center).copy()
    next_event = SimpleNamespace(
        type="mouse_press",
        button=2,
        pos=(80.0, 75.0),
        handled=False,
    )
    next_drag = widget._seed_mouse_callback(viewer, next_event)
    next(next_drag)
    np.testing.assert_allclose(
        viewer.camera.center,
        next_center,
    )
    next_event.type = "mouse_move"
    next_event.pos = (95.0, 90.0)
    next(next_drag)
    np.testing.assert_allclose(image.plane.position, next_position)
    next_event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(next_drag)


def test_middle_drag_pans_without_changing_slab(make_napari_viewer) -> None:
    from filament_tracer._widget import FilamentTracerWidget

    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((20, 32, 32)), scale=(2, 3, 4))
    widget = FilamentTracerWidget(viewer)
    widget._set_image_layer(image)
    viewer.camera.set_view_direction(
        view_direction=(-1, 0, 0), up_direction=(0, -1, 0)
    )
    viewer.camera.zoom = 2
    centre = np.asarray(viewer.camera.center).copy()
    angles = viewer.camera.angles
    position = np.asarray(image.plane.position).copy()
    normal = np.asarray(image.plane.normal).copy()
    event = SimpleNamespace(
        type="mouse_press", button=3, pos=(100, 100), handled=False
    )
    drag = widget._seed_mouse_callback(viewer, event)
    next(drag)
    event.type = "mouse_move"
    event.pos = (120, 110)
    next(drag)
    np.testing.assert_allclose(viewer.camera.center, centre + (0, -5, -10))
    np.testing.assert_allclose(viewer.camera.angles, angles)
    np.testing.assert_allclose(image.plane.position, position)
    np.testing.assert_allclose(image.plane.normal, normal)
    assert event.handled
    drag.close()
    assert viewer.camera.mouse_pan
    # Starting another rotation must not undo the pan.
    centre = np.asarray(viewer.camera.center).copy()
    event.type = "mouse_press"
    event.button = 2
    drag = widget._seed_mouse_callback(viewer, event)
    next(drag)
    np.testing.assert_allclose(viewer.camera.center, centre)
    drag.close()


def test_selected_pivot_and_reset(make_napari_viewer) -> None:
    from filament_tracer._widget import PIVOT_LAYER, FilamentTracerWidget

    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((20, 32, 32)), scale=(2, 3, 4))
    widget = FilamentTracerWidget(viewer)
    widget._set_image_layer(image)
    points = viewer.add_points([[6, 12, 20]], scale=(2, 3, 4))
    points.selected_data = {0}
    widget._use_selected_pivot()
    np.testing.assert_allclose(image.plane.position, (6, 12, 20))
    np.testing.assert_allclose(viewer.layers[PIVOT_LAYER].data, [[12, 36, 80]])
    widget._reset_slab_pivot()
    np.testing.assert_allclose(image.plane.position, (6, 15.5, 15.5))

    event = SimpleNamespace(
        type="mouse_press", button=2, modifiers=("Control",),
        position=image.data_to_world((6, 10, 11)),
        view_direction=(1, 0, 0), handled=False,
    )
    mouse_press_callbacks(viewer, event)
    np.testing.assert_allclose(image.plane.position, (6, 10, 11))
    np.testing.assert_allclose(image.plane.normal, (1, 0, 0))
    assert event.handled


def test_slab_controls_apply_projection_and_thickness(
    make_napari_viewer,
) -> None:
    from filament_tracer._widget import FilamentTracerWidget

    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((12, 16, 20), dtype=np.float32))
    widget = FilamentTracerWidget(viewer)
    widget._set_image_layer(image)

    widget._slab_thickness.setValue(7.5)
    widget._slab_projection.setCurrentIndex(1)

    assert image.plane.thickness == pytest.approx(7.5)
    assert image.rendering == "mip"


def test_shift_left_translates_plane_without_marking_manual_point(
    make_napari_viewer,
) -> None:
    from filament_tracer._widget import (
        MANUAL_SEED_LAYER,
        FilamentTracerWidget,
    )

    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((24, 32, 32), dtype=np.float32))
    widget = FilamentTracerWidget(viewer)
    widget._set_image_layer(image)
    widget._start_manual_trace()
    image.plane.position = (5.0, 16.0, 16.0)
    image.plane.normal = (1.0, 0.0, 0.0)

    plain_event = SimpleNamespace(
        type="mouse_press",
        button=1,
        modifiers=(),
        position=image.data_to_world((5.0, 14.0, 15.0)),
        view_direction=np.array((1.0, 0.0, 0.0)),
        handled=False,
    )
    mouse_press_callbacks(viewer, plain_event)
    assert len(viewer.layers[MANUAL_SEED_LAYER].data) == 1

    shift_drag_event = SimpleNamespace(
        type="mouse_press",
        button=1,
        modifiers=("Shift",),
        pos=(100.0, 100.0),
        handled=False,
    )
    drag = widget._seed_mouse_callback(viewer, shift_drag_event)
    next(drag)
    assert shift_drag_event.handled
    assert not image.mouse_pan
    shift_drag_event.type = "mouse_move"
    shift_drag_event.pos = (103.0, 103.0)
    next(drag)
    np.testing.assert_allclose(image.plane.position, (5.0, 16.0, 16.0))
    shift_drag_event.pos = (100.0, 60.0)
    next(drag)
    np.testing.assert_allclose(image.plane.position, (10.0, 16.0, 16.0))
    assert len(viewer.layers[MANUAL_SEED_LAYER].data) == 1
    shift_drag_event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(drag)
    assert image.mouse_pan

    shift_click_event = SimpleNamespace(
        type="mouse_press",
        button=1,
        modifiers=("Shift",),
        pos=(80.0, 80.0),
        handled=False,
    )
    click_drag = widget._seed_mouse_callback(viewer, shift_click_event)
    next(click_drag)
    assert shift_click_event.handled
    assert not image.mouse_pan
    shift_click_event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(click_drag)
    assert image.mouse_pan
    np.testing.assert_allclose(image.plane.position, (10.0, 16.0, 16.0))
    assert len(viewer.layers[MANUAL_SEED_LAYER].data) == 1

    control_event = SimpleNamespace(
        type="mouse_press",
        button=1,
        modifiers=("Control",),
        position=image.data_to_world((10.0, 13.0, 14.0)),
        view_direction=np.array((1.0, 0.0, 0.0)),
        handled=False,
    )
    mouse_press_callbacks(viewer, control_event)
    assert len(viewer.layers[MANUAL_SEED_LAYER].data) == 1


def test_individual_filaments_click_finish_new_and_undo(make_napari_viewer, tmp_path):
    from filament_tracer._widget import FilamentTracerWidget

    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((24, 32, 32), dtype=np.float32))
    widget = FilamentTracerWidget(viewer)
    widget._set_image_layer(image)
    image.plane.normal = (1.0, 0.0, 0.0)

    def click(z):
        image.plane.position = (z, 16.0, 16.0)
        event = SimpleNamespace(
            type="mouse_press",
            button=1,
            modifiers=(),
            position=image.data_to_world((z, 12.0, 12.0)),
            view_direction=np.array((1.0, 0.0, 0.0)),
            handled=False,
        )
        mouse_press_callbacks(viewer, event)
        return event.handled

    assert widget._manual_workflow.currentIndex() == 0
    widget._new_individual_filament()
    assert click(4)
    widget._undo_individual_point()
    assert widget._project is None
    assert click(4)
    assert click(8)
    widget._finish_individual_filament()
    assert not click(10)
    assert len(widget._project.filaments[0].points) == 2

    widget._new_individual_filament()
    assert click(8)  # Same location must create a separate ID, not reconnect.
    assert click(12)
    assert [f.filament_id for f in widget._project.filaments] == [1, 2]
    widget._undo_individual_point()
    assert [len(f.points) for f in widget._project.filaments] == [2, 1]
    widget._finish_individual_filament()
    widget._undo_individual_point()
    assert [len(f.points) for f in widget._project.filaments] == [2, 1]
    widget._new_individual_filament()
    widget._manual_workflow.setCurrentIndex(1)
    assert widget._individual_id is None
    assert not click(16)
    saved = tmp_path / "individual.ftskeleton.json"
    widget._project.save(saved)
    widget._manual_workflow.setCurrentIndex(0)
    widget.load_project(saved)
    assert widget._individual_id is None
    widget._new_individual_filament()
    assert click(16)
    assert [f.filament_id for f in widget._project.filaments] == [1, 2, 3]
    assert [len(f.points) for f in widget._project.filaments] == [2, 1, 1]


def test_smoothing_selection_undo_and_edit_protection(make_napari_viewer):
    from filament_tracer._widget import SKELETON_POINT_LAYER, FilamentTracerWidget

    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((24, 32, 32), dtype=np.float32))
    widget = FilamentTracerWidget(viewer)
    widget._set_image_layer(image)
    for y in (10, 20):
        widget._new_individual_filament()
        for p in ((2, y, 10), (8, y + 3, 10), (16, y, 10)):
            widget._append_individual_point(np.array(p))
        widget._finish_individual_filament()
    original = widget._project.model_dump_json()
    widget._smoothing_id.setValue(1)
    widget._smoothing_scale.setValue(10)
    widget._smooth_project()
    assert widget._project.filaments[0].points[1].position_zyx[1] < 13
    assert widget._project.filaments[1].points[1].position_zyx[1] == 23
    widget._undo_smoothing()
    assert widget._project.model_dump_json() == original
    widget._smoothing_id.setValue(0)
    widget._smooth_project()
    assert widget._project.filaments[1].points[1].position_zyx[1] < 23
    layer = viewer.layers[SKELETON_POINT_LAYER]
    data = layer.data.copy()
    data[1, 1] += 1
    layer.data = data
    widget._undo_smoothing()
    assert "Cannot undo" in widget._smoothing_status.text()
    assert widget._project.filaments[0].points[1].position_zyx[1] == data[1, 1]
    widget._new_individual_filament()
    before = widget._project.model_dump_json()
    widget._smooth_project()
    assert widget._project.model_dump_json() == before
    assert "Finish" in widget._smoothing_status.text()


def test_manual_mode_connects_unordered_marks_across_planes(
    make_napari_viewer,
) -> None:
    from filament_tracer._widget import (
        MANUAL_SEED_LAYER,
        SKELETON_PATH_LAYER,
        FilamentTracerWidget,
    )

    viewer = make_napari_viewer()
    image = viewer.add_image(
        np.zeros((20, 32, 32), dtype=np.float32),
        scale=(10.0, 10.0, 10.0),
    )
    widget = FilamentTracerWidget(viewer)
    widget._set_image_layer(image)
    widget._start_manual_trace()

    def click(point: tuple[float, float, float]) -> None:
        event = SimpleNamespace(
            type="mouse_press",
            button=1,
            modifiers=(),
            position=image.data_to_world(point),
            view_direction=np.array((1.0, 0.0, 0.0)),
            handled=False,
        )
        mouse_press_callbacks(viewer, event)
        assert event.handled

    image.plane.position = (4.0, 16.0, 16.0)
    image.plane.normal = (1.0, 0.0, 0.0)
    click((4.0, 12.0, 12.0))
    click((4.0, 20.0, 20.0))
    widget._commit_manual_plane()

    assert widget._project is not None
    assert len(widget._project.filaments) == 2
    assert all(len(filament.points) == 1 for filament in widget._project.filaments)
    assert all(
        filament.points[0].provenance == "manual"
        for filament in widget._project.filaments
    )
    assert len(viewer.layers[MANUAL_SEED_LAYER].data) == 0

    image.plane.position = (8.0, 16.0, 16.0)
    click((8.0, 20.0, 20.0))
    click((8.0, 12.0, 12.0))
    widget._commit_manual_plane()

    assert all(len(filament.points) == 2 for filament in widget._project.filaments)
    np.testing.assert_allclose(
        [point.position_zyx for point in widget._project.filaments[0].points],
        [(4.0, 12.0, 12.0), (8.0, 12.0, 12.0)],
    )
    np.testing.assert_allclose(
        [point.position_zyx for point in widget._project.filaments[1].points],
        [(4.0, 20.0, 20.0), (8.0, 20.0, 20.0)],
    )
    assert len(viewer.layers[SKELETON_PATH_LAYER].data) == 2

    image.plane.position = (12.0, 16.0, 16.0)
    click((12.0, 12.0, 12.0))
    click((12.0, 20.0, 20.0))
    widget._commit_manual_plane()
    assert all(len(filament.points) == 3 for filament in widget._project.filaments)

    widget._undo_manual_plane()
    assert all(len(filament.points) == 2 for filament in widget._project.filaments)
    assert len(viewer.layers[MANUAL_SEED_LAYER].data) == 2
    np.testing.assert_allclose(image.plane.position, (12.0, 16.0, 16.0))
