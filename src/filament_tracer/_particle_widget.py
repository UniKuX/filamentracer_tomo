"""Read-only particle STAR display tab, separate from editable skeletons."""

from pathlib import Path

import numpy as np
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from filament_tracer.relion_particles import particle_positions, read_particles

PARTICLE_LAYER = "FT RELION particles"


class ParticleDisplayWidget(QWidget):
    def __init__(self, host):
        super().__init__()
        self.host = host
        self.table = self.optics = None
        self.source = None
        layout = QVBoxLayout(self)
        help_text = QLabel(
            "Open the matching tomogram in tab 1, then load a refined particle STAR. "
            "Select its exact tomogram name. Coordinates are shown as read-only "
            "markers, separate from the skeleton. Verify voxel sizes in tab 1."
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        load = QPushButton("Load particles STAR…")
        load.clicked.connect(self.open_dialog)
        layout.addWidget(load)
        self.filename = QLabel("No STAR loaded")
        self.filename.setWordWrap(True)
        layout.addWidget(self.filename)
        form = QFormLayout()
        self.tomogram = QComboBox()
        self.tomogram.currentIndexChanged.connect(self.update_classes)
        form.addRow("Tomogram", self.tomogram)
        self.classes = QComboBox()
        form.addRow("Class", self.classes)
        self.shifts = QCheckBox("Apply refinement origin shifts")
        self.shifts.setChecked(True)
        form.addRow(self.shifts)
        center_row = QHBoxLayout()
        self.center = []
        for axis in "ZYX":
            spin = QDoubleSpinBox()
            spin.setRange(-1e7, 1e7)
            spin.setDecimals(4)
            spin.setPrefix(axis + " ")
            self.center.append(spin)
            center_row.addWidget(spin)
        form.addRow("RELION center in volume voxels", center_row)
        self.size = QDoubleSpinBox()
        self.size.setRange(0.1, 1000)
        self.size.setValue(6)
        form.addRow("Marker diameter (voxels)", self.size)
        self.coloring = QComboBox()
        self.coloring.addItems(["Tube ID", "Class", "Single color"])
        form.addRow("Color by", self.coloring)
        layout.addLayout(form)
        warning = QLabel(
            "Center defaults to shape/2 for a full centered reconstruction. "
            "For cropped/shifted volumes enter the original RELION center's "
            "location in this volume. Axis flips/rotations are not inferred. "
            "Press Display / update after changing settings."
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)
        show = QPushButton("Display / update particles")
        show.clicked.connect(self.display)
        layout.addWidget(show)
        clear = QPushButton("Remove particle overlay")
        clear.clicked.connect(lambda: host._remove_layer(PARTICLE_LAYER))
        layout.addWidget(clear)
        self.status = QLabel("No particles displayed")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def image_changed(self):
        self.host._remove_layer(PARTICLE_LAYER)
        image = self.host._image_layer
        for spin, size in zip(self.center, image.data.shape, strict=True):
            spin.setValue(size / 2)
        self.status.setText(
            "Tomogram changed; select its STAR name and Display / update."
        )

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load refined particles", "", "STAR files (*.star)"
        )
        if path:
            try:
                self.load_star(path)
            except (ValueError, OSError, KeyError) as error:
                self.status.setText(f"Cannot load STAR: {error}")

    def load_star(self, path):
        table, optics = read_particles(path)
        self.table, self.optics, self.source = table, optics, str(path)
        self.host._remove_layer(PARTICLE_LAYER)
        self.tomogram.clear()
        self.tomogram.addItems(sorted(table.rlnTomoName.astype(str).unique()))
        self.filename.setText(Path(path).name)
        self.status.setText(
            f"Loaded {len(table)} particles. Select tomogram and display."
        )

    def update_classes(self, *_):
        self.classes.clear()
        self.classes.addItem("All classes", None)
        if self.table is not None and "rlnClassNumber" in self.table:
            rows = self.table[self.table.rlnTomoName == self.tomogram.currentText()]
            for value in sorted(rows.rlnClassNumber.dropna().unique()):
                self.classes.addItem(str(value), value)

    def display(self):
        image = self.host._require_image()
        if image is None:
            return
        if self.table is None:
            self.status.setText("Load a particle STAR first.")
            return
        rows = self.table[self.table.rlnTomoName == self.tomogram.currentText()]
        if self.classes.currentData() is not None:
            rows = rows[rows.rlnClassNumber == self.classes.currentData()]
        try:
            positions = particle_positions(
                rows,
                self.optics,
                self.host._voxel_size(),
                [s.value() for s in self.center],
                self.shifts.isChecked(),
            )
        except (ValueError, KeyError) as error:
            self.host._remove_layer(PARTICLE_LAYER)
            self.status.setText(f"Cannot display particles: {error}")
            return
        if not len(positions):
            self.host._remove_layer(PARTICLE_LAYER)
            self.status.setText("No particles match this selection.")
            return
        key = {"Tube ID": "rlnHelicalTubeID", "Class": "rlnClassNumber"}.get(
            self.coloring.currentText()
        )
        colors = "cyan"
        if key in rows:
            values = rows[key].fillna("unknown").astype(str)
            palette = {
                v: self.host._filament_color(i + 1)
                for i, v in enumerate(sorted(values.unique()))
            }
            colors = [palette[v] for v in values]
        outside = np.any(
            (positions < 0) | (positions > np.array(image.data.shape) - 1), axis=1
        )
        self.host._remove_layer(PARTICLE_LAYER)
        layer = self.host.viewer.add_points(
            positions,
            name=PARTICLE_LAYER,
            ndim=3,
            size=self.size.value(),
            face_color=colors,
            border_color="white",
            border_width=0.08,
            out_of_slice_display=True,
            features=rows.reset_index(drop=True),
            metadata={
                "source_star": self.source,
                "apply_origin_shifts": self.shifts.isChecked(),
            },
            **self.host._overlay_transform(),
        )
        layer.editable = False
        self.host.viewer.layers.selection.active = image
        self.status.setText(
            f"Showing {len(positions)} particles for {self.tomogram.currentText()}; "
            f"{outside.sum()} outside volume. "
            "Use the layer visibility checkbox to hide/show."
        )
