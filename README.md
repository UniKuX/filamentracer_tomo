# Filament Tracer

A napari plugin for cryo-ET filament workflows, separated into:

1. filament tracing and editable skeleton generation;
2. skeleton resampling and RELION 5 STAR export.

## WSL development environment

For standalone, reviewed perpendicular training crops from manually picked
microtubules, see [training data preparation](docs/training_data.md).

The supported development environment is Ubuntu 24.04 under WSL2. Miniconda is
installed in `/home/weilong/miniconda3`, and the project environment is named
`filament-tracer`.

From PowerShell:

```powershell
wsl.exe -d Ubuntu -- bash /mnt/d/filament_tracer/scripts/bootstrap_wsl.sh
wsl.exe -d Ubuntu -- bash /mnt/d/filament_tracer/scripts/test_wsl.sh
wsl.exe -d Ubuntu -- bash /mnt/d/filament_tracer/scripts/launch_wsl.sh
```

From Ubuntu:

```bash
bash /mnt/d/filament_tracer/scripts/bootstrap_wsl.sh
conda activate filament-tracer
cd /mnt/d/filament_tracer
pytest
napari
```

Use `scripts/launch_wsl.sh` for interactive work. It enables Mesa software
rendering because the current WSLg Zink path displays image planes but silently
drops napari's 3D Points overlays, including filament seeds.

## Part 1 workflow

The plugin has five tabs:

- **1. Tomogram & seeds**: tomogram display, slab settings, parallel planes,
  and bundle seed selection/matching.
- **2. Manual tracing**: full manual multi-plane tracing.
- **3. Assisted tracing**: detector selection (Template matching / Neural
  network), detector-specific settings, shared tracing controls, and diagnostics.
  Template matching is available; neural model/device selection is a UI preview
  and neural inference is not implemented yet.
- **4. Model training**: Data, Train, and Evaluate views for the future training
  workflow. Training, model saving, evaluation, and Use for tracing are disabled
  until the neural backend is implemented.
- **5. Particle STAR display**: load refined RELION particles, select a tomogram
  and class, and overlay positions with optional refinement origin shifts.
  See [particle display](docs/particle_display.md).

RELION export is a standalone CLI; see [RELION export](docs/relion_export.md).
For Warp/Miss-Alignment reconstructions, use the standalone
[WarpTools coordinate export](docs/warp_export.md) to extract in Warp and refine
the resulting particle series in RELION 5.

Skeleton load/save and vertex synchronization controls appear beneath the
manual and assisted tracing tabs and operate on the same skeleton project.

1. Launch the plugin with `scripts/launch_wsl.sh`.
2. Use **Open MRC/REC** to memory-map a tomogram, or select an existing 3D
   Image layer and click **Use active image**.
3. In the 3D view, right-drag to rotate the image slab until the bundled
   filaments appear as dots or rings. Adjust **Slab** thickness and
   Average/Maximum/Minimum projection if needed, then click
   **Capture plane A**. **Keep slab face-on** makes the camera follow the
   rotating slab; **Face slab toward camera** resets the slab to the current
   view direction. The magenta crosshair marks the rotation pivot. Ctrl+right-click
   places it on the slab's central plane; to use an exact 3D seed or tracing
   vertex, select that point and click **Use selected point as pivot** (this
   moves the slab to pass through that point). **Reset pivot to slice centre**
   restores the central pivot without changing the slice depth or angle.
   Hold the mouse wheel (middle button) and drag to pan without changing
   the slab angle or depth. Right-drag preserves that pan instead of snapping
   back to the centre. Moving the slab along its normal also moves the pivot.
   Shift+left-drag translates continuously along the slab normal; dragging
   upward moves forward and dragging downward moves backward. A stationary
   Shift+left-click, including small mouse jitter, is suppressed because
   napari's default image-plane callback otherwise repositions the slab on
   mouse-down.
4. Select **Mark plane A** and Ctrl+click each cross-section.
5. Move the image plane with napari or the plugin's **Move −/+** buttons, click
   **Capture plane B**, and Ctrl+click the same bundle again.
6. Click **Match seeds and initialize skeleton**. Correct a bad match by editing
   either seed layer and matching again.
7. Select a filament preset and tracing policy, then click
   **Trace / extend skeleton**.
   The default matching template is cropped from each filament's real manual
   seed: plane B supplies the forward template and plane A supplies the
   backward template. After every accepted step, the template is replaced by
   the newly detected, centered 2D cross-section for that filament and
   direction. Its in-plane coordinate frame is parallel-transported between
   steps so small tangent changes cannot rotate the patch by 90 degrees.
   Ideal dot/ring templates are explicit first-step fallbacks only.
   For noisy or strongly bending filaments, enable **Robust slab** to average
   3–5 parallel cross-sections and/or **Robust orientation** to test a small
   cone of nearby perpendicular planes. `Auto` slab spacing uses the smallest
   voxel size. Orientation search is ranked primarily by template score;
   a circularity weight around `0.05–0.10` can help when oblique sections look
   oval. These options cost more per tracing step and are off by default.
   While tracing runs, a yellow cross on the tomogram marks the prediction
   currently being evaluated. Coloured dots show seed and accepted points as
   the trace grows, after bundle collision checks. Updates leave the camera
   and slab unchanged; the completed skeleton replaces these temporary marks.
8. Use **Tracing diagnostics** to inspect each attempted step. The yellow cross
   is the predicted center and the cyan/red circle is the accepted/rejected
   correlation peak. Compare the perpendicular patch, selected template, and
   score map before changing detector parameters. The template source reports
   `seed crop` for the first step and `adaptive step N` thereafter. Diagnostics
   are ordered by filament, then direction and step; **Previous step** and
   **Next step** follow this order. Selecting an entry highlights that filament
   and labels its predicted step in the viewer. **Centre on step** pans to it
   without changing the slab. The patch heading distinguishes a single slice
   from an averaged slab and gives the slice count and spacing recorded during
   detection, independently of the current viewing-slab settings. Diagnostics
   also report the number of tested orientations, selected angular offset,
   circularity, and combined selection score.
9. Correct vertices in the `FT skeleton vertices` layer, click
   **Sync vertex edits**, and save the result as `.ftskeleton.json`.

### Full manual tracing

The shared **Editable skeleton project** controls offer **Smooth skeleton**.
Finish manual/assisted tracing first, select a filament ID (0 means all), and
set the smoothing scale in ångströms (default 100 Å = 10 nm). Larger scales smooth
more strongly. The operation uses physical arclength, including anisotropic
voxel spacing, and keeps endpoints, vertex counts, IDs and point metadata intact.
Paths with fewer than three points are unchanged. Inspect the result: excessive
smoothing can flatten genuine bends. **Undo smoothing** restores the most recent
smoothing operation only if no subsequent skeleton edits have occurred. Save the
skeleton to use its smoothed coordinates in training-data preparation. Applying
smoothing repeatedly is cumulative; undo before comparing different scales.

Tab **2. Manual tracing** defaults to **Individual filament**:

1. Click **New filament** to start a separate path without replacing existing ones.
2. Left-click its cross-section. Each click immediately records a skeleton point.
3. Move the slab (Shift+left-drag) or rotate it, and click the next cross-section.
4. Use **Undo last point** to remove a mistaken click on the active filament.
5. Click **Finish filament** to stop adding points. Click **New filament** for the
   next microtubule; it receives a separate ID even if it is near an existing path.
6. Use **Save skeleton** to save all paths to disk. Finishing does not save a file.

No plane commit or endpoint matching is used in this mode. Switching modes ends
the active individual filament; it keeps its recorded points. A loaded skeleton
has no active filament, and **New filament** appends a new path to it.

Select **Multi-filament plane matching** for the original workflow below. This
mode matches against existing endpoints, including previously drawn paths, so
stay in Individual filament mode when paths must remain explicitly separate:

1. Click **Start new manual skeleton**. Manual marking becomes active and plain
   unmodified left-clicks add yellow points on the current slab.
2. Mark every visible filament cross-section and click **Commit marked plane**.
   The first plane starts one path per point.
3. Move the slab with **Move +** and mark the next cross-sections. Click
   **Commit marked plane** again, or use **Commit and move +**.
4. Each new set is matched to the previous path endpoints in physical
   coordinates, so the points do not need to be clicked in the same order.
   Connected paths and vertices update after every commit.
5. Use **Undo last mark** before committing, or **Undo last committed plane** to
   return that plane's points to the yellow marker layer for correction.
6. Click **Finish manual mode**, then save the skeleton normally. To append
   manual planes to a loaded or automatically traced project, click
   **Continue current skeleton**.

The match residual limit in the seed section also controls manual plane
matching. Increase it when the slab spacing or filament curvature produces a
larger inter-plane displacement residual.

The tracing output is a collection of ordered, non-branching polylines. The
saved skeleton is the input to `scripts/export_relion.py` for RELION coordinate
STAR export at a user-specified inter-box distance.
