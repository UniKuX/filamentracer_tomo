# Filament Tracer

A napari plugin for cryo-ET filament workflows, separated into:

1. filament tracing and editable skeleton generation;
2. skeleton resampling and RELION 5 STAR export.

## WSL development environment

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

1. Launch the plugin with `scripts/launch_wsl.sh`.
2. Use **Open MRC/REC** to memory-map a tomogram, or select an existing 3D
   Image layer and click **Use active image**.
3. In the 3D view, right-drag to rotate the image slab until the bundled
   filaments appear as dots or rings. Adjust **Slab** thickness and
   Average/Maximum/Minimum projection if needed, then click
   **Capture plane A**. **Keep slab face-on** makes the camera follow the
   rotating slab; **Face slab toward camera** resets the slab to the current
   view direction. Every new right-drag uses the slab's current center as a
   fresh rotation pivot, so moving the slab before the next drag also moves
   the pivot.
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
   direction. Ideal dot/ring templates are explicit first-step fallbacks only.
8. Use **Tracing diagnostics** to inspect each attempted step. The yellow cross
   is the predicted center and the cyan/red circle is the accepted/rejected
   correlation peak. Compare the perpendicular patch, selected template, and
   score map before changing detector parameters. The template source reports
   `seed crop` for the first step and `adaptive step N` thereafter.
9. Correct vertices in the `FT skeleton vertices` layer, click
   **Sync vertex edits**, and save the result as `.ftskeleton.json`.

### Full manual tracing

Use **Full manual multi-plane tracing** when every skeleton point should be
placed manually:

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
saved skeleton is the formal input to the later RELION export module.
