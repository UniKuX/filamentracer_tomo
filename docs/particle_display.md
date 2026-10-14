# Display refined particles in the tomogram

1. Open the corresponding reconstruction in tab 1. Verify its **voxel sizes**
   (the binned reconstruction spacing, not the tilt-series pixel size).
2. Open **5. Particle STAR display** and click **Load particles STAR…**.
   For the supplied example choose `run_data.star`.
3. Select the matching `rlnTomoName`, e.g. `Position_1` for
   `rec_Position_1_half1.mrc`. Choose all classes or a single class.
4. Click **Display / update particles**. They appear in the separate, read-only
   `FT RELION particles` Points layer. Adjust marker diameter and color by tube
   ID, class, or a single color, then click Display / update again.
5. Toggle **Apply refinement origin shifts** and update to compare positions
   before and after refinement shifts. Hide/show via the layer visibility icon.

The STAR is never modified, and these markers are not editable skeleton points.
All selected particles are displayed in 3D, not just those near the current
slab. The status reports particles outside the volume; they are not silently
clipped or rescaled. Loading another image clears the old overlay to avoid
showing particles over an unrelated tomogram.

## Coordinate handling

RELION 5 `rlnCenteredCoordinateX/Y/ZAngst` are converted to displayed-volume
voxels with the voxel sizes in tab 1. The **RELION center in volume voxels**
defaults to `shape / 2` in ZYX order, assuming a full centered reconstruction.
If the volume was cropped or shifted, enter the location of the original
RELION center in the displayed array instead. The STAR alone cannot establish
that registration. MRC origin shifts and arbitrary axis flips/rotations are
not automatically inferred; verify alignment against recognizable features.

Refinement shifts use:

`XYZ_refined_angstrom = XYZ_centered_angstrom - A_subtomogram @ origin_shift`

where `A_subtomogram` is constructed from `rlnTomoSubtomogramRot/Tilt/Psi`,
or identity when absent. This follows RELION's
[ParticleSet::getPosition](https://github.com/3dem/relion/blob/master/src/jaz/tomography/particle_set.cpp)
and [Euler::anglesToMatrix3](https://github.com/3dem/relion/blob/master/src/jaz/math/Euler_angles_relion.h).
The particle's `rlnAngleRot/Tilt/Psi` are not used to rotate the origin shifts.
Missing origin triplets mean zero shift; partially missing triplets or invalid
numeric values cause an error rather than a guessed position.

Legacy `rlnCoordinateX/Y/Z` are also supported if their unbinned tilt-series
pixel sizes are present in the particle table or matching optics groups.
Legacy origins expressed in pixels are not applied; convert them to angstroms
or disable origin shifts explicitly. The center control only affects centered
coordinates. Refinement orientation glyphs and per-frame motion trajectories
are not displayed in this version.

For `run_data.star`, select the actual matching reconstruction and tomogram name;
do not overlay all five tomograms into a single volume. Its 1.67 Å optics pixel
size is not the reconstruction voxel size (e.g. 13.36 Å for an 8x-binned map).
