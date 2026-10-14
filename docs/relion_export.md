# Standalone RELION coordinate export

Save the skeleton from napari first. The CLI needs no GUI or tomogram image:

```bash
conda activate filament-tracer
cd /mnt/d/filament_tracer
python scripts/export_relion.py \
  --skeleton manual.ftskeleton.json \
  --output coordinates.star \
  --inter-box-distance 100 \
  --relion-pixel-size 1.67 \
  --tomo-name TS_01
```

Replace **1.67** with your actual unbinned motion-corrected tilt-series pixel
size in Å, and **TS_01** with the exact `rlnTomoName` from your RELION tomogram set.
Do not substitute the binned reconstruction voxel size for this parameter.
For example, 13.36 Å reconstruction voxels and 1.67 Å tilt-series pixels imply
8-fold scaling of skeleton pixel coordinates. Voxel spacing is read from the
skeleton JSON and supports unequal Z/Y/X spacing.

`--inter-box-distance` is physical distance along the polyline in Å (100 Å =
10 nm), not straight-line separation across a bend. Sampling starts at the first
vertex. The final endpoint is included only when it is a spacing multiple; no
short final interval is appended. A nonzero path shorter than the spacing yields
one coordinate. Duplicate consecutive vertices are ignored. Single-point and
zero-length paths, invalid coordinates, and vertices outside the volume stop
export with an error rather than being silently omitted. Existing output files
are never overwritten.

The STAR contains a `data_particles` coordinate table with:

- `rlnTomoName`
- `rlnCoordinateX`, `rlnCoordinateY`, `rlnCoordinateZ` in unbinned pixels
- `rlnHelicalTubeID`, preserving the skeleton filament IDs
- `rlnHelicalTrackLengthAngst`, distance from the start of each filament

Conversion is `XYZ = (ZYX * skeleton_voxel_size)[reversed axes] / relion_pixel_size`.
No centering, axis flips, origin translations, or one-pixel shifts are guessed.
This assumes the picked reconstruction and RELION tomogram share the same
physical origin and axis orientation. Cropped, shifted, or reoriented volumes
need an explicit registration before using this exporter. Visually verify the
imported coordinates in RELION before extraction.

This is a **coordinate-import file**, not a complete extracted particle set:
it deliberately does not invent optics, CTF metadata, Euler angles, polarity,
or helical priors. Import it with RELION's coordinate importer and the matching
tomogram set, for example:

```bash
relion_tomo_import_particles --i coordinates.star \
  --t tomograms.star --o Import/filament_coordinates/
```

RELION's [coordinate importer](https://relion.readthedocs.io/en/latest/Reference/STA/Programs/import_particles.html)
documents this single-STAR input layout. The imported RELION 5 particle set uses
[centered angstrom coordinates](https://relion.readthedocs.io/en/release-5.0/Reference/STA/Datatypes/particle_set.html),
which are different from this import file's pixel coordinates. Do not pass this
coordinate-only file directly as a finished particle set to refinement.

Export uses the saved vertices as-is; it does not smooth automatically. Smooth
and save in napari first if desired. Use `--help` for the CLI argument summary.
