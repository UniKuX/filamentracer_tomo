# Prepare single-microtubule training patches

This standalone tool uses manually picked `.ftskeleton.json` skeletons. It does
not train a model or change the plugin GUI. PyTorch is not required.

In WSL, activate the existing environment and run from the repository:

```bash
conda activate filament-tracer
cd /mnt/d/filament_tracer
python scripts/prepare_training_data.py prepare \
  --tomogram /path/to/tomogram.mrc \
  --skeleton /path/to/manual.ftskeleton.json \
  --output /path/to/patch_review \
  --patch-size 97 --pixel-spacing 5 --diameter 250 \
  --step 100 --tangent-window 200 --variants 3 --seed 0

QT_QPA_PLATFORM=xcb LIBGL_ALWAYS_SOFTWARE=1 \
  python scripts/prepare_training_data.py review /path/to/patch_review

python scripts/prepare_training_data.py export /path/to/patch_review \
  --output /path/to/training.npz
```

All physical distances are in **angstroms**. Voxel spacing comes from the
skeleton project; ensure it matches the tomogram. Coordinates use array ZYX.
This version uses individual slices, not slabs. Patch size is odd (default 97)
to reuse the existing perpendicular-plane extractor exactly. The local tangent
is estimated from positions half a `tangent-window` before/after each sample
along the physical skeleton. In-plane axes are transported between samples.

Each sample has three independently shifted crops by default. Offsets are
uniform in both in-plane directions, bounded by `--offset-fraction` (default
0.35 of patch half-width) and the space needed to keep a complete microtubule
inside the crop. Labels shift with the crop. `--seed` makes this reproducible.
The `--diameter` setting is the expected outer microtubule diameter and controls
conservative boundary/ambiguity exclusions. Increase field of view when needed.

Supply all available microtubule skeletons in the project, not just the target
one. Every suitable intersection with the plane is labeled. Known near-parallel
or strongly oblique tubes (>30 degrees), partially visible tubes, nearby tube
segments without reliable intersections, and out-of-volume patches are excluded.
This is intentionally conservative for unbundled microtubules. Automatic vertices
are refused unless `--allow-corrected-automatic` explicitly confirms they were
manually reviewed/corrected.

**Unpicked microtubules cannot be ruled out from skeletons alone.** Preparation
therefore creates pending patches, not approved training examples. In the review
window, add/move/delete centers using the Points tools. Label every additional
complete cross-section, or reject the patch if any neighbor is uncertain,
oblique, or partial. Acceptance explicitly confirms that all visible tubes have
labels. Skip leaves a patch pending; Previous lets you revise a decision.
Decisions save immediately, and rerunning review resumes at the first pending
patch. Unaccepted edits are not saved. No empty negative examples are exported.

The review directory contains raw interpolated float32 images (`patch_*.npz`)
and `manifest.json` with editable reviewed labels, decision status, source paths,
physical plane origin/axes, voxel/pixel spacing, arclength, random offsets, and
exclusion reasons. The original labels in each patch NPZ are proposals; the
manifest contains authoritative reviewed labels. Export includes only accepted
patches and preserves the full geometry for those records.

The exported archive has:

- `images`: `(N, H, W)` raw densities; choose/fix normalization when training.
- `centers_rc`: all reviewed `(row, column)` centers, zero-based float coordinates.
- `center_offsets`: length `N+1`; patch `i` uses centers in
  `[center_offsets[i]:center_offsets[i+1]]`.
- `metadata`: JSON provenance and reviewed record list, loadable with
  `np.load(..., allow_pickle=False)`.

Crop center in physical ZYX is `origin_physical_zyx`. A label maps to 3D as
`origin + spacing * ((row-half)*basis_v + (column-half)*basis_u)`, where
`half=(patch_size-1)/2`. Divide componentwise by voxel spacing for data coordinates.

Use separate source tomograms for training and validation. Offset variants and
nearby samples of one skeleton must stay in the same split to avoid leakage.
This first version holds prepared images in memory; use moderate sampling density
for long skeletons. It never overwrites an existing output directory or archive.
