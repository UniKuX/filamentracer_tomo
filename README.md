# Filament Trace TOMO

Classical, voxel-value based utilities for tracing filaments in CryoET tomograms.

## Setup On A New Machine

These steps assume you have `git` and Conda or Miniconda already installed.

### 1. Clone the repository

```bash
git clone https://github.com/UniKuX/filamentracer_tomo.git
cd filamentracer_tomo
```

### 2. Create the Conda environment

The repository includes an `environment.yml` file with the Python packages used
by the project and the local package installed in editable mode.

```bash
conda env create -f environment.yml
```

### 3. Activate the environment

```bash
conda activate filament-trace-tomo
```

### 4. Verify the install

Run the test suite:

```bash
python -m pytest
```

You can also verify that the package imports correctly:

```bash
python -c "import filament_trace_tomo; print('import ok')"
```

## Running Local Test Scripts

The `filter_test/` folder contains small scripts for experimenting on local
tomograms in VS Code or from the terminal.

Examples:

### Generate a cylindrical ROI mask

Edit the config block in:

```text
filter_test/generate_cylindrical_roi_mask.py
```

Then run:

```bash
python filter_test/generate_cylindrical_roi_mask.py
```

### Apply ROI masking and preprocessing

Edit the config block in:

```text
filter_test/apply_cylindrical_roi_and_filter.py
```

Then run:

```bash
python filter_test/apply_cylindrical_roi_and_filter.py
```

### Compare ridge filters visually

```bash
python filter_test/plot_ridge_filter.py
```

## Notes

- The repository `.gitignore` excludes large tomogram and mask outputs such as
  `.mrc` files, so you will need to copy your own tomograms onto the new
  machine.
- Run commands from the repository root so that relative paths and the editable
  install behave as expected.

The first implemented piece is a RELION 5 filament particle STAR writer built on
the `starfile` package. The intended pipeline is:

1. Trace an ordered centerline from voxel data inside a user-provided ROI mask.
2. Resample that centerline by the requested inter-box distance.
3. Export the sampled coordinates as a RELION 5-compatible `data_particles`
   STAR file.

## Install

```bash
python3 -m pip install -e ".[test]"
```

## RELION 5 STAR Export

```python
from filament_trace_tomo.relion5 import Relion5FilamentParticle, write_relion5_filament_star

particles = [
    Relion5FilamentParticle(
        tomo_name="Position_51_2",
        helical_tube_id=1,
        helical_track_length_angst=0.0,
        centered_coordinate_x_angst=-3974.394070,
        centered_coordinate_y_angst=4168.292866,
        centered_coordinate_z_angst=96.189681,
        tomo_subtomogram_rot=90.0,
        tomo_subtomogram_tilt=123.880889,
        tomo_subtomogram_psi=142.172287,
        angle_rot=0.0,
        angle_tilt=90.0,
        angle_psi=0.0,
        angle_tilt_prior=90.0,
        angle_psi_prior=0.0,
    )
]

write_relion5_filament_star("particles.star", particles)
```
