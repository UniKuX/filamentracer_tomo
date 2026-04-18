# Filament Trace TOMO

Classical, voxel-value based utilities for tracing filaments in CryoET tomograms.

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
