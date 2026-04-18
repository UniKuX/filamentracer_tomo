import numpy as np
from filament_trace_tomo.inputs import load_tracing_inputs_from_roi_bounds
from skimage import filters, restoration

inputs = load_tracing_inputs_from_roi_bounds(
    "rec_Position_1_3.mrc",
    "100:125,150:255,88:89",
    "110,200,88",
)

volume = inputs.volume
mask = inputs.roi_mask

# Work only on the bounding ROI region if you want faster experiments.
roi_values = volume[mask]

# Option 1: Gaussian smoothing
gaussian = filters.gaussian(
    volume,
    sigma=1.0,
    preserve_range=True,
).astype(np.float32)

# Option 2: Total variation denoising
tv = restoration.denoise_tv_chambolle(
    volume,
    weight=0.05,
).astype(np.float32)

# Option 3: Non-local means, heavier but worth testing on cropped ROIs
sigma_est = np.mean(restoration.estimate_sigma(volume, channel_axis=None))
nlm = restoration.denoise_nl_means(
    volume,
    h=0.8 * sigma_est,
    fast_mode=True,
    patch_size=3,
    patch_distance=5,
    channel_axis=None,
).astype(np.float32)
