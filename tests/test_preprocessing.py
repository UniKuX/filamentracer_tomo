import numpy as np
import pytest
import mrcfile

from filament_trace_tomo.inputs import TracingInputs
from filament_trace_tomo.preprocessing import (
    PreprocessingOptions,
    preprocess_tracing_inputs,
    preprocess_volume,
    write_preprocessed_mrc,
)


def _volume_and_mask():
    volume = np.arange(27, dtype=np.float32).reshape((3, 3, 3))
    roi_mask = np.zeros_like(volume, dtype=bool)
    roi_mask[1:, 1:, 1:] = True
    return volume, roi_mask


def test_preprocess_volume_defaults_leave_roi_values_unchanged_and_zero_outside_roi():
    volume, roi_mask = _volume_and_mask()

    preprocessed = preprocess_volume(volume, roi_mask)

    np.testing.assert_array_equal(preprocessed.volume[roi_mask], volume[roi_mask])
    assert np.all(preprocessed.volume[~roi_mask] == 0)
    assert preprocessed.volume.dtype == np.float32


def test_preprocess_volume_can_preserve_values_outside_roi():
    volume, roi_mask = _volume_and_mask()

    preprocessed = preprocess_volume(
        volume,
        roi_mask,
        options=PreprocessingOptions(fill_outside_roi=None),
    )

    np.testing.assert_array_equal(preprocessed.volume, volume)


def test_preprocess_volume_can_invert_density():
    volume, roi_mask = _volume_and_mask()

    preprocessed = preprocess_volume(
        volume,
        roi_mask,
        options=PreprocessingOptions(invert_density=True),
    )

    np.testing.assert_array_equal(preprocessed.volume[roi_mask], -volume[roi_mask])


def test_preprocess_volume_zscore_normalizes_using_roi_statistics():
    volume, roi_mask = _volume_and_mask()

    preprocessed = preprocess_volume(
        volume,
        roi_mask,
        options=PreprocessingOptions(normalization="zscore"),
    )

    assert float(np.mean(preprocessed.volume[roi_mask])) == pytest.approx(0.0, abs=1e-6)
    assert float(np.std(preprocessed.volume[roi_mask])) == pytest.approx(1.0, abs=1e-6)


def test_preprocess_volume_robust_normalizes_using_roi_statistics():
    volume, roi_mask = _volume_and_mask()

    preprocessed = preprocess_volume(
        volume,
        roi_mask,
        options=PreprocessingOptions(normalization="robust"),
    )

    roi_values = preprocessed.volume[roi_mask]
    assert float(np.median(roi_values)) == pytest.approx(0.0, abs=1e-6)
    q1, q3 = np.percentile(roi_values, [25, 75])
    assert float(q3 - q1) == pytest.approx(1.0, abs=1e-6)


def test_preprocess_volume_clips_using_roi_percentiles():
    volume, roi_mask = _volume_and_mask()
    volume = volume.copy()
    volume[1, 1, 1] = -100
    volume[2, 2, 2] = 100

    preprocessed = preprocess_volume(
        volume,
        roi_mask,
        options=PreprocessingOptions(clip_percentiles=(10, 90)),
    )

    roi_values = preprocessed.volume[roi_mask]
    assert roi_values.min() > -100
    assert roi_values.max() < 100


def test_preprocess_volume_applies_gaussian_filter():
    volume = np.zeros((5, 5, 5), dtype=np.float32)
    volume[2, 2, 2] = 1.0
    roi_mask = np.ones_like(volume, dtype=bool)

    preprocessed = preprocess_volume(
        volume,
        roi_mask,
        options=PreprocessingOptions(filter_method="gaussian", gaussian_sigma=1.0),
    )

    assert 0 < preprocessed.volume[2, 2, 2] < 1
    assert preprocessed.volume[2, 2, 1] > 0


def test_preprocess_volume_applies_median_filter():
    volume = np.zeros((5, 5, 5), dtype=np.float32)
    volume[2, 2, 2] = 100.0
    roi_mask = np.ones_like(volume, dtype=bool)

    preprocessed = preprocess_volume(
        volume,
        roi_mask,
        options=PreprocessingOptions(filter_method="median", median_size=3),
    )

    assert preprocessed.volume[2, 2, 2] == 0


def test_preprocess_volume_applies_hessian_filter():
    volume = np.zeros((9, 9, 9), dtype=np.float32)
    volume[:, 4, 4] = -1.0
    roi_mask = np.ones_like(volume, dtype=bool)

    preprocessed = preprocess_volume(
        volume,
        roi_mask,
        options=PreprocessingOptions(
            filter_method="hessian",
            hessian_sigmas=(1.0, 2.0),
            hessian_black_ridges=True,
        ),
    )

    centerline_value = float(preprocessed.volume[4, 4, 4])
    off_axis_value = float(preprocessed.volume[4, 2, 2])
    assert centerline_value > off_axis_value
    assert np.isfinite(preprocessed.volume).all()


def test_preprocess_tracing_inputs_uses_loaded_volume_and_mask(tmp_path):
    volume, roi_mask = _volume_and_mask()
    tracing_inputs = TracingInputs(
        volume=volume,
        roi_mask=roi_mask,
        seed_point=(1, 1, 1),
        voxel_size_angst=13.1,
        tomo_name="example",
        volume_path=tmp_path / "volume.mrc",
        mask_path=None,
    )

    preprocessed = preprocess_tracing_inputs(
        tracing_inputs,
        PreprocessingOptions(normalization="zscore"),
    )

    assert preprocessed.volume.shape == volume.shape
    assert preprocessed.roi_mask is roi_mask


def test_write_preprocessed_mrc_writes_float32_volume_with_voxel_size(tmp_path):
    volume, roi_mask = _volume_and_mask()
    preprocessed = preprocess_volume(volume, roi_mask)
    output_path = tmp_path / "preprocessed.mrc"

    write_preprocessed_mrc(output_path, preprocessed, voxel_size_angst=16.88)

    with mrcfile.open(output_path, permissive=True) as mrc:
        assert mrc.data.dtype == np.float32
        assert mrc.data.shape == volume.shape
        assert mrc.voxel_size.x == pytest.approx(16.88)


def test_preprocess_volume_rejects_nonfinite_roi_values():
    volume, roi_mask = _volume_and_mask()
    volume = volume.copy()
    volume[1, 1, 1] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        preprocess_volume(volume, roi_mask)


def test_preprocess_volume_rejects_invalid_filter_options():
    volume, roi_mask = _volume_and_mask()

    with pytest.raises(ValueError, match="Gaussian sigma"):
        preprocess_volume(volume, roi_mask, options=PreprocessingOptions(gaussian_sigma=0))

    with pytest.raises(ValueError, match="Median size must be odd"):
        preprocess_volume(volume, roi_mask, options=PreprocessingOptions(median_size=2))
    with pytest.raises(ValueError, match="Hessian sigmas must contain at least one scale"):
        preprocess_volume(volume, roi_mask, options=PreprocessingOptions(hessian_sigmas=()))
    with pytest.raises(ValueError, match="Hessian sigmas must all be positive"):
        preprocess_volume(volume, roi_mask, options=PreprocessingOptions(hessian_sigmas=(0.0, 1.0)))
