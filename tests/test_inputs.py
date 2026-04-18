import mrcfile
import numpy as np
import pytest

from filament_trace_tomo.inputs import (
    create_roi_mask_from_bounds,
    load_tracing_inputs_from_roi_bounds,
    load_tracing_inputs,
    parse_roi_bounds,
    parse_seed_point,
    validate_roi_mask,
    validate_seed_in_mask,
    write_roi_mask_mrc,
)


def _write_mrc(path, data, voxel_size=13.1):
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.asarray(data))
        mrc.voxel_size = voxel_size


def test_parse_seed_point_from_text_and_sequence():
    assert parse_seed_point("1, 2.5, 3") == (1.0, 2.5, 3.0)
    assert parse_seed_point([4, 5, 6]) == (4.0, 5.0, 6.0)


@pytest.mark.parametrize("seed", ["1,2", "1,2,nope", [1, 2]])
def test_parse_seed_point_rejects_invalid_values(seed):
    with pytest.raises(ValueError):
        parse_seed_point(seed)


def test_parse_roi_bounds_from_text_and_sequence():
    assert parse_roi_bounds("10:20, 30:45, 5:9") == ((10, 20), (30, 45), (5, 9))
    assert parse_roi_bounds([(1, 4), (2, 6), (3, 8)]) == ((1, 4), (2, 6), (3, 8))


@pytest.mark.parametrize("bounds", ["1:2,3:4", "1:2,3,nope", "2:1,3:4,5:6"])
def test_parse_roi_bounds_rejects_invalid_values(bounds):
    with pytest.raises(ValueError):
        parse_roi_bounds(bounds)


def test_create_roi_mask_from_bounds_uses_xyz_bounds_against_zyx_array():
    roi_mask = create_roi_mask_from_bounds((6, 8, 10), "2:5,3:7,1:4")

    assert roi_mask.shape == (6, 8, 10)
    assert roi_mask.sum() == 3 * 4 * 3
    assert roi_mask[1, 3, 2]
    assert roi_mask[3, 6, 4]
    assert not roi_mask[4, 6, 4]
    assert not roi_mask[3, 7, 4]
    assert not roi_mask[3, 6, 5]


def test_create_roi_mask_from_bounds_rejects_out_of_bounds_by_default():
    with pytest.raises(ValueError, match="outside volume bounds"):
        create_roi_mask_from_bounds((6, 8, 10), "-1:5,3:7,1:4")


def test_create_roi_mask_from_bounds_can_clip_to_volume():
    roi_mask = create_roi_mask_from_bounds((6, 8, 10), "-1:5,3:20,1:4", clip=True)

    assert roi_mask.sum() == 5 * 5 * 3


def test_validate_roi_mask_binarizes_positive_values():
    mask = np.array(
        [
            [[0, 0], [0, 1]],
            [[0, 2], [0, 0]],
        ],
        dtype=np.float32,
    )

    roi_mask = validate_roi_mask(mask, (2, 2, 2))

    assert roi_mask.dtype == bool
    assert roi_mask.sum() == 2


def test_validate_seed_in_mask_uses_xyz_seed_order_against_zyx_array():
    roi_mask = np.zeros((4, 5, 6), dtype=bool)
    roi_mask[3, 2, 1] = True

    validate_seed_in_mask((1, 2, 3), roi_mask)


def test_validate_seed_in_mask_rejects_seed_outside_roi():
    roi_mask = np.zeros((4, 5, 6), dtype=bool)

    with pytest.raises(ValueError, match="outside the ROI mask"):
        validate_seed_in_mask((1, 2, 3), roi_mask)


def test_load_tracing_inputs_loads_volume_mask_seed_and_metadata(tmp_path):
    volume = np.arange(27, dtype=np.float32).reshape((3, 3, 3))
    mask = np.zeros((3, 3, 3), dtype=np.float32)
    mask[1, 1, 1] = 1
    volume_path = tmp_path / "Position_51_2.mrc"
    mask_path = tmp_path / "Position_51_2_mask.mrc"
    _write_mrc(volume_path, volume, voxel_size=13.11267)
    _write_mrc(mask_path, mask, voxel_size=13.11267)

    inputs = load_tracing_inputs(volume_path, mask_path, "1,1,1")

    assert inputs.volume.shape == (3, 3, 3)
    assert inputs.roi_mask.dtype == bool
    assert inputs.seed_point == (1.0, 1.0, 1.0)
    assert inputs.voxel_size_angst == pytest.approx(13.11267)
    assert inputs.tomo_name == "Position_51_2"


def test_write_roi_mask_mrc_writes_binary_mask_with_voxel_size(tmp_path):
    roi_mask = create_roi_mask_from_bounds((3, 4, 5), "1:4,1:3,0:2")
    mask_path = tmp_path / "roi_mask.mrc"

    write_roi_mask_mrc(mask_path, roi_mask, voxel_size_angst=16.88)

    with mrcfile.open(mask_path, permissive=True) as mrc:
        assert mrc.data.dtype == np.int8
        assert mrc.data.sum() == roi_mask.sum()
        assert mrc.voxel_size.x == pytest.approx(16.88)


def test_load_tracing_inputs_from_roi_bounds_can_write_mask(tmp_path):
    volume = np.arange(5 * 6 * 7, dtype=np.float32).reshape((5, 6, 7))
    volume_path = tmp_path / "rec_Position_1_3.mrc"
    mask_path = tmp_path / "box_roi.mrc"
    _write_mrc(volume_path, volume, voxel_size=16.88)

    inputs = load_tracing_inputs_from_roi_bounds(
        volume_path,
        "2:5,1:4,1:3",
        "3,2,1",
        mask_output_path=mask_path,
    )

    assert inputs.roi_mask.sum() == 3 * 3 * 2
    assert inputs.seed_point == (3.0, 2.0, 1.0)
    assert inputs.mask_path == mask_path
    assert mask_path.exists()
