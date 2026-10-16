import numpy as np
import pytest
import starfile
from test_relion_export import make_project

from filament_tracer.warp_export import main, warp_coordinate_table


def test_warp_physical_coordinates_and_crop_offset():
    table = warp_coordinate_table(make_project(), 10, 2, "TS_01", (100, 20, 8))
    assert table.rlnMicrographName.tolist() == ["TS_01"] * 6
    assert "rlnTomoName" not in table
    # Warp multiplies coordinates by coords_angpix to recover physical XYZ.
    xyz = table[[f"rlnCoordinate{a}" for a in "XYZ"]].to_numpy() * 2
    np.testing.assert_allclose(xyz[:3], [[103, 24, 12], [103, 24, 22], [103, 24, 32]])
    assert table.rlnHelicalTubeID.tolist() == [1, 1, 1, 2, 2, 2]
    assert table.rlnHelicalTrackLengthAngst.tolist() == [0, 10, 20, 0, 10, 20]


@pytest.mark.parametrize("offset", [(0, 0), (0, np.nan, 0), (-100, 0, 0)])
def test_invalid_offsets(offset):
    with pytest.raises(ValueError):
        warp_coordinate_table(make_project(), 10, 2, "TS_01", offset)


@pytest.mark.parametrize("pixel", [0, -1, np.inf, np.nan])
def test_invalid_pixel_size(pixel):
    with pytest.raises(ValueError, match="pixel size"):
        warp_coordinate_table(make_project(), 10, pixel, "TS_01")


def test_cli_roundtrip_and_no_overwrite(tmp_path, capsys):
    source = tmp_path / "test.ftskeleton.json"
    output = tmp_path / "warp.star"
    make_project().save(source)
    args = [
        "--skeleton", str(source), "--output", str(output),
        "--inter-box-distance", "10", "--coords-angpix", "2",
        "--tomo-name", "TS_01", "--offset-xyz-angst", "100", "20", "8",
    ]
    main(args)
    table = starfile.read(output)
    assert len(table) == 6
    assert table.rlnMicrographName.tolist() == ["TS_01"] * 6
    np.testing.assert_allclose(table.rlnCoordinateX[:3], [51.5] * 3)
    assert "--coords_angpix 2" in capsys.readouterr().out
    original = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert output.read_bytes() == original
