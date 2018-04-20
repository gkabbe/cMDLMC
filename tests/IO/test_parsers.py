from io import StringIO

import daiquiri
import numpy as np
import pytest

from mdlmc.IO.trajectory_parser import XYZTrajectory, filter_lines, filter_selection, Frame
from mdlmc.atoms.numpy_atom import dtype_xyz


logger = daiquiri.getLogger(__name__)
daiquiri.setup(level=daiquiri.logging.DEBUG)


MOCK_XYZ = """
3
comment
O 0 0 0
H 0 1 0
H 1 0 0
3
comment
O 0 0 0
H 0 1 0
H 1.2 0 0
3
comment
O 0 0 0
H 0 1 0
H 1.4 0 0
""".strip()


@pytest.fixture
def xyz_array():
    """Return an array of dtype xyz_dtype"""
    return np.array([("O", [0, 0, 0]),
                     ("H", [0, 1, 0]),
                     ("H", [1, 0, 0])], dtype=dtype_xyz)


@pytest.fixture
def xyz_file():
    """Return a StringIO object containing a small xyz trajectory"""
    return StringIO(MOCK_XYZ)


def test_frame(xyz_array):
    frame = Frame.from_recarray(xyz_array)

    np.testing.assert_equal(frame["H"].atom_positions, xyz_array["pos"][xyz_array["name"] == "H"]), \
        "Frame did not select array of atoms properly by name"

    np.testing.assert_equal(frame[[0, -1]].atom_positions, xyz_array["pos"][[0, -1]]), \
        "Frame did not select array of atoms properly by index selection"

    assert frame.atom_number == 3


def test_frame_append(xyz_array):
    f1 = Frame.from_recarray(xyz_array)
    f2 = Frame.from_recarray(xyz_array)

    result = f1.append(f2)

    assert result.atom_number == 6
    np.testing.assert_array_equal(result.atom_names, ["O", "H", "H", "O", "H", "H"])


def test_filter_lines(xyz_file):
    fl = filter_lines(xyz_file, frame_len=5)
    for line in fl:
        assert line.strip() not in ("3", "comment")


def test_xyz_trajectory(xyz_file):
    """Assert that the xyz generator yields three frames with three atoms each"""
    parser = XYZTrajectory(xyz_file, number_of_atoms=3)

    frames = list(parser)

    for frame in frames:
        assert frame.atom_names.shape == (3,), "XYZ generator yielded wrong shape"
    assert len(frames) == 3, "XYZ generator did not yield all frames"


@pytest.mark.parametrize("selection, expected_shape",
                         [((0, 2), (2,)),
                          (("O", "H"), (3,))
                          ])
def test_xyz_selection(xyz_file, selection, expected_shape):
    # Assert that a selection works
    # Select O and second H
    mock_file = StringIO(MOCK_XYZ)
    parser = XYZTrajectory(mock_file, number_of_atoms=3, selection=selection)

    frames = list(parser)
    logger.debug("Output of XYZTrajectory: %s", frames)
    assert frames[0].atom_names.shape == expected_shape
