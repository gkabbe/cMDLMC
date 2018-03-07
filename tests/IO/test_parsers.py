from io import StringIO

import daiquiri

from mdlmc.IO.trajectory_parser import xyz_generator, filter_lines, filter_selection


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


def test_filter_lines():
    mock_file = StringIO(MOCK_XYZ)
    fl = filter_lines(mock_file, frame_len=5)
    for line in fl:
        assert line.strip() not in ("3", "comment")


def test_xyz_generator():

    # Assert that the xyz generator yields three frames with three atoms each

    mock_file = StringIO(MOCK_XYZ)
    parser = xyz_generator(mock_file, number_of_atoms=3)

    frames = list(parser)

    assert frames[0].shape == (3,), "XYZ generator yielded wrong shape"
    assert len(frames) == 3, "XYZ generator did not yield all frames"

    # Assert that a selection works
    # Select O and second H
    selection = [0, 2]
    mock_file = StringIO(MOCK_XYZ)
    parser = xyz_generator(mock_file, number_of_atoms=3, selection=selection)

    frames = list(parser)

    assert frames[0].shape == (2,)
    assert frames[0][0]["name"] == b"O"
    assert frames[0][1]["name"] == b"H"
