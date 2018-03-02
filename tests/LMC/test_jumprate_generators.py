from itertools import count


import numpy as np

from mdlmc.LMC.jumprate_generators import jumprate_generator


def test_jumprate_generator():
    def mock_xyz_generator():
        """Create mock generator which yields trajectory of three atoms.
        Two atoms are fixed at (0, 0) and (2, 0), while the third one oscillates
        between (1, 1) and (1, -1)."""
        pos = np.zeros((3, 3))
        pos[1, 0] = 2
        pos[2, 0] = 1
        for t in count():
            pos[2, 1] = np.cos(t * np.pi / 2)
            yield pos

    jumpgen = jumprate_generator(mock_xyz_generator())



