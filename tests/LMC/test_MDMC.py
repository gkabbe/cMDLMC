import unittest
from unittest.mock import MagicMock, patch, mock_open
from io import StringIO
import numpy as np

from mdlmc.LMC.MDMC import MDMC


fake_output = StringIO()

def fake_load_configfile(*args, **kwargs):
    cfg = {"filename": "blub",
         "o_neighbor": "P",
         "auxiliary_file": None,
         "clip_trajectory": 0,
         "verbose": True,
         "seed": 0,
         "box_multiplier": [1, 1, 1],
         "jumprate_params_fs": dict(A=0.07, a=0.1, b=0.2, c=0.3),
         "md_timestep_fs": 0.5,
         "pbc": "10 20 30",
         "output": fake_output
         }
    return cfg


def fake_load_atoms(*args, **kwargs):
    return [np.random.random((20, 3)) for i in range(len(args) - 1)]


class TestMDMC(unittest.TestCase):
    @patch("mdlmc.LMC.MDMC.load_atoms", side_effect=fake_load_atoms)
    @patch("mdlmc.LMC.MDMC.load_configfile", side_effect=fake_load_configfile)
    def test_print_settings(self, fk_load_cfgf, fk_load_at):
        mdmc = MDMC("bla")
        mdmc.print_settings()
        print(fake_output.getvalue())
