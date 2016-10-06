import sys
import copy
import unittest
from unittest.mock import patch, MagicMock, call
from io import StringIO

import numpy as np
from mdlmc.LMC.MDMC import cmd_lmc_run, initialize_oxygen_lattice


class MockLMCHelper:
    def LMCRoutine(self, *args, **kwargs):
        return "bullshit"


class MockPBCHelper:
    def AtomBoxMonoclinic(self, *args, **kwargs):
        return "quark"


class Setting:
    pass


class TestMDMC(unittest.TestCase):

    def setUp(self):
        self.out = StringIO()
        self.mock_settings = Setting()
        self.mock_settings.nonortho = True
        self.mock_settings.pbc = np.asfarray([1, 2, 3])
        self.mock_settings.box_multiplier = (1, 1, 1)
        self.mock_settings.oxygen_number = 10
        self.mock_settings.oxygen_number_extended = 10
        self.mock_settings.proton_number = 5
        self.mock_settings.higher_msd = False
        self.mock_settings.jumprate_params_fs = dict(A=0.07, a=0.01, b=0.02, c=0.03)
        self.mock_settings.cutoff_radius = 5.0
        self.mock_settings.angle_threshold = 1.5
        self.mock_settings.neighbor_search_radius = 5.0
        self.mock_settings.jumprate_type = "ActivationEnergy"
        self.mock_settings.verbose = True
        self.mock_settings.md_timestep_fs = 0.5
        self.mock_settings.sweeps = 10000
        self.mock_settings.equilibration_sweeps = 1000
        self.mock_settings.variance_per_proton = False
        self.mock_settings.output = self.out
        self.mock_settings.skip_frames = 0
        self.mock_settings.shuffle = False
        self.mock_settings.xyz_output = False
        self.mock_settings.reset_freq = 100
        self.mock_settings.print_freq = 1
        self.mock_settings.jumpmatrix_filename = None

        self.oxy_traj = np.random.random((10, 20, 3))
        self.phos_traj = np.random.random((10, 10, 3))
        self.oxy_lattice = initialize_oxygen_lattice(self.mock_settings.oxygen_number,
                                                     self.mock_settings.proton_number)

    def test_initialize_oxygen_lattice(self):
        np.random.seed(0)
        for i in range(20):
            oxy_number = np.random.randint(1, 100)
            prot_number = np.random.randint(1, oxy_number)
            oxy_lattice = initialize_oxygen_lattice(oxy_number, prot_number)
            # Make sure the number of protons in the lattice is correct
            self.assertEqual((oxy_lattice > 0).sum(), prot_number)
            # Make sure the protons are enumerated from 1 to prot_number
            self.assertTrue(oxy_lattice.sum() == int(prot_number * (prot_number + 1) / 2))

    @patch("mdlmc.LMC.MDMC.ObservableManager")
    @patch("mdlmc.LMC.MDMC.LMCHelper.LMCRoutine")
    def test_cmdlmc_sweep_calls(self, mock_rout, mock_obs):
        settings = copy.deepcopy(self.mock_settings)
        # Make sure that the number of sweeps is correct
        settings.jumpmatrix_filename = None

        cmd_lmc_run(self.oxy_traj, self.oxy_lattice, mock_rout, mock_obs, settings)
        self.assertEqual(mock_rout.sweep.call_count,
                         settings.sweeps + settings.equilibration_sweeps)

    @patch("mdlmc.LMC.MDMC.np.savetxt")
    @patch("mdlmc.LMC.MDMC.ObservableManager")
    @patch("mdlmc.LMC.MDMC.LMCHelper.LMCRoutine")
    def test_cmdlmc_sweep_with_jumpmatrix_calls(self, mock_rout, mock_obs, mock_savetxt):
        settings = copy.deepcopy(self.mock_settings)
        settings.jumpmatrix_filename = "sth"
        cmd_lmc_run(self.oxy_traj, self.oxy_lattice, mock_rout, mock_obs, settings)
        # During the equilibration nothing is written to the jumpmatrix
        self.assertEqual(mock_rout.sweep_with_jumpmatrix.call_count, settings.sweeps)
        # In the end, it will be written to disk
        self.assertEqual(mock_savetxt.call_count, 1)

    @patch("mdlmc.LMC.MDMC.ObservableManager")
    @patch("mdlmc.LMC.MDMC.LMCHelper.LMCRoutine")
    def test_cmdlmc_xyz_output_calls(self, mock_lmc_routine, mock_observable_manager):
        np.random.seed(1)
        settings = copy.deepcopy(self.mock_settings)
        settings.xyz_output = True
        print_freq = np.random.randint(1, 50, size=5)
        sweeps = np.random.randint(2, 10) * print_freq

        for s, r in zip(sweeps, print_freq):
            settings.sweeps = s
            settings.print_freq = r
            cmd_lmc_run(self.oxy_traj, self.oxy_lattice, mock_lmc_routine, mock_observable_manager,
                        settings)
            self.assertEqual(mock_observable_manager.print_xyz.call_count,
                             settings.sweeps // settings.print_freq)
            mock_observable_manager.reset_mock()

    @patch.object("mdlmc.LMC.MDMC.ObservableManager", "proton_pos_snapshot")
    def test_observable_manager_calculate_displacement(self, mock_proton_pos_snapshot):
        mock_proton_pos_snapshot = np.random.uniform(0, 20, size=(20, 3))
