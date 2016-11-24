import unittest
import numpy as np

from mdlmc.analysis.excess_charge_analyzer import excess_charge_collective_variable


class TestExcessChargeAnalyzer(unittest.TestCase):
    def test_collective_variable_invariance(self):
        """Assert that the collective variable is invariant for h2o movement"""
        h2o_angle = np.radians(104.45)
        h2o = np.array([[0, 0, 0],
                        [0.96, 0, 0],
                        [np.cos(h2o_angle) * 0.96, np.sin(h2o_angle) * 0.96, 0]])

        colvars = []
        for i in range(20):
            waterbox = np.vstack([h2o + np.random.uniform(-20, 20, size=3) for i in range(20)])
            oxygens = waterbox[::3]
            protons = waterbox[np.array([i % 3 != 0 for i in range(waterbox.shape[0])])]
            colvars.append(excess_charge_collective_variable(oxygens, protons))
        for a, b in zip(colvars[:-1], colvars[1:]):
            self.assertTrue(np.allclose(a, b))

    def test_collective_variable_correlates_with_excess_charge(self):
        """Assert that the collective variable actually tracks the excess charge movement
        (Move excess charge around, but keep oxygens fixed)"""
        h2o_angle = np.radians(104.45)
        h2o = np.array([[0, 0, 0],
                        [0.96, 0, 0],
                        [np.cos(h2o_angle) * 0.96, np.sin(h2o_angle) * 0.96, 0]])

        waterbox = np.vstack([h2o + np.random.uniform(-20, 20, size=3) for i in range(20)])
        oxygens = waterbox[::3]
        protons = waterbox[np.array([i % 3 != 0 for i in range(waterbox.shape[0])])]
        colvars = []
        excess_charge_positions = []
        for i in range(20):
            excess_charge = np.random.uniform(-20, 20, size=3)
            colvars.append(excess_charge_collective_variable(oxygens, np.vstack(
                [protons, excess_charge[None, :]])))
            excess_charge_positions.append(excess_charge)
        colvars = np.array(colvars)
        excess_charge_positions = np.array(excess_charge_positions)
        for cv, ec in zip(colvars, excess_charge_positions):
            cv_diff = cv - colvars[0]
            ec_diff = ec - excess_charge_positions[0]
            np.testing.assert_allclose(cv_diff, ec_diff)
