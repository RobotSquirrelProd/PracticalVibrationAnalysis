"""Tests for stress-to-angle torsional helper."""

import unittest

import numpy as np

from vibration_analysis.torsional.calc_angle_from_stress import calc_angle_from_stress


class CalcAngleFromStressTests(unittest.TestCase):
    ABS_TOL = 1e-15

    def test_scalar_case_uscs(self) -> None:
        # 3 in diameter 4140 steel, 4 in long
        d_dia_outer = 3.0
        d_ro = d_dia_outer / 2.0
        d_L = 4.0
        d_taumax = 17200.0
        d_G = 11603e3

        d_phi = calc_angle_from_stress(d_ro, d_L, d_taumax, d_G)
        d_phi_deg = np.rad2deg(d_phi)

        d_expected_deg = np.array([[0.226490254273324]])

        self.assertTrue(np.allclose(d_phi_deg, d_expected_deg, atol=self.ABS_TOL, rtol=0.0))

    def test_scalar_case_si(self) -> None:
        # 7.54 cm diameter 4140 steel, 11 cm long
        d_dia_outer = 7.54
        d_ro = d_dia_outer / 2.0
        d_L = 11.0
        d_taumax = 121e6
        d_G = 80e9

        d_phi = calc_angle_from_stress(d_ro, d_L, d_taumax, d_G)
        d_phi_deg = np.rad2deg(d_phi)

        d_expected_deg = np.array([[0.252853721922787]])

        self.assertTrue(np.allclose(d_phi_deg, d_expected_deg, atol=self.ABS_TOL, rtol=0.0))

    def test_vector_stress_input_si(self) -> None:
        # 7.54 cm diameter 4140 steel, 11 cm long
        # Vector stress input in MPa
        d_dia_outer = 7.54
        d_ro = d_dia_outer / 2.0
        d_L = 11.0
        d_taumax = [121e6, 120e6, 119e6]
        d_G = 80e9

        d_phi = calc_angle_from_stress(d_ro, d_L, d_taumax, d_G)
        d_phi_deg = np.rad2deg(d_phi)

        d_expected_deg = np.array(
            [
                [0.252853721922787],
                [0.250764021741607],
                [0.248674321560427],
            ]
        )

        self.assertTrue(np.allclose(d_phi_deg, d_expected_deg, atol=self.ABS_TOL, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
