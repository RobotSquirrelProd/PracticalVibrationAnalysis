"""Tests for free-free torsional response helper."""

import unittest

from matplotlib.figure import Figure

from vibration_analysis.torsional.calc_free_free_tors_resp import calc_free_free_tors_resp


class CalcFreeFreeTorsRespTests(unittest.TestCase):
    ABS_TOL_MODE_SHAPE = 1e-6
    ABS_TOL_MODE_1 = 1e-12
    ABS_TOL_MODE_2 = 1e-12

    def test_three_mass_free_free_response(self) -> None:
        # Define polar mass moments of inertia
        d_moip = [1.0, 2.0, 3.0]

        # Define torsional stiffness (last entry is zero for free-free system)
        d_kt = [1.0, 2.0, 0.0]

        # Define uniform element lengths
        d_len = [1.0, 1.0, 1.0]

        h, d_eigvec, omega_cpm, h_mass_elastic, ss_rotor, *_ = calc_free_free_tors_resp(
            d_moip,
            d_kt,
            d_len,
        )

        # Analytical solution from Mathematica reference
        omega_analytical_cpm1 = 8.9138119652212137794
        omega_analytical_cpm2 = 14.467526728136422258

        # Verify mode shape
        self.assertLess(abs(float(d_eigvec[0, 0]) + 1.0), self.ABS_TOL_MODE_SHAPE)

        # Verify plot handles were returned
        self.assertIsNotNone(h)
        self.assertIsInstance(h, Figure)
        self.assertIsNotNone(h_mass_elastic)

        # Verify natural frequencies
        self.assertAlmostEqual(float(omega_cpm[0]), omega_analytical_cpm1, delta=self.ABS_TOL_MODE_1)
        self.assertAlmostEqual(float(omega_cpm[1]), omega_analytical_cpm2, delta=self.ABS_TOL_MODE_2)

        # Verify state-space model was returned
        self.assertIsNotNone(ss_rotor)


if __name__ == "__main__":
    unittest.main()
