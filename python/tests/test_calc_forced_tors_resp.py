"""Tests for forced torsional response helper."""

import unittest

from matplotlib.figure import Figure

from vibration_analysis.torsional.calc_forced_tors_resp import calc_forced_tors_resp


class CalcForcedTorsRespTests(unittest.TestCase):
    ABS_TOL_MODE_SHAPE = 1e-6
    ABS_TOL_FREQ_3MASS_1 = 1e-12
    ABS_TOL_FREQ_3MASS_2 = 1e-12
    ABS_TOL_FREQ_2MASS_1 = 1e-12

    def test_three_mass_forced_model_returns_expected_free_free_response(self) -> None:
        d_moip = [1.0, 2.0, 3.0]
        d_kt = [1.0, 2.0, 0.0]
        d_len = [1.0, 1.0, 1.0]

        d_damp_int = [1e-2, 2e-2, 0.0]
        d_damp_ext = [1e-10, 2e-10, 3e-10]
        d_obs = [1, 1, 1, 0, 0, 0]

        h, d_eigvec, omega_cpm, h_mass_elastic, ss_rotor, *_ = calc_forced_tors_resp(
            d_moip,
            d_kt,
            d_len,
            d_damp_int,
            d_damp_ext,
            d_obs,
        )

        omega_analytical_cpm1 = 8.9138119652212137794
        omega_analytical_cpm2 = 14.467526728136422258

        self.assertLess(abs(float(d_eigvec[0, 0]) + 1.0), self.ABS_TOL_MODE_SHAPE)
        self.assertIsNotNone(h)
        self.assertIsInstance(h, Figure)
        self.assertIsNotNone(h_mass_elastic)
        self.assertAlmostEqual(float(omega_cpm[0]), omega_analytical_cpm1, delta=self.ABS_TOL_FREQ_3MASS_1)
        self.assertAlmostEqual(float(omega_cpm[1]), omega_analytical_cpm2, delta=self.ABS_TOL_FREQ_3MASS_2)
        self.assertIsNotNone(ss_rotor)

    def test_two_mass_forced_model_returns_expected_free_free_response(self) -> None:
        d_moip = [1.0, 2.0]
        d_kt = [3.0, 0.0]
        d_len = [1.0, 1.0]

        d_damp_int = [3e-2, 0.0]
        d_damp_ext = [1e-4, 2e-4]
        d_obs = [1, 1, 0, 0]

        h, d_eigvec, omega_cpm, h_mass_elastic, ss_rotor, *_ = calc_forced_tors_resp(
            d_moip,
            d_kt,
            d_len,
            d_damp_int,
            d_damp_ext,
            d_obs,
            1,
            -1,
            b_no_plots=False,
        )

        omega_analytical_cpm1 = 20.25711711353489

        self.assertLess(abs(float(d_eigvec[0, 0]) + 1.0), self.ABS_TOL_MODE_SHAPE)
        self.assertIsNotNone(h)
        self.assertIsInstance(h, Figure)
        self.assertIsNotNone(h_mass_elastic)
        self.assertAlmostEqual(float(omega_cpm[0]), omega_analytical_cpm1, delta=self.ABS_TOL_FREQ_2MASS_1)
        self.assertIsNotNone(ss_rotor)

    def test_two_mass_with_external_stiffness_and_partial_uncertainty(self) -> None:
        d_moip = [1.0, 2.0]
        d_u_moip = [0.01, 0.02]
        d_kt = [3.0, 0.0]
        d_u_kt = [0.04, 0.0]
        d_kt_ext = [0.5, 0.25]

        d_len = [1.0, 1.0]
        d_damp_int = [3e-2, 0.0]
        d_damp_ext = [1e-4, 2e-4]
        d_obs = [1, 1, 0, 0]

        h, d_eigvec, omega_cpm, h_mass_elastic, ss_rotor, *_ = calc_forced_tors_resp(
            d_moip,
            d_kt,
            d_len,
            d_damp_int,
            d_damp_ext,
            d_obs,
            1,
            -1,
            b_no_plots=False,
            b_supr_degen=False,
            d_kt_ext=d_kt_ext,
            d_u_MoIp=d_u_moip,
            d_u_kt=d_u_kt,
        )

        self.assertLess(abs(float(d_eigvec[0, 0]) - 1.0), self.ABS_TOL_MODE_SHAPE)
        self.assertIsNotNone(h)
        self.assertIsInstance(h, Figure)
        self.assertIsNotNone(h_mass_elastic)
        self.assertIsNotNone(ss_rotor)
        self.assertTrue(len(omega_cpm) > 0)

    def test_two_mass_with_external_stiffness_and_full_uncertainty(self) -> None:
        d_moip = [1.0, 2.0]
        d_u_moip = [0.01, 0.02]
        d_kt = [3.0, 0.0]
        d_u_kt = [0.04, 0.0]
        d_kt_ext = [0.5, 0.25]
        d_u_kt_ext = [0.04, 0.02]

        d_len = [1.0, 1.0]
        d_damp_int = [3e-2, 0.0]
        d_damp_ext = [1e-4, 2e-4]
        d_obs = [1, 1, 0, 0]

        h, d_eigvec, omega_cpm, h_mass_elastic, ss_rotor, *_ = calc_forced_tors_resp(
            d_moip,
            d_kt,
            d_len,
            d_damp_int,
            d_damp_ext,
            d_obs,
            1,
            -1,
            b_no_plots=False,
            b_supr_degen=False,
            d_kt_ext=d_kt_ext,
            d_u_MoIp=d_u_moip,
            d_u_kt=d_u_kt,
            d_u_kt_ext=d_u_kt_ext,
        )

        self.assertLess(abs(float(d_eigvec[0, 0]) - 1.0), self.ABS_TOL_MODE_SHAPE)
        self.assertIsNotNone(h)
        self.assertIsInstance(h, Figure)
        self.assertIsNotNone(h_mass_elastic)
        self.assertIsNotNone(ss_rotor)
        self.assertTrue(len(omega_cpm) > 0)


if __name__ == "__main__":
    unittest.main()
