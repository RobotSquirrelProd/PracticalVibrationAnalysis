%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2026 Robot Squirrel Productions.
%
%          (\_   _/)
%          ( o   o )
%          (  ^_^  )
%             \_/_
%
% ABOUT
% This test harness excercises the forced torsional test harness.
%
% DEVELOPMENT NOTES
% The primary function, CalcForcedTorsResp, depends on the
% CalcFreeFreeTorsResp function so the test harness excercises both
% functions.
%
% This file has been structured to run in MATLAB's unit test environment:
%   results = runtests("CalcForcedTorsResp_Tests.m"); table(results)
%
% Last run: 18 Apr 2026 
% Totals:
%    4 Passed, 0 Failed, 0 Incomplete.
%    1.384 seconds testing time.
% 
% The MATLAB test environment requires each section of code inside the main
% comment header to run independently. For this reason, some code pieces
% repeat.
%

classdef CalcForcedTorsResp_Tests < matlab.unittest.TestCase

    properties (Constant, Access = private)
        d_abs_tol_mode_shape = 1e-6
        d_abs_tol_freq_3mass_1 = 1e-15
        d_abs_tol_freq_3mass_2 = 2e-15
        d_abs_tol_freq_2mass_1 = 4e-15
    end

    methods (TestMethodSetup)
        function closeFigures(testCase)
            close all
        end
    end

    methods (TestMethodTeardown)
        function closeFiguresAfterEachTest(testCase)
            close all
        end
    end

    methods (Test)
        function testThreeMassForcedModelReturnsExpectedFreeFreeResponse(testCase)

            d_MoIp = [1, 2, 3];
            d_kt = [1, 2, 0];
            d_len = ones(size(d_kt));

            d_damp_int = [1e-2, 2e-2, 0];
            d_damp_ext = [1e-10, 2e-10, 3e-10];
            d_obs = [1, 1, 1, 0, 0, 0];

            [h, d_eigvec, omega_cpm, h_mass_elastic, ss_rotor] = ...
                CalcForcedTorsResp(d_MoIp, d_kt, d_len, ...
                d_damp_int, d_damp_ext, d_obs);

            omega_analytical_cpm1 = 8.9138119652212137794;
            omega_analytical_cpm2 = 14.467526728136422258;

            testCase.verifyLessThan(abs(d_eigvec(1) + 1.000), ...
                testCase.d_abs_tol_mode_shape, ...
                "Failed to return correct mode shape.");

            testCase.verifyNotEmpty(h, ...
                "Failed to create main plot handle.");

            testCase.verifyNotEmpty(h_mass_elastic, ...
                "Failed to create mass elastic plot handle.");

            testCase.verifyEqual(omega_cpm(1), omega_analytical_cpm1, ...
                "AbsTol", testCase.d_abs_tol_freq_3mass_1, ...
                "Failed to find mode I natural frequency.");

            testCase.verifyEqual(omega_cpm(2), omega_analytical_cpm2, ...
                "AbsTol", testCase.d_abs_tol_freq_3mass_2, ...
                "Failed to find mode II natural frequency.");

            testCase.verifyNotEmpty(ss_rotor, ...
                "Failed to return rotor state space model.");

            localPlotBodeIfAvailable(ss_rotor, 3, false);
        end

        function testTwoMassForcedModelReturnsExpectedFreeFreeResponse(testCase)

            d_MoIp = [1, 2];
            d_kt = [3, 0];
            d_len = ones(size(d_kt));

            d_damp_int = [3e-2, 0];
            d_damp_ext = [1e-4, 2e-4];
            d_obs = [1, 1, 0, 0];

            i_station_skip = 1;
            y_max_tick_input = -1;
            b_no_plots = false;

            [h, d_eigvec, omega_cpm, h_mass_elastic, ss_rotor] = ...
                CalcForcedTorsResp(d_MoIp, d_kt, d_len, ...
                d_damp_int, d_damp_ext, d_obs, ...
                i_station_skip, y_max_tick_input, ...
                'b_no_plots', b_no_plots);

            omega_analytical_cpm1 = 20.25711711353489;

            testCase.verifyLessThan(abs(d_eigvec(1) + 1.000), ...
                testCase.d_abs_tol_mode_shape, ...
                "Failed to return correct mode shape.");

            testCase.verifyNotEmpty(h, ...
                "Failed to create main plot handle.");

            testCase.verifyNotEmpty(h_mass_elastic, ...
                "Failed to create mass elastic plot handle.");

            testCase.verifyEqual(omega_cpm(1), omega_analytical_cpm1, ...
                "AbsTol", testCase.d_abs_tol_freq_2mass_1, ...
                "Failed to find mode I natural frequency.");

            testCase.verifyNotEmpty(ss_rotor, ...
                "Failed to return rotor state space model.");

            localPlotBodeIfAvailable(ss_rotor, 2, false);
        end

        function testTwoMassWithExternalStiffnessAndPartialUncertainty(testCase)

            d_MoIp = [1, 2];
            d_u_MoIp = [0.01, 0.02];

            d_kt = [3, 0];
            d_u_kt = [0.04, 0.00];

            d_kt_ext = [0.5, 0.25];
            d_u_kt_ext = [0.004, 0.002]; %#ok<NASGU>

            d_len = ones(size(d_kt));
            d_damp_int = [3e-2, 0];
            d_damp_ext = [1e-4, 2e-4];
            d_obs = [1, 1, 0, 0];

            i_station_skip = 1;
            y_max_tick_input = -1;
            b_no_plots = false;
            b_supr_degen = false;

            [h, d_eigvec, omega_cpm, h_mass_elastic, ss_rotor] = ...
                CalcForcedTorsResp(d_MoIp, d_kt, d_len, ...
                d_damp_int, d_damp_ext, d_obs, ...
                i_station_skip, y_max_tick_input, ...
                'b_no_plots', b_no_plots, ...
                'b_supr_degen', b_supr_degen, ...
                'd_kt_ext', d_kt_ext, ...
                'd_u_MoIp', d_u_MoIp, ...
                'd_u_kt', d_u_kt);

            testCase.verifyLessThan(abs(d_eigvec(1) - 1.000), ...
                testCase.d_abs_tol_mode_shape, ...
                "Failed to return correct mode shape.");

            testCase.verifyNotEmpty(h, ...
                "Failed to create main plot handle.");

            testCase.verifyNotEmpty(h_mass_elastic, ...
                "Failed to create mass elastic plot handle.");

            testCase.verifyNotEmpty(ss_rotor, ...
                "Failed to return rotor state space model.");

            testCase.verifyNotEmpty(omega_cpm, ...
                "Failed to return modal frequencies.");

            localPlotBodeIfAvailable(ss_rotor, 2, false);
        end

        function testTwoMassWithExternalStiffnessAndFullUncertainty(testCase)

            d_MoIp = [1, 2];
            d_u_MoIp = [0.01, 0.02];

            d_kt = [3, 0];
            d_u_kt = [0.04, 0.00];

            d_kt_ext = [0.5, 0.25];
            d_u_kt_ext = [0.04, 0.02];

            d_len = ones(size(d_kt));
            d_damp_int = [3e-2, 0];
            d_damp_ext = [1e-4, 2e-4];
            d_obs = [1, 1, 0, 0];

            i_station_skip = 1;
            y_max_tick_input = -1;
            b_no_plots = false;
            b_supr_degen = false;

            [h, d_eigvec, omega_cpm, h_mass_elastic, ss_rotor] = ...
                CalcForcedTorsResp(d_MoIp, d_kt, d_len, ...
                d_damp_int, d_damp_ext, d_obs, ...
                i_station_skip, y_max_tick_input, ...
                'b_no_plots', b_no_plots, ...
                'b_supr_degen', b_supr_degen, ...
                'd_kt_ext', d_kt_ext, ...
                'd_u_MoIp', d_u_MoIp, ...
                'd_u_kt', d_u_kt, ...
                'd_u_kt_ext', d_u_kt_ext);

            testCase.verifyLessThan(abs(d_eigvec(1) - 1.000), ...
                testCase.d_abs_tol_mode_shape, ...
                "Failed to return correct mode shape.");

            testCase.verifyNotEmpty(h, ...
                "Failed to create main plot handle.");

            testCase.verifyNotEmpty(h_mass_elastic, ...
                "Failed to create mass elastic plot handle.");

            testCase.verifyNotEmpty(ss_rotor, ...
                "Failed to return rotor state space model.");

            testCase.verifyNotEmpty(omega_cpm, ...
                "Failed to return modal frequencies.");

            localPlotBodeIfAvailable(ss_rotor, 2, false);
        end
    end
end

function localPlotBodeIfAvailable(ss_rotor, i_input_station, b_lbl_peaks)
%LOCALPLOTBODEIFAVAILABLE
% Build a transfer function from the state space model and call the bode
% plotting helper only when that helper exists in the path.

    [A, B, C, D] = ssdata(ss_rotor);
    [b_tf, a_tf] = ss2tf(A, B, C, D, i_input_station);
    tf_theta1 = tf(b_tf(1, :), a_tf);

    [mag, phase, wout] = bode(tf_theta1);

    if exist('PlotBodeTors', 'file') > 0
        if nargin < 3
            PlotBodeTors(mag, phase, wout);
        else
            PlotBodeTors(mag, phase, wout, 'b_lbl_peaks', b_lbl_peaks);
        end
    end
end