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
% CalcFreeFreeTorsResp function so you must run that test harness anytime 
% changes are made here.
%
% This file has been structured to run in MATLAB's unit test environment:
%   results = runtests("CalcFreeFreeTorsResp_Tests.m");
%
% Last run: 18 Apr 2026
% Totals:
%    2 Passed, 0 Failed, 0 Incomplete.
%    2.6598 seconds testing time.
% 
% The MATLAB test environment requires each section of code inside the main
% comment header to run independently. For this reason, some code pieces
% repeat.
%
classdef CalcFreeFreeTorsResp_Tests < matlab.unittest.TestCase

    properties (Constant, Access = private)
        d_abs_tol_mode_shape = 1e-6
        d_abs_tol_mode_1 = 1e-15
        d_abs_tol_mode_2 = 2e-15
    end

    methods (Test)
        function testThreeMassFreeFreeResponse(testCase)
            % Define polar mass moments of inertia
            d_MoIp = [1, 2, 3];

            % Define torsional stiffness
            % The last stiffness is zero for the free free system.
            d_kt = [1, 2, 0];

            % Define uniform element lengths
            d_len = ones(size(d_kt));

            % Call function under test
            [h, d_eigvec, omega_cpm, h_mass_elastic, ss_rotor] = ...
                CalcFreeFreeTorsResp(d_MoIp, d_kt, d_len);

            % Analytical solution from Mathematica reference
            omega_analytical_cpm1 = 8.9138119652212137794;
            omega_analytical_cpm2 = 14.467526728136422258;

            % Verify mode shape
            testCase.verifyLessThan(abs(d_eigvec(1) + 1.000), ...
                testCase.d_abs_tol_mode_shape, ...
                "Failed to return correct mode shape.");

            % Verify plot handles were returned
            testCase.verifyNotEmpty(h, ...
                "Failed to create main plot.");

            testCase.verifyNotEmpty(h_mass_elastic, ...
                "Failed to create mass elastic plot.");

            % Verify natural frequencies
            testCase.verifyEqual(omega_cpm(1), omega_analytical_cpm1, ...
                "AbsTol", testCase.d_abs_tol_mode_1, ...
                "Failed to find mode I natural frequency.");

            testCase.verifyEqual(omega_cpm(2), omega_analytical_cpm2, ...
                "AbsTol", testCase.d_abs_tol_mode_2, ...
                "Failed to find mode II natural frequency.");

            % Verify state space model was returned
            testCase.verifyNotEmpty(ss_rotor, ...
                "Failed to return the rotor state space model.");
        end
    end
end