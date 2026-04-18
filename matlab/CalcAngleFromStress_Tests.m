%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2026 Robot Squirrel Productions.
%
%          (\_   _/)
%          ( o   o )
%          (  ^_^  )
%             \_/_
%
% ABOUT
% This test harness excercises the calculation of angle of twist from the
% torque applied to a shaft section
%
% DEVELOPMENT NOTES
% This file has been structured to run in MATLAB's unit test environment:
%   results = runtests("CalcAngleFromStress_Tests.m");
%   table(results)
%
% Last run: 18 Apr 2026
% Totals:
%    3 Passed, 0 Failed, 0 Incomplete.
%    0.013054 seconds testing time.
% 
%
classdef CalcAngleFromStress_Tests < matlab.unittest.TestCase

    properties (Constant, Access = private)
        d_abs_tol = 1e-15
    end

    methods (Test)
        function testScalarCaseUSCS(testCase)
            % 3 in diameter 4140 steel, 4 in long

            d_dia_outer = 3.0;
            d_ro = d_dia_outer / 2.0;
            d_L = 4.0;
            d_taumax = 17200.0;
            d_G = 11603e3;

            d_phi = CalcAngleFromStress(d_ro, d_L, d_taumax, d_G);
            d_phi_deg = rad2deg(d_phi);

            d_expected_deg = 0.226490254273324;

            testCase.verifyEqual(d_phi_deg, d_expected_deg, ...
                "AbsTol", testCase.d_abs_tol, ...
                "USCS scalar case returned the wrong twist angle.");
        end

        function testScalarCaseSI(testCase)
            % 7.54 cm diameter 4140 steel, 11 cm long

            d_dia_outer = 7.54;
            d_ro = d_dia_outer / 2.0;
            d_L = 11.0;
            d_taumax = 121e6;
            d_G = 80e9;

            d_phi = CalcAngleFromStress(d_ro, d_L, d_taumax, d_G);
            d_phi_deg = rad2deg(d_phi);

            d_expected_deg = 0.252853721922787;

            testCase.verifyEqual(d_phi_deg, d_expected_deg, ...
                "AbsTol", testCase.d_abs_tol, ...
                "SI scalar case returned the wrong twist angle.");
        end

        function testVectorStressInputSI(testCase)
            % 7.54 cm diameter 4140 steel, 11 cm long
            % Vector stress input in MPa

            d_dia_outer = 7.54;
            d_ro = d_dia_outer / 2.0;
            d_L = 11.0;
            d_taumax = [121e6, 120e6, 119e6];
            d_G = 80e9;

            d_phi = CalcAngleFromStress(d_ro, d_L, d_taumax, d_G);
            d_phi_deg = rad2deg(d_phi);

            d_expected_deg = [ ...
                0.252853721922787; ...
                0.250764021741607; ...
                0.248674321560427];

            testCase.verifyEqual(d_phi_deg, d_expected_deg, ...
                "AbsTol", testCase.d_abs_tol, ...
                "SI vector case returned the wrong twist angles.");
        end
    end
end