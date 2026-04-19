%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%          (\_   _/)
%          ( o   o )
%          (  ^_^  )
%             \_/_
%
% Confidential and proprietary computer code and data.
%
% Copyright 2026 Robot Squirrel Productions.  All rights reserved. This
% computer code is proprietary to Robot Squirrel Productions and/or its
% affiliate(s) and may be covered by patents. It may not be used,
% disclosed, modified, transferred, or reproduced without prior written
% consent.

clearvars
close all
clc
format long

str_plot_file = mfilename;

%%
% Three mass system with internal and external damping, focused
% only validating the free-free response. 

% Define polar mass moment of inertia, MR2
d_MoIp = [];
d_MoIp(1) = 1;
d_MoIp(2) = 2;
d_MoIp(3) = 3;

% Define torsional stiffness. This vector should have one less element than
% the polar mass (the last stifness will always be zero)
d_kt = [];
d_kt(1) = 1;
d_kt(2) = 2;
d_kt(3) = 0;

% Define uniform length between polar moment masses
d_len = ones(size(d_kt));

% Define internal damping
d_damp_int = [];
d_damp_int(1) = 0;
d_damp_int(2) = 0;
d_damp_int(3) = 0;

% Define external damping
d_damp_ext = [];
d_damp_ext(1) = 0;
d_damp_ext(2) = 0;
d_damp_ext(3) = 0;

% Define observations
d_obs = [1 1 1 0 0 0];

% Call the forced torsional vibration function, but with return values that
% include the free-free solution
[h_plot, d_eigvec, omega_cpm, h_mass_elastic, ss_rotor] =...
    CalcForcedTorsResp(d_MoIp, d_kt, d_len,...
    d_damp_int, d_damp_ext, d_obs); 

% This test uses the analytical solution from the
% "Torsional_ThreeMass_ExtCoupling.nb" Mathematica worksheet.
omega_analytical_cpm1 = 8.9138119652212137794;
omega_analytical_cpm2 = 14.467526728136422258;

% Extract the transfer function looking for a response at station 1 with
% impulse at station 3
[A,B,C,D] = ssdata(ss_rotor);
[b_tf, a_tf] = ss2tf(A,B,C,D,3);
tf_theta1 = tf(b_tf(1,:), a_tf);

% Calculate the response for this transfer function
[mag,phase,wout] = bode(tf_theta1);

% Output the plot (export ONCE after all subplots are created)
set(h_plot, 'Units', 'inches')
set(h_plot, 'Position', [0 0 16 9]*0.5)
set(h_plot, 'PaperSize', [16 9]*0.5)
exportgraphics(h_plot, [str_plot_file '.pdf'], 'ContentType', 'vector')

set(h_mass_elastic, 'Units', 'inches')
set(h_mass_elastic, 'Position', [0 0 16 9]*0.5)
set(h_mass_elastic, 'PaperSize', [16 9]*0.5)
exportgraphics(h_mass_elastic, [str_plot_file '_mass_elastic.pdf'],...
    'ContentType', 'vector')
