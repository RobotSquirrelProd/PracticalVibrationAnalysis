%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2026 Robot Squirrel Productions.
%
%          (\_   _/)
%          ( o   o )
%          (  ^_^  )
%             \_/_
%
% ABOUT
% Full torsional analysis of a string with internal and external damping.
% The model outputs free-free as well as forced-free response. The code
% also handles branched systems so even the forced torsional vibration
% parameters should be passed through this function to be consolidated into
% a single system.
%
% EXAMPLE
% function [h_plot, d_eigvec, d_ft, h_mass_elastic, ss_rotor] =...
%    CalcForcedTorsResp(J, K, varargin) 
%
% DEVELOPER NOTES
% This function explicitly depends on the "FreeFreeTorsResp.m" to construct
% the matrices for the state-space solution. Rather than create a separate
% test harness, the test harness for this function also excercises the
% free-free test harness.
%
% DEPENDENCIES
% This function requires the following toolboxe(s):
% - Control System Toolbox
% 
% INPUT
% d_MoIp (REQUIRED, NUMERIC VECTOR) - Vector of shaft polar mass moment 
%   inertias (in-lb-s^2).
% d_kt (REQUIRED, NUMERIC VECTOR) - Vector with torsional
%   stiffness (in-lb/rad). Set this to zero at the gear mesh station to
%   force infinite mesh stiffness solution.
% d_len (OPTIONAL, NUMERIC VECTOR) - distance between masses and stiffness.
%   Also used to set horizontal scaling when plotting out station numbers.
% d_damp_int (OPTIONAL, NUMERIC) - Vector of internal damping 
%   (in-lb-s/rad). Defaults to zero.
% d_damp_ext (OPTIONAL, NUMERIC) - Vector of external damping 
%   (in-lb-s/rad). Defaults to zero.
% d_obs (OPTIONAL, NUMERIC) - Vector of observational measurements 
%   (1 = observed, otherwise not included in the "C" state space matrix. 
%   Default is all ones (all variables observed). variables are ordered as:
%   (?1, ?2, ... ?n, ?dot1, ?dot2, ... ?dotn)
% i_station_skip (OPTIONAL, INTEGER) - Defaults to 1 which labels each 
%           station. Describes the spacing between station labels on plots.
% y_max_tick_input (OPTIONAL, NUMERIC)- If greater than zero this is the 
%   maximum tick label for the y-axis on the mass-elastic diagram. Defaults 
%   to -1. 
% d_kt_ext - (PARAMETER, NUMERIC) Vector of external stiffness
%   (in-lb/rad). Default is zero.
% b_supr_degen - (PARAMETER, LOGICAL). Defaults to true which 
%   removes the degenerate mode from the calculated natural frequencies if
%   true. Set to false for systems with external stiffness to display all
%   modes.
% b_no_plots - (PARAMETER, LOGICAL). Defaults to true. Determines if
%   plots are generated.
% str_plot_file (Keyed, char). Defaults to ''. Sets the name for plots to be
%   written to file.
% d_gear_ratio - (PARAMETER, NUMERIC). Defaults to and empty value, []. 
%   Vector of gear ratios. Values should be -1 to indicate absence of gear.
% i_mode_max - (PARAMETER, NUMERIC). Defaults to 4. Number of modes to 
%   plot, including the degenerate mode and independent of b_supr_degen 
%   value.
% d_u_MoIp (PARAMETER, POSITIVE SCALAR) - Uncertainty in the polar mass 
%   moment of inertia calculation. This has the same units as d_MoIp 
%   (absolute uncertainty)
% d_u_kt (PARAMETER, POSITIVE SCALAR) - Uncertainty torsional stiffness 
%   vector. This has the same units as d_kt (absolute uncertainty)
% d_u_kt_ext (PARAMETER, POSITIVE SCALAR) - Uncertainty for the external 
%   torsional stiffness vector. This has the same units as d_kt_ext 
%   (absolute uncertainty)
%
% OUTPUT
% h_plot - (MATLAB FIGURE HANDLE) Handle to mode shape plot
% d_eigvec - (NUMERIC MATRIX) Matrix of top eigenvectors mode where the 
%   rows correspond to station numbers and the columns to mode numbers.
%   The variable b_supr_degen determines if the matrix includes the
%   degenerate mode or not.
% d_ft - (NUMERIC VECTOR) Vector with the natural frequency (CPM)
%   corresponding to the mode shape in d_eigvec. Naming convention follows 
%   2529RS0011 to avoid naming collisions. 
% h_mass_elastic - (MATLAB FIGURE HANDLE) Handle to mass elastic diagram
% ss_rotor - (NUMERIC MATRIX) Rotor state-space matrix
% d_MoIp - (NUMERIC VECTOR) Vector of shaft rotational inertias
%   (in-lb-s^2). These inertias are collapsed to account for branching in 
%   geared systems.
% d_kt - (NUMERIC VECTOR) Vector of shaft torsional stiffness.
%   Values have units of (in-lb/rad).
% d_damp_int - (PARAMETER, NUMERIC) Vector of internal damping. Values 
%   have units of (in-lb-s/rad). This vector defaults to zero. This vector
%   is collapsed to account for branching in the system.
% d_damp_ext - (PARAMETER, NUMERIC) Vector of external damping. The 
%   values have units of (in-lb-s/rad).  This vector defaults to zero. 
%   This vector is collapsed to account for branching in the system.
% d_obs - (PARAMETER, NUMERIC) Vector of observational measurements 
%   (1 = observed, otherwise not included in the "C" state space matrix. 
%   Default is all ones (all variables observed). variables are ordered as:
%   (?1, ?2, ... ?n, ?dot1, ?dot2, ... ?dotn). This vector is collapsed to
%   account for branching in the system.
%
% B. Howard
% 2 Sep 2024
%
function [h_plot, d_eigvec, d_ft, h_mass_elastic, ss_rotor, d_MoIp,...
    d_kt, d_damp_int, d_damp_ext, d_obs] =...
    CalcForcedTorsResp(d_MoIp, d_kt, varargin) 

% Set the defaults for the inputs
d_default_len = ones(size(d_MoIp));
d_default_damp_int = zeros(size(d_MoIp));
d_default_damp_ext = zeros(size(d_MoIp));
d_default_obs = ones(1, 2*length(d_MoIp));
i_default_station_skip = 1;
i_default_y_max_tick_input = -1;
d_default_kt_ext = zeros(size(d_MoIp));

% Validation rules
validNonNegNum = @(x) assert(isnumeric(x) && all(x >= 0),...
    'Must be a positive numeric quantity');

% Parse the inputs
p = inputParser;
addRequired(p, 'd_MoIp', @isvector);
addRequired(p, 'd_kt', @isvector);
addOptional(p, 'd_len', d_default_len, @isvector)
addOptional(p, 'd_damp_int', d_default_damp_int, @isvector)
addOptional(p, 'd_damp_ext', d_default_damp_ext, @isvector)
addOptional(p, 'd_obs', d_default_obs, @isvector)
addOptional(p, 'i_station_skip', i_default_station_skip, @isreal)
addOptional(p, 'y_max_tick_input', i_default_y_max_tick_input, @isreal)
addParameter(p, 'd_kt_ext', d_default_kt_ext, @isvector)
addParameter(p, 'b_supr_degen', true, @islogical)
addParameter(p, 'b_no_plots', false, @islogical)
addParameter(p, 'str_plot_file', '', @ischar)
addParameter(p, 'd_gear_ratio', [], @isvector);
addParameter(p, 'i_mode_max', 4, @isnumeric);
addParameter(p, 'd_u_MoIp', [], validNonNegNum);
addParameter(p, 'd_u_kt', [], validNonNegNum);
addParameter(p, 'd_u_kt_ext', [], validNonNegNum);

% Instatiate the parser and process the inputs
parse(p, d_MoIp, d_kt, varargin{:});

% Normalize vector orientation
d_MoIp = d_MoIp(:)';
d_kt = d_kt(:)';

% Extract optional arguements
d_len = p.Results.d_len;
d_len = d_len(:)';
d_damp_int = p.Results.d_damp_int;
d_damp_ext = p.Results.d_damp_ext;
d_obs = p.Results.d_obs;
i_station_skip = p.Results.i_station_skip;
y_max_tick_input = p.Results.y_max_tick_input;
d_kt_ext = p.Results.d_kt_ext;
b_supr_degen = p.Results.b_supr_degen;
b_no_plots = p.Results.b_no_plots;
str_plot_file = p.Results.str_plot_file;
d_gear_ratio = p.Results.d_gear_ratio(:)';
i_mode_max = p.Results.i_mode_max;
d_u_MoIp = p.Results.d_u_MoIp;
d_u_kt = p.Results.d_u_kt;
d_u_kt_ext = p.Results.d_u_kt_ext;

% Validate dependencies
assert( license('test', 'control_toolbox')==1,...
    'The Control System Toolbox must be licensed for this function to run')

% Validate the inputs
i_pmm = length(d_MoIp);
assert(length(d_obs) == 2*i_pmm, 'Not enough observation values in O')

% Free-free response. This also re-structures polar mass moments (J) and
% torsional stiffness (K) for branched systems. For this reason we have to
% perform counts below.
disp(d_len)
[h_plot, d_eigvec, d_ft, h_mass_elastic, d_MoIp, d_kt,...
    d_damp_int, d_damp_ext, d_obs] = CalcFreeFreeTorsResp(d_MoIp,...
    d_kt, d_len, i_station_skip, y_max_tick_input,...
    'd_kt_ext', d_kt_ext, 'b_supr_degen', b_supr_degen,...
    'b_no_plots', b_no_plots, 'str_plot_file', str_plot_file,...
    'd_gear_ratio', d_gear_ratio, 'i_mode_max', i_mode_max,...
    'd_damp_int', d_damp_int, 'd_damp_ext', d_damp_ext, 'd_obs', d_obs,...
    'd_u_MoIp', d_u_MoIp, 'd_u_kt', d_u_kt, 'd_u_kt_ext', d_u_kt_ext);

% Validate the collapse of branched system was successful
i_pmm = length(d_MoIp);
i_obs = length(d_obs(d_obs==1));
assert(length(d_obs) == 2*i_pmm, 'Not enough observation values in O')

% Assemble the state-space matrixes where the matrices are defined as:
A = zeros(2*i_pmm, 2*i_pmm);
B = zeros(2*i_pmm, i_pmm);
C = eye(2*i_pmm, 2*i_pmm);
D = zeros(i_obs, i_pmm);

%%
% Contruct the A matrix
%
% State space development for most turbomachinery systems follows a
% pattern defined in 145M1773. These lines define the first submatrix (all
% zeros) and the second submatrix (identity matrix).
idx_row = 1;
for idx = (i_pmm+1):(2*i_pmm)
    A(idx_row, idx) = 1;
    idx_row = idx_row + 1;
end

% Document 2529RS0019 describes the methodology for constructing the 
% mass-stiffness and mass-damping diagonal terms
idx_row = (i_pmm + 1);
for idx = 1:i_pmm
    switch idx
        case 1
            A(idx_row, idx) = -(d_kt(idx)+d_kt_ext(idx))/d_MoIp(idx);
            A(idx_row, idx+i_pmm) = -(d_damp_int(idx) + d_damp_ext(idx))/d_MoIp(idx);
        case i_pmm
            A(idx_row, idx) = -(d_kt(idx-1)+d_kt_ext(idx))/d_MoIp(idx);
            A(idx_row, idx+i_pmm) = -(d_damp_int(idx-1) + d_damp_ext(idx))/d_MoIp(idx);
        otherwise
            A(idx_row, idx) = -(d_kt(idx-1) + d_kt(idx) + d_kt_ext(idx) )/d_MoIp(idx);
            A(idx_row, idx+i_pmm) = -(d_damp_int(idx-1) + d_damp_ext(idx) + d_damp_int(idx))/d_MoIp(idx);
    end
    idx_row = idx_row + 1;
end

% Mass-stiffness and mass-damping off-diagonal terms
idx_row = (i_pmm + 1);
for idx = 2:i_pmm
    
    % mass-stiffness terms
    A(idx_row, idx) = d_kt(idx-1)/d_MoIp(idx-1);
    A(idx_row+1, idx-1) = d_kt(idx-1)/d_MoIp(idx);
    
    % mass-damping terms
    A(idx_row, idx+i_pmm) = d_damp_int(idx-1)/d_MoIp(idx-1);
    A(idx_row+1, idx-1+i_pmm) = d_damp_int(idx-1)/d_MoIp(idx);

    % Next row
    idx_row = idx_row + 1;
    
end

%%
% Populate the "B" matrix
for idx = 1:i_pmm
   B(idx+i_pmm, idx) = 1/d_MoIp(idx); 
end

%%
% Populate the C matrix
C(d_obs==0,:) = []; 

%%
% Create Matlab's state-space model
disp("Observation (O) vector:")
disp(d_obs)
disp("C matrix:")
disp(C)
disp(D)
ss_rotor = ss(A,B,C,D);

return