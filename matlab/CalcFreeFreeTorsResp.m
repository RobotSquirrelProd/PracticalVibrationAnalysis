%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2026 Robot Squirrel Productions.
%
%          (\_   _/)
%          ( o   o )
%          (  ^_^  )
%             \_/_
%
% ABOUT
% Eigenvector analysis of a single torsional model. The code also handles
% branched systems so even the forced torsional vibration parameters should
% be passed through this function to be consolidated into a single system.
%
% DEPENDENCIES
% This function requires the following toolboxe(s):
% - Control System Toolbox
%
% EXAMPLE
% [h, d_eigvec, d_ft, h_mass_elastic] =...
%   CalcFreeFreeTorsResp(d_MoIp, d_kt, varargin)
%
% INPUT
% d_MoIp - (REQUIRED, NUMERIC VECTOR) Vector of shaft rotational
%   inertias (in-lb-s^2).
% d_kt - (REQUIRED, NUMERIC VECTOR)  Vector of shaft torsional
%   stiffness (in-lb/rad). Set this to zero at the gear mesh station to
%   force infinite mesh stiffness solution.
% d_len - (OPTIONAL, NUMERIC VECTOR) length between masses and stiffness
% i_station_skip - (OPTIONAL, INTEGER). Defaults to 1 which labels each
%   station. Describes the spacing between station labels on plots.
% y_max_tick_input - (OPTIONAL, NUMERIC). Defaults to -1. If greater than
%   zero this is the maximum tick label for the y-axis on the mass-elastic
%   diagram.
% d_kt_ext - (OPTIONAL_KEYED, NUMERIC) Vector of external stiffness
%   (in-lb/rad). Default is zero.
% b_supr_degen (OPTIONAL_KEYED, LOGICAL). Defaults to true which
%   removes the degenerate mode from the calculated natural frequencies if
%   true. Set to false for systems with external stiffness to display all
%   modes.
% b_no_plots (OPTIONAL_KEYED, LOGICAL). Defaults to true. Determines if
%   plots are generated.
% str_plot_file (OPTIONAL_KEYED, CHAd_gear_ratio). Defaults to ''. Sets the
%   name for plots to be written to file.
% d_gear_ratio - (OPTIONAL_KEYED, NUMERIC). Defaults to and empty value, [].
%   Vector of gear ratios. Values should be -1 to indicate absence of gear.
% i_mode_max - (OPTIONAL_KEYED, NUMERIC). Defaults to 4. Number of modes to
%   plot, including the degenerate mode and independent of b_supr_degen
%   value.
% d_damp_int - (OPTIONAL_KEYED, NUMERIC) Vector of internal damping
%   (in-lb-s/rad). Defaults to zero.
% d_damp_ext - (OPTIONAL_KEYED, NUMERIC) Vector of external damping
%   (in-lb-s/rad). Defaults to zero.
% d_obs - (OPTIONAL_KEYED, NUMERIC) Vector of observational measurements
%   (1 = observed, otherwise not included in the "C" state space matrix.
%   Default is all ones (all variables observed). variables are ordered as:
%   (?1, ?2, ... ?n, ?dot1, ?dot2, ... ?dotn)
% b_supr_out - (OPTIONAL_KEYED, LOGICAL). Defaults to false. Determines if
%   intermediate matrices are displayed.
% d_u_MoIp (PARAMETER, POSITIVE SCALAR) - Uncertainty in the polar mass
%   moment of inertia calculation. This has the same units as d_MoIp
%   (absolute uncertainty)
% d_u_kt (PARAMETER, POSITIVE SCALAR ) - Uncertainty torsional stiffness
%   vector. This has the same units as d_kt (absolute uncertainty)
% d_u_kt_ext (PARAMETER, POSITIVE SCALAR) - Uncertainty for the external 
%   torsional stiffness vector. This has the same units as d_kt_ext 
%   (absolute uncertainty)
%
% OUTPUT
% h - (MATLAB FIGURE HANDLE) Handle to mode shape plot
% d_eigvec - (NUMERIC MATRIX) - Matrix of top mode shapes (eigen vectors)
%   where the rows correspond to station numbers and the columns to mode
%   numbers. If b_supr_degen has been set to true the matrix will not
%   includes degenerate mode. This should be false for systems coupled to
%   inertia ground.
% d_ft - (NUMERIC VECTOR) Vector with the natural frequency (CPM)
%   corresponding to the mode shape in d_eigvec. Naming convention follows
%   2529RS0011 to avoid naming collisions.
% h_mass_elastic - (MATLAB FIGURE HANDLE) Handle to mass elastic diagram
% d_MoIp - (NUMERIC VECTOR) Vector of shaft rotational inertias
%   (in-lb-s^2). These inertias are collapsed to account for branching in
%   geared systems.
% d_kt - (NUMERIC VECTOR) Vector of shaft torsional stiffness.
%   Values have units of (in-lb/rad).
% d_damp_int - (OPTIONAL_KEYED, NUMERIC) Vector of internal damping. Values
%   have units of (in-lb-s/rad). This vector defaults to zero. This vector
%   is collapsed to account for branching in the system.
% d_damp_ext - (OPTIONAL_KEYED, NUMERIC) Vector of external damping. The
%   values have units of (in-lb-s/rad).  This vector defaults to zero.
%   This vector is collapsed to account for branching in the system.
% d_obs - (OPTIONAL_KEYED, NUMERIC) Vector of observational measurements
%   (1 = observed, otherwise not included in the "C" state space matrix.
%   Default is all ones (all variables observed). variables are ordered as:
%   (?1, ?2, ... ?n, ?dot1, ?dot2, ... ?dotn). This vector is collapsed to
%   account for branching in the system.
%
%
% B. Howard
% 2 Sep 2024
%
function [h_mode_shapes, mat_eig_sorted, d_ft, h_mass_elastic, d_MoIp,...
    d_kt, d_damp_int, d_damp_ext, d_obs] = CalcFreeFreeTorsResp(d_MoIp,...
    d_kt, varargin)

% Set the defaults
d_default_len = ones(size(d_MoIp));
i_default_station_skip = 1;
d_default_y_max_tick_input = -1;
d_default_kt = zeros(size(d_MoIp));
d_default_damp_int = zeros(size(d_MoIp));
d_default_damp_ext = zeros(size(d_MoIp));
d_default_obs = ones(1, 2*length(d_MoIp));

% Validation rules
validNonNegNum = @(x) assert(isnumeric(x) && all(x >= 0),...
    'Must be a positive numeric quantity');

% Instantiate the parser
p = inputParser;

% Define the inputs
addRequired(p, 'd_MoIp', @isvector);
addRequired(p, 'd_kt', @isvector);
addOptional(p, 'd_len', d_default_len, @isvector)
addOptional(p, 'i_station_skip', i_default_station_skip, @isreal)
addOptional(p, 'y_max_tick_input', d_default_y_max_tick_input, @isreal)
addParameter(p, 'd_kt_ext', d_default_kt, @isvector)
addParameter(p, 'b_supr_degen', true, @islogical)
addParameter(p, 'b_no_plots', false, @islogical)
addParameter(p, 'str_plot_file', '', @ischar)
addParameter(p, 'd_gear_ratio', [], @isvector);
addParameter(p, 'i_mode_max', 4, @isnumeric);
addParameter(p, 'd_damp_int', d_default_damp_int, @isvector)
addParameter(p, 'd_damp_ext', d_default_damp_ext, @isvector)
addParameter(p, 'd_obs', d_default_obs, @isvector)
addParameter(p, 'b_supr_out', false, @islogical)
addParameter(p, 'd_u_MoIp', [], validNonNegNum);
addParameter(p, 'd_u_kt', [], validNonNegNum);
addParameter(p, 'd_u_kt_ext', [], validNonNegNum);

parse(p, d_MoIp, d_kt, varargin{:});

% Extract arguements
d_MoIp = p.Results.d_MoIp(:)';
d_kt = p.Results.d_kt(:)';
d_len = p.Results.d_len(:)';
i_station_skip = p.Results.i_station_skip;
y_max_tick_input = p.Results.y_max_tick_input;
d_kt_ext = p.Results.d_kt_ext;
b_supr_degen = p.Results.b_supr_degen;
b_no_plots = p.Results.b_no_plots;
str_plot_file = p.Results.str_plot_file;
d_gear_ratio = p.Results.d_gear_ratio(:)';
i_mode_max = p.Results.i_mode_max;
d_damp_int = p.Results.d_damp_int;
d_damp_ext = p.Results.d_damp_ext;
d_obs = p.Results.d_obs;
b_supr_out = p.Results.b_supr_out;
d_u_MoIp = p.Results.d_u_MoIp(:)';
d_u_kt = p.Results.d_u_kt(:)';
d_u_kt_ext = p.Results.d_u_kt_ext(:)';

% Validate toolboxes
assert( license('test', 'control_toolbox')==1,...
    'The Control System Toolbox must be licensed for this function to run')

% How many input nodes?
i_stat_nos=size( d_kt, 2 );

% Validate uncertainty vector sizes
i_seq_steps = 0;
if ( ~isempty(d_u_MoIp) )
    assert( length(d_u_MoIp) == i_stat_nos,...
        'd_u_MoIp must have the same number of elements as d_MoIp')
    assert( length(d_u_kt) == i_stat_nos,...
        'd_u_kt must have the same number of elements as d_kt')
    
    % Because models may not always have external stiffness, the user can
    % skip the uncertianty for external stiffness
    if( isempty(d_u_kt_ext))
        d_u_kt_ext = zeros(size(d_u_MoIp));
    end

    % Validate the lengths
    assert( length(d_u_kt_ext) == i_stat_nos,...
        'd_u_kt_ext must have the same number of elements as d_kt_ext')
    
    % Save off the number of steps
    i_seq_steps = length(d_u_kt);
end

% Handle the gear ratio vector, then process for gears
if length(d_gear_ratio) < 1
    d_gear_ratio = ones(size(d_kt)) * -1;
end

% Does the system have a gear?
b_gear_present = false;
[~, idx_gear] = find(d_gear_ratio>0);
assert(length(idx_gear) <= 1,...
    'More than one gear found. Code only supports one gearset.')
if ( idx_gear > 0 )
    b_gear_present = true;
else
    % used for the plotting functions
    idx_gear = i_stat_nos;
end

% Save off the vectors, used later in the mass-elastic diagram
d_kt_inp = d_kt;
d_MoIp_inp = d_MoIp;

% Restructure the vectors to handle the gear
if b_gear_present

    % Extract the gear ratio
    r = d_gear_ratio(idx_gear);

    % Case I - Treat gear as infinitely stiff, collapse system
    if d_kt(idx_gear) <= 0

        % Consolidate the gearset into a single node
        d_kt(idx_gear) = d_kt(idx_gear) + d_kt(idx_gear+1) / (r^2);

        % Collapse the rest of the string, calculated reflected inertias
        for idx = (idx_gear+1):(i_stat_nos-1)
            d_kt(idx) = d_kt(idx+1)  / (r^2);
            d_MoIp(idx) = d_MoIp(idx+1)  / (r^2);
            d_len(idx) = d_len(idx+1);
            d_damp_int(idx) = d_damp_int(idx+1);
            d_damp_ext(idx) = d_damp_ext(idx+1);
            d_obs(idx) = d_obs(idx+1);
            d_obs(idx+i_stat_nos) = d_obs(idx+1+i_stat_nos);

            % Do the uncertainties need to be adjusted?
            if ( i_seq_steps > 0 )
                d_u_MoIp(idx) = d_u_MoIp(idx + 1);
                d_u_kt(idx) = d_u_kt(idx + 1);
            end

        end

        % remove the end points
        d_kt(end) = [];
        d_MoIp(end) = [];
        d_len(end) = [];
        d_damp_int(end) = [];
        d_damp_ext(end) = [];
        d_obs(i_stat_nos) = [];
        d_obs(end) = [];

        % Do the uncertainties need to be adjusted?
        if ( i_seq_steps > 0 )
            d_u_MoIp(end) = [];
            d_u_kt(end) = [];
        end

        % Case II - Gear stiffness is present, do not collapse the system
    else

        % Calculate reflected inertias
        for idx = (idx_gear+1):i_stat_nos
            d_kt(idx) = d_kt(idx)  / (r^2);
            d_MoIp(idx) = d_MoIp(idx)  / (r^2);
            % Needed for plots later
            d_MoIp_inp(idx+1) = d_MoIp_inp(idx);
            d_len(idx) = d_len(idx);
            d_damp_int(idx) = d_damp_int(idx);
            d_damp_ext(idx) = d_damp_ext(idx);
            d_obs(idx) = d_obs(idx);
            d_obs(idx+i_stat_nos) = d_obs(idx+i_stat_nos);
        end

    end
end

% New length (station numbers)
i_stat_nos = size( d_kt, 2 );

% Generate cumulative length
d_len_sum = cumsum(d_len);

% Harmonize units. Polar mass momement has units of in-lb-s^2 and converted
% to m-N-s^2 for display.See "GeometricProperties.nb" for calculation of
% constants.
d_MoIp_convert = 0.11298;
d_MoIp_si = d_MoIp * d_MoIp_convert;
d_MoIp_in_si = d_MoIp_inp * d_MoIp_convert;
d_kt_convert = 0.11298;
d_kt_in_si = d_kt_inp * d_kt_convert;



%%
% Define array from equations of motion.
d_A = FormA(d_MoIp, d_kt, d_kt_ext, i_stat_nos, b_supr_out);

%%
% Calculate the eigen values and eigen vectors.
[mat_eigen, d] = eig(d_A);

% Calculate the frequencies
d_ft = sqrt( diag(-d) );

% Sort the frequencies
[d_ft_sorted, ~] = sort(d_ft);

%%
% Perform the sequential perturbation for uncertainty analysis
d_ft_delR = zeros( i_stat_nos, 1 );
b_supr_out_pert = true;
for idx_pert = 1:i_seq_steps

    %----------------------------------------------------------------------
    %----------------------------------------------------------------------
    % Positive side of the stiffness
    d_kt_pert = d_kt;
    d_kt_pert(idx_pert) = ( d_kt_pert(idx_pert) + d_u_kt(idx_pert));

    % Define array from equations of motion, calculate frequencies
    d_A_pert = FormA(d_MoIp, d_kt_pert, d_kt_ext, i_stat_nos,...
        b_supr_out_pert);
    [~, d_pert] = eig(d_A_pert);
    d_ft_pert = sqrt( diag(-d_pert) );
    d_ft_pert = sort(d_ft_pert);
    
    % Calculate the finite difference, positive side
    d_delR_plus = (d_ft_pert - d_ft_sorted);

    %----------------------------------------------------------------------
    % Negative side of the stiffness
    d_kt_pert = d_kt;
    d_kt_pert(idx_pert) = ( d_kt_pert(idx_pert) - d_u_kt(idx_pert));

    % Define array from equations of motion, calculate frequencies
    d_A_pert = FormA(d_MoIp, d_kt_pert, d_kt_ext, i_stat_nos,...
        b_supr_out_pert);
    [~, d_pert] = eig(d_A_pert);
    d_ft_pert = sqrt( diag(-d_pert) );
    d_ft_pert = sort(d_ft_pert);
    
    % Calculate the finite difference, positive side
    d_delR_neg = (d_ft_pert - d_ft_sorted);

    %----------------------------------------------------------------------
    % Accumulate the difference
    d_ft_delR = d_ft_delR +...
        ( ( abs(d_delR_plus) + abs(d_delR_neg) ) / 2.0 ).^2;

    %----------------------------------------------------------------------
    %----------------------------------------------------------------------
    % Positive side of the polar mass uncertainty
    d_MoIp_pert = d_MoIp;
    d_MoIp_pert(idx_pert) = ( d_MoIp_pert(idx_pert) + d_u_MoIp(idx_pert));

    % Define array from equations of motion, calculate frequencies
    d_A_pert = FormA(d_MoIp_pert, d_kt, d_kt_ext, i_stat_nos,...
        b_supr_out_pert);
    [~, d_pert] = eig(d_A_pert);
    d_ft_pert = sqrt( diag(-d_pert) );
    d_ft_pert = sort(d_ft_pert);
    
    % Calculate the finite difference, positive side
    d_delR_plus = (d_ft_pert - d_ft_sorted);

    %----------------------------------------------------------------------
    % Negative side of the polar mass uncertainty
    d_MoIp_pert = d_MoIp;
    d_MoIp_pert(idx_pert) = ( d_MoIp_pert(idx_pert) - d_u_MoIp(idx_pert));

    % Define array from equations of motion, calculate frequencies
    d_A_pert = FormA(d_MoIp_pert, d_kt, d_kt_ext, i_stat_nos,...
        b_supr_out_pert);
    [~, d_pert] = eig(d_A_pert);
    d_ft_pert = sqrt( diag(-d_pert) );
    d_ft_pert = sort(d_ft_pert);
    
    % Calculate the finite difference, positive side
    d_delR_neg = (d_ft_pert - d_ft_sorted);
    
    %----------------------------------------------------------------------
    % Accumulate the difference
    d_ft_delR = d_ft_delR +...
        ( ( abs(d_delR_plus) + abs(d_delR_neg) ) / 2.0 ).^2;


    %----------------------------------------------------------------------
    %----------------------------------------------------------------------
    % Positive side of the external stiffness
    d_kt_ext_pert = d_kt_ext;
    d_kt_ext_pert(idx_pert) = ( d_kt_ext_pert(idx_pert) +...
        d_u_kt_ext(idx_pert));

    % Define array from equations of motion, calculate frequencies
    d_A_pert = FormA(d_MoIp, d_kt, d_kt_ext_pert, i_stat_nos,...
        b_supr_out_pert);
    [~, d_pert] = eig(d_A_pert);
    d_ft_pert = sqrt( diag(-d_pert) );
    d_ft_pert = sort(d_ft_pert);
    
    % Calculate the finite difference, positive side
    d_delR_plus = (d_ft_pert - d_ft_sorted);

    %----------------------------------------------------------------------
    % Negative side of the external stiffness
    d_kt_ext_pert = d_kt_ext;
    d_kt_ext_pert(idx_pert) = ( d_kt_ext_pert(idx_pert) -...
        d_u_kt_ext(idx_pert));

    % Define array from equations of motion, calculate frequencies
    d_A_pert = FormA(d_MoIp, d_kt, d_kt_ext_pert, i_stat_nos,...
        b_supr_out_pert);
    [~, d_pert] = eig(d_A_pert);
    d_ft_pert = sqrt( diag(-d_pert) );
    d_ft_pert = sort(d_ft_pert);
    
    % Calculate the finite difference, positive side
    d_delR_neg = (d_ft_pert - d_ft_sorted);

    %----------------------------------------------------------------------
    % Accumulate the difference
    d_ft_delR = d_ft_delR +...
        ( ( abs(d_delR_plus) + abs(d_delR_neg) ) / 2.0 ).^2;    
end

% Finally, take the square root of the sum of squares
d_ft_delR = ( d_ft_delR .^ 0.5 );

%%
% Display the system
if( ~b_supr_out)
    disp('A matrix'); disp(d_A)
    disp('Eigenvalues'); disp(diag(d)')
    disp('Eigenvectors (each column is an eigenvector)'); disp(mat_eigen)
    disp('Frequencies, d_ft'); disp(d_ft);
    disp('d'); disp(d)
    disp('d_ftSorted'); disp(d_ft_sorted)
end

% Select the lowest modes for display
if( i_stat_nos > i_mode_max )

    % Select modes to plot
    d_ft_plot = d_ft_sorted(1:i_mode_max);
    d_ft_delR_plot = d_ft_delR(1:i_mode_max);
    i_num_modes = i_mode_max;

else
    
    % Display all modes
    d_ft_plot = d_ft_sorted;
    d_ft_delR_plot = d_ft_delR;
    i_num_modes = i_stat_nos;

end

% Initialize storage buffers
mat_eig_sorted = zeros(i_stat_nos, i_num_modes);

% Copy eigenvactors and values for the lowest modes.
for idx_mode = (1:i_num_modes)

    % Iterate over the station numbers
    for idx_station = 1:i_stat_nos

        % Select and scale the eigenvectors (mode shapes)
        if( ( d_ft(idx_station) == d_ft_sorted(idx_mode) ) )

            % Normalize curve by mapping the minimum value to -1 and the
            % maximum value in the vector to +1 using point-slope (delta 
            % is fixed at 2 for the normalized data).
            d_vec_min = min( mat_eigen( :, idx_station ) );
            d_vec_max = max( mat_eigen( :, idx_station ) );
            d_vec_delta =  ( d_vec_max - d_vec_min );

            % The degenerate mode results in a division by zero so avoid
            % that case.
            d_norm_slope = 1;
            if( abs( d_vec_delta ) > 1e-6 )

                d_norm_slope = ( 2.0 / d_vec_delta );
            
            end

            % Store the specific vector
            mat_eig_sorted(:, idx_mode) =...
                ( mat_eigen(:,idx_station) - d_vec_min ) * d_norm_slope - 1;
        end

    end

end

% Remove the degenerate row
if b_supr_degen
    mat_eig_sorted(:,1) = [];
    d_ft_plot(1) = [];
    d_ft_delR_plot(1) = [];
    i_num_modes = i_num_modes-1;
end

% Just output the required modes, but in CPM
d_ft = d_ft_plot.*30/pi;

%%
% Plot the mass-elastic schematic and station numbers
h_mass_elastic = [];

% Branch if plots desired
if ~b_no_plots

    % Instantiate the plot
    h_mass_elastic = figure('Name', 'Schematic',...
        'position',[1 1 1000 600]);
    i_panes = 20;

    % Setup the plotting limits
    y_max = round(max(d_MoIp_si)*1.1, 2, 'Significant');
    if y_max < 1e-5
        y_max = max(d_MoIp_si)*1.1;
    end
    y_max_tick = y_max_tick_input;
    if (y_max_tick_input <= 0)
        y_max_tick = round(y_max/10)*10;
    end
    if y_max_tick < 1e-5
        y_max_tick = max(d_MoIp_si);
    end
    ytick_spaced = linspace(-y_max_tick, y_max_tick, 5);

    % USCS units. Bit of code to shift to kin-lbf
    y_max_USCS = y_max/d_MoIp_convert;
    str_prefix_polar_mass_USCS = '';
    d_scale = 1.0;
    if y_max_USCS > 1e4
        d_scale = 1/1e3;
        y_max_USCS = y_max_USCS*d_scale;
        str_prefix_polar_mass_USCS = 'k';
    end

    % Set USCS ticks
    ytick_spaced_USCS = linspace(-d_scale*y_max_tick/d_MoIp_convert,...
        d_scale*y_max_tick/d_MoIp_convert, 5);

    % Station numbers
    d_offset = d_len_sum(end)*0.05;

    % Secondary left axis
    subplot(2,i_panes,1);
    ax = gca;
    ax.YColor = 'k';
    ax.XColor = 'none';
    ylabel({'Polar Mass Moment of Inertia'...
        ['\bfI\rm_p, (' str_prefix_polar_mass_USCS 'in-lb-s^2)']})

    % Make room for the driven string
    if (b_gear_present)
        ylim([-2*y_max_USCS 2*y_max_USCS])
    else
        ylim([-y_max_USCS y_max_USCS])
    end
    set(gca,'YTick',ytick_spaced_USCS);
    set(gca,'color','none')
    ytickformat('%,.0f')

    % Main plot
    subplot(2,i_panes,[ 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18]);

    % Polar mass moment of inertia
    yyaxis left
    ax = gca;
    ax.YColor = 'k';

    % Grid and keep previously plotted data points
    hold on
    grid on

    % Setup the scale factor for the dots on the end of inertial elements
    i_marker_range = [2 6];
    d_inertias = [min(d_MoIp_in_si) max(d_MoIp_in_si)];
    d_coeff = polyfit(d_inertias, i_marker_range, 1);

    % Plot the driver string starting with points on top
    d_len_plot = d_len_sum(1:idx_gear-1);
    d_MoIp_plot = d_MoIp_si(1:idx_gear-1);
    for idx = 1:length(d_len_plot)

        % Set marker size
        d_mkr_size = polyval(d_coeff, d_MoIp_plot(idx));

        % top points
        stem(d_len_plot(idx), d_MoIp_plot(idx), 'k',...
            'filled', 'LineWidth', 1.1, 'MarkerSize', d_mkr_size,...
            'LineStyle', '-', 'Marker', 'o');

        % Now plot the points on bottom
        stem(d_len_plot(idx), -d_MoIp_plot(idx), '-k',...
            'filled', 'LineWidth', 1.1, 'MarkerSize', d_mkr_size,...
            'LineStyle', '-', 'Marker', 'o');
    end

    % Plot the driven string
    if (b_gear_present)

        % Driver gear inertia using the input values
        % Set marker size
        d_MoIp_plot = d_MoIp_in_si(idx_gear);
        d_mkr_size = polyval(d_coeff, d_MoIp_plot);
        stem(d_len_sum(idx_gear), d_MoIp_plot,...
            '-k', 'filled', 'LineWidth',1.1, 'MarkerSize', d_mkr_size,...
            'LineStyle', '-', 'Marker', 'o');
        stem(d_len_sum(idx_gear), -d_MoIp_plot,...
            '-k', 'filled', 'LineWidth',1.1, 'MarkerSize', d_mkr_size,...
            'LineStyle', '-', 'Marker', 'o');

        % Plot the driven string starting with the line connecting the
        % driven gear to the first inertia
        plot(d_len_sum(idx_gear:end),...
            -y_max*ones(size(d_len_sum(idx_gear:end))), '-k')

        % This is a pain - stem plots always start from zero, but the
        % driven system is offset from axis so I have to revert to
        % different plot commands. The MarkerSize is also interpreted
        % differently for 'stem' and 'plot' commands so I have to scale the
        % linear fit. a MarkerSize of 27 for 'plot' roughly corresponds to
        % a MarkerSize of 6 for 'stem'
        d_len_plot = d_len_sum(idx_gear:end);
        for idx = 0:(length(d_len_plot)-1)

            % Set marker size
            d_MoIp_plot = d_MoIp_in_si(idx_gear+1+idx);
            d_mkr_size = polyval(d_coeff, d_MoIp_plot);

            plot(d_len_sum(idx_gear+idx), d_MoIp_plot - y_max,...
                '.k', 'MarkerSize', d_mkr_size * 27/6);
            plot(d_len_sum(idx_gear+idx), -d_MoIp_plot - y_max,...
                '.k', 'MarkerSize', d_mkr_size * 27/6);

        end

        % Vertical lines for the driven string inertial elements
        nConnect = length(d_len_sum(idx_gear:end));
        for idx = idx_gear:((nConnect+idx_gear)-1)
            plot( [d_len_sum(idx) d_len_sum(idx)],...
                [-y_max -y_max+d_MoIp_in_si(idx+1)], '-k')
            plot( [d_len_sum(idx) d_len_sum(idx)],...
                [-y_max -y_max-d_MoIp_in_si(idx+1)], '-k')
        end

        % Connector between driver and driven string
        plot( [d_len_sum(idx_gear) d_len_sum(idx_gear)],...
            [-y_max 0], '-.k')


    else
        stem(d_len_sum(idx_gear), d_MoIp_si(idx_gear),...
            '-k', 'filled', 'LineWidth',1.1);
        stem(d_len_sum(idx_gear), -d_MoIp_si(idx_gear),...
            '-k', 'filled', 'LineWidth',1.1);
    end

    ylabel('\bfI\rm_p, (m-N-s^2)')
    if (b_gear_present)
        ylim([-2*y_max 2*y_max])
    else
        ylim([-y_max y_max])
    end
    set(gca,'YTick',ytick_spaced);
    ytickformat('%,.0f')
    ylimfinal = ylim;

    yyaxis right

    % Skipping the mesh stiffness
    dTempK = d_kt(idx_gear);
    if( d_kt(idx_gear)>0)
        d_kt_in_si(idx_gear) = 0;
        d_kt(idx_gear)=0;
    end

    % Set maximum limits
    d_exp = round(log10(max(d_kt_in_si)));
    y_max_K = round(2*max(d_kt_in_si)/10^d_exp)*10^d_exp;

    % Create the rotor springs
    for idx = 1:length(d_kt)

        if d_kt(idx)>0
            d_x_bar = ( d_len_sum(idx+1) - d_len_sum(idx) ) / 2;
            d_y_rect = d_kt_in_si(idx);
            i_spring_elements = 7;
            d_x_start = d_len_sum(idx)+d_x_bar*0.5;
            x_coords = linspace(0,d_x_bar,i_spring_elements);
            x_coords(1) = (x_coords(2) - x_coords(1))/2;
            x_coords(end) = x_coords(end) - x_coords(1);
            x_coords = x_coords + d_x_start ;
            ax = gca;
            ax.ColorOrderIndex = 1;
            hold on
            dKoffset = 0;
            if (idx >= idx_gear && b_gear_present)
                dKoffset = (y_max*(y_max_K/max(ylimfinal)));
            end
            plot([d_len_sum(idx) x_coords(1)], [0 0]-dKoffset, '-k')
            plot(x_coords,...
                [0 d_y_rect -d_y_rect d_y_rect -d_y_rect d_y_rect 0]-dKoffset, '-k',...
                'LineWidth',0.8)
            plot([x_coords(end) d_len_sum(idx + 1) ], [0 0]-dKoffset, '-k')
        end
    end
    d_kt(idx_gear) = dTempK;

    ax = gca;
    ax.YColor = 'k';
    ylabel('\bfk\rm, (m-N/rad)')
    ylim([-y_max_K y_max_K])
    ytick_spaced_K = linspace(-y_max_K , y_max_K , 5);
    set(gca,'YTick',ytick_spaced_K);

    % Label the x-axis and ticks
    set_x_tick_labels(d_len_sum, i_stat_nos, i_station_skip, b_supr_out)
    xlim([0-d_offset d_len_sum(end)+d_offset]);

    % Schematic title
    title('Lumped Parameter Model Schematic')

    % Secondary right axis
    subplot(2,i_panes,i_panes);
    yyaxis left
    ax = gca;
    ax.YColor = 'none';
    yyaxis right
    ax = gca;
    ax.YColor = 'k';
    ax.XColor = 'none';
    ylabel({'\bfk\rm, (in-lb/rad)' 'Stiffness'})
    ylim([-y_max_K/d_kt_convert y_max_K/d_kt_convert])
    set(gca,'YTick',ytick_spaced_K/d_kt_convert);
    set(gca,'color','none')
    ytickformat('%,.1f')

    % Create the plot for the schematic
    subplot(2,i_panes,[ 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38]);

    L_rotor = [0 d_len_sum];
    d_dia = d_MoIp_inp.^0.25;
    d_dia = d_dia/max(d_dia);
    d_dia = [d_dia d_dia(end)];
    stairs(L_rotor(1:idx_gear+1), [d_dia(1:idx_gear) d_dia(idx_gear)], 'k')
    hold on
    grid on
    stem(L_rotor(1:idx_gear+1), [d_dia(1:idx_gear) d_dia(idx_gear)], 'k','Marker','none')
    stairs(L_rotor(1:idx_gear+1), -[d_dia(1:idx_gear) d_dia(idx_gear)], 'k')
    stem(L_rotor(1:idx_gear+1), -[d_dia(1:idx_gear) d_dia(idx_gear)], 'k','Marker','none')

    % Handle the gear
    if (b_gear_present)

        % Horizontal line segments
        stairs(L_rotor(idx_gear:end), d_dia(idx_gear+1:end) - 2, 'k')
        stairs(L_rotor(idx_gear:end), -d_dia(idx_gear+1:end) - 2, 'k')

        % Vertical lines for the driven string
        nConnect = length(d_len_sum(idx_gear:end));
        for idx = idx_gear:(nConnect+idx_gear)
            plot( [L_rotor(idx) L_rotor(idx)],...
                [-2 -2+d_dia(idx+1)], '-k')
            plot( [L_rotor(idx) L_rotor(idx)],...
                [-2 -2-d_dia(idx+1)], '-k')
        end
    end

    % Label the x-axis and ticks
    set_x_tick_labels(d_len_sum, i_stat_nos, i_station_skip, b_supr_out)
    xlim([0-d_offset d_len_sum(end)+d_offset]);


    % y-axis limits and labels
    ylim([ -1.1 1.1])
    if( b_gear_present )
        ylim([ -3.1 1.1])
    end
    ylabel('Normalized Equivalent Diameter')

    title('Station number schematic')

    if length(str_plot_file) > 1
        set(h_mass_elastic, 'PaperOrientation', 'portrait');
        set(h_mass_elastic,'PaperUnits','normalized');
        set(h_mass_elastic,'PaperPosition', [0 0 0.8 0.5]);
        saveas(h_mass_elastic, [str_plot_file '_mass_elastic'], 'pdf')
        saveas(h_mass_elastic, [str_plot_file '_mass_elastic'], 'png')
    end

end

%%
% Plot the mode shapes
if ~b_no_plots
    h_mode_shapes = figure('Name', 'Mode Shape', 'position',[1 1 1000 600]);

    for idx_mode=1:i_num_modes

        % Create the plot and put the mode outline on the plot
        subplot(i_num_modes,1,idx_mode)

        plot(d_len_sum(1:idx_gear), mat_eig_sorted(1:idx_gear,idx_mode), 'k');
        hold on
        grid on

        % Plot the dots
        plot(d_len_sum(1:idx_gear), mat_eig_sorted(1:idx_gear,idx_mode),'.k','MarkerSize',12);

        % Driven string
        if( b_gear_present )
            plot(d_len_sum(idx_gear:end), mat_eig_sorted(idx_gear:end,idx_mode)-0.1, 'k');
            plot(d_len_sum(idx_gear:end), mat_eig_sorted(idx_gear:end,idx_mode)-0.1,...
                '.k','MarkerSize',12);
        end

        % Label the x-axis and ticks
        set_x_tick_labels(d_len_sum, i_stat_nos, i_station_skip, b_supr_out)

        % Label the y-axis and set limits
        ylabel('Mode amplitude');
        ylim([-1.1 1.1])
        set(gca, 'ytick', -1:0.5:1);

        % Setup the title
        str_title = ['Mode no. ' num2str(idx_mode-(~b_supr_degen))];
        t = title(str_title);
        set(t, 'horizontalAlignment', 'left')
        set(t, 'units', 'normalized')
        h1 = get(t, 'position');
        set(t, 'position', [0 h1(2) h1(3)])

        % Create the mode frequency labels in hertz and CPM
        str_uncert_hz = '      ';
        if ( i_seq_steps > 0 )
            str_uncert_hz = [' ± ' num2str(d_ft_delR_plot(idx_mode)/(2*pi), '%5.3f') ' Hz      '];
        end
        str_uncert_cpm = '';
        if ( i_seq_steps > 0 )
            str_uncert_cpm = [' ± ' num2str(d_ft_delR_plot(idx_mode)*(30/pi), '%5.1f') ' CPM'];
        end
        str_mode_freq = [ num2str( d_ft_plot(idx_mode)/(2*pi), '%4.2f') ' Hz'...
            str_uncert_hz...
            num2str( d_ft_plot(idx_mode)*(30/pi), '%4.1f') ' CPM'...
            str_uncert_cpm];
        tx = text(1, h1(2)+0.05, str_mode_freq, 'Units','Normalized',...
            'FontWeight', 'bold');
        set(tx, 'horizontalAlignment', 'right')

        % Group title
        if idx_mode == 1
            tg = text(0.5, h1(2)+0.20, 'Normalize Mode Shapes (Eigenvectors)',...
                'Units','Normalized');
            set(tg, 'horizontalAlignment', 'center', 'FontSize',11,...
                'FontWeight','bold')
        end

    end

    if length(str_plot_file) > 1
        set(h_mode_shapes, 'PaperOrientation', 'portrait');
        set(h_mode_shapes,'PaperUnits','normalized');
        set(h_mode_shapes,'PaperPosition', [0 0 0.8 0.5]);
        saveas(h_mode_shapes, [str_plot_file '_mode_shapes'], 'pdf')
        saveas(h_mode_shapes, [str_plot_file '_mode_shapes'], 'png')
    end

else
    h_mode_shapes = [];
end

return

% ABOUT
% This function constructs the A matrix
function d_A = FormA(d_MoIp, d_kt, d_kt_ext, i_stat_nos, b_supr_out)

% Form the stiffness matrix
d_MoIp_diag = diag(d_MoIp);

d_stiffness = zeros(i_stat_nos, i_stat_nos);

for i_row_index=1:i_stat_nos

    for iColIndex=1:i_stat_nos

        if( iColIndex == 1 && i_row_index == 1 )
            d_stiffness(i_row_index, iColIndex) = d_kt(1) + d_kt_ext(1);
        else

            if( iColIndex == i_row_index )
                % Diagonal terms
                d_stiffness(i_row_index, iColIndex) = (d_kt( i_row_index - 1 ) +...
                    d_kt( i_row_index ) + d_kt_ext(i_row_index) );
            end

            if( ( i_row_index + 1 ) == iColIndex )

                % Off-diagonal terms
                d_stiffness(i_row_index, iColIndex) = -d_kt(i_row_index);
                d_stiffness(iColIndex, i_row_index) = -d_kt(i_row_index);
            end


        end

    end

end

% Display the stiffness matrix
if( ~b_supr_out)
    disp('Stiffness')
    disp(d_stiffness)
end

% Create the matrix
d_A=-d_MoIp_diag\d_stiffness;

return

% Helper function to get the x-axis ticks and labels in the right spot
function set_x_tick_labels(L_sum, N, i_station_skip, b_supr_out)

% Label the x-axis and ticks
xlabel('Station Number');
d_offset = L_sum(end)*0.05;
xlim([L_sum(1)-d_offset L_sum(end)+d_offset]);
if( ~b_supr_out)
    disp(L_sum)
end
set(gca,'XTick',L_sum);
str_x_tick_label = string(1:N);
str_x_tick_label(1:1:N) = ' ';
str_x_tick_label(i_station_skip:i_station_skip:N) = string(i_station_skip:i_station_skip:N);
set(gca,'XTickLabels', str_x_tick_label)

return