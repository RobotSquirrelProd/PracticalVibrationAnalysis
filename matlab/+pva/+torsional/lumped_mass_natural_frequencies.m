function [frequencies_hz, mode_shapes, M, K] = lumped_mass_natural_frequencies(inertias, shafts, rigidBodyToleranceHz)
%LUMPED_MASS_NATURAL_FREQUENCIES Compute torsional natural frequencies.
%
% [frequencies_hz, mode_shapes, M, K] = pva.torsional.lumped_mass_natural_frequencies(...)
% solves K*phi = omega^2*M*phi for a lumped-mass torsional model.
%
% Inputs:
%   inertias              - Nx1 or 1xN inertias [kg*m^2], positive values.
%   shafts                - Struct array with fields:
%                           .node_i (1-based index), .node_j, .stiffness [N*m/rad]
%   rigidBodyToleranceHz  - Optional frequency cutoff for filtering rigid-body modes.
%
% Outputs:
%   frequencies_hz        - Column vector of natural frequencies [Hz], ascending.
%   mode_shapes           - Corresponding mode shape matrix.
%   M, K                  - Mass and stiffness matrices.
%
% Example:
%   inertias = [0.1, 0.2, 0.15];
%   shafts = struct('node_i',{1,2}, 'node_j',{2,3}, 'stiffness',{1200,900});
%   [f, phi] = pva.torsional.lumped_mass_natural_frequencies(inertias, shafts);

    if nargin < 3 || isempty(rigidBodyToleranceHz)
        rigidBodyToleranceHz = 1e-6;
    end

    inertias = inertias(:);
    if isempty(inertias) || any(inertias <= 0)
        error('inertias must be a non-empty vector of positive values.');
    end

    n = numel(inertias);
    M = diag(inertias);
    K = zeros(n, n);

    for idx = 1:numel(shafts)
        i = shafts(idx).node_i;
        j = shafts(idx).node_j;
        k = shafts(idx).stiffness;

        if i == j
            error('shaft node_i and node_j must be different.');
        end
        if i < 1 || j < 1 || i > n || j > n
            error('shaft node indices must be within inertia vector bounds.');
        end
        if k <= 0
            error('shaft stiffness must be positive.');
        end

        K(i, i) = K(i, i) + k;
        K(j, j) = K(j, j) + k;
        K(i, j) = K(i, j) - k;
        K(j, i) = K(j, i) - k;
    end

    [mode_shapes, D] = eig(K, M);
    omega2 = max(real(diag(D)), 0.0);
    frequencies_hz = sqrt(omega2) / (2*pi);

    keep = frequencies_hz > rigidBodyToleranceHz;
    frequencies_hz = frequencies_hz(keep);
    mode_shapes = mode_shapes(:, keep);

    [frequencies_hz, order] = sort(frequencies_hz, 'ascend');
    mode_shapes = mode_shapes(:, order);
end
