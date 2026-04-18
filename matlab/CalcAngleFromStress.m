%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2026 Robot Squirrel Productions.
%
%          (\_   _/)
%          ( o   o )
%          (  ^_^  )
%             \_/_
%
% ABOUT
% Given shaft section properties and applied torque, this function returns
% the total angle of twist in radians.
% 
% The companion Mathematica worksheet ("GeometricProperties.nb") has
% pictures and defintions of the terms used in this function. This code
% follows the nomenclature defined in 2529RS0011
%
% EXAMPLE
% d_phi = CalcAngleFromStress(d_ro, d_L, d_taumax, d_G);
%
% INPUT
% d_ro (REQUIRED, POSITIVE SCALAR) - Outer radius of circular section,
%   engineering units of m or in.
% d_L (REQUIRED, NON-NEGATIVE SCALAR) - Shaft section length, engineering
%   units of m or in.
% d_taumax (REQUIRED, NUMERIC) - Torsional shear stress. If a vector then
%   the output will also be vector.
% d_G (REQUIRED, POSITIVE SCALAR) - Shear modulus of elasticity, in
%   engineering units of Pascal or psi. This function does not convert
%   units so it must match length and torque units.
%
% OUPUT
% d_phi (SCALAR) - Angle of twist, radians 
%
% B. Howard
% 20 Aug 2024
%
function d_phi = CalcAngleFromStress(d_ro, d_L, d_taumax, d_G, varargin)

% Instantiate the parser
p = inputParser;

% Validation rules
validScalarPosNum = @(x) assert(isnumeric(x) && isscalar(x) && (x > 0),...
    'Must be a positive scalar numeric quantity');

% Define the inputs
addRequired(p, 'd_ro', validScalarPosNum)
addRequired(p, 'd_L', validScalarPosNum)
addRequired(p, 'd_taumax', @isnumeric)
addRequired(p, 'd_G', validScalarPosNum)

% Parse the inputs
parse(p, d_ro, d_L, d_taumax, d_G, varargin{:});

% Bring the inputs into the function
d_ro = p.Results.d_ro;
d_L = p.Results.d_L;
d_taumax = p.Results.d_taumax(:);
d_G = p.Results.d_G;

% Perform the calculation
d_phi = (d_L .* d_taumax  ) ./ ( d_G .* d_ro );

return