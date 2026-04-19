# AppliedVibrationAnalysis

## Introduction

Computation tools for looking at real-world signals for [applied vibration analysis](https://robotsquirrelproductions.com/applied-vibration-analysis/):

![Vibration analysis tools](https://robotsquirrelproductions.com/wp-content/uploads/2026/04/VibAnalysisProcess_2k_wht.png)

As of this writing, the package is a work-in-progress and heavily in development. Structure and syntax will likely change prior to a formal release.

If you do scare up the courage to install this and try it, I would love your feedback, please log issues and concerns here in the GitHub repository.

## Repository Layout

- `python/vibration_analysis/`: Maintained Python package source.
- `python/tests/`: Primary automated test suite.
- `python/tests/legacy/`: Legacy tests that exercise older modules and hardware integrations.
- `python/tests/data/`: Test fixture data files used by test modules.
- `matlab/`: MATLAB vibration analysis functions and test harnesses.
- `notebooks/`: Exploratory notebooks and test harness notebooks.
- `scripts/`: Local helper scripts and launchers.
- `docs/architecture/`: Architecture and UML artifacts.
- `docs/figures/`: Generated figures and plot exports.

## MATLAB Torsional Vibration Analysis
The `matlab/` folder contains the reference torsional-analysis implementation used for development and validation.

### Functions

- `CalcAngleFromStress.m`
    - Purpose: Calculates shaft twist angle from stress, geometry, and shear modulus.
    - Core relationship: `phi = (L * tau_max) / (G * r_o)`
    - Inputs: outer radius (`d_ro`), section length (`d_L`), max shear stress (`d_taumax`, scalar or vector), shear modulus (`d_G`).
    - Output: angle of twist in radians (`d_phi`).

- `CalcFreeFreeTorsResp.m`
    - Purpose: Computes free-free torsional natural frequencies and mode shapes from a lumped inertia/stiffness model.
    - Handles: optional damping, external stiffness to ground, observation vector, gear-ratio-based model collapse/reflection, and plotting controls.
    - Outputs include mode-shape figure handle, eigenvectors, natural frequencies (CPM), mass-elastic diagram handle, and collapsed model vectors.

- `CalcForcedTorsResp.m`
    - Purpose: Extends the free-free model into a forced response formulation by assembling a rotor state-space model.
    - Depends on: `CalcFreeFreeTorsResp.m` (for model preprocessing and modal data).
    - Outputs include state-space model (`ss_rotor`) in addition to modal/plot outputs.

### Typical MATLAB Workflow

1. Build a torsional model (vectors for inertia and stiffness, optionally damping and gear ratio).
2. Run `CalcFreeFreeTorsResp` to inspect natural frequencies and mode shapes.
3. Run `CalcForcedTorsResp` to build a forced-response state-space model for transfer-function/Bode workflows.
4. Use `CalcAngleFromStress` for section-level twist calculations from stress inputs.

### Quick Start Example

```matlab
% 3-mass free-free model
d_MoIp = [1, 2, 3];
d_kt = [1, 2, 0];
d_len = ones(size(d_kt));

[h_modes, eigvec, f_cpm, h_mass] = CalcFreeFreeTorsResp(d_MoIp, d_kt, d_len);

% Add damping and build forced-response state-space model
d_damp_int = [1e-2, 2e-2, 0];
d_damp_ext = [1e-10, 2e-10, 3e-10];
d_obs = [1, 1, 1, 0, 0, 0];

[~, ~, ~, ~, ss_rotor] = CalcForcedTorsResp(d_MoIp, d_kt, d_len, d_damp_int, d_damp_ext, d_obs);
```

### Running MATLAB Unit Tests

From MATLAB in the `matlab/` folder:

```matlab
table(runtests("CalcAngleFromStress_Tests.m"))
table(runtests("CalcFreeFreeTorsResp_Tests.m"))
table(runtests("CalcForcedTorsResp_Tests.m"))
```

### Dependencies

- MATLAB
- Control System Toolbox (required by the torsional response solvers)

## Python Torsional Vibration Analysis
The repository includes python code for free-free and forced torsional vibration analysis. As of Apr 2026 I am still porting over the functions from MATLAB. I have a hello world notebook with examples in the root of the repository:

`ForcedTorsionalResponse_HelloWorld.ipynb`

This code will be integrated into the applied vibration analysis, but until all tests are written and it passes all the tests the code is outside the library.

## Python Vibration Analysis Library
This section describes how to install and use the library for applied vibration analysis

### Installation

In this early stage I clone the repository to a local folder and then:

-   Create and activate the dev environment:

    `virtualenv dev_env`

    `source dev_env/bin/activate`

-   Navigate to the local folder with the cloned contents from Github:

    `cd “C:\Local Documents\Github\AppliedVibrationAnalysis”`

-   Use `pip` to install the package (note the ‘`.`’ at the end):

    `pip install -e .`

I verified these instructions in Windows 11, but it should be similar for other operating systems.

### Quick Examples

This section has some examples to get you startedd.

#### Timebase plot

A timebase plot shows an amplitude versus time waveform in a rectangular grid for one data series. This plot format can be thought of as a high-resolution trend with 1000s of points over a few hundred milliseconds. 

The repo includes a [hello world notebook](notebooks/AppliedVibrationAnalysis%20HelloWorld.ipynb) with more examples. But to get started follow these 3 steps to get the plot up:

-   Load the libraries:

```
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
import numpy as np
import math
from datetime import datetime, timedelta, timezone
from scipy import signal
import matplotlib
from appvib import ClSigFeatures
	
```	
	
-   Synthesize the signal:


```
d_fs_even = 2048
i_ns = (d_fs_even*2)`
d_freq_sig = 10./(float(i_ns)/d_fs_even)
print('Signal frequency, hertz: ' + '%0.10f' % d_freq_sig)
d_time_ext = np.linspace(0,(i_ns-1),i_ns)/float(d_fs_even)
np_d_test_even = np.sin(2 * math.pi * d_freq_sig * d_time_ext )
```
	
-   Create the plot starting with defining the data features:

```
ClSigEven = ClSigFeatures(np_d_test_even, d_fs_even,
    dt_timestamp=datetime.fromisoformat('2020-01-01T00:00:00-00:00'))
ClSigEven.str_plot_desc = "Sinusoid"
ClSigEven.str_machine_name_set("Simulated Data", idx=0)
ClSigEven.str_eu_set("g's",idx=0)

```

```
plt.rcParams['figure.figsize'] = [8, 4.5]
lst_sig_even = ClSigEven.plt_sigs()
lst_sig_even.savefig('HelloWorld_Sinusoid.pdf')
```
	
The code should produce a plot like this:

![timebase plot](https://robotsquirrelproductions.com/wp-content/uploads/2024/11/Timebase-HelloWorld_Sinusoid_2k_241127_170919.png)