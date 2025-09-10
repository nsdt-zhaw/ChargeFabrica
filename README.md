# ChargeFabrica
A Python-based Finite Difference Multi-dimensional Electro-ionic Drift Diffusion Simulator for Perovskite Solar Cells 

<img src="https://github.com/user-attachments/assets/8fefefaf-dfd4-4be8-8c48-8add559b91da" width="200" alt="DOI" data-canonical-src="" style="max-width: 1%;">

##### Table of Contents  
[installation](#installation)  
[quickstart guide](#quickstart)  
This Python-based solver uses [fipy](https://github.com/usnistgov/fipy) to solve the semiconductor equations.

<div class="grid cards" markdown>

-   1D Simulation plot for FTO (Boundary)|TiO2 (50 nm)|MAPbI3 (1600 nm)|Carbon (Boundary) Cell

    ---
    <img src="https://github.com/user-attachments/assets/b30c3bcc-a48d-4661-9de1-8ffa65a27a80" width="400">

-   2D Simulation plot for carbon-based triple mesoscopic HTL-free device: FTO (Boundary)|TiO2 (50 nm)|m-TiO2/MAPbI3 (150 nm)|m-ZrO2/MAPbI3 (1000 nm)|MAPbI3 (100 nm)|Carbon (Boundary) Cell

    ---
    <img src="https://github.com/user-attachments/assets/24107cfe-7f70-4a8e-926a-eea8055bd6e9" width="400">

</div>

<a name="installation"/>
## Installation
The code has been tested on [miniforge](https://github.com/conda-forge/miniforge) with Python2.7 installed, which is necessary to be compatible with [pysparse](https://github.com/PythonOptimizers/pysparse).

ChargeFabrica requires the following packages to be installed: [numpy](https://github.com/numpy/numpy), [scipy](https://github.com/scipy/scipy), [fipy](https://github.com/usnistgov/fipy) (ideally with [pysparse](https://github.com/PythonOptimizers/pysparse) for optimum performance), [joblib](https://github.com/joblib/joblib), [pandas](https://github.com/pandas-dev/pandas), [xlrd](https://github.com/python-excel/xlrd), [matplotlib](https://github.com/matplotlib/matplotlib)

To further enhance 2D pysparse performance (especially if elements >100k), it is best to disable partial pivoting by navigating to the pysparse LU solver file:
(For miniforge this would be (miniforge install directory)/envs/(environment name)/lib/python2.7/site-packages/fipy/solvers/pysparse/linearLUsolver.py)
and then replacing the code line "LU = superlu.factorize(L.matrix.to_csr())" with "superlu.factorize(L.matrix.to_csr(), diag_pivot_thresh=0)".

# Computation Time
The 1D compute time with ions enabled on a Intel(R) Core(TM) i9-12900 desktop PC is roughly 1-2 minutes.

The 2D compute time with ions enabled on a dedicated server with AMD EPYC 74F3 processor for ~100k elements is roughly 2 hours.

It is therefore **strongly** recommended to test the code in 1D before moving to 2D.

# Numerics and Damping
Drift-Diffusion problems can be quite challenging to solve numerically, depending on the material parameters used and the geometry employed.

If the desired residual isn't achieved due to frequent residual instabilities, then the DampingFactor ratio must be decreased. This comes at a cost of decreasing the effective time step, which often requires increasing the number of iterations necessary for convergence.

**Note:** The residual may increase briefly for certain problems as the timestep is being dynamically increased. This usually does not require an adjustment of the DampingFactor.
For very stiff problems, it may be necessary to sweep the Poisson and electronic continuity equations multiple times per time step. However, the computational overhead of sweeping is very significant, and it is usually better to adjust the DampingFactor.




