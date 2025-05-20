# ChargeFabrica
A Finite Difference Multi-dimensional Electro-ionic Drift Diffusion Simulator for Perovskite Solar Cells

<div class="grid cards" markdown>

-   1D Simulation plot for FTO (Boundary)|TiO2 (50 nm)|MAPbI3 (1600 nm)|Carbon (Boundary) Cell

    ---
    <img src="https://github.com/user-attachments/assets/b30c3bcc-a48d-4661-9de1-8ffa65a27a80" width="400">

-   2D Simulation plot for carbon-based triple mesoscopic HTL-free device: FTO (Boundary)|TiO2 (50 nm)|m-TiO2/MAPbI3 (150 nm)|m-ZrO2/MAPbI3 (1000 nm)|MAPbI3 (100 nm)|Carbon (Boundary) Cell

    ---
    <img src="https://github.com/user-attachments/assets/24107cfe-7f70-4a8e-926a-eea8055bd6e9" width="400">

</div>

# Installation
The code has been tested on [miniforge](https://github.com/conda-forge/miniforge) with Python2.7 installed, which is necessary to be compatible with [pysparse](https://github.com/PythonOptimizers/pysparse).

ChargeFabrica requires the following packages to be installed: numpy, scipy, fipy (ideally with pysparse for optimum performance), joblib, pandas, xlrd, opencv, matplotlib

To further enhance 2D pysparse performance (especially if elements >100k), it is best to disable partial pivoting by navigating to the pysparse LU solver file:
(For miniforge this would be (miniforge install directory)/envs/(environment name)/lib/python2.7/site-packages/fipy/solvers/pysparse/linearLUsolver.py)
and then replacing the code line "LU = superlu.factorize(L.matrix.to_csr())" with "superlu.factorize(L.matrix.to_csr(), diag_pivot_thresh=0)".

# Computation Time
The 1D compute time with ions enabled on a Intel(R) Core(TM) i9-12900 desktop PC is roughly 1-2 minutes.

The 2D compute time for ~100k elements is roughly 2 hours on a dedicated server with AMD EPYC 74F3 processor.

It is therefore **strongly** recommended to test the code in 1D before moving to 2D.
