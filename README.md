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

ChargeFabrica requires the following packages to be installed: numpy, scipy, fipy (ideally with pysparse for optimum performance), joblib, multiprocessing, pandas, opencv, matplotlib

To further enhance 2D pysparse performance (especially if elements >100k), it is best to disable partial pivoting by navigating to the pysparse LU solver file:
(For miniforge this would be ./envs/(environment name)/lib/python2.7/site-packages/fipy/solvers/pysparse/linearLUsolver.py)
and then replacing the code line "LU = superlu.factorize(L.matrix.to_csr())" with "superlu.factorize(L.matrix.to_csr(), diag_pivot_thresh=0)".
