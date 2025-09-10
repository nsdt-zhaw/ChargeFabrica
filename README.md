# ChargeFabrica
A Python-based Finite Difference Multi-dimensional Electro-ionic Drift Diffusion Simulator for Perovskite Solar Cells 

<img src="https://github.com/user-attachments/assets/8fefefaf-dfd4-4be8-8c48-8add559b91da" width="200" alt="DOI" data-canonical-src="" style="max-width: 1%;">

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [QuickStart](#quickstart)
4. [Computation Time](#computation-time)
5. [Numerics and Damping](#numerics-and-damping)
6. [How to cite](#how-to-cite)

## Introduction
ChargeFabrica uses [fipy](https://github.com/usnistgov/fipy) to solve the semiconductor equations in 1D and 2D, thereby determining the electrostatic potential, charge density distributions for electrons, holes and mobile ions and the resulting current-voltage relationships. Furthermore, Beer–Lambert generation, various recombination mechanisms, and PL Yield can be modelled. The 2D solver is designed to handle arbitrary semiconductor geometries which can be defined within a numpy array.
### Example Problems:
<div class="grid cards" markdown>

-   1D Simulation plot for FTO (Boundary)|TiO2 (50 nm)|MAPbI3 (1600 nm)|Carbon (Boundary) Cell

    ---
    <img src="https://github.com/user-attachments/assets/b30c3bcc-a48d-4661-9de1-8ffa65a27a80" width="400">

-   2D Simulation plot for carbon-based triple mesoscopic HTL-free device: FTO (Boundary)|TiO2 (50 nm)|m-TiO2/MAPbI3 (150 nm)|m-ZrO2/MAPbI3 (1000 nm)|MAPbI3 (100 nm)|Carbon (Boundary) Cell

    ---
    <img src="https://github.com/user-attachments/assets/24107cfe-7f70-4a8e-926a-eea8055bd6e9" width="400">

</div>

## Installation
ChargeFabrica requires the following packages to be installed: [numpy](https://github.com/numpy/numpy), [scipy](https://github.com/scipy/scipy), [fipy](https://github.com/usnistgov/fipy), [joblib](https://github.com/joblib/joblib), [pandas](https://github.com/pandas-dev/pandas), [xlrd](https://github.com/python-excel/xlrd), [matplotlib](https://github.com/matplotlib/matplotlib)

The prerequisite packages can be installed using pip: 
```console
pip install numpy scipy fipy pandas joblib xlrd matplotlib
```
The ChargeFabrica repo can then be cloned using the command:
```console
git clone https://github.com/nsdt-zhaw/ChargeFabrica.git
```
## QuickStart
It is recommended to start with the script [Drift_Diffusion_1D_IV_IONS_NIP_Example.py](Drift_Diffusion_1D_IV_IONS_NIP_Example.py) by executing it.

Once the simulation is completed, the results are saved as .npy files in the ./Outputs folder

The results can then be plotted using the [PlottingResults1D.py](PlottingResults1D.py) script.

## Computation Time
The 1D compute time with ions enabled on a Intel(R) Core(TM) i9-12900 desktop PC is roughly 1-2 minutes.

The 2D compute time with ions enabled on a dedicated server with AMD EPYC 74F3 processor for ~100k elements is roughly 2 hours.

It is therefore **strongly** recommended to test the code in 1D before moving to 2D.

## Numerics and Damping
Drift-Diffusion problems can be quite challenging to solve numerically, depending on the material parameters used and the geometry employed.

If the desired residual isn't achieved due to frequent residual instabilities, then the DampingFactor ratio must be decreased. This comes at a cost of decreasing the effective time step, which often requires increasing the number of iterations necessary for convergence.

**Note:** The residual may increase briefly for certain problems as the timestep is being dynamically increased. This usually does not require an adjustment of the DampingFactor.
For very stiff problems, it may be necessary to sweep the Poisson and electronic continuity equations multiple times per time step. However, the computational overhead of sweeping is very significant, and it is usually better to adjust the DampingFactor.

## How to cite
ChargeFabrica has been published as:

A Python-based Finite Difference Multi-dimensional Electro-ionic Drift Diffusion Simulator for Perovskite Solar Cells




