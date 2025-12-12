# ChargeFabrica
A Python-based Finite Difference Multidimensional Electro-Ionic Drift Diffusion Simulator for Perovskite Solar Cells 

[<img width="300" height="36" alt="ChargeFabricaDOI" src="https://github.com/user-attachments/assets/15c0cebb-f51d-4c34-b045-e77347802357" />](https://iopscience.iop.org/article/10.1088/2752-5724/ae27e9)

Authors: Tristan Sachsenweger, Miguel A. Torre Cachafeiro, Wolfgang Tress

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [QuickStart](#quickstart)
4. [Computation Time](#computation-time)
5. [Units and Formatting](#units-and-formatting)
6. [Numerics and Damping](#numerics-and-damping)
7. [How to cite](#how-to-cite)

## Introduction
ChargeFabrica uses [fipy](https://github.com/usnistgov/fipy) to solve the semiconductor equations in 1D, 2D or 3D, thereby determining the electrostatic potential, charge density distributions for electrons, holes and mobile ions and the resulting current-voltage relationships. Furthermore, Beer–Lambert generation, various recombination mechanisms, PL Yield, external quantum efficiency (EQE), spatial collection efficiency (SCE) and ion preconditioning can be modelled. The solver is designed to handle arbitrary semiconductor geometries, which can be defined within a numpy array.

### Example Problems:
<div class="grid cards" markdown>

-   1D Simulation plot for FTO (Boundary)|TiO2 (50 nm)|MAPbI3 (1600 nm)|Carbon (Boundary) Cell

    ---
    
    <img src="https://github.com/user-attachments/assets/b30c3bcc-a48d-4661-9de1-8ffa65a27a80" width="400">

- 1D Simulation plots for EQE and SCE for NIP cells:
  
  ---
  <img height="250" alt="image" src="https://github.com/user-attachments/assets/6e9a66e9-aa1e-49d2-8d65-5dbf7aff0128" />
  <img height="250" alt="image" src="https://github.com/user-attachments/assets/4594abeb-013a-4853-aa4d-bb2e6532e6bb" />

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

## Units and Formatting
For the semiconductors, the units are defined as follows:

name: uniquename

GenRate: prefactor in suns (unitless)

epsilon: relative permittivity (unitless)

pmob: hole mobility (m^2/Vs)

nmob: electron mobility (m^2/Vs)

Eg: band gap (eV)

chi: electron affinity (eV)

cationmob: cation mobility (m^2/Vs)

anionmob: anion mobility (m^2/Vs)

Recombination_Langevin: prefactor to enable Langevin recombination (unitless)

Recombination_Bimolecular: bimolecular recombination prefactor (m^3/s)

Nc: effective density of states in the conduction band (1/m^3)

Nv: effective density of states in the valence band (1/m^3)

Chi_a: anionic energetic offset (eV)

Chi_c: cationic energetic offset (eV)

a_initial_level: initial mobile anion density (1/m^3)

c_initial_level: initial mobile cation density (1/m^3)

Nd: donor density (1/m^3)

Na: acceptor density (1/m^3)

For the electrodes, the work function must be provided (in eV)

## Numerics and Damping
Drift-Diffusion problems can be quite challenging to solve numerically, depending on the material parameters used and the geometry employed.

If the desired residual isn't achieved due to frequent residual instabilities, then the DampingFactor ratio must be decreased. This comes at a cost of decreasing the effective time step, which often requires increasing the number of iterations necessary for convergence.

The Newton method can greatly accelerate convergence with the correct DampingFactor. However, it is more susceptible to instabilities than the Gummel method if the wrong DampingFactor is chosen (The Gummel method is more tolerant in this regard). Furthermore, for some problems, high bias voltages cannot be directly solved using Newton's method without a good initial guess, which can be fixed by reducing the multiprocessing chunk_size variable to (4-8), thereby allowing the bias voltage to be slowly stepped up with the previous highest-bias solution acting as the initial guess for the next chunk.

**Note:** The residual may increase briefly for certain problems as the timestep is being dynamically increased. This usually does not require an adjustment of the DampingFactor.
For very stiff problems, it may be necessary to sweep the Poisson and electronic continuity equations multiple times per time step. However, the computational overhead of sweeping is very significant, and it is usually better to adjust the DampingFactor.

## How to cite
ChargeFabrica has been published as:
@article{10.1088/2752-5724/ae27e9,
	author={Sachsenweger, Tristan and A. Torre Cachafeiro, Miguel and Tress, Wolfgang},
	title={ChargeFabrica: A Python-based Finite Difference Multidimensional Electro-Ionic Drift Diffusion Simulator applied to Mesoporous Perovskite Solar Cells},
	journal={Materials Futures},
	url={http://iopscience.iop.org/article/10.1088/2752-5724/ae27e9},
	year={2025}
}




