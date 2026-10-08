# ChargeFabrica
A Python-based Finite Difference Multidimensional Electro-Ionic Drift Diffusion Simulator for Perovskite Solar Cells 

[<img width="300" height="36" alt="ChargeFabricaDOI" src="https://github.com/user-attachments/assets/15c0cebb-f51d-4c34-b045-e77347802357" />](https://iopscience.iop.org/article/10.1088/2752-5724/ae27e9)

Authors: Tristan Sachsenweger, Miguel A. Torre Cachafeiro, Wolfgang Tress

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [QuickStart](#quickstart)
4. [Video Tutorials](#video-tutorials)
5. [Computation Time](#computation-time-using-newton-method)
6. [Units and Formatting](#units-and-formatting)
7. [Numerics and Damping](#numerics-and-damping)
8. [How to cite](#how-to-cite)
9. [Publications using ChargeFabrica](#publications-using-chargefabrica)

## Introduction
ChargeFabrica uses [fipy](https://github.com/usnistgov/fipy) to solve the semiconductor equations in 1D, 2D or 3D, thereby determining the electrostatic potential, charge density distributions for electrons, holes and mobile ions and the resulting current-voltage relationships. Furthermore, Beer–Lambert generation, various recombination mechanisms, PL Yield, external quantum efficiency (EQE), spatial collection efficiency (SCE) and ion preconditioning can be modelled. The solver is designed to handle arbitrary semiconductor geometries, which can be defined within a numpy array.

### Example Problems:
---
<div class="grid cards" markdown>

-   1D Simulation plot for FTO (Boundary)|TiO2 (50 nm)|MAPbI3 (1600 nm)|Carbon (Boundary) Cell
-   Source file: [1D_HTL_Free_Carbon_Device_IONS_Newton_Example.py](1D_HTL_Free_Carbon_Device_IONS_Newton_Example.py)
<img src="https://github.com/user-attachments/assets/b30c3bcc-a48d-4661-9de1-8ffa65a27a80" width="400">

---

-  1D Simulation plots for EQE and SCE for NIP cells:
-  Source files: [1D_NIP_EQE.py](1D_NIP_EQE.py) & [1D_NIP_SCE_Example.py](1D_NIP_SCE_Example.py)
  <img height="250" alt="image" src="https://github.com/user-attachments/assets/6e9a66e9-aa1e-49d2-8d65-5dbf7aff0128" />
  <img height="250" alt="image" src="https://github.com/user-attachments/assets/4594abeb-013a-4853-aa4d-bb2e6532e6bb" />
  
---

-   2D Simulation plot for carbon-based triple mesoscopic HTL-free device: FTO (Boundary)|TiO2 (50 nm)|m-TiO2/MAPbI3 (150 nm)|m-ZrO2/MAPbI3 (1000 nm)|MAPbI3 (100 nm)|Carbon (Boundary) Cell
-	Source file: [2D_HTL_Free_Carbon_Device_IONS_Newton_Example.py](2D_HTL_Free_Carbon_Device_IONS_Newton_Example.py)
<img src="https://github.com/user-attachments/assets/24107cfe-7f70-4a8e-926a-eea8055bd6e9" width="400">

---

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
It is recommended to start with the script [1D_IONS_NIP_Example.py](1D_IONS_NIP_Example.py) by executing it.

Once the simulation is completed, the results are saved as .npy files in the ./Outputs folder

The results can then be plotted using the [PlottingResults1D.py](PlottingResults1D.py) script.

## Video Tutorials
Here we provide a list of instructional videos to help new users get familiar with the program:

1  [Installation and QuickStart Tutorial](https://www.youtube.com/watch?v=Io8mPTLpUPw)

## Computation Time using Newton method
The 1D compute time with ions enabled on a Intel(R) Core(TM) i9-12900 desktop PC is roughly 30 seconds.

The 2D compute time with ions enabled on a dedicated server with AMD EPYC 74F3 processor for ~100k elements is roughly 30 minutes.

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

## Publications using ChargeFabrica
Ji, F., Sachsenweger Ballantyne, T., Meraji, K. et al. Simultaneous optimization of film morphology and structure dimensionality in Cs3Sb2I9 through butylamine gas treatment. Commun Mater (2026). https://doi.org/10.1038/s43246-026-01240-8

## 1. Physical unknowns and conventions

At each applied voltage, the solver seeks stationary fields

$$
\mathbf u=(\phi,n,p,a,c)^T.
$$

Here $\phi$ is the electrostatic potential, $n$ and $p$ are electron and hole
densities, and $a$ and $c$ are singly negative and positive ionic densities.
Redistribution of these charges changes the potential; the potential changes
transport, which feeds back into the charge distribution.

| Symbol | Meaning and units | Code |
| --- | --- | --- |
| $\phi$ | Electrostatic potential, V | `philocal` |
| $n,p,a,c$ | Particle densities, m⁻³ | `nlocal`, `plocal`, `alocal`, `clocal` |
| $q$ | Positive elementary charge, C | `q` in `constantsfile.py` |
| $\epsilon_r$ | Relative permittivity | `epsilon` |
| $\epsilon_0$ | Vacuum permittivity, F/m | `epsilon_0` |
| $\mu_s$ | Species mobility, m²/(V s) | `nmob`, `pmob`, `anionmob`, `cationmob` |
| $V_T=k_BT/q$ | Thermal voltage, V | **`D`** in `constantsfile.py` |
| $N_c,N_v$ | Conduction/valence densities of states, m⁻³ | `Nc`, `Nv`, `LogNcCell`, `LogNvCell` |
| $\chi,E_g$ | Band parameters in the code's numerical eV/V convention | `ChiCell`, `EgCell` |
| $G,R$ | Pair generation/net recombination, m⁻³ s⁻¹ | `gen_rate`, `Recombination_Combined` |

The code variable **`D` is the thermal voltage**, not a diffusion coefficient.
The Einstein relation gives $\mathcal D_s=\mu_sV_T$. FiPy continuity equations
are charge-weighted, so their diffusion coefficients also contain $q$.

The model is isothermal and uses nondegenerate semiconductor statistics.
Material maps supply band parameters, doping, permittivity, mobilities and
recombination parameters. `flatten_and_smooth_all()` smooths selected maps
before equation assembly; its settings affect the represented interfaces.

## 2. Stationary equations and their implementation

### Poisson equation

The charge density and electric field are

$$
\rho=q(p-n+c-a+N_D-N_A),\qquad \mathbf E=-\nabla\phi.
$$

Using relative permittivity, Poisson's equation is

$$
\nabla\cdot(\epsilon_r\nabla\phi)
+\frac{q}{\epsilon_0}(p-n+c-a+N_D-N_A)=0.
$$

`eqpoisson` implements this charge sum using `plocal`, `nlocal`, `clocal`,
`alocal`, `NdCell` and `NaCell`. Electrons and anions contribute negative
charge; holes and cations contribute positive charge. Its `TransientTerm`
is an artificial relaxation term, discussed in Section 5; it vanishes in
the stationary physical target.

### Electronic transport

Define the effective transport potentials

$$
U_n=\phi+\chi+V_T\ln N_c,\qquad
U_p=\phi+\chi+E_g-V_T\ln N_v.
$$

Only gradients of these logarithms enter transport; a fixed density reference
is implicit when taking logarithms of dimensional densities of states.
Including the band and density-of-states gradients matters at material
interfaces.

Using **conventional charge-current** signs,

$$
\mathbf J_n=q\mu_n(V_T\nabla n-n\nabla U_n),\qquad
\mathbf J_p=-q\mu_p(V_T\nabla p+p\nabla U_p).
$$

The stationary electronic continuity equations are

$$
0=\nabla\cdot\mathbf J_n+q(G-R),\qquad
0=-\nabla\cdot\mathbf J_p+q(G-R).
$$

`eqn` and `eqp` contain these terms plus pseudo-time storage. For example,
the electron equation is assembled as

```python
eqn = (
    TransientTerm(coeff=q, var=nlocal)
    == DiffusionTerm(coeff=q * D * nmob.harmonicFaceValue, var=nlocal)
    - ExponentialConvectionTerm(
        coeff=q * nmob.harmonicFaceValue
              * (philocal.faceGrad + ChiCell.faceGrad + D * LogNcCell.faceGrad),
        var=nlocal)
    + q * gen_rate - q * Recombination_Combined
)
```

The hole equation uses the opposite convection sign and includes
`EgCell.faceGrad - D * LogNvCell.faceGrad`. Electron particle flux is
$-\mathbf J_n/q$, whereas hole particle flux is $\mathbf J_p/q$.

### Ionic transport and inventory

For the negative and positive ionic species, define

$$
U_a=\phi+\chi_a,\qquad U_c=\phi+\chi_c.
$$

In this code convention their particle fluxes are

$$
\mathbf N_a=-\mu_aV_T\nabla a+\mu_a a\nabla U_a,\qquad
\mathbf N_c=-\mu_cV_T\nabla c-\mu_c c\nabla U_c.
$$

There are no ionic reaction sources in this example. The stationary equations
are $\nabla\cdot\mathbf N_a=0$ and $\nabla\cdot\mathbf N_c=0$:

$$
0=\nabla\cdot(q\mu_aV_T\nabla a)
-\nabla\cdot(q\mu_a a\nabla U_a),
$$

$$
0=\nabla\cdot(q\mu_cV_T\nabla c)
+\nabla\cdot(q\mu_c c\nabla U_c).
$$

These are `eqa` and `eqc`. `ChiCell_a` and `ChiCell_c` allow species-specific
energy gradients. Their charge currents are $\mathbf J_a=-q\mathbf N_a$ and
$\mathbf J_c=q\mathbf N_c$.

Natural no-flux boundaries block ionic exchange with the exterior. Zero face
mobility can also isolate regions. Each connected blocking region has a
conserved inventory:

$$
I_a=\int_{\Omega_a}a\,dV,\qquad I_c=\int_{\Omega_c}c\,dV.
$$

The initial ionic levels select these inventories. Pseudo-time continuation
retains them rather than solving an unconstrained, singular stationary ion
system from scratch. Different disconnected regions require their own
inventories. Conservation should be checked when changing boundaries or maps.

### Optical generation and recombination

The example calculates optical generation from its absorption spectrum and
solar illumination using `calculate_absorption_above_bandgap()`. The resulting
spatial generation profile is supplied through `gen_rate`.

The implemented recombination sum is

$$
R=R_{\mathrm{rad}}+R_{\mathrm{SRH,bulk}}+R_{\mathrm{SRH,interface}},\qquad
R_{\mathrm{rad}}=B(np-n_i^2),
$$

$$
n_i^2=N_cN_v\exp(-E_g/V_T).
$$

For each SRH contribution, including its spatial activation factor $Z$,

$$
R_{\mathrm{SRH}}=Z\frac{np-n_i^2}{H},\qquad
H=\tau_p(n+n_1)+\tau_n(p+p_1).
$$

`n_hat`, `p_hat`, `n_hat_mixed` and `p_hat_mixed` specify the bulk and
interfacial trap populations. The interfacial term is represented in a marked
cell region, not as a separate boundary equation.

The exact fixed-temperature derivatives are

$$
\frac{\partial R_{\mathrm{SRH}}}{\partial n}
=Z\frac{pH-\tau_p(np-n_i^2)}{H^2},\qquad
\frac{\partial R_{\mathrm{SRH}}}{\partial p}
=Z\frac{nH-\tau_n(np-n_i^2)}{H^2}.
$$

`srh_rate_and_carrier_derivatives()` in
[electrical_numerics.py](electrical_numerics.py) supplies these expressions.
`net_dR_dn` and `net_dR_dp` add the radiative derivatives $Bp$ and $Bn$ and
both enabled SRH contributions.

Inspect `Recombination_Combined` to identify the mechanisms actually used.
For example, the script defines a Langevin rate, but it is not included in
the current recombination sum.

### Ohmic contacts

[BoundaryConditions.py](BoundaryConditions.py), function `ohmic()`, calculates
contact carrier densities from the adjacent semiconductor and metal work
function $W$:

$$
n_{\mathrm{contact}}=N_c\exp[(\chi-W)/V_T],\qquad
p_{\mathrm{contact}}=N_v\exp[(W-\chi-E_g)/V_T].
$$

`contact_bcs` fixes the top potential to zero and the bottom potential to
$-(V_{\mathrm{bi}}-V_{\mathrm{app}})$. The electronic densities are constrained
at both contacts. The corresponding Newton corrections are zero there because
the boundary values are fixed during each voltage-point solve.

## 3. The fully coupled Newton iteration

### Residual and linear system

The discretized equations form a nonlinear residual vector

$$
\mathbf F(\mathbf u)=0.
$$

At Newton iteration $m$, solve

$$
\mathbf J(\mathbf u^{(m)})\delta\mathbf u=-\mathbf F(\mathbf u^{(m)}),
\qquad \mathbf J=\frac{\partial\mathbf F}{\partial\mathbf u}.
$$

The code implements this step by solving directly for candidate physical fields
$\mathbf u^*$:

$$
\mathbf J(\mathbf u^{(m)})\mathbf u^*
=\mathbf J(\mathbf u^{(m)})\mathbf u^{(m)}-\mathbf F(\mathbf u^{(m)}),
\qquad \delta\mathbf u=\mathbf u^*-\mathbf u^{(m)}.
$$

There are five unknowns per cell, so $N$ cells give a $5N\times5N$ system.
The field order is `(philocal, nlocal, plocal, alocal, clocal)`. There are no
separate FiPy correction variables: `_solve_coupled()` saves the current values,
solves for the candidate fields, computes the NumPy array `deltas`, and restores
the current values before applying damping. The mathematical Newton update
$\delta\mathbf u$ is still needed; it is represented by this array.

The Jacobian has the structure

$$
\begin{pmatrix}
J_{\phi\phi}&J_{\phi n}&J_{\phi p}&J_{\phi a}&J_{\phi c}\\
J_{n\phi}&J_{nn}&J_{np}&0&0\\
J_{p\phi}&J_{pn}&J_{pp}&0&0\\
J_{a\phi}&0&0&J_{aa}&0\\
J_{c\phi}&0&0&0&J_{cc}
\end{pmatrix}.
$$

| Block | Meaning | Code |
| --- | --- | --- |
| $J_{\phi\phi}$ | Dielectric Laplacian and potential pseudo-time regularization | `eqpoisson` |
| $J_{\phi s}$ | Signed charge response to each density update | `CoupledChargeTerm` in `eqpoisson` |
| $J_{nn},J_{pp}$ | Storage, fixed-potential transport and self-recombination derivative | `eqn`, `eqp`, `recombination_derivative()` |
| $J_{np},J_{pn}$ | Cross-carrier recombination derivatives | `net_dR_dp`, `net_dR_dn` |
| $J_{s\phi}$ | Potential derivative of the fitted transport flux | `responses`, `update_transport_response()` and potential diffusion blocks |
| $J_{aa},J_{cc}$ | Ionic storage and fixed-potential transport | `eqa`, `eqc` |

### Adding Jacobian terms without changing the physical equations

Before adding derivatives, the example saves independent copies of the original
equations in `physical_equations` for convergence verification. It then adds
matrix contributions to the equations acting on the physical fields:

```python
def jacobian_only(term):
    return term + ResidualTerm(equation=term, underRelaxation=-1.)
```

Here `ResidualTerm` is the example's alias for `IndependentResidualTerm` from
`newton_solver.py`. At the current state, the explicit residual subtraction
cancels the added term's residual, while retaining its matrix contribution.
For a linear term with matrix $K$, this adds $K$ to the matrix and
$K\mathbf u^{(m)}$ to the right-hand side. It changes the Jacobian used for the
candidate solve without changing the physical residual at the current state.

### Poisson charge blocks

The examples write the Poisson relaxation as `TransientTerm == DiffusionTerm + charge`.
FiPy assembles the left side minus the right side, so its residual variation is

$$
\delta F_\phi=\frac{\delta\phi}{\Delta\tau}
-\nabla\cdot(\epsilon_r\nabla\delta\phi)
-\frac{q}{\epsilon_0}(\delta p-\delta n+\delta c-\delta a).
$$

The device adds these charge blocks explicitly:

```python
eqpoisson += jacobian_only(sum(
    CoupledChargeTerm(coeff=sign*q/epsilon_0, var=field)
    for sign, field in (
        (-1, plocal), (1, nlocal), (-1, clocal), (1, alocal)
    )
))
```

`CoupledChargeTerm` retains either sign in the implicit matrix. Ordinary
source-term sign splitting can otherwise move an intended cross derivative
to the explicit side.

### Recombination and potential–transport blocks

Both carrier rows include

$$
q\,\delta R=qR_n\delta n+qR_p\delta p.
$$

`recombination_derivative()` creates fresh FiPy source terms for each row,
acting on `nlocal` and `plocal`. Both carrier equations receive these terms
through `jacobian_only()`.
The potential–transport blocks differentiate the fitted drift flux, rather
than substituting an equilibrium density response. For example:

```python
eqn += jacobian_only(DiffusionTerm(
    coeff=q * nmob.harmonicFaceValue * responses[0], var=philocal))
eqp += jacobian_only(-DiffusionTerm(
    coeff=q * pmob.harmonicFaceValue * responses[1], var=philocal))
```

`update_transport_response()` refreshes all four face response fields before
assembly. The anion and cation potential blocks have the corresponding opposite
signs and mask exterior faces to preserve blocking boundaries.

### FiPy assembly and solve

`solve_for_voltage()` supplies all five physical fields, the five linearized
equations with Jacobian additions, the original `physical_equations`, and the
response callback to `solve_newton_coupled()`.
The common `_solve_coupled()` engine assembles them through FiPy:

```python
block = equations[0]
for equation in equations[1:]:
    block = block & equation
block(fields)  # Fix the physical-field ordering explicitly.
block.sweep(dt=dt, solver=linear_solver, cacheResidual=True)
```

FiPy's `&` combines the supplied blocks; it does not generate missing nonlinear
derivatives. Its public ordering API fixes the field order. See
[FiPy's coupled-equation documentation](https://pages.nist.gov/fipy/en/latest/USAGE.html).
The shared default uses the SciPy-based `MeshOrderedLinearLUSolver`.

The sweep writes candidate values into the physical fields. The driver derives
`deltas` from those candidates and uses the existing pseudo-time damping policy
to accept the update. Final convergence is checked against `physical_equations`,
with effectively infinite `dt` to test the stationary residual without
pseudo-time storage. All residual-only assembly explicitly uses SciPy.

Depending on which side of `==` a term appears, FiPy can use an overall row
sign; residual and derivative conventions must remain consistent.

Coupling resolves potential, transport, recombination and charge feedback in
one linear solve. A sequential iteration delays parts of this feedback until
later field updates.

## 4. Pseudo-time, damping and convergence

### Numerical relaxation toward a stationary state

`TransientTerm` in this steady solver represents **pseudo-time** $\tau$.
For example, the potential relaxation is written as

$$
-\frac{\partial\phi}{\partial\tau}
+\nabla\cdot(\epsilon_r\nabla\phi)+\rho/\epsilon_0=0.
$$

It regularizes the iteration and disappears at the stationary solution.
Pseudo-time is a numerical parameter, not an experimental time scale. The
potential storage term is artificial and has no physical capacitance meaning.
A backward-Euler relaxation step contributes storage proportional to
$(u-u_{\mathrm{old}})/\Delta\tau$; `TransientTerm(coeff=q)` applies the
charge factor for the carrier and ionic equations.

`solve_newton_coupled()` uses the current `dt`, `min_dt` and `max_dt` to
control relaxation. The shared engine decreases `dt` after sufficiently worse
residuals, otherwise increases it up to `max_dt`, and advances `.old` after
each update. This retains the established steady continuation policy.
