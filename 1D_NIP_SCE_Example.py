# -*- coding: utf-8 -*-
#This code is a simulation of a NIP planar perovskite solar cell using the finite volume method with the FiPy library.
#Device architecture: FTO (Boundary)|TiO2 (50 nm)|MAPbI3 (400 nm)|Spiro-OMeTAD (50 nm)|Gold (Boundary)
#This code calculates the spatial collection efficiency at 0V bias, which can be plotted using PlottingSCE_1D
import os
os.environ["OMP_NUM_THREADS"] = "1" #Really important! Pysparse doesnt benefit from multithreading.
import numpy as np
from mark_interface_file import mark_interfaces, mark_interfaces_mixed
from calculate_absorption import calculate_absorption_above_bandgap
from fipy import TransientTerm, DiffusionTerm, ImplicitSourceTerm
import fipy
from fipy.tools import numerix
from newton_solver import (solve_newton_coupled, CoupledChargeTerm, exponential_flux_response, MeshOrderedLinearLUSolver)
from newton_solver import LiveExponentialConvectionTerm as ExponentialConvectionTerm
from newton_solver import IndependentResidualTerm as ResidualTerm, fresh_equation
from scipy.ndimage import zoom
from joblib import Parallel, delayed
import multiprocessing
from functools import partial
from material_maps import Semiconductors, map_semiconductor_property, map_electrode_property, name_to_code_SC, name_to_code_EL
from BoundaryConditions import ohmic
from constantsfile import q, epsilon_0, D
from LoadSolarSpectrum import SolarSpectrumWavelength, SolarSpectrumIrradiance
from workflow_utils import run_sweep, prepare_sweep_output, solve_and_save_point
from electrical_numerics import device_field, as_cell_array, cell_variable, conservative_internal_face_currents, srh_rate_and_carrier_derivatives, terminal_current_densities

Gold_ID = name_to_code_EL["Gold"]
Spiro_ID = name_to_code_SC["Spiro"]
PS_ID = name_to_code_SC["PS"]
TiO2_ID = name_to_code_SC["mTiO2_2"]
FTO_ID = name_to_code_EL["FTO2"]

StretchFactor = 1 #Can help convergence if a finer mesh is needed
SmoothFactor = 0.2 #Some smoothing helps with convergence

dx = 1.00e-9/StretchFactor #Pixel Width in meters
dy = 1.00e-9/StretchFactor #Pixel Width in meters

#Importing Absorbance Coefficient Spectrum for MAPbI3
AbsorptionData = np.genfromtxt("MAPI_tailfit_nk 1.txt", delimiter=",", skip_header=1)
kdata = AbsorptionData[:, 2]
alphadata = 4 * np.pi * kdata / (AbsorptionData[:, 0] * 1.00e-9)

######Define Device Architecture
DeviceArchitecture = np.empty((500, 1))
DeviceArchitecture[0:50,:] = Spiro_ID #50 nm Spiro HTL
DeviceArchitecture[50:450,:] = PS_ID #400nm PS Absorber
DeviceArchitecture[450:500,:] = TiO2_ID #50nm TiO2 ETL

TopElectrode = FTO_ID
TopLocationSC = DeviceArchitecture[-1,:].flatten() #Semiconducting materials adjacent to the top electrode
BottomLocationSC = DeviceArchitecture[0,:].flatten() #Semiconducting materials adjacent to the bottom electrode
BottomElectrode = Gold_ID

EffectiveMediumApproximationVolumeFraction = 1.00
GenRate_values_default = map_semiconductor_property(DeviceArchitecture, 'GenRate') #Binary array for whether generation is enabled or not

GenMode = 1
if GenMode == 1:
    #Lambert-Beer Law
    GenRate_values_default, ThermalisationHeat, PhotonFluxArray, TransmittedEnergy = calculate_absorption_above_bandgap(SolarSpectrumWavelength, SolarSpectrumIrradiance, AbsorptionData[:, 0], alphadata * EffectiveMediumApproximationVolumeFraction,GenRate_values_default, dy*StretchFactor, map_semiconductor_property(PS_ID, "Eg"))
else:
    #Constant Generation Rate
    GenRate_values_default = GenRate_values_default * 2.20e27

#Stretching in case finer meshing is needed (Stretching of generation array is done afterwards since the 3D hyperspectral generation array may exhaust RAM on machine)
zoom_factor = [StretchFactor] + ([StretchFactor] if DeviceArchitecture.shape[1] > 1 else [1])
DeviceArchitecture = zoom(DeviceArchitecture, zoom_factor, order=0)
GenRate_values_default = zoom(GenRate_values_default, zoom_factor, order=0)

print(DeviceArchitecture.shape)

ny, nx = DeviceArchitecture.shape
mesh = fipy.Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)

#mark_interfaces() places the interface inside the absorber

#mark_interfaces_mixed() places the interface in the middle of the absorber and the transport layer
LocationETL_Exact = mark_interfaces_mixed(DeviceArchitecture, TiO2_ID, PS_ID, 3*StretchFactor)

SRH_Interfacial_Recombination_Zone = LocationETL_Exact

print("Number of ETL interface nm: ", 1.00e9*dx*(np.count_nonzero(LocationETL_Exact)-1)/(nx))

SRH_Bulk_Recombination_Zone = map_semiconductor_property(DeviceArchitecture, 'GenRate') - SRH_Interfacial_Recombination_Zone
#Make negative values zero
SRH_Bulk_Recombination_Zone = np.where(SRH_Bulk_Recombination_Zone < 0, 0.00, SRH_Bulk_Recombination_Zone)

make_field = partial(device_field, mesh, DeviceArchitecture, smoothing=SmoothFactor*StretchFactor)
# Smooth transport energies and electronic mobilities; keep ions and sources sharp.
epsilon, nmob, pmob = [make_field(prop) for prop in ('epsilon', 'nmob', 'pmob')]
ChiCell, ChiCell_a, ChiCell_c, EgCell = [make_field(prop) for prop in ('chi', 'Chi_a', 'Chi_c', 'Eg')]
Nc, Nv = [make_field(prop).value for prop in ('Nc', 'Nv')]
LogNcCell, LogNvCell = [make_field(prop, logarithm=True) for prop in ('Nc', 'Nv')]
anionmob, cationmob, NdCell, NaCell = [make_field(prop, smoothing=0.) for prop in ('anionmob', 'cationmob', 'Nd', 'Na')]
Recombination_Langevin_Cell, Recombination_Bimolecular_Cell = [make_field(prop, smoothing=0.) for prop in ('Recombination_Langevin', 'Recombination_Bimolecular')]
a_initial_values, c_initial_values = [map_semiconductor_property(DeviceArchitecture, prop) for prop in ('a_initial_level', 'c_initial_level')]
Recombination_Interfacial_SRH_Cell = make_field('interface SRH zone', SRH_Interfacial_Recombination_Zone)
Recombination_Bulk_SRH_Cell = make_field('bulk SRH zone', SRH_Bulk_Recombination_Zone)
GenRate_values_default = GenRate_values_default.flatten()
gen_rate = make_field('Generation Rate', GenRate_values_default, smoothing=0.)

nTop, pTop = ohmic(TopLocationSC, TopElectrode)
nBottom, pBottom = ohmic(BottomLocationSC, BottomElectrode)

Vbi = (map_electrode_property(BottomElectrode, "WF") - map_electrode_property(TopElectrode, "WF"))

############Recombination Constants############
#Charge Carrier Lifetimes in the bulk (s)
tau_p_bulk = 35 * 1.00e-9
tau_n_bulk = 35 * 1.00e-9
#Charge Carrier Lifetimes at the interface (s)
tau_p_interface = 0.02 * 1.00e-9
tau_n_interface = 0.02 * 1.00e-9

absorber, transport_layer = Semiconductors[PS_ID], Semiconductors[TiO2_ID]
Etrap = absorber.chi + absorber.Eg/2 #Mid-bandgap trap energy level in eV
Etrap_interface = transport_layer.chi + ((absorber.chi + absorber.Eg)-transport_layer.chi)/2

#Here we define the mid-bandgap SRH trap energy level
n_hat = absorber.Nc * np.exp((absorber.chi - Etrap) / D)
p_hat = absorber.Nv * np.exp((Etrap - absorber.chi - absorber.Eg) / D)

#Here we define the mixed band PS-HOMO/TiO2-LUMO SRH trap level
n_hat_mixed = absorber.Nc * np.exp((transport_layer.chi - Etrap_interface) / D)
p_hat_mixed = absorber.Nv * np.exp((Etrap_interface - absorber.chi - absorber.Eg) / D)

niPS = np.sqrt(Nc * Nv * np.exp(-EgCell.value / D))

def solve_for_excitation_site(excitation_index, n_values, p_values, a_values, c_values, phi_values):

    state_names = ("electrostatic potential", "electron density", "hole density", "anion density", "cation density")
    state_values = (phi_values, n_values, p_values, a_values, c_values)
    philocal, nlocal, plocal, alocal, clocal = [cell_variable(mesh, name, value, True) for name, value in zip(state_names, state_values)]

    if excitation_index == -1:
        SCERegion = np.zeros(np.size(DeviceArchitecture))
    else:
        UniqueIDs = np.reshape(np.arange(np.size(DeviceArchitecture)), DeviceArchitecture.shape)
        #Set UniqueID entries which are not equal to voltage to zero
        SCERegion = (np.array([UniqueIDs == excitation_index], dtype=float) * 1.0).flatten()

    GenRate_values_default_modified = GenRate_values_default + SCERegion*1.0e28

    gen_rate = cell_variable(mesh, "Generation Rate", GenRate_values_default_modified)

    voltage = 0.00

    for boundary, n_contact, p_contact, phi_contact in (
            (mesh.facesTop, nTop, pTop, Vbi + voltage),
            (mesh.facesBottom, nBottom, pBottom, 0)):
        nlocal.constrain(n_contact, where=boundary)
        plocal.constrain(p_contact, where=boundary)
        philocal.constrain(phi_contact, where=boundary)

    #Band-to-band recombination models
    Recombination_Langevin_EQ = (Recombination_Langevin_Cell * q * (pmob + nmob) * (nlocal * plocal - niPS * niPS) / (epsilon.value * epsilon_0))
    Recombination_Bimolecular_EQ = (Recombination_Bimolecular_Cell * (nlocal * plocal - niPS * niPS))

    # SRH rate and exact fixed-temperature carrier derivatives.
    niPS_squared = niPS * niPS
    bulk_srh = srh_rate_and_carrier_derivatives(Recombination_Bulk_SRH_Cell, nlocal, plocal, niPS_squared, tau_p_bulk, tau_n_bulk, n_hat, p_hat)

    bimolecular = (Recombination_Bimolecular_EQ, Recombination_Bimolecular_Cell*plocal, Recombination_Bimolecular_Cell*nlocal)
    # Each model supplies (rate, dR/dn, dR/dp); edit this list to enable mechanisms.
    Recombination_Combined, net_dR_dn, net_dR_dp = [sum(parts[1:], parts[0]) for parts in zip(bimolecular, bulk_srh)]

    # Drift-driving gradients for the four mobile species.
    electron_drift = philocal.faceGrad + ChiCell.faceGrad + D*LogNcCell.faceGrad
    hole_drift = philocal.faceGrad + ChiCell.faceGrad + EgCell.faceGrad - D*LogNvCell.faceGrad
    anion_drift = philocal.faceGrad + ChiCell_a.faceGrad
    cation_drift = philocal.faceGrad + ChiCell_c.faceGrad
    eqn = (TransientTerm(coeff=q, var=nlocal) == DiffusionTerm(coeff=q * D * nmob.harmonicFaceValue, var=nlocal) - ExponentialConvectionTerm(coeff=q * nmob.harmonicFaceValue * electron_drift, var=nlocal) + q*gen_rate - q*Recombination_Combined)
    eqp = (TransientTerm(coeff=q, var=plocal) == DiffusionTerm(coeff=q * D * pmob.harmonicFaceValue, var=plocal) + ExponentialConvectionTerm(coeff=q * pmob.harmonicFaceValue * hole_drift, var=plocal) + q*gen_rate - q*Recombination_Combined)
    eqa = (TransientTerm(coeff=q, var=alocal) == DiffusionTerm(coeff=q * D * anionmob.harmonicFaceValue, var=alocal) - ExponentialConvectionTerm(coeff=q * anionmob.harmonicFaceValue * anion_drift, var=alocal))
    eqc = (TransientTerm(coeff=q, var=clocal) == DiffusionTerm(coeff=q * D * cationmob.harmonicFaceValue, var=clocal) + ExponentialConvectionTerm(coeff=q * cationmob.harmonicFaceValue * cation_drift, var=clocal))
    eqpoisson = (TransientTerm(var=philocal) == DiffusionTerm(coeff=epsilon, var=philocal) + (q/epsilon_0) * (plocal - nlocal + clocal - alocal + NdCell - NaCell))

    physical_equations = tuple(fresh_equation(eq) for eq in (eqpoisson, eqn, eqp, eqa, eqc))

    # Add a derivative to the matrix without changing the physical residual.
    # FiPy assembles the subtraction at the current state before solving.
    def jacobian_only(term):
        return term + ResidualTerm(equation=term, underRelaxation=-1.)

    # Fresh terms per row: FiPy terms carry mutable assembly caches (issue #1235).
    def recombination_derivative():
        return (ImplicitSourceTerm(coeff=q*net_dR_dn, var=nlocal)
                + ImplicitSourceTerm(coeff=q*net_dR_dp, var=plocal))
    eqn += jacobian_only(recombination_derivative())
    eqp += jacobian_only(recombination_derivative())
    eqpoisson += jacobian_only(sum(CoupledChargeTerm(coeff=sign*q/epsilon_0, var=field)
        for sign, field in ((-1, plocal), (1, nlocal), (-1, clocal), (1, alocal))))

    responses = [fipy.FaceVariable(mesh=mesh, value=0.) for _ in range(4)]
    gradients = (electron_drift, hole_drift, anion_drift, cation_drift)
    def update_transport_response():
        for response, field, gradient, sign in zip(responses, (nlocal, plocal, alocal, clocal), gradients, (1, -1, 1, -1)):
            response.setValue(exponential_flux_response(field, gradient, sign, D))
    eqn += jacobian_only(DiffusionTerm(coeff=q*nmob.harmonicFaceValue*responses[0], var=philocal))
    eqp += jacobian_only(-DiffusionTerm(coeff=q*pmob.harmonicFaceValue*responses[1], var=philocal))
    eqa += jacobian_only(DiffusionTerm(coeff=q*anionmob.harmonicFaceValue*responses[2]*(~mesh.exteriorFaces), var=philocal))
    eqc += jacobian_only(-DiffusionTerm(coeff=q*cationmob.harmonicFaceValue*responses[3]*(~mesh.exteriorFaces), var=philocal))

    desired_residual = 1e-12
    # The full block uses unit damping, with pseudo-time regularization retained.
    # Separate continuity tolerances prevent the Poisson norm masking slow ions.
    residual, SweepCounter, residualarray = solve_newton_coupled(
        fields=(philocal, nlocal, plocal, alocal, clocal),
        equations=(eqpoisson, eqn, eqp, eqa, eqc),
        dt=1e-7, max_dt=1e-5, tolerance=desired_residual,
        damping=1.0, sweeps=1, max_steps=2000, enable_ions=True,
        physical_equations=physical_equations,
        update_transport_response=update_transport_response,
        linear_solver=MeshOrderedLinearLUSolver(mesh, tolerance=1e-12, iterations=1),
        equation_tolerances=(desired_residual, desired_residual, desired_residual, desired_residual, desired_residual))

    # Here the electron and hole quasi-fermi levels are calculated
    # Legacy output names are paired directly with their physical fields.
    result = dict((name, as_cell_array(field, DeviceArchitecture.shape)) for name, field in (
        ('NMatrix', nlocal), ('PMatrix', plocal), ('RecombinationMatrix', Recombination_Combined),
        ('GenValues_Matrix', gen_rate), ('PotentialMatrix', philocal), ('ChiMatrix', ChiCell), ('EgMatrix', EgCell),
        ('psinvarmatrix', philocal + ChiCell - D * (numerix.log(nlocal) - LogNcCell)),
        ('psipvarmatrix', philocal + ChiCell + EgCell + D * (numerix.log(plocal) - LogNvCell)),
        ('Recombination_Bimolecular_EQMatrix', Recombination_Bimolecular_EQ)))
    jn, jp = conservative_internal_face_currents(
        *[as_cell_array(field, DeviceArchitecture.shape) for field in
          (nlocal, plocal, philocal, ChiCell, EgCell, LogNcCell, LogNvCell, nmob, pmob)], axis=0, spacing=dy, thermal_voltage=D)
    bottom_current, top_current, terminal_current = terminal_current_densities(jn, jp)
    result.update(
                  Efield_matrix=(-philocal.grad.globalValue).reshape((mesh.dim,) + DeviceArchitecture.shape),
                  n=nlocal.globalValue.copy(), p=plocal.globalValue.copy(), phi=philocal.globalValue.copy(),
                  AnionDensityMatrix=alocal.globalValue.copy(), CationDensityMatrix=clocal.globalValue.copy(),
                  ResidualMatrix=residual, SweepCounterMatrix=SweepCounter, ResidualArray=residualarray, ConservativeJnInternal=jn,
                  ConservativeJpInternal=jp, TerminalCurrentDensity=terminal_current, BottomTerminalCurrentDensity=bottom_current,
                  TopTerminalCurrentDensity=top_current, Converged=bool(residual <= desired_residual))
    return result

def simulate_device(output_dir):
    prepare_sweep_output(output_dir)

    excitation_indices = np.arange(-1,np.size(DeviceArchitecture))

    chunk_size = min(len(excitation_indices), max(1, multiprocessing.cpu_count() - 1))

    # State order: electrons, holes, anions, cations, potential.
    state = (1.00e-30, 1.00e-30, a_initial_values.flatten(), c_initial_values.flatten(), 1.00e-30)

    # Process excitation sites in sequential chunks.
    for start in range(0, len(excitation_indices), chunk_size):
        chunk_excitation_indices = excitation_indices[start:start + chunk_size]

        chunk_results = Parallel(n_jobs=chunk_size, backend="multiprocessing")(
            delayed(solve_and_save_point)(
                solve_for_excitation_site, output_dir, start + offset,
                (excitation_index,) + state,
                {'applied_voltage': 0., 'excitation_index': excitation_index})
            for offset, excitation_index in enumerate(chunk_excitation_indices))

        state = tuple(chunk_results[-1][key] for key in
                      ('n', 'p', 'AnionDensityMatrix', 'CationDensityMatrix', 'phi'))
    return chunk_results

def main_workflow():
    return run_sweep(simulate_device, __file__, "VoltageSweep", "Starting standard voltage sweep...", "Voltage sweep completed.")

# Fix for multiprocessing on Windows
if __name__ == '__main__':
    main_workflow()

