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
from SmoothingFunction import flatten_and_smooth_all
from joblib import Parallel, delayed
import multiprocessing
from material_maps import Semiconductors, Electrodes, map_semiconductor_property, map_electrode_property, map_props, name_to_code_SC, name_to_code_EL
from BoundaryConditions import ohmic
from constantsfile import TInfinite, q, epsilon_0, D
from LoadSolarSpectrum import SolarSpectrumWavelength, SolarSpectrumIrradiance
from workflow_utils import run_sweep, prepare_sweep_output, solve_and_save_point
from electrical_numerics import as_cell_array, cell_variable, conservative_internal_face_currents, srh_rate_and_carrier_derivatives, terminal_current_densities

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
DeviceArchitechture = np.empty((500, 1))
DeviceArchitechture[0:50,:] = Spiro_ID #50 nm Spiro HTL
DeviceArchitechture[50:450,:] = PS_ID #400nm PS Absorber
DeviceArchitechture[450:500,:] = TiO2_ID #50nm TiO2 ETL

TopElectrode = FTO_ID
TopLocationSC = DeviceArchitechture[-1,:].flatten() #Semiconducting materials adjacent to the top electrode
BottomLocationSC = DeviceArchitechture[0,:].flatten() #Semiconducting materials adjacent to the bottom electrode
BottomElectrode = Gold_ID

EffectiveMediumApproximationVolumeFraction = 1.00
GenRate_values_default = map_semiconductor_property(DeviceArchitechture, 'GenRate') #Binary array for whether generation is enabled or not

GenMode = 1
if GenMode == 1:
    #Lambert-Beer Law
    GenRate_values_default, ThermalisationHeat, PhotonFluxArray, TransmittedEnergy = calculate_absorption_above_bandgap(SolarSpectrumWavelength, SolarSpectrumIrradiance, AbsorptionData[:, 0], alphadata * EffectiveMediumApproximationVolumeFraction,GenRate_values_default, dy*StretchFactor, map_semiconductor_property(PS_ID, "Eg"))
else:
    #Constant Generation Rate
    GenRate_values_default = GenRate_values_default * 2.20e27

#Stretching in case finer meshing is needed (Stretching of generation array is done afterwards since the 3D hyperspectral generation array may exhaust RAM on machine)
zoom_factor = [StretchFactor] + ([StretchFactor] if DeviceArchitechture.shape[1] > 1 else [1])
DeviceArchitechture = zoom(DeviceArchitechture, zoom_factor, order=0)
GenRate_values_default = zoom(GenRate_values_default, zoom_factor, order=0)

print(DeviceArchitechture.shape)

sc_props = ['epsilon','pmob','nmob','Eg','chi','cationmob','anionmob', 'Recombination_Langevin','Recombination_Bimolecular','Nc','Nv', 'Chi_a','Chi_c','a_initial_level','c_initial_level','Nd','Na']
(epsilon_values, pmob_values, nmob_values, Eg, chi, cation_mob_values, anion_mob_values, Recombination_Langevin_values, Recombination_Bimolecular_values, Nc, Nv, chi_a, chi_c, a_initial_values, c_initial_values, Nd_values, Na_values) = map_props(DeviceArchitechture, sc_props, Semiconductors)

ny, nx = DeviceArchitechture.shape

#LocationSRH_HTL = mark_interfaces(DeviceArchitechture, 50, PS_ID)
#mark_interfaces() places the interface inside the absorber
LocationSRH_ETL = mark_interfaces(DeviceArchitechture, TiO2_ID, PS_ID)

#LocationHTL_Exact = mark_interfaces_mixed(DeviceArchitechture, 50, PS_ID, 0*StretchFactor)
#mark_interfaces_mixed() places the interface in the middle of the absorber and the transport layer
LocationETL_Exact = mark_interfaces_mixed(DeviceArchitechture, TiO2_ID, PS_ID, 3*StretchFactor)

SRH_Interfacial_Recombination_Zone = LocationETL_Exact

print("Number of ETL interface nm: ", 1.00e9*dx*(np.count_nonzero(LocationETL_Exact)-1)/(nx))
#print("Number of HTL interface nm: ", 1.00e9*dx*(np.count_nonzero(LocationHTL_Exact)-1)/(nx)) NO HTL in this simulation!

SRH_Bulk_Recombination_Zone = map_semiconductor_property(DeviceArchitechture, 'GenRate') - SRH_Interfacial_Recombination_Zone
#Make negative values zero
SRH_Bulk_Recombination_Zone = np.where(SRH_Bulk_Recombination_Zone < 0, 0.00, SRH_Bulk_Recombination_Zone)

#Flatten and smoothen variables to improve numerical stability
(epsilon_values, pmob_values, nmob_values, chi, chi_a, chi_c,Nc, LogNc, Nv, LogNv, Eg, SRH_Interfacial_Recombination_Zone, SRH_Bulk_Recombination_Zone) = flatten_and_smooth_all([epsilon_values, pmob_values, nmob_values, chi, chi_a, chi_c,Nc, np.log(Nc), Nv, np.log(Nv), Eg, SRH_Interfacial_Recombination_Zone, SRH_Bulk_Recombination_Zone],SmoothFactor * StretchFactor)
(GenRate_values_default, Recombination_Langevin_values, Recombination_Bimolecular_values, anion_mob_values, cation_mob_values, Nd_values, Na_values) = flatten_and_smooth_all([GenRate_values_default, Recombination_Langevin_values, Recombination_Bimolecular_values, anion_mob_values, cation_mob_values, Nd_values, Na_values],0.00)

mesh = fipy.Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)


(Recombination_Langevin_Cell, Recombination_Bimolecular_Cell, Recombination_Interfacial_SRH_Cell, Recombination_Bulk_SRH_Cell) = [cell_variable(mesh, name, values) for name, values in (("Recombination_Langevin_Cell", Recombination_Langevin_values), ("Recombination_Bimolecular_Cell", Recombination_Bimolecular_values), ("Recombination_SRH_Cell", SRH_Interfacial_Recombination_Zone), ("Recombination_SRH_Cell", SRH_Bulk_Recombination_Zone))]
(nmob, pmob, anionmob, cationmob) = [cell_variable(mesh, name, values) for name, values in (("electron mobility", nmob_values), ("hole mobility", pmob_values), ("anion mobility", anion_mob_values), ("cation mobility", cation_mob_values))]
(epsilon, LogNcCell, LogNvCell) = [cell_variable(mesh, name, values) for name, values in (("dielectric permittivity", epsilon_values), ("Log Effective Density of States CB", LogNc), ("Log Effective Density of States VB", LogNv))]
(ChiCell, ChiCell_a, ChiCell_c, EgCell, NdCell, NaCell) = [cell_variable(mesh, name, values) for name, values in (("Electron Affinity", chi), ("Electron Affinity", chi_a), ("Electron Affinity", chi_c), ("Band Gap", Eg), ("Fixed Ionised Donors", Nd_values), ("Fixed Ionised Acceptor", Na_values))]

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

Etrap = map_semiconductor_property(PS_ID, "chi") + map_semiconductor_property(PS_ID, "Eg")/2 #Mid-bandgap trap energy level in eV
Etrap_interface = map_semiconductor_property(TiO2_ID, "chi") + ((map_semiconductor_property(PS_ID, "chi") + map_semiconductor_property(PS_ID, "Eg"))-map_semiconductor_property(TiO2_ID, 'chi'))/2

#Here we define the mid-bandgap SRH trap energy level
n_hat = map_semiconductor_property(PS_ID, 'Nc') * np.exp((map_semiconductor_property(PS_ID, "chi") - Etrap) / D)
p_hat = map_semiconductor_property(PS_ID, 'Nv') * np.exp((Etrap - map_semiconductor_property(PS_ID, "chi") - map_semiconductor_property(PS_ID, "Eg")) / D)

#Here we define the mixed band PS-HOMO/TiO2-LUMO SRH trap level
n_hat_mixed = map_semiconductor_property(PS_ID, 'Nc') * np.exp((map_semiconductor_property(TiO2_ID, "chi") - Etrap_interface) / D)
p_hat_mixed = map_semiconductor_property(PS_ID, 'Nv') * np.exp((Etrap_interface - map_semiconductor_property(PS_ID, "chi") - map_semiconductor_property(PS_ID, "Eg")) / D)

niPS = np.sqrt(Nc * Nv * np.exp(-Eg / D))

def solve_for_excitation_site(excitation_index, n_values, p_values, a_values, c_values, phi_values):

    state_names = ("electrostatic potential", "electron density", "hole density", "anion density", "cation density")
    state_values = (phi_values, n_values, p_values, a_values, c_values)
    philocal, nlocal, plocal, alocal, clocal = [cell_variable(mesh, name, value, True) for name, value in zip(state_names, state_values)]

    if excitation_index == -1:
        SCERegion = np.zeros(np.size(DeviceArchitechture))
    else:
        UniqueIDs = np.reshape(np.arange(np.size(DeviceArchitechture)), DeviceArchitechture.shape)
        #Set UniqueID entries which are not equal to voltage to zero
        SCERegion = (np.array([UniqueIDs == excitation_index], dtype=float) * 1.0).flatten()

    GenRate_values_default_modified = GenRate_values_default + SCERegion*1.0e28

    gen_rate = cell_variable(mesh, "Generation Rate", GenRate_values_default_modified)

    voltage = 0.00

    contact_bcs = [
        {'boundary': mesh.facesTop, 'n': nTop, 'p': pTop, 'phi': (Vbi + voltage)},
        {'boundary': mesh.facesBottom, 'n': nBottom, 'p': pBottom, 'phi': 0}
    ]

    for bc in contact_bcs:
            nlocal.constrain(bc['n'], where=bc['boundary'])
            plocal.constrain(bc['p'], where=bc['boundary'])
            philocal.constrain(bc['phi'], where=bc['boundary'])

    #Band-to-band recombination models
    Recombination_Langevin_EQ = (Recombination_Langevin_Cell * q * (pmob + nmob) * (nlocal * plocal - niPS * niPS) / (epsilon_values * epsilon_0))
    Recombination_Bimolecular_EQ = (Recombination_Bimolecular_Cell * (nlocal * plocal - niPS * niPS))

    # SRH rate and exact fixed-temperature carrier derivatives.
    niPS_squared = niPS * niPS
    Recombination_SRH_Bulk_EQ, SRH_Bulk_dR_dn, SRH_Bulk_dR_dp = srh_rate_and_carrier_derivatives(Recombination_Bulk_SRH_Cell, nlocal, plocal, niPS_squared, tau_p_bulk, tau_n_bulk, n_hat, p_hat)

    Recombination_Combined = (Recombination_Bimolecular_EQ + Recombination_SRH_Bulk_EQ) #Include more recombination mechanisms by adding them to this line

    eqn = (TransientTerm(coeff=q, var=nlocal) == DiffusionTerm(coeff=q * D * nmob.harmonicFaceValue, var=nlocal) - ExponentialConvectionTerm(coeff=q * nmob.harmonicFaceValue * (philocal.faceGrad + ChiCell.faceGrad + D * LogNcCell.faceGrad), var=nlocal) + q*gen_rate - q*Recombination_Combined)
    eqp = (TransientTerm(coeff=q, var=plocal) == DiffusionTerm(coeff=q * D * pmob.harmonicFaceValue, var=plocal) + ExponentialConvectionTerm(coeff=q * pmob.harmonicFaceValue * (philocal.faceGrad + ChiCell.faceGrad + EgCell.faceGrad - D * LogNvCell.faceGrad), var=plocal) + q*gen_rate - q*Recombination_Combined)
    eqa = (TransientTerm(coeff=q, var=alocal) == DiffusionTerm(coeff=q * D * anionmob.harmonicFaceValue, var=alocal) - ExponentialConvectionTerm(coeff=q * anionmob.harmonicFaceValue * (philocal.faceGrad + ChiCell_a.faceGrad), var=alocal))
    eqc = (TransientTerm(coeff=q, var=clocal) == DiffusionTerm(coeff=q * D * cationmob.harmonicFaceValue, var=clocal) + ExponentialConvectionTerm(coeff=q * cationmob.harmonicFaceValue * (philocal.faceGrad + ChiCell_c.faceGrad), var=clocal))
    eqpoisson = (0.00 == -TransientTerm(var=philocal) + DiffusionTerm(coeff=epsilon, var=philocal) + (q/epsilon_0) * (plocal - nlocal + clocal - alocal + NdCell - NaCell))

    net_dR_dn = Recombination_Bimolecular_Cell * plocal + SRH_Bulk_dR_dn
    net_dR_dp = Recombination_Bimolecular_Cell * nlocal + SRH_Bulk_dR_dp
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
        for sign, field in ((1, plocal), (-1, nlocal), (1, clocal), (-1, alocal))))

    responses = [fipy.FaceVariable(mesh=mesh, value=0.) for _ in range(4)]
    gradients = (philocal.faceGrad+ChiCell.faceGrad+D*LogNcCell.faceGrad,
                 philocal.faceGrad+ChiCell.faceGrad+EgCell.faceGrad-D*LogNvCell.faceGrad,
                 philocal.faceGrad+ChiCell_a.faceGrad, philocal.faceGrad+ChiCell_c.faceGrad)
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
    psinvar = philocal + ChiCell - D * (numerix.log(nlocal) - LogNcCell)
    psipvar = philocal + ChiCell + EgCell + D * (numerix.log(plocal) - LogNvCell)

    # Here the electric field is calculated
    E = -philocal.grad.globalValue
    Efield_matrix = np.reshape(E, (E.shape[0], ny, nx))

    n_array, p_array, phi_array, chi_array, eg_array, log_nc_array, log_nv_array, nmob_array, pmob_array = [as_cell_array(field, DeviceArchitechture.shape) for field in (nlocal, plocal, philocal, ChiCell, EgCell, LogNcCell, LogNvCell, nmob, pmob)]
    ConservativeJnInternal, ConservativeJpInternal = conservative_internal_face_currents(n_array, p_array, phi_array, chi_array, eg_array, log_nc_array, log_nv_array, nmob_array, pmob_array, axis=0, spacing=dy, thermal_voltage=D)
    BottomTerminalCurrentDensity, TopTerminalCurrentDensity, TerminalCurrentDensity = terminal_current_densities(ConservativeJnInternal, ConservativeJpInternal)

    (PotentialMatrix, GenValues_Matrix, RecombinationMatrix, Recombination_Bimolecular_EQMatrix, NMatrix, PMatrix, chiMatrix, EgMatrix, psinvarmatrix, psipvarmatrix) = [np.reshape(arr,(ny, nx)) for arr in (philocal, gen_rate, Recombination_Combined, Recombination_Bimolecular_EQ, nlocal, plocal, ChiCell, EgCell, psinvar, psipvar)]

    return {"NMatrix": NMatrix, "PMatrix": PMatrix, "RecombinationMatrix": RecombinationMatrix, "GenValues_Matrix": GenValues_Matrix, "PotentialMatrix": PotentialMatrix, "Efield_matrix": Efield_matrix, "n": nlocal.globalValue.copy(), "p": plocal.globalValue.copy(), "phi": philocal.globalValue.copy(), "ChiMatrix": chiMatrix, "EgMatrix": EgMatrix, "psinvarmatrix": psinvarmatrix, "psipvarmatrix": psipvarmatrix, "AnionDensityMatrix": alocal.globalValue.copy(), "CationDensityMatrix": clocal.globalValue.copy(), "ResidualMatrix": residual, "SweepCounterMatrix": SweepCounter, "Recombination_Bimolecular_EQMatrix": Recombination_Bimolecular_EQMatrix, "ResidualArray": residualarray, "ConservativeJnInternal": ConservativeJnInternal, "ConservativeJpInternal": ConservativeJpInternal, "TerminalCurrentDensity": TerminalCurrentDensity, "BottomTerminalCurrentDensity": BottomTerminalCurrentDensity, "TopTerminalCurrentDensity": TopTerminalCurrentDensity, "Converged": bool(residual <= desired_residual)}

def simulate_device(output_dir):
    prepare_sweep_output(output_dir)

    excitation_indices = np.arange(-1,np.size(DeviceArchitechture))

    chunk_size = min(len(excitation_indices), max(1, multiprocessing.cpu_count() - 1))

    n_values = 1.00e-30
    p_values = 1.00e-30
    a_values = a_initial_values.flatten()
    c_values = c_initial_values.flatten()
    phi_values = 1.00e-30

    # Process excitation sites in sequential chunks.
    for start in range(0, len(excitation_indices), chunk_size):
        # Create a chunk of excitation sites to simulate in parallel.
        chunk_excitation_indices = excitation_indices[start:start + chunk_size]

        # Parallel computation within the chunk
        chunk_results = Parallel(n_jobs=chunk_size, backend="multiprocessing")(
            delayed(solve_and_save_point)(
                solve_for_excitation_site, output_dir, start + offset,
                (excitation_index, n_values, p_values, a_values, c_values, phi_values),
                {'applied_voltage': 0., 'excitation_index': excitation_index})
            for offset, excitation_index in enumerate(chunk_excitation_indices))

        # Update initial conditions using results from the last voltage in the chunk to speed up convergence of the next chunk
        last_result = chunk_results[-1]  # The last result in the current chunk
        n_values, p_values = last_result["n"], last_result["p"]
        a_values, c_values = last_result["AnionDensityMatrix"], last_result["CationDensityMatrix"]
        phi_values = last_result["phi"]
    return chunk_results

def main_workflow():
    return run_sweep(simulate_device, __file__, "VoltageSweep", "Starting standard voltage sweep...", "Voltage sweep completed.")

# Fix for multiprocessing on Windows
if __name__ == '__main__':
    main_workflow()


