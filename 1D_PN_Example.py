# -*- coding: utf-8 -*-
#This code is a simulation of a PN planar silicon solar cell in the dark using the finite volume method with the FiPy library.
#Device architecture: Si_Na_Doped (1000 nm)|Si_Nd_Doped (2000 nm)
import os
os.environ["OMP_NUM_THREADS"] = "1" #Really important! Pysparse doesnt benefit from multithreading.
import numpy as np
from fipy import TransientTerm, DiffusionTerm, ExponentialConvectionTerm
import fipy
from fipy.tools import numerix
from gummel_solver import solve_gummel
from scipy.ndimage import zoom
from SmoothingFunction import flatten_and_smooth_all
from joblib import Parallel, delayed
import multiprocessing
from material_maps import Semiconductors, Electrodes, map_semiconductor_property, map_electrode_property, map_props, name_to_code_SC, name_to_code_EL
from constantsfile import TInfinite, q, epsilon_0, D
from LoadSolarSpectrum import SolarSpectrumWavelength, SolarSpectrumIrradiance
from workflow_utils import run_sweep, prepare_voltage_output, solve_and_save_voltage
from electrical_numerics import as_cell_array, cell_variable, conservative_internal_face_currents, terminal_current_densities

Si_NaDoping_ID = name_to_code_SC["Si_NaDoping"]
Si_NdDoping_ID = name_to_code_SC["Si_NdDoping"]

StretchFactor = 1 #Can help convergence if a finer mesh is needed
SmoothFactor = 0.2 #Some smoothing helps with convergence

dx = 1.00e-9/StretchFactor #Pixel Width in meters
dy = 1.00e-9/StretchFactor #Pixel Width in meters

######Define Device Architecture
DeviceArchitechture = np.empty((3000, 1))

DeviceArchitechture[0:1000,:] = Si_NaDoping_ID
DeviceArchitechture[1000:3000,:] = Si_NdDoping_ID

TopLocationSC = DeviceArchitechture[-1,:].flatten() #Semiconducting materials adjacent to the top electrode
BottomLocationSC = DeviceArchitechture[0,:].flatten() #Semiconducting materials adjacent to the bottom electrode

EffectiveMediumApproximationVolumeFraction = 1.00

GenRate_values_default = map_semiconductor_property(DeviceArchitechture, 'GenRate') #Binary array for whether generation is enabled or not
GenRate_values_default = GenRate_values_default * 0.00e27

#Stretching in case finer meshing is needed (Stretching of generation array is done afterwards since the 3D hyperspectral generation array may exhaust RAM on machine)
zoom_factor = [StretchFactor] + ([StretchFactor] if DeviceArchitechture.shape[1] > 1 else [1])
DeviceArchitechture = zoom(DeviceArchitechture, zoom_factor, order=0)
GenRate_values_default = zoom(GenRate_values_default, zoom_factor, order=0)

print(DeviceArchitechture.shape)

sc_props = ['epsilon','pmob','nmob','Eg','chi','cationmob','anionmob', 'Recombination_Langevin','Recombination_Bimolecular','Nc','Nv', 'Chi_a','Chi_c','a_initial_level','c_initial_level','Nd','Na']
(epsilon_values, pmob_values, nmob_values, Eg, chi, cation_mob_values, anion_mob_values, Recombination_Langevin_values, Recombination_Bimolecular_values, Nc, Nv, chi_a, chi_c, a_initial_values, c_initial_values, Nd_values, Na_values) = map_props(DeviceArchitechture, sc_props, Semiconductors)

ny, nx = DeviceArchitechture.shape

#Flatten and smoothen variables to improve numerical stability
(epsilon_values, pmob_values, nmob_values, chi, chi_a, chi_c,Nc, LogNc, Nv, LogNv, Eg) = flatten_and_smooth_all([epsilon_values, pmob_values, nmob_values, chi, chi_a, chi_c,Nc, np.log(Nc), Nv, np.log(Nv), Eg],SmoothFactor * StretchFactor)
(GenRate_values_default, Recombination_Langevin_values, Recombination_Bimolecular_values, anion_mob_values, cation_mob_values, Nd_values, Na_values) = flatten_and_smooth_all([GenRate_values_default, Recombination_Langevin_values, Recombination_Bimolecular_values, anion_mob_values, cation_mob_values, Nd_values, Na_values],0.00)

mesh = fipy.Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)

(gen_rate, Recombination_Langevin_Cell, Recombination_Bimolecular_Cell) = [cell_variable(mesh, name, values) for name, values in (("Generation Rate", GenRate_values_default), ("Recombination_Langevin_Cell", Recombination_Langevin_values), ("Recombination_Bimolecular_Cell", Recombination_Bimolecular_values))]
(nmob, pmob, anionmob, cationmob) = [cell_variable(mesh, name, values) for name, values in (("electron mobility", nmob_values), ("hole mobility", pmob_values), ("anion mobility", anion_mob_values), ("cation mobility", cation_mob_values))]
(epsilon, LogNcCell, LogNvCell) = [cell_variable(mesh, name, values) for name, values in (("dielectric permittivity", epsilon_values), ("Log Effective Density of States CB", LogNc), ("Log Effective Density of States VB", LogNv))]
(ChiCell, ChiCell_a, ChiCell_c, EgCell, NdCell, NaCell) = [cell_variable(mesh, name, values) for name, values in (("Electron Affinity", chi), ("Electron Affinity", chi_a), ("Electron Affinity", chi_c), ("Band Gap", Eg), ("Fixed Ionised Donors", Nd_values), ("Fixed Ionised Acceptor", Na_values))]

niPS = np.sqrt(Nc * Nv * np.exp(-Eg / D))

niPSmax = np.max(niPS)

#Here we define the boundary conditions
nTop = 1.00e22
pTop = (niPSmax*niPSmax)/nTop

pBottom = 1.00e23
nBottom = (niPSmax*niPSmax)/pBottom

Vbi = D*np.log(1.00e22*1.00e23/(niPSmax*niPSmax))

def solve_for_voltage(voltage, n_values, p_values, a_values, c_values, phi_values):

    state_names = ("electrostatic potential", "electron density", "hole density", "anion density", "cation density")
    state_values = (phi_values, n_values, p_values, a_values, c_values)
    philocal, nlocal, plocal, alocal, clocal = [cell_variable(mesh, name, value, True) for name, value in zip(state_names, state_values)]

    contact_bcs = [
        {'boundary': mesh.facesTop, 'n': nTop, 'p': pTop, 'phi': (Vbi - voltage)},
        {'boundary': mesh.facesBottom, 'n': nBottom, 'p': pBottom, 'phi': 0}
    ]

    for bc in contact_bcs:
            nlocal.constrain(bc['n'], where=bc['boundary'])
            plocal.constrain(bc['p'], where=bc['boundary'])
            philocal.constrain(bc['phi'], where=bc['boundary'])

    #Band-to-band recombination models
    Recombination_Langevin_EQ = (Recombination_Langevin_Cell * q * (pmob + nmob) * (nlocal * plocal - niPS * niPS) / (epsilon_values * epsilon_0))
    Recombination_Bimolecular_EQ = (Recombination_Bimolecular_Cell * (nlocal * plocal - niPS * niPS))

    Recombination_Combined = (Recombination_Bimolecular_EQ) #Include more recombination mechanisms by adding them to this line

    eqn = (0.00 == -TransientTerm(coeff=q, var=nlocal) + DiffusionTerm(coeff=q * D * nmob.harmonicFaceValue, var=nlocal) - ExponentialConvectionTerm(coeff=q * nmob.harmonicFaceValue * (philocal.faceGrad + ChiCell.faceGrad + D * LogNcCell.faceGrad), var=nlocal) + q*gen_rate)
    eqp = (0.00 == -TransientTerm(coeff=q, var=plocal) + DiffusionTerm(coeff=q * D * pmob.harmonicFaceValue, var=plocal) + ExponentialConvectionTerm(coeff=q * pmob.harmonicFaceValue * (philocal.faceGrad + ChiCell.faceGrad + EgCell.faceGrad - D * LogNvCell.faceGrad), var=plocal) + q*gen_rate)
    eqa = (0.00 == -TransientTerm(coeff=q, var=alocal) + DiffusionTerm(coeff=q * D * anionmob.harmonicFaceValue, var=alocal) - ExponentialConvectionTerm(coeff=q * anionmob.harmonicFaceValue * (philocal.faceGrad + ChiCell_a.faceGrad), var=alocal))
    eqc = (0.00 == -TransientTerm(coeff=q, var=clocal) + DiffusionTerm(coeff=q * D * cationmob.harmonicFaceValue, var=clocal) + ExponentialConvectionTerm(coeff=q * cationmob.harmonicFaceValue * (philocal.faceGrad + ChiCell_c.faceGrad), var=clocal))
    eqpoisson = (0.00 == -TransientTerm(var=philocal) + DiffusionTerm(coeff=epsilon, var=philocal) + (q/epsilon_0) * (plocal - nlocal + clocal - alocal + NdCell - NaCell))

    # Shared iteration; device physics and equations remain above.
    residual, SweepCounter, residualarray = solve_gummel(
        fields=(philocal, nlocal, plocal, alocal, clocal),
        equations=(eqpoisson, eqn, eqp, eqa, eqc),
        dt=1e-11, max_dt=1e-06, tolerance=1e-15,
        damping=0.01, sweeps=1, max_steps=1000, enable_ions=False,
        min_dt=1e-11, min_steps=500, residual_growth_limit=1.1, timestep_growth=1.02, min_damping=0.01)

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

    return {"NMatrix": NMatrix, "PMatrix": PMatrix, "RecombinationMatrix": RecombinationMatrix, "GenValues_Matrix": GenValues_Matrix, "PotentialMatrix": PotentialMatrix, "Efield_matrix": Efield_matrix, "n": nlocal.globalValue.copy(), "p": plocal.globalValue.copy(), "phi": philocal.globalValue.copy(), "ChiMatrix": chiMatrix, "EgMatrix": EgMatrix, "psinvarmatrix": psinvarmatrix, "psipvarmatrix": psipvarmatrix, "AnionDensityMatrix": alocal.globalValue.copy(), "CationDensityMatrix": clocal.globalValue.copy(), "ResidualMatrix": residual, "SweepCounterMatrix": SweepCounter, "Recombination_Bimolecular_EQMatrix": Recombination_Bimolecular_EQMatrix, "ResidualArray": residualarray, "ConservativeJnInternal": ConservativeJnInternal, "ConservativeJpInternal": ConservativeJpInternal, "TerminalCurrentDensity": TerminalCurrentDensity, "BottomTerminalCurrentDensity": BottomTerminalCurrentDensity, "TopTerminalCurrentDensity": TopTerminalCurrentDensity}

def simulate_device(output_dir):
    prepare_voltage_output(output_dir)

    applied_voltages = np.arange(0.0, 0.6, 0.05)

    chunk_size = min(len(applied_voltages), max(1, multiprocessing.cpu_count() - 1))

    n_values = 1.00e-30
    p_values = 1.00e-30
    a_values = a_initial_values.flatten()
    c_values = c_initial_values.flatten()
    phi_values = 1.00e-30

    # Process voltages in sequential chunks
    for start in range(0, len(applied_voltages), chunk_size):
        # Create a chunk of voltages to simulate in parallel
        chunk_voltages = applied_voltages[start:start + chunk_size]

        # Parallel computation within the chunk
        chunk_results = Parallel(n_jobs=chunk_size, backend="multiprocessing")(delayed(solve_and_save_voltage)(solve_for_voltage, output_dir, start + offset, voltage, n_values, p_values, a_values, c_values, phi_values) for offset, voltage in enumerate(chunk_voltages))

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
