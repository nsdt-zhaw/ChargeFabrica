# -*- coding: utf-8 -*-
#This code is a simulation of a 2D carbon-based triple mesoscopic HTL-free device using the finite volume method with the FiPy library.
#Device architecture: FTO (Boundary)|TiO2 (50 nm)|m-TiO2/MAPbI3 (150 nm)|m-ZrO2.txt/MAPbI3 (1000 nm)|MAPbI3 (100 nm)|Carbon (Boundary)
import os
os.environ["OMP_NUM_THREADS"] = "1" #Really important! Pysparse doesnt benefit from multithreading.
import numpy as np
from mark_interface_file import mark_interfaces, mark_interfaces_mixed
from calculate_absorption import calculate_absorption_above_bandgap
from fipy import TransientTerm, DiffusionTerm, ExponentialConvectionTerm
import fipy
from fipy.tools import numerix
from gummel_solver import solve_gummel
from scipy.ndimage import zoom
from joblib import Parallel, delayed
import multiprocessing
from functools import partial
from material_maps import Semiconductors, map_semiconductor_property, map_electrode_property, name_to_code_SC, name_to_code_EL
from BoundaryConditions import ohmic
from constantsfile import q, epsilon_0, D
from LoadSolarSpectrum import SolarSpectrumWavelength, SolarSpectrumIrradiance
from workflow_utils import run_sweep, prepare_voltage_output, solve_and_save_voltage
from electrical_numerics import device_field, as_cell_array, cell_variable, conservative_internal_face_currents, terminal_current_densities

Carbon_ID = name_to_code_EL["Carbon"]
PS_ID = name_to_code_SC["PS"]
TiO2_ID = name_to_code_SC["mTiO2"]
FTO_ID = name_to_code_EL["FTO"]
ZrO2_ID = name_to_code_SC["ZrO2"]

StretchFactor = 1 #Can help convergence if a finer mesh is needed
SmoothFactor = 0.2 #Some smoothing helps with convergence

dx = 1.00e-9/StretchFactor #Pixel Width in meters
dy = 1.00e-9/StretchFactor #Pixel Width in meters

#Importing Absorbance Coefficient Spectrum for MAPbI3
AbsorptionData = np.genfromtxt("MAPI_tailfit_nk 1.txt", delimiter=",", skip_header=1)
kdata = AbsorptionData[:, 2]
alphadata = 4 * np.pi * kdata / (AbsorptionData[:, 0] * 1.00e-9)

######Define 2D Device Architecture
ZirconiaLength = 1000 #1000nm of m-ZrO2.txt
MesoLength = 1150 #1150nm of m-TiO2 + m-ZrO2.txt

# Initialize array parameters for the sinosoidal structure
SinusoidalArray = np.ones((MesoLength, 100), dtype=int) * PS_ID  # You can modify shape
Amplitude = 20
dwidth = 35
desired_wavelength = 60  # explicitly set wavelength (the length of one full sine wave period)
phase = 0  # fixed phase to start from the top of the sine wave

# Compute Frequency based on desired fixed wavelength
Frequency = SinusoidalArray.shape[0] / desired_wavelength

# Calculate the vertical coordinate
y_coords = np.arange(SinusoidalArray.shape[0])

# Calculate horizontal positions for the sine wave:
center_x = SinusoidalArray.shape[1] // 2

x_positions = center_x + (Amplitude * np.sin(2 * np.pi * Frequency * y_coords / SinusoidalArray.shape[0] + phase)).astype(int)

# Insert sine wave into your SinusoidalArray
for dyyy in range(0, MesoLength-ZirconiaLength):
    xpos = x_positions[dyyy]

    x_start = max(xpos - dwidth // 2, 0)
    x_end = min(xpos + dwidth // 2 + 1, SinusoidalArray.shape[1])  # +1 to include endpoint

    SinusoidalArray[dyyy, x_start:x_end] = TiO2_ID

# Insert sine wave into your SinusoidalArray
for dyyy in range(MesoLength-ZirconiaLength, MesoLength):
    xpos = x_positions[dyyy]

    x_start = max(xpos - dwidth // 2, 0)
    x_end = min(xpos + dwidth // 2 + 1, SinusoidalArray.shape[1])  # +1 to include endpoint

    SinusoidalArray[dyyy, x_start:x_end] = ZrO2_ID

#flip the SinusoidalArray
SinusoidalArray = np.flip(SinusoidalArray, axis=0)

DeviceArchitecture = np.empty((MesoLength + 150, 100))

DeviceArchitecture[0:100,:] = PS_ID
DeviceArchitecture[100:MesoLength+100,:] = SinusoidalArray
DeviceArchitecture[(MesoLength+100):(MesoLength + 150),:] = TiO2_ID

TopElectrode = FTO_ID
TopLocationSC = DeviceArchitecture[-1,:].flatten() #Semiconducting materials adjacent to the top electrode
BottomLocationSC = DeviceArchitecture[0,:].flatten() #Semiconducting materials adjacent to the bottom electrode
BottomElectrode = Carbon_ID

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
tau_p_bulk = 5 * 1.00e-9
tau_n_bulk = 5 * 1.00e-9
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

def solve_for_voltage(voltage, n_values, p_values, a_values, c_values, phi_values):

    state_names = ("electrostatic potential", "electron density", "hole density", "anion density", "cation density")
    state_values = (phi_values, n_values, p_values, a_values, c_values)
    philocal, nlocal, plocal, alocal, clocal = [cell_variable(mesh, name, value, True) for name, value in zip(state_names, state_values)]

    for boundary, n_contact, p_contact, phi_contact in (
            (mesh.facesTop, nTop, pTop, 0),
            (mesh.facesBottom, nBottom, pBottom, -(Vbi - voltage))):
        nlocal.constrain(n_contact, where=boundary)
        plocal.constrain(p_contact, where=boundary)
        philocal.constrain(phi_contact, where=boundary)

    #Band-to-band recombination models
    Recombination_Langevin_EQ = (Recombination_Langevin_Cell * q * (pmob + nmob) * (nlocal * plocal - niPS * niPS) / (epsilon.value * epsilon_0))
    Recombination_Bimolecular_EQ = (Recombination_Bimolecular_Cell * (nlocal * plocal - niPS * niPS))

    #SRH trap assisted recombination models
    Recombination_SRH_Interfacial_EQ = (Recombination_Interfacial_SRH_Cell * (nlocal * plocal - niPS * niPS) / (tau_p_interface * (nlocal + n_hat) + tau_n_interface * (plocal + p_hat)))
    Recombination_SRH_Interfacial_Mixed_EQ = (Recombination_Interfacial_SRH_Cell * (nlocal * plocal - niPS * niPS) / (tau_p_interface * (nlocal + n_hat_mixed) + tau_n_interface * (plocal + p_hat_mixed)))
    Recombination_SRH_Bulk_EQ = (Recombination_Bulk_SRH_Cell * (nlocal * plocal - niPS * niPS) / (tau_p_bulk * (nlocal + n_hat) + tau_n_bulk * (plocal + p_hat)))

    Recombination_Combined = (Recombination_Bimolecular_EQ + Recombination_SRH_Bulk_EQ + Recombination_SRH_Interfacial_Mixed_EQ) #Include more recombination mechanisms by adding them to this line

    # Drift-driving gradients for the four mobile species.
    electron_drift = philocal.faceGrad + ChiCell.faceGrad + D*LogNcCell.faceGrad
    hole_drift = philocal.faceGrad + ChiCell.faceGrad + EgCell.faceGrad - D*LogNvCell.faceGrad
    anion_drift = philocal.faceGrad + ChiCell_a.faceGrad
    cation_drift = philocal.faceGrad + ChiCell_c.faceGrad
    eqn = (0.00 == -TransientTerm(coeff=q, var=nlocal) + DiffusionTerm(coeff=q * D * nmob.harmonicFaceValue, var=nlocal) - ExponentialConvectionTerm(coeff=q * nmob.harmonicFaceValue * electron_drift, var=nlocal) + q*gen_rate - q*Recombination_Combined)
    eqp = (0.00 == -TransientTerm(coeff=q, var=plocal) + DiffusionTerm(coeff=q * D * pmob.harmonicFaceValue, var=plocal) + ExponentialConvectionTerm(coeff=q * pmob.harmonicFaceValue * hole_drift, var=plocal) + q*gen_rate - q*Recombination_Combined)
    eqa = (0.00 == -TransientTerm(coeff=q, var=alocal) + DiffusionTerm(coeff=q * D * anionmob.harmonicFaceValue, var=alocal) - ExponentialConvectionTerm(coeff=q * anionmob.harmonicFaceValue * anion_drift, var=alocal))
    eqc = (0.00 == -TransientTerm(coeff=q, var=clocal) + DiffusionTerm(coeff=q * D * cationmob.harmonicFaceValue, var=clocal) + ExponentialConvectionTerm(coeff=q * cationmob.harmonicFaceValue * cation_drift, var=clocal))
    eqpoisson = (0.00 == -TransientTerm(var=philocal) + DiffusionTerm(coeff=epsilon, var=philocal) + (q/epsilon_0) * (plocal - nlocal + clocal - alocal + NdCell - NaCell))

    # Shared iteration; device physics and equations remain above.
    residual, SweepCounter, residualarray = solve_gummel(
        fields=(philocal, nlocal, plocal, alocal, clocal),
        equations=(eqpoisson, eqn, eqp, eqa, eqc),
        dt=1e-09, max_dt=1e-07, tolerance=1e-10,
        damping=0.01, sweeps=1, max_steps=2000, enable_ions=True)

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
                  TopTerminalCurrentDensity=top_current)
    return result

def simulate_device(output_dir):
    prepare_voltage_output(output_dir)

    applied_voltages = np.arange(0.0, 1.15, 0.05)

    chunk_size = min(len(applied_voltages), max(1, multiprocessing.cpu_count() - 1))

    # State order: electrons, holes, anions, cations, potential.
    state = (1.00e-30, 1.00e-30, a_initial_values.flatten(), c_initial_values.flatten(), 1.00e-30)

    # Process voltages in sequential chunks
    for start in range(0, len(applied_voltages), chunk_size):
        chunk_voltages = applied_voltages[start:start + chunk_size]

        chunk_results = Parallel(n_jobs=chunk_size, backend="multiprocessing")(delayed(solve_and_save_voltage)(solve_for_voltage, output_dir, start + offset, voltage, *state) for offset, voltage in enumerate(chunk_voltages))

        state = tuple(chunk_results[-1][key] for key in
                      ('n', 'p', 'AnionDensityMatrix', 'CationDensityMatrix', 'phi'))
    return chunk_results

def main_workflow():
    return run_sweep(simulate_device, __file__, "VoltageSweep", "Starting standard voltage sweep...", "Voltage sweep completed.")

# Fix for multiprocessing on Windows
if __name__ == '__main__':

    main_workflow()
