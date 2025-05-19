# -*- coding: utf-8 -*-
import os
os.environ["OMP_NUM_THREADS"] = "1" #Really important! Pysparse doesnt benefit from multithreading.
import numpy as np
import matplotlib
if os.name == 'nt':
    print("Windows")
else:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mark_interface_file import mark_interfaces, mark_interfaces_mixed
from calculate_absorption import calculate_absorption_above_bandgap
from fipy import CellVariable, TransientTerm, DiffusionTerm, ExponentialConvectionTerm
import time
import fipy
import pandas as pd
import cv2 as cv
from SmoothingFunction import smooth
from fipy.tools import numerix
from scipy.interpolate import griddata
from joblib import Parallel, delayed
import multiprocessing
from material_maps import MATERIALS
import copy

q = 1.60217646e-19 # Elementary charge, in Coulombs
k_B = 1.3806503e-23 #J/K
epsilon_0 = 8.85418782e-12
TInfinite = 300.0
D = k_B * TInfinite / q #=== kT in eV

Carbon_ID = 0
PS_ID = 100
TiO2_ID = 150
FTO_ID = 200
ZrO2_ID = 225
def map_material_property(devarray, prop):
    return np.vectorize(lambda x: getattr(MATERIALS[x], prop))(devarray)

StretchFactor = 1 #Can help convergence if a finer mesh is needed
SmoothFactor = 0.2 #Some smoothing helps with convergence

dx = 1.00e-9/StretchFactor #Pixel Width in meters
dy = 1.00e-9/StretchFactor #Pixel Width in meters

# Extract current file name
current_file = os.path.basename(__file__)[0:-3]
print("Current file name", current_file)

# Read the Excel file and skip the first two rows
data = pd.read_excel('./Solar_Spectrum.xls', skiprows=2)

SolarSpectrumWavelength = data.iloc[:, 0] #Units are nm
SolarSpectrumIrradiance = data.iloc[:, 2] #Units are W/(m2*nm)

# Round wavelengths to nearest integer
rounded_wavelengths = np.round(SolarSpectrumWavelength).astype(int)

# Get the indices to sort the rounded wavelengths
sorted_indices = np.argsort(rounded_wavelengths)

# Sort both wavelengths and irradiances, so they are aligned
sorted_wavelengths = rounded_wavelengths[sorted_indices]
sorted_irradiances = SolarSpectrumIrradiance[sorted_indices]

# Sum the irradiance for each unique rounded wavelength to get the average
unique_wavelengths, inverse_indices = np.unique(sorted_wavelengths, return_inverse=True)

# Initialize the array for average irradiances
averaged_irradiances = np.zeros_like(unique_wavelengths, dtype=float)

# Sum the irradiance values for each unique wavelength
for i, wavelength in enumerate(unique_wavelengths):
    averaged_irradiances[i] = np.mean(sorted_irradiances[inverse_indices == i])

SolarSpectrumWavelength = unique_wavelengths
SolarSpectrumIrradiance = averaged_irradiances

#Importing Absorbance Coefficient Spectrum for MAPbI3
AbsorptionData = np.genfromtxt("MAPI_tailfit_nk 1.txt", delimiter=",", skip_header=1)
kdata = AbsorptionData[:, 2]
alphadata = 4 * np.pi * kdata / (AbsorptionData[:, 0] * 1.00e-9)

######Define Device Architecture
ProblemDimension = 1 #1D or 2D
DeviceArchitechture = np.empty((1650, 1))
DeviceArchitechture[0:1600,:] = 100 #1600nm PS Absorber
DeviceArchitechture[1600:1650,:] = 150 #50nm TiO2 ETL

GenRate_values_default = map_material_property(DeviceArchitechture, 'GenRate')
epsilon_values = map_material_property(DeviceArchitechture, 'epsilon')
pmob_values = map_material_property(DeviceArchitechture, 'pmob')
nmob_values = map_material_property(DeviceArchitechture, 'nmob')
Eg = map_material_property(DeviceArchitechture, 'Eg')
Eg_PS = map_material_property(PS_ID, 'Eg')
chi = map_material_property(DeviceArchitechture, 'chi')
cation_mob_values = map_material_property(DeviceArchitechture, 'cationmob')
anion_mob_values = map_material_property(DeviceArchitechture, 'anionmob')
Recombination_Langevin_values = map_material_property(DeviceArchitechture, 'Recombination_Langevin')
Recombination_Bimolecular_values = map_material_property(DeviceArchitechture, 'Recombination_Bimolecular')
Nc = map_material_property(DeviceArchitechture, 'Nc')
Nv = map_material_property(DeviceArchitechture, 'Nv')
chi_a = map_material_property(DeviceArchitechture, 'Chi_a')
chi_c = map_material_property(DeviceArchitechture, 'Chi_c')
a_initial_values = map_material_property(DeviceArchitechture, 'a_initial_level')
c_initial_values = map_material_property(DeviceArchitechture, 'c_initial_level')

EffectiveMediumApproximationVolumeFraction = 1.00

GenMode = 1
if GenMode == 1:
    #Lambert-Beer Law
    GenRate_values_default, ThermalisationHeat, PhotonFluxArray, TransmittedEnergy = calculate_absorption_above_bandgap(SolarSpectrumWavelength, SolarSpectrumIrradiance, AbsorptionData[:, 0], alphadata * EffectiveMediumApproximationVolumeFraction,GenRate_values_default, dx*StretchFactor, Eg_PS)
else:
    #Constant Generation Rate
    GenRate_values_default = GenRate_values_default * 2.20e27
    ThermalisationHeat = np.zeros_like(GenRate_values_default)
    PhotonFluxArray = np.zeros_like(GenRate_values_default)
    TransmittedEnergy = np.zeros_like(GenRate_values_default)

#Stretching in case finer meshing is needed
if ProblemDimension == 1:
    DeviceArchitechture = cv.resize(DeviceArchitechture, None, fx=1.0, fy=StretchFactor, interpolation=cv.INTER_NEAREST)
    GenRate_values_default = cv.resize(GenRate_values_default, None, fx=1.0, fy=StretchFactor, interpolation=cv.INTER_NEAREST)
    PhotonFluxArray = cv.resize(PhotonFluxArray, None, fx=1.0, fy=StretchFactor, interpolation=cv.INTER_NEAREST)
else:
    DeviceArchitechture = cv.resize(DeviceArchitechture, None, fx=StretchFactor, fy=StretchFactor, interpolation=cv.INTER_NEAREST)
    GenRate_values_default = cv.resize(GenRate_values_default, None, fx=StretchFactor, fy=StretchFactor, interpolation=cv.INTER_NEAREST)
    PhotonFluxArray = cv.resize(PhotonFluxArray, None, fx=StretchFactor, fy=StretchFactor, interpolation=cv.INTER_NEAREST)

def flatten_smooth(arr, smoothF):
    if smoothF > 0.0:
        arr = smooth(arr, smoothF)
    return arr.flatten()

print(DeviceArchitechture.shape)

nx = DeviceArchitechture.shape[1]
ny = DeviceArchitechture.shape[0]

#Smoothen Variables to improve numerical stability
epsilon_values = flatten_smooth(epsilon_values, SmoothFactor*StretchFactor)
pmob_values = flatten_smooth(pmob_values, SmoothFactor*StretchFactor)
nmob_values = flatten_smooth(nmob_values, SmoothFactor*StretchFactor)
chi = flatten_smooth(chi, SmoothFactor*StretchFactor)
chi_a = flatten_smooth(chi_a, SmoothFactor*StretchFactor)
chi_c = flatten_smooth(chi_c, SmoothFactor*StretchFactor)
LogNc = np.log(Nc)
Nc = flatten_smooth(Nc, SmoothFactor*StretchFactor)
LogNc = flatten_smooth(LogNc, SmoothFactor*StretchFactor)
LogNv = np.log(Nv)
Nv = flatten_smooth(Nv, SmoothFactor*StretchFactor)
LogNv = flatten_smooth(LogNv, SmoothFactor*StretchFactor)
Eg = flatten_smooth(Eg, SmoothFactor*StretchFactor)

gold_mask = np.where(DeviceArchitechture == Carbon_ID, 1.00, 0.00)
fto_mask = np.where(DeviceArchitechture == FTO_ID, 1.00, 0.00)

gold_mask = gold_mask.flatten()
fto_mask = fto_mask.flatten()

ElectrodeMask = gold_mask + fto_mask

#LocationSRH_HTL = mark_interfaces(DeviceArchitechture, 50, PS_ID)
#mark_interfaces() places the interface inside the absorber
LocationSRH_ETL = mark_interfaces(DeviceArchitechture, TiO2_ID, PS_ID)

#LocationHTL_Exact = mark_interfaces_mixed(DeviceArchitechture, 50, PS_ID, 0*StretchFactor)
#mark_interfaces_mixed() places the interface in the middle of the absorber and the transport layer
LocationETL_Exact = mark_interfaces_mixed(DeviceArchitechture, TiO2_ID, PS_ID, 3*StretchFactor)

SRH_Interfacial_Recombination_Zone = LocationETL_Exact

print("Number of ETL interface nm: ", 1.00e9*dx*(np.count_nonzero(LocationETL_Exact)-1)/(nx))
#print("Number of HTL interface nm: ", 1.00e9*dx*(np.count_nonzero(LocationHTL_Exact)-1)/(nx))

SRH_Bulk_Recombination_Zone = map_material_property(DeviceArchitechture, 'GenRate') - SRH_Interfacial_Recombination_Zone
#Make negative values zero
SRH_Bulk_Recombination_Zone = np.where(SRH_Bulk_Recombination_Zone < 0, 0.00, SRH_Bulk_Recombination_Zone)

#Here we define the Ohmic boundary conditions
nFTO = map_material_property(TiO2_ID, 'Nc') * np.exp(((map_material_property(TiO2_ID, 'chi') - map_material_property(FTO_ID, "WF")) / D))
nCarbon = map_material_property(PS_ID, 'Nc') * np.exp(((map_material_property(PS_ID, 'chi') - map_material_property(Carbon_ID, "WF")) / D))

pCarbon = map_material_property(PS_ID, 'Nv') * np.exp(((map_material_property(Carbon_ID, "WF") - (map_material_property(PS_ID, "chi") +map_material_property(PS_ID, "Eg"))) / D))
pFTO = map_material_property(TiO2_ID, 'Nv') * np.exp(((map_material_property(FTO_ID, "WF") - (map_material_property(TiO2_ID, "chi") +map_material_property(TiO2_ID, "Eg"))) / D))

############Recombination Constants############
tau_p_bulk = 5 * 1.00e-9
tau_n_bulk = 5 * 1.00e-9

tau_p_interface = 0.02 * 1.00e-9
tau_n_interface = 0.02 * 1.00e-9

n_hat = map_material_property(PS_ID, 'Nc') * np.exp((1.00 / 2.00) * (-Eg_PS / D))
p_hat = map_material_property(PS_ID, 'Nv') * np.exp((1.00 / 2.00) * (-Eg_PS / D))

n_hat_mixed = map_material_property(PS_ID, 'Nc') * np.exp((1.00 / 2.00) * (-((map_material_property(PS_ID, "chi") + map_material_property(PS_ID, "Eg"))-map_material_property(TiO2_ID, 'chi')) / D))
p_hat_mixed = map_material_property(PS_ID, 'Nv') * np.exp((1.00 / 2.00) * (-((map_material_property(PS_ID, "chi") + map_material_property(PS_ID, "Eg"))-map_material_property(TiO2_ID, 'chi')) / D))

niPS = np.sqrt(Nc * Nv * np.exp(-Eg / D))

def solve_for_voltage(voltage, dx, dy, nx, ny, ProblemDimension, SmoothFactor, StretchFactor, D, nFTO, nCarbon, pFTO, pCarbon , GenRate_values_default, Recombination_Langevin_values, Recombination_Bimolecular_values, SRH_Interfacial_Recombination_Zone, SRH_Bulk_Recombination_Zone, epsilon_values, n_values, nmob_values, p_values, pmob_values , a_values, anion_mob_values, c_values, cation_mob_values, phi_values, gold_mask, fto_mask, Nc, Nv, chi, chi_a, chi_c, Eg, TInfinite, tau_p_interface, tau_n_interface, tau_p_bulk, tau_n_bulk, epsilon_0, n_hat, p_hat, n_hat_mixed, p_hat_mixed, q, Eg_PS, niPS):
    if os.name == 'nt':
        print("Windows")
        solver = fipy.solvers.LinearLUSolver(precon=None, iterations=1) #Works out of the box with simple fipy installation, but slower than pysparse
    else:
        solver = fipy.solvers.pysparse.linearLUSolver.LinearLUSolver(precon=None, iterations=1) #Very fast solver

    # Step 6: Create the Mesh
    if ProblemDimension == 1:
        mesh = fipy.Grid1D(dx=dx, nx=ny)
        CarbonContactLocation = mesh.facesLeft
        FTOContactLocation = mesh.facesRight

    else:
        mesh = fipy.Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
        CarbonContactLocation = mesh.facesBottom
        FTOContactLocation = mesh.facesTop

    # Reshape GenRate_values to 1D array and set the values on the grid
    Upmesh_default = CellVariable(name="generation minus recombination rate", mesh=mesh, value=0.000)
    GenRate_values_default = GenRate_values_default.flatten()
    Upmesh_default.setValue(GenRate_values_default)

    Recombination_Langevin_Cell = CellVariable(name="Recombination_Langevin_Cell", mesh=mesh, value=0.000)
    Recombination_Langevin_values = Recombination_Langevin_values.flatten()
    Recombination_Langevin_Cell.setValue(Recombination_Langevin_values)

    Recombination_Bimolecular_Cell = CellVariable(name="Recombination_Bimolecular_Cell", mesh=mesh, value=0.000)
    Recombination_Bimolecular_values = Recombination_Bimolecular_values.flatten()
    Recombination_Bimolecular_Cell.setValue(Recombination_Bimolecular_values)

    Recombination_Interfacial_SRH_Cell = CellVariable(name="Recombination_SRH_Cell", mesh=mesh, value=0.000)
    SRH_Interfacial_Recombination_Zone = flatten_smooth(SRH_Interfacial_Recombination_Zone, SmoothFactor * StretchFactor)
    Recombination_Interfacial_SRH_Cell.setValue(SRH_Interfacial_Recombination_Zone)

    Recombination_Bulk_SRH_Cell = CellVariable(name="Recombination_SRH_Cell", mesh=mesh, value=0.000)
    SRH_Bulk_Recombination_Zone = flatten_smooth(SRH_Bulk_Recombination_Zone, SmoothFactor * StretchFactor)
    Recombination_Bulk_SRH_Cell.setValue(SRH_Bulk_Recombination_Zone)

    anionmob = CellVariable(name="anion mobility", mesh=mesh, value=0.00)
    cationmob = CellVariable(name="cation mobility", mesh=mesh, value=0.00)
    anion_mob_values = anion_mob_values.flatten()
    cation_mob_values = cation_mob_values.flatten()
    anionmob.setValue(anion_mob_values)
    cationmob.setValue(cation_mob_values)

    pmob_default = CellVariable(name="hole mobility", mesh=mesh, value=0.00)
    nmob_default = CellVariable(name="electron mobility", mesh=mesh, value=0.00)
    pmob_default.setValue(pmob_values)
    nmob_default.setValue(nmob_values)

    epsilon = CellVariable(name="dielectric permittivity", mesh=mesh, value=0.00)
    epsilon.setValue(epsilon_values)

    gold_mask_cellvar = CellVariable(name="gold", mesh=mesh, value=0.00)
    gold_mask = gold_mask.flatten()
    gold_mask_cellvar.setValue(gold_mask)

    fto_mask_cellvar = CellVariable(name="fto", mesh=mesh, value=0.00)
    fto_mask = fto_mask.flatten()
    fto_mask_cellvar.setValue(fto_mask)

    pmob = pmob_default  # Default value
    nmob = nmob_default
    gen_rate = Upmesh_default  # Placeholder for example extension
    NcCell = CellVariable(name="Effective Density of States CB", mesh=mesh, value=0.00)
    NcCell.setValue(Nc.flatten())
    NvCell = CellVariable(name="Effective Density of States VB", mesh=mesh, value=0.00)
    NvCell.setValue(Nv.flatten())
    LogNcCell = CellVariable(name="Log Effective Density of States CB", mesh=mesh, value=0.00)
    LogNcCell.setValue(LogNc)
    LogNvCell = CellVariable(name="Log Effective Density of States VB", mesh=mesh, value=0.00)
    LogNvCell.setValue(LogNv)
    philocal = CellVariable(name="solution variable", mesh=mesh, value=0.00, hasOld=True)
    philocal.setValue(phi_values)
    nlocal = CellVariable(name="electron density", mesh=mesh, value=niPS, hasOld=True)
    plocal = CellVariable(name="hole density", mesh=mesh, value=niPS, hasOld=True)
    nlocal.setValue(n_values)
    plocal.setValue(p_values)

    alocal = CellVariable(name="anion density", mesh=mesh, value=0.00)
    alocal.setValue(a_values)

    clocal = CellVariable(name="cation density", mesh=mesh, value=0.00)
    clocal.setValue(c_values)

    LoadOldSolution = False

    if LoadOldSolution:
        voltage_sweep_output_dir_old = "./SimulateFIPY2DReal/OutputsIntermediate/Miguel_Eq_2D_24_03_2025_150nm_2k_Iter_Jn_Jp_10ns_SRH_0dot9V_PS_Nc_Nv1e25_Gummel_3XSweep_Fixed1e-9dt_Smooth0dot2/VoltageSweep"

        phi_values_old = np.load(voltage_sweep_output_dir_old + "/phi.npy")
        n_values_old = np.load(voltage_sweep_output_dir_old + "/n.npy")
        p_values_old = np.load(voltage_sweep_output_dir_old + "/p.npy")
        a_values_old = np.load(voltage_sweep_output_dir_old + "/AnionDensityMatrix.npy")
        c_values_old = np.load(voltage_sweep_output_dir_old + "/CationDensityMatrix.npy")
        applied_voltages_old = np.load(voltage_sweep_output_dir_old + "/applied_voltages.npy")

        voltagevalue = np.where(applied_voltages_old == voltage)[0][0]
        philocal.setValue(phi_values_old[voltagevalue])
        nlocal.setValue(n_values_old[voltagevalue])
        plocal.setValue(p_values_old[voltagevalue])
        alocal.setValue(a_values_old[voltagevalue])
        clocal.setValue(c_values_old[voltagevalue])

    nlocal.constrain(nFTO, where=FTOContactLocation)
    nlocal.constrain(nCarbon, where=CarbonContactLocation)
    plocal.constrain(pFTO, where=FTOContactLocation)
    plocal.constrain(pCarbon, where=CarbonContactLocation)

    ChiCell = CellVariable(name="Electron Affinity", mesh=mesh, value=0.00)
    ChiCell.setValue(chi.flatten())
    ChiCell_a = CellVariable(name="Electron Affinity", mesh=mesh, value=0.00)
    ChiCell_a.setValue(chi_a.flatten())
    ChiCell_c = CellVariable(name="Electron Affinity", mesh=mesh, value=0.00)
    ChiCell_c.setValue(chi_c.flatten())
    EgCell = CellVariable(name="Band Gap", mesh=mesh, value=0.00)
    EgCell.setValue(Eg.flatten())

    ZerosCellVariable = CellVariable(name="Zeros", mesh=mesh, value=0.00)

    phih = map_material_property(Carbon_ID, "WF")
    phin = map_material_property(FTO_ID, "WF")
    Vbi = (phih - phin)

    philocal.constrain(0.00, where=FTOContactLocation)
    philocal.constrain(-(Vbi - voltage), where=CarbonContactLocation)

    Recombination_Langevin_EQ = (Recombination_Langevin_Cell * q * (pmob + nmob) * (nlocal * plocal - niPS * niPS) / (epsilon_values * epsilon_0))
    Recombination_Bimolecular_EQ = (Recombination_Bimolecular_Cell * (nlocal * plocal - niPS * niPS))
    Recombination_SRH_Interfacial_EQ = (Recombination_Interfacial_SRH_Cell * (nlocal * plocal - niPS * niPS) / (tau_p_interface * (nlocal + n_hat) + tau_n_interface * (plocal + p_hat)))
    Recombination_SRH_Interfacial_Mixed_EQ = (Recombination_Interfacial_SRH_Cell * (nlocal * plocal - niPS * niPS) / (tau_p_interface * (nlocal + n_hat_mixed) + tau_n_interface * (plocal + p_hat_mixed)))
    Recombination_SRH_Bulk_EQ = (Recombination_Bulk_SRH_Cell * (nlocal * plocal - niPS * niPS) / (tau_p_bulk * (nlocal + n_hat) + tau_n_bulk * (plocal + p_hat)))

    Recombination_Langevin_EQ_ReLU = numerix.fmax(Recombination_Langevin_EQ, ZerosCellVariable)
    Recombination_Bimolecular_EQ_ReLU = numerix.fmax(Recombination_Bimolecular_EQ, ZerosCellVariable)
    Recombination_SRH_Interfacial_EQ_ReLU = numerix.fmax(Recombination_SRH_Interfacial_EQ, ZerosCellVariable)
    Recombination_SRH_Interfacial_Mixed_EQ_ReLU = numerix.fmax(Recombination_SRH_Interfacial_Mixed_EQ, ZerosCellVariable)
    Recombination_SRH_Bulk_EQ_ReLU = numerix.fmax(Recombination_SRH_Bulk_EQ, ZerosCellVariable)

    Recombination_Combined = (Recombination_Bimolecular_EQ_ReLU + Recombination_SRH_Bulk_EQ_ReLU + Recombination_SRH_Interfacial_Mixed_EQ_ReLU) #Include more recombination mechanisms by adding them to this line

    LUMO = philocal + ChiCell
    HOMO = philocal + ChiCell + EgCell

    LUMO_a = philocal + ChiCell_a
    LUMO_c = philocal + ChiCell_c

    eq1 = (0.00 == -TransientTerm(coeff=q, var=nlocal) + DiffusionTerm(coeff=q * D * nmob.harmonicFaceValue, var=nlocal) - ExponentialConvectionTerm(coeff=q * nmob.harmonicFaceValue * (LUMO + D*LogNcCell).faceGrad, var=nlocal) + q*gen_rate - q*Recombination_Combined)
    eq2 = (0.00 == -TransientTerm(coeff=q, var=plocal) + DiffusionTerm(coeff=q * D * pmob.harmonicFaceValue, var=plocal) + ExponentialConvectionTerm(coeff=q * pmob.harmonicFaceValue * (HOMO - D*LogNvCell).faceGrad, var=plocal) + q*gen_rate - q*Recombination_Combined)
    eq3 = (0.00 == -TransientTerm(coeff=q, var=alocal) + DiffusionTerm(coeff=q * D * anionmob.harmonicFaceValue, var=alocal) - ExponentialConvectionTerm(coeff=q * anionmob.harmonicFaceValue * LUMO_a.faceGrad, var=alocal))
    eq4 = (0.00 == -TransientTerm(coeff=q, var=clocal) + DiffusionTerm(coeff=q * D * cationmob.harmonicFaceValue, var=clocal) + ExponentialConvectionTerm(coeff=q * cationmob.harmonicFaceValue * LUMO_c.faceGrad, var=clocal))
    eq5 = (0.00 == -TransientTerm(var=philocal) + DiffusionTerm(coeff=epsilon, var=philocal) + (q/epsilon_0) * (plocal - nlocal + clocal - alocal))

    eqconteh = eq1 & eq2 #Electron and hole continuity equations
    eqcontac = eq3 & eq4 #Anion and cation continuity equations
    eqpoisson = eq5 #Poisson equation

    max_iterations = 1000 # Maximum iterations
    dt = 1.00e-9 #Starting time step should be small
    dt_old = dt
    MaxTimeStep = 1.00e-5 #Increasing above 1.00e-5 sometimes leads to artefacts in the solution even if the residual is small
    desired_residual = 1.00e-10
    SweepCounter = 0
    residual = 1.00
    TotalTime = 0.00
    residualarray = [1000]
    residual_old = 1.00e10
    DampingFactor = 0.05 #Very important parameter!, stiff problems may require a smaller value
    NumberofSweeps = 1 #Number of sweeps at same time step, for very stiff problems it can be increased to 2 or 3, but at the cost of compute time.

    nold = nlocal.value
    pold = plocal.value
    phiold = philocal.value
    aold = alocal.value
    cold = clocal.value

    while SweepCounter < max_iterations and residual > desired_residual:

        t0 = time.time()

        for i in range(NumberofSweeps):
            eqpoisson.sweep(dt = dt, solver=solver)
            philocal.setValue(DampingFactor * philocal.value + (1 - DampingFactor) * phiold) #The potential should be damped BEFORE passing to the continuity equations!
            phiold = np.copy(philocal.value)

            residual = eqconteh.sweep(dt = dt, solver=solver)
            nlocal.setValue(np.maximum(nlocal, 1.00e-30))
            plocal.setValue(np.maximum(plocal, 1.00e-30))
            nlocal.setValue(DampingFactor * nlocal.value + (1 - DampingFactor) * nold)
            plocal.setValue(DampingFactor * plocal.value + (1 - DampingFactor) * pold)
            nold = np.copy(nlocal.value)
            pold = np.copy(plocal.value)

        EnableIons = True
        if EnableIons:
            #Here the ionic continuity equations are solved
            residualac = eqcontac.sweep(dt=dt, solver=solver)
            residual = residual + residualac
            alocal.setValue(DampingFactor * alocal.value + (1 - DampingFactor) * aold)
            clocal.setValue(DampingFactor * clocal.value + (1 - DampingFactor) * cold)
            aold = np.copy(alocal.value)
            cold = np.copy(clocal.value)

        PercentageImprovementPerSweep = (1 - (residual / residual_old) * dt_old / dt) * 100

        if residual > residual_old * 1.20:
            dt = dt * 0.1
            if dt < 1.00e-11:
                dt = 1.00e-11
        else:
            dt = dt * 1.05
            if dt > MaxTimeStep:
                dt = MaxTimeStep

        dt_old = dt
        residual_old = residual

        #Update old
        nlocal.updateOld()
        plocal.updateOld()
        philocal.updateOld()

        TotalTime = TotalTime + dt
        residualarray.append(residual/dt)

        print("Sweep: ", SweepCounter, "TotalTime: ", TotalTime, "Residual: ", 1.00e-9*residual/dt, "Time for sweep: ", time.time() - t0, "dt: ", dt, "Percentage Improvement: ", PercentageImprovementPerSweep, "Damping: ", DampingFactor)
        SweepCounter += 1

    #plot the residual vs total computational time
    residualarray.pop(0)

    # Here the electron and hole quasi-fermi levels are calculated
    psinvar = LUMO - D * numerix.log((nlocal / (NcCell)))
    psipvar = HOMO + D * numerix.log((plocal / (NvCell)))

    #Here the electron and hole current densities are calculated
    Jn = (q * nmob.globalValue * nlocal.globalValue * -psinvar.grad.globalValue) #Vector Quantity
    Jp = (q * pmob.globalValue * plocal.globalValue * -psipvar.grad.globalValue) #Vector Quantity
    Jph = Jn + Jp #Vector Quantity

    # Calculation of E-field
    E = -philocal.grad  #Vector Quantity

    # Calculation of E-field magnitude
    Efield = E.mag  #Scalar Quantity

    #Reshaping variables back into square grid
    Efield_matrix = np.reshape(Efield.globalValue, (ny, nx))
    PotentialMatrix = np.reshape(philocal.globalValue, (ny, nx))
    GenValues_Matrix = np.reshape(gen_rate.globalValue, (ny, nx))
    RecombinationMatrix = (np.reshape(Recombination_Combined.globalValue,(ny, nx)))
    Recombination_Bimolecular_EQ_ReLUMatrix = (np.reshape(Recombination_Bimolecular_EQ_ReLU.globalValue,(ny, nx)))
    NMatrix = np.reshape(nlocal.globalValue, (ny, nx))
    PMatrix = np.reshape(plocal.globalValue, (ny, nx))

    if ProblemDimension == 1:
        J_Total_Y = np.reshape(numerix.dot(Jph, [1]), (ny, nx))
        Jn_Matrix = np.reshape(numerix.dot(Jn, [1]), (ny, nx))
        Jp_Matrix = np.reshape(numerix.dot(Jp, [1]), (ny, nx))
    else:
        J_Total_Y = np.reshape(Jph[1], (ny, nx))
        Jn_Matrix = np.reshape(Jn[1], (ny, nx))
        Jp_Matrix = np.reshape(Jp[1], (ny, nx))

    chiMatrix = np.reshape(ChiCell.globalValue, (ny, nx))
    EgMatrix = np.reshape(EgCell.globalValue, (ny, nx))

    psinvarmatrix = np.reshape(psinvar.globalValue, (ny, nx))
    psipvarmatrix = np.reshape(psipvar.globalValue, (ny, nx))

    return {"E": E, "NMatrix": NMatrix, "PMatrix": PMatrix, "RecombinationMatrix": RecombinationMatrix, "GenValues_Matrix": GenValues_Matrix, "PotentialMatrix": PotentialMatrix, "Efield_matrix": Efield_matrix, "J_Total_Y": J_Total_Y, "n": nlocal.globalValue, "p": plocal.globalValue, "phi": philocal.globalValue, "ChiMatrix": chiMatrix, "EgMatrix": EgMatrix, "psinvarmatrix": psinvarmatrix, "psipvarmatrix": psipvarmatrix, "AnionDensityMatrix": alocal.globalValue, "CationDensityMatrix": clocal.globalValue, "ResidualMatrix": residual, "SweepCounterMatrix": SweepCounter, "Jn_Matrix": Jn_Matrix, "Jp_Matrix": Jp_Matrix, "Recombination_Bimolecular_EQ_ReLUMatrix": Recombination_Bimolecular_EQ_ReLUMatrix}

def simulate_device(output_dir, additional_voltages=None, GenRate_values_default=GenRate_values_default, Recombination_Langevin_values=Recombination_Langevin_values, Recombination_Bimolecular_values=Recombination_Bimolecular_values, SRH_Interfacial_Recombination_Zone=SRH_Interfacial_Recombination_Zone, SRH_Bulk_Recombination_Zone=SRH_Bulk_Recombination_Zone):

    # Determine voltages to simulate
    if additional_voltages is not None:
        applied_voltages = additional_voltages
    else:
        applied_voltages = np.arange(0.0, 1.2, 0.1)

    if len(applied_voltages) < multiprocessing.cpu_count() - 1:
        chunk_size = len(applied_voltages)
    else:
        chunk_size = multiprocessing.cpu_count() - 1

    n_values = 0.00
    p_values = 0.00
    a_values = a_initial_values.flatten()
    c_values = c_initial_values.flatten()
    phi_values = 0.00

    def append_to_npy(filename, new_data):
        new_data = np.expand_dims(new_data, axis=0)

        if os.path.isfile(os.path.join(output_dir, filename)):
            existing_data = np.load(os.path.join(output_dir, filename))
            fulldata = np.concatenate((existing_data, new_data), axis=0)
            np.save(os.path.join(output_dir, filename), fulldata)
        else:
            # Create the file if it doesn't exist
            np.save(os.path.join(output_dir, filename), new_data)

    # Process voltages in sequential chunks
    for start in range(0, len(applied_voltages), chunk_size):
        # Create a chunk of voltages to simulate in parallel
        chunk_voltages = applied_voltages[start:start + chunk_size]

        # Parallel computation within the chunk
        chunk_results = Parallel(n_jobs=chunk_size, backend="multiprocessing")(delayed(solve_for_voltage)(voltage, dx, dy, nx, ny, ProblemDimension, SmoothFactor, StretchFactor, D, nFTO, nCarbon, pFTO, pCarbon, GenRate_values_default, Recombination_Langevin_values, Recombination_Bimolecular_values, SRH_Interfacial_Recombination_Zone, SRH_Bulk_Recombination_Zone, epsilon_values, n_values, nmob_values, p_values, pmob_values , a_values, anion_mob_values, c_values, cation_mob_values, phi_values, gold_mask, fto_mask, Nc, Nv, chi, chi_a, chi_c, Eg, TInfinite, tau_p_interface, tau_n_interface, tau_p_bulk, tau_n_bulk, epsilon_0, n_hat, p_hat, n_hat_mixed, p_hat_mixed, q, Eg_PS, niPS) for voltage in chunk_voltages)

        #DeepCopy To avoid overwriting the results in next loop
        copied_result = [copy.deepcopy(r) for r in chunk_results]

        # Save dictionary of chunk_results as .npy files named after the key
        for result in copied_result:
            for key, value in result.items():
                append_to_npy(f"{key}.npy", value)

        #Save an array of all the voltages applied so far
        np.save(os.path.join(output_dir, "applied_voltages.npy"), applied_voltages[:start + len(chunk_voltages)])

        # Update initial conditions using results from the last voltage in the chunk to speed up convergence of the next chunk
        last_result = chunk_results[-1]  # The last result in the current chunk
        n_values = last_result["n"]  # Update `n` from the last result
        p_values = last_result["p"]  # Update `p` from the last result
        a_values = last_result["AnionDensityMatrix"]  # Update `a` from the last result
        c_values = last_result["CationDensityMatrix"]  # Update `c` from the last result
        phi_values = last_result["phi"]  # Update `phi` from the last result
    return copied_result

def main_workflow():
    voltage_sweep_output_dir = "./SimulateFIPY2DReal/OutputsIntermediate/" + current_file + "/VoltageSweep"

    if not os.path.exists(voltage_sweep_output_dir):
        os.makedirs(voltage_sweep_output_dir)

    print("Starting standard voltage sweep...")
    results = simulate_device(output_dir=voltage_sweep_output_dir)
    print("Voltage sweep completed.")
    return results

# Fix for multiprocessing on Windows
if __name__ == '__main__':
    main_workflow()