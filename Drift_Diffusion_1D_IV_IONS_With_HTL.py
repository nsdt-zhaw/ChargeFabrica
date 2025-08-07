# -*- coding: utf-8 -*-
#This code is a simulation of a NIP planar perovskite solar cell using the finite volume method with the FiPy library.
#Device architecture: FTO (Boundary)|TiO2 (50 nm)|MAPbI3 (400 nm)|Spiro-OMeTAD (50 nm)|Gold (Boundary)
import os
os.environ["OMP_NUM_THREADS"] = "1" #Really important! Pysparse doesnt benefit from multithreading.
import numpy as np
from mark_interface_file import mark_interfaces, mark_interfaces_mixed
from calculate_absorption import calculate_absorption_above_bandgap
from fipy import CellVariable, TransientTerm, DiffusionTerm, ExponentialConvectionTerm, Variable
import fipy
from fipy.tools import numerix
import time
import pandas as pd
from scipy.ndimage import zoom
from SmoothingFunction import flatten_and_smooth_all
from joblib import Parallel, delayed
import multiprocessing
from material_maps import Semiconductors, Electrodes
import copy

q = 1.60217646e-19 # Elementary charge, in Coulombs
k_B = 1.3806503e-23 #J/K
epsilon_0 = 8.85418782e-12
TInfinite = 300.0
D = k_B * TInfinite / q #=== kT in eV

name_to_code_SC = {mat.name: mat.code for mat in Semiconductors.values()}
name_to_code_EL = {mat.name: mat.code for mat in Electrodes.values()}

Gold_ID = name_to_code_EL["Gold"]
Spiro_ID = name_to_code_SC["Spiro"]
PS_ID = name_to_code_SC["PS"]
TiO2_ID = name_to_code_SC["mTiO2_2"]
FTO_ID = name_to_code_EL["FTO2"]

def map_semiconductor_property(devarray, prop):
    return np.vectorize(lambda x: getattr(Semiconductors[x], prop))(devarray)

def map_electrode_property(devarray, prop):
    return np.vectorize(lambda x: getattr(Electrodes[x], prop))(devarray)

StretchFactor = 1 #Can help convergence if a finer mesh is needed
SmoothFactor = 0.2 #Some smoothing helps with convergence

dx = 1.00e-9/StretchFactor #Pixel Width in meters
dy = 1.00e-9/StretchFactor #Pixel Width in meters

data = pd.read_excel('./Solar_Spectrum.xls', skiprows=2)
data['RoundedWavelength'] = data.iloc[:,0].round().astype(int)
grouped = data.groupby('RoundedWavelength').mean()
SolarSpectrumWavelength = grouped.index.values
SolarSpectrumIrradiance = grouped.iloc[:, 2].values

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
TopLocationSC = DeviceArchitechture[-1,:] #Semiconducting materials adjacent to the top electrode
BottomLocationSC = DeviceArchitechture[0,:] #Semiconducting materials adjacent to the bottom electrode
BottomElectrode = Gold_ID

EffectiveMediumApproximationVolumeFraction = 1.00
GenRate_values_default = map_semiconductor_property(DeviceArchitechture, 'GenRate') #Binary array for whether generation is enabled or not

GenMode = 1
if GenMode == 1:
    #Lambert-Beer Law
    GenRate_values_default, ThermalisationHeat, PhotonFluxArray, TransmittedEnergy = calculate_absorption_above_bandgap(SolarSpectrumWavelength, SolarSpectrumIrradiance, AbsorptionData[:, 0], alphadata * EffectiveMediumApproximationVolumeFraction,GenRate_values_default, dx*StretchFactor, map_semiconductor_property(PS_ID, "Eg"))
else:
    #Constant Generation Rate
    GenRate_values_default = GenRate_values_default * 2.20e27

#Stretching in case finer meshing is needed (Stretching of generation array is done afterwards since the 3D hyperspectral generation array may exhaust RAM on machine)
if DeviceArchitechture.shape[1] == 1:
    DeviceArchitechture = zoom(DeviceArchitechture, [StretchFactor, 1], order=0)
    GenRate_values_default = zoom(GenRate_values_default, [StretchFactor, 1], order=0)
else:
    DeviceArchitechture = zoom(DeviceArchitechture, [StretchFactor, StretchFactor], order=0)
    GenRate_values_default = zoom(GenRate_values_default, [StretchFactor, StretchFactor], order=0)

epsilon_values = map_semiconductor_property(DeviceArchitechture, 'epsilon') #Dielectric constant (Unitless)
pmob_values = map_semiconductor_property(DeviceArchitechture, 'pmob') #Hole Mobility (m^2/Vs)
nmob_values = map_semiconductor_property(DeviceArchitechture, 'nmob') #Electron Mobility (m^2/Vs)
Eg = map_semiconductor_property(DeviceArchitechture, 'Eg') #Band Gap (eV)
chi = map_semiconductor_property(DeviceArchitechture, 'chi') #Electron Affinity (eV)
cation_mob_values = map_semiconductor_property(DeviceArchitechture, 'cationmob') #Cation Mobility (m^2/Vs)
anion_mob_values = map_semiconductor_property(DeviceArchitechture, 'anionmob') #Anion Mobility (m^2/Vs)
Recombination_Langevin_values = map_semiconductor_property(DeviceArchitechture, 'Recombination_Langevin')
Recombination_Bimolecular_values = map_semiconductor_property(DeviceArchitechture, 'Recombination_Bimolecular') #Bimolecular Recombination Prefactor (m^3/s)
Nc = map_semiconductor_property(DeviceArchitechture, 'Nc') #Effective Density of States in the Conduction Band (1/m^3)
Nv = map_semiconductor_property(DeviceArchitechture, 'Nv') #Effective Density of States in the Valence Band (1/m^3)
chi_a = map_semiconductor_property(DeviceArchitechture, 'Chi_a') #(Untested!) mobile anion Band Offset
chi_c = map_semiconductor_property(DeviceArchitechture, 'Chi_c') #(Untested!) mobile cation Band Offset
a_initial_values = map_semiconductor_property(DeviceArchitechture, 'a_initial_level') #mobile anion concentration (1/m^3)
c_initial_values = map_semiconductor_property(DeviceArchitechture, 'c_initial_level') #mobile cation concentration (1/m^3)
Nd_values = map_semiconductor_property(DeviceArchitechture, 'Nd') #Ionised Dopant Density (1/m^3)
Na_values = map_semiconductor_property(DeviceArchitechture, 'Na') #Ionised Acceptor Density (1/m^3)

print(DeviceArchitechture.shape)

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

gen_rate = CellVariable(name="Generation Rate", mesh=mesh, value=GenRate_values_default)
Recombination_Langevin_Cell = CellVariable(name="Recombination_Langevin_Cell", mesh=mesh, value=Recombination_Langevin_values)
Recombination_Bimolecular_Cell = CellVariable(name="Recombination_Bimolecular_Cell", mesh=mesh, value=Recombination_Bimolecular_values)
Recombination_Interfacial_SRH_Cell = CellVariable(name="Recombination_SRH_Cell", mesh=mesh, value=SRH_Interfacial_Recombination_Zone)
Recombination_Bulk_SRH_Cell = CellVariable(name="Recombination_SRH_Cell", mesh=mesh, value=SRH_Bulk_Recombination_Zone)

nmob = CellVariable(name="electron mobility", mesh=mesh, value=nmob_values)
pmob = CellVariable(name="hole mobility", mesh=mesh, value=pmob_values)
anionmob = CellVariable(name="anion mobility", mesh=mesh, value=anion_mob_values)
cationmob = CellVariable(name="cation mobility", mesh=mesh, value=cation_mob_values)

epsilon = CellVariable(name="dielectric permittivity", mesh=mesh, value=epsilon_values)
LogNcCell = CellVariable(name="Log Effective Density of States CB", mesh=mesh, value=LogNc)
LogNvCell = CellVariable(name="Log Effective Density of States VB", mesh=mesh, value=LogNv)

ChiCell = CellVariable(name="Electron Affinity", mesh=mesh, value=chi)
ChiCell_a = CellVariable(name="Electron Affinity", mesh=mesh, value=chi_a)
ChiCell_c = CellVariable(name="Electron Affinity", mesh=mesh, value=chi_c)
EgCell = CellVariable(name="Band Gap", mesh=mesh, value=Eg)

NdCell = CellVariable(name="Fixed Ionised Donors", mesh=mesh, value=Nd_values)
NaCell = CellVariable(name="Fixed Ionised Acceptor", mesh=mesh, value=Na_values)

#Here we define the Ohmic boundary conditions
nTop = map_semiconductor_property(TopLocationSC, 'Nc') * np.exp(((map_semiconductor_property(TopLocationSC, 'chi') - map_electrode_property(TopElectrode, "WF")) / D))
pTop = map_semiconductor_property(TopLocationSC, 'Nv') * np.exp(((map_electrode_property(TopElectrode, "WF") - (map_semiconductor_property(TopLocationSC, "chi") + map_semiconductor_property(TopLocationSC, "Eg"))) / D))

nBottom = map_semiconductor_property(BottomLocationSC, 'Nc') * np.exp(((map_semiconductor_property(BottomLocationSC, 'chi') - map_electrode_property(BottomElectrode, "WF")) / D))
pBottom = map_semiconductor_property(BottomLocationSC, 'Nv') * np.exp(((map_electrode_property(BottomElectrode, "WF") - (map_semiconductor_property(BottomLocationSC, "chi") +map_semiconductor_property(BottomLocationSC, "Eg"))) / D))

Va = Variable(name="Applied Voltage", value=0.00) #This variable will be used to set the voltage across the device

Vbi = (map_electrode_property(BottomElectrode, "WF") - map_electrode_property(TopElectrode, "WF"))

contact_bcs = [
    {'boundary': mesh.facesTop,    'n': nTop, 'p': pTop, 'phi': 0},
    {'boundary': mesh.facesBottom, 'n': nBottom, 'p': pBottom, 'phi': -(Vbi - Va) }
]

############Recombination Constants############
#Charge Carrier Lifetimes in the bulk (s)
tau_p_bulk = 500 * 1.00e-9
tau_n_bulk = 500 * 1.00e-9
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

def solve_for_voltage(voltage, nx, ny, D, epsilon_values, n_values, nmob_values, p_values, pmob_values , a_values, anion_mob_values, c_values, cation_mob_values, phi_values, chi, chi_a, chi_c, Eg, TInfinite, tau_p_interface, tau_n_interface, tau_p_bulk, tau_n_bulk, epsilon_0, n_hat, p_hat, n_hat_mixed, p_hat_mixed, q, niPS, Nd_values, Na_values):

    #solver = fipy.solvers.LinearLUSolver(precon=None, iterations=1) #Works out of the box with simple fipy installation, but slower than pysparse
    solver = fipy.solvers.pysparse.linearLUSolver.LinearLUSolver(precon=None, iterations=1) #Very fast solver

    philocal = CellVariable(name="electrostatic potential", mesh=mesh, value=phi_values, hasOld=True)
    nlocal = CellVariable(name="electron density", mesh=mesh, value=n_values, hasOld=True)
    plocal = CellVariable(name="hole density", mesh=mesh, value=p_values, hasOld=True)
    alocal = CellVariable(name="anion density", mesh=mesh, value=a_values)
    clocal = CellVariable(name="cation density", mesh=mesh, value=c_values)

    Va.setValue(voltage)

    for bc in contact_bcs:
            nlocal.constrain(bc['n'], where=bc['boundary'])
            plocal.constrain(bc['p'], where=bc['boundary'])
            philocal.constrain(bc['phi'], where=bc['boundary'])

    #Band-to-band recombination models
    Recombination_Langevin_EQ = (Recombination_Langevin_Cell * q * (pmob + nmob) * (nlocal * plocal - niPS * niPS) / (epsilon_values * epsilon_0))
    Recombination_Bimolecular_EQ = (Recombination_Bimolecular_Cell * (nlocal * plocal - niPS * niPS))

    #SRH trap assisted recombination models
    Recombination_SRH_Interfacial_EQ = (Recombination_Interfacial_SRH_Cell * (nlocal * plocal - niPS * niPS) / (tau_p_interface * (nlocal + n_hat) + tau_n_interface * (plocal + p_hat)))
    Recombination_SRH_Interfacial_Mixed_EQ = (Recombination_Interfacial_SRH_Cell * (nlocal * plocal - niPS * niPS) / (tau_p_interface * (nlocal + n_hat_mixed) + tau_n_interface * (plocal + p_hat_mixed)))
    Recombination_SRH_Bulk_EQ = (Recombination_Bulk_SRH_Cell * (nlocal * plocal - niPS * niPS) / (tau_p_bulk * (nlocal + n_hat) + tau_n_bulk * (plocal + p_hat)))

    Recombination_Combined = (Recombination_Bimolecular_EQ + Recombination_SRH_Bulk_EQ) #Include more recombination mechanisms by adding them to this line

    LUMO = philocal + ChiCell
    HOMO = philocal + ChiCell + EgCell

    LUMO_a = philocal + ChiCell_a
    LUMO_c = philocal + ChiCell_c

    eq1 = (0.00 == -TransientTerm(coeff=q, var=nlocal) + DiffusionTerm(coeff=q * D * nmob.harmonicFaceValue, var=nlocal) - ExponentialConvectionTerm(coeff=q * nmob.harmonicFaceValue * (LUMO + D*LogNcCell).faceGrad, var=nlocal) + q*gen_rate - q*Recombination_Combined)
    eq2 = (0.00 == -TransientTerm(coeff=q, var=plocal) + DiffusionTerm(coeff=q * D * pmob.harmonicFaceValue, var=plocal) + ExponentialConvectionTerm(coeff=q * pmob.harmonicFaceValue * (HOMO - D*LogNvCell).faceGrad, var=plocal) + q*gen_rate - q*Recombination_Combined)
    eq3 = (0.00 == -TransientTerm(coeff=q, var=alocal) + DiffusionTerm(coeff=q * D * anionmob.harmonicFaceValue, var=alocal) - ExponentialConvectionTerm(coeff=q * anionmob.harmonicFaceValue * LUMO_a.faceGrad, var=alocal))
    eq4 = (0.00 == -TransientTerm(coeff=q, var=clocal) + DiffusionTerm(coeff=q * D * cationmob.harmonicFaceValue, var=clocal) + ExponentialConvectionTerm(coeff=q * cationmob.harmonicFaceValue * LUMO_c.faceGrad, var=clocal))
    eq5 = (0.00 == -TransientTerm(var=philocal) + DiffusionTerm(coeff=epsilon, var=philocal) + (q/epsilon_0) * (plocal - nlocal + clocal - alocal + NdCell - NaCell))

    eqconteh = eq1 & eq2 #Electron and hole continuity equations
    eqcontac = eq3 & eq4 #Anion and cation continuity equations
    eqpoisson = eq5 #Poisson equation

    max_iterations = 2000 # Maximum iterations
    dt = 1.00e-9 #Starting time step should be small
    dt_old = dt
    MaxTimeStep = 1.00e-6 #Increasing above 1.00e-5 sometimes leads to artefacts in the solution even if the residual is small
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
    psinvar = LUMO - D * (numerix.log(nlocal) - LogNcCell)
    psipvar = HOMO + D * (numerix.log(plocal) - LogNvCell)

    #Here the electron and hole current densities are calculated
    Jn = (q * nmob.globalValue * nlocal.globalValue * -psinvar.grad.globalValue) #Vector Quantity
    Jp = (q * pmob.globalValue * plocal.globalValue * -psipvar.grad.globalValue) #Vector Quantity
    Jph = Jn + Jp #Vector Quantity

    # Calculation of E-field
    E = -philocal.grad  #Vector Quantity

    Efield_matrix = np.reshape(E.globalValue, (E.shape[0], ny, nx))

    PotentialMatrix = np.reshape(philocal.globalValue, (ny, nx))
    GenValues_Matrix = np.reshape(gen_rate.globalValue, (ny, nx))
    RecombinationMatrix = (np.reshape(Recombination_Combined.globalValue,(ny, nx)))
    Recombination_Bimolecular_EQMatrix = (np.reshape(Recombination_Bimolecular_EQ.globalValue,(ny, nx)))
    NMatrix = np.reshape(nlocal.globalValue, (ny, nx))
    PMatrix = np.reshape(plocal.globalValue, (ny, nx))

    J_Total_Y = np.reshape(Jph[1], (ny, nx))
    Jn_Matrix = np.reshape(Jn, (Jn.shape[0], ny, nx))
    Jp_Matrix = np.reshape(Jp, (Jp.shape[0], ny, nx))

    chiMatrix = np.reshape(ChiCell.globalValue, (ny, nx))
    EgMatrix = np.reshape(EgCell.globalValue, (ny, nx))

    psinvarmatrix = np.reshape(psinvar.globalValue, (ny, nx))
    psipvarmatrix = np.reshape(psipvar.globalValue, (ny, nx))

    return {"NMatrix": NMatrix, "PMatrix": PMatrix, "RecombinationMatrix": RecombinationMatrix, "GenValues_Matrix": GenValues_Matrix, "PotentialMatrix": PotentialMatrix, "Efield_matrix": Efield_matrix, "J_Total_Y": J_Total_Y, "n": nlocal.globalValue, "p": plocal.globalValue, "phi": philocal.globalValue, "ChiMatrix": chiMatrix, "EgMatrix": EgMatrix, "psinvarmatrix": psinvarmatrix, "psipvarmatrix": psipvarmatrix, "AnionDensityMatrix": alocal.globalValue, "CationDensityMatrix": clocal.globalValue, "ResidualMatrix": residual, "SweepCounterMatrix": SweepCounter, "Jn_Matrix": Jn_Matrix, "Jp_Matrix": Jp_Matrix, "Recombination_Bimolecular_EQMatrix": Recombination_Bimolecular_EQMatrix}

def simulate_device(output_dir):

    applied_voltages = np.arange(0.0, 1.3, 0.05)

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
        chunk_results = Parallel(n_jobs=chunk_size, backend="multiprocessing")(delayed(solve_for_voltage)(voltage, nx, ny, D, epsilon_values, n_values, nmob_values, p_values, pmob_values , a_values, anion_mob_values, c_values, cation_mob_values, phi_values, chi, chi_a, chi_c, Eg, TInfinite, tau_p_interface, tau_n_interface, tau_p_bulk, tau_n_bulk, epsilon_0, n_hat, p_hat, n_hat_mixed, p_hat_mixed, q, niPS, Nd_values, Na_values) for voltage in chunk_voltages)

        #DeepCopy To avoid overwriting the results in next loop
        copied_result = [copy.deepcopy(r) for r in chunk_results]

        # Save dictionary of chunk_results as .npy files named after the key
        for result in copied_result:
            for key, value in result.items():
                append_to_npy(key + ".npy", value)

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
    current_file = os.path.basename(__file__)[0:-3]
    voltage_sweep_output_dir = "./Outputs/" + current_file + "/VoltageSweep"

    if not os.path.exists(voltage_sweep_output_dir):
        os.makedirs(voltage_sweep_output_dir)

    print("Starting standard voltage sweep...")
    results = simulate_device(output_dir=voltage_sweep_output_dir)
    print("Voltage sweep completed.")
    return results

# Fix for multiprocessing on Windows
if __name__ == '__main__':
    main_workflow()