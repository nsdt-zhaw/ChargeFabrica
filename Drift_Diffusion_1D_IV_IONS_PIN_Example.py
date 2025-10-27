# -*- coding: utf-8 -*-
#This code is a simulation of a PIN planar perovskite solar cell using the finite volume method with the FiPy library.
#Device architecture: ITO/SAM (Boundary)|MAPbI3 (450 nm)|C60 (50 nm)|Silver (Boundary)
#Due to the non-selective contact at the ITO boundary (even with WF modulation by the SAM), there are losses due to electrons being extracted at ITO.
import os
os.environ["OMP_NUM_THREADS"] = "1" #Really important! Pysparse doesnt benefit from multithreading.
import numpy as np
from mark_interface_file import mark_interfaces, mark_interfaces_mixed
from calculate_absorption import calculate_absorption_above_bandgap
from fipy import CellVariable, TransientTerm, DiffusionTerm, ExponentialConvectionTerm
import fipy
from fipy.tools import numerix
import time
import pandas as pd
from scipy.ndimage import zoom
from scipy import constants
from SmoothingFunction import flatten_and_smooth_all
from joblib import Parallel, delayed
import multiprocessing
from material_maps import Semiconductors, Electrodes
import copy

TInfinite = 300.0
(q, epsilon_0, D) = (constants.electron_volt, constants.epsilon_0, (constants.Boltzmann * TInfinite) / constants.electron_volt)

name_to_code_SC = {mat.name: mat.code for mat in Semiconductors.values()}
name_to_code_EL = {mat.name: mat.code for mat in Electrodes.values()}

Ag_ID = name_to_code_EL["Ag"]
PS_ID = name_to_code_SC["PS"]
C60_ID = name_to_code_SC["C60"]
ITO2_ID = name_to_code_EL["ITO2"]

def map_semiconductor_property(devarray, prop):
    return np.vectorize(lambda x: getattr(Semiconductors[x], prop))(devarray)

def map_electrode_property(devarray, prop):
    return np.vectorize(lambda x: getattr(Electrodes[x], prop))(devarray)

def map_props(arr, props, table):
    getter = np.vectorize(lambda x, p: getattr(table[x], p))
    return [getter(arr, p) for p in props]

StretchFactor = 1 #Can help convergence if a finer mesh is needed
SmoothFactor = 0.2 #Some smoothing helps with convergence

dx = 1.00e-9/StretchFactor #Pixel Width in meters
dy = 1.00e-9/StretchFactor #Pixel Width in meters

data = pd.read_excel('./Solar_Spectrum.xls', skiprows=2)
grouped = data.groupby(data.iloc[:,0].round().astype(int)).mean()
SolarSpectrumWavelength = grouped.index.values
SolarSpectrumIrradiance = grouped.iloc[:, 2].values

#Importing Absorbance Coefficient Spectrum for MAPbI3
AbsorptionData = np.genfromtxt("MAPI_tailfit_nk 1.txt", delimiter=",", skip_header=1)
kdata = AbsorptionData[:, 2]
alphadata = 4 * np.pi * kdata / (AbsorptionData[:, 0] * 1.00e-9)

######Define Device Architecture
DeviceArchitechture = np.empty((500, 1))

DeviceArchitechture[0:50,:] = C60_ID #50nm C60 ETL
DeviceArchitechture[50:500,:] = PS_ID #450nm PS Absorber

TopElectrode = ITO2_ID
TopLocationSC = DeviceArchitechture[-1,:] #Semiconducting materials adjacent to the top electrode
BottomLocationSC = DeviceArchitechture[0,:] #Semiconducting materials adjacent to the bottom electrode
BottomElectrode = Ag_ID

SAMWFModifier = 0.6 #Self-assembled monolayer workfunction modifier in eV

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
zoom_factor = [StretchFactor] + ([StretchFactor] if DeviceArchitechture.shape[1] > 1 else [1])
DeviceArchitechture = zoom(DeviceArchitechture, zoom_factor, order=0)
GenRate_values_default = zoom(GenRate_values_default, zoom_factor, order=0)

print(DeviceArchitechture.shape)

sc_props = ['epsilon','pmob','nmob','Eg','chi','cationmob','anionmob', 'Recombination_Langevin','Recombination_Bimolecular','Nc','Nv', 'Chi_a','Chi_c','a_initial_level','c_initial_level','Nd','Na']
(epsilon_values, pmob_values, nmob_values, Eg, chi, cation_mob_values, anion_mob_values, Recombination_Langevin_values, Recombination_Bimolecular_values, Nc, Nv, chi_a, chi_c, a_initial_values, c_initial_values, Nd_values, Na_values) = map_props(DeviceArchitechture, sc_props, Semiconductors)

ny, nx = DeviceArchitechture.shape

#LocationSRH_HTL = mark_interfaces(DeviceArchitechture, 50, PS_ID)
#mark_interfaces() places the interface inside the absorber
LocationSRH_ETL = mark_interfaces(DeviceArchitechture, C60_ID, PS_ID)

#LocationHTL_Exact = mark_interfaces_mixed(DeviceArchitechture, 50, PS_ID, 0*StretchFactor)
#mark_interfaces_mixed() places the interface in the middle of the absorber and the transport layer
LocationETL_Exact = mark_interfaces_mixed(DeviceArchitechture, C60_ID, PS_ID, 3*StretchFactor)

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
nTop = map_semiconductor_property(TopLocationSC, 'Nc') * np.exp(((map_semiconductor_property(TopLocationSC, 'chi') - (map_electrode_property(TopElectrode, "WF") + SAMWFModifier)) / D))
pTop = map_semiconductor_property(TopLocationSC, 'Nv') * np.exp((((map_electrode_property(TopElectrode, "WF") + SAMWFModifier) - (map_semiconductor_property(TopLocationSC, "chi") + map_semiconductor_property(TopLocationSC, "Eg"))) / D))

nBottom = map_semiconductor_property(BottomLocationSC, 'Nc') * np.exp(((map_semiconductor_property(BottomLocationSC, 'chi') - map_electrode_property(BottomElectrode, "WF")) / D))
pBottom = map_semiconductor_property(BottomLocationSC, 'Nv') * np.exp(((map_electrode_property(BottomElectrode, "WF") - (map_semiconductor_property(BottomLocationSC, "chi") +map_semiconductor_property(BottomLocationSC, "Eg"))) / D))

Vbi = (map_electrode_property(BottomElectrode, "WF") - (map_electrode_property(TopElectrode, "WF") + SAMWFModifier))

############Recombination Constants############
#Charge Carrier Lifetimes in the bulk (s)
tau_p_bulk = 500 * 1.00e-9
tau_n_bulk = 500 * 1.00e-9
#Charge Carrier Lifetimes at the interface (s)
tau_p_interface = 0.02 * 1.00e-9
tau_n_interface = 0.02 * 1.00e-9

Etrap = map_semiconductor_property(PS_ID, "chi") + map_semiconductor_property(PS_ID, "Eg")/2 #Mid-bandgap trap energy level in eV
Etrap_interface = map_semiconductor_property(C60_ID, "chi") + ((map_semiconductor_property(PS_ID, "chi") + map_semiconductor_property(PS_ID, "Eg"))-map_semiconductor_property(C60_ID, 'chi'))/2

#Here we define the mid-bandgap SRH trap energy level
n_hat = map_semiconductor_property(PS_ID, 'Nc') * np.exp((map_semiconductor_property(PS_ID, "chi") - Etrap) / D)
p_hat = map_semiconductor_property(PS_ID, 'Nv') * np.exp((Etrap - map_semiconductor_property(PS_ID, "chi") - map_semiconductor_property(PS_ID, "Eg")) / D)

#Here we define the mixed band PS-HOMO/TiO2-LUMO SRH trap level
n_hat_mixed = map_semiconductor_property(PS_ID, 'Nc') * np.exp((map_semiconductor_property(C60_ID, "chi") - Etrap_interface) / D)
p_hat_mixed = map_semiconductor_property(PS_ID, 'Nv') * np.exp((Etrap_interface - map_semiconductor_property(PS_ID, "chi") - map_semiconductor_property(PS_ID, "Eg")) / D)

niPS = np.sqrt(Nc * Nv * np.exp(-Eg / D))

def solve_for_voltage(voltage, n_values, p_values, a_values, c_values, phi_values):

    solver = fipy.solvers.LinearLUSolver(precon=None, iterations=1) #Works out of the box with simple fipy installation, but slower than pysparse
    #solver = fipy.solvers.pysparse.linearLUSolver.LinearLUSolver(precon=None, iterations=1) #Very fast solver

    philocal = CellVariable(name="electrostatic potential", mesh=mesh, value=phi_values, hasOld=True)
    nlocal = CellVariable(name="electron density", mesh=mesh, value=n_values, hasOld=True)
    plocal = CellVariable(name="hole density", mesh=mesh, value=p_values, hasOld=True)
    alocal = CellVariable(name="anion density", mesh=mesh, value=a_values, hasOld=True)
    clocal = CellVariable(name="cation density", mesh=mesh, value=c_values, hasOld=True)

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

    #SRH trap assisted recombination models
    Recombination_SRH_Interfacial_EQ = (Recombination_Interfacial_SRH_Cell * (nlocal * plocal - niPS * niPS) / (tau_p_interface * (nlocal + n_hat) + tau_n_interface * (plocal + p_hat)))
    Recombination_SRH_Interfacial_Mixed_EQ = (Recombination_Interfacial_SRH_Cell * (nlocal * plocal - niPS * niPS) / (tau_p_interface * (nlocal + n_hat_mixed) + tau_n_interface * (plocal + p_hat_mixed)))
    Recombination_SRH_Bulk_EQ = (Recombination_Bulk_SRH_Cell * (nlocal * plocal - niPS * niPS) / (tau_p_bulk * (nlocal + n_hat) + tau_n_bulk * (plocal + p_hat)))

    Recombination_Combined = (Recombination_Bimolecular_EQ + Recombination_SRH_Bulk_EQ) #Include more recombination mechanisms by adding them to this line

    LUMO = philocal + ChiCell
    HOMO = philocal + ChiCell + EgCell

    LUMO_a = philocal + ChiCell_a
    LUMO_c = philocal + ChiCell_c

    eqn = (0.00 == -TransientTerm(coeff=q, var=nlocal) + DiffusionTerm(coeff=q * D * nmob.harmonicFaceValue, var=nlocal) - ExponentialConvectionTerm(coeff=q * nmob.harmonicFaceValue * (LUMO + D*LogNcCell).faceGrad, var=nlocal) + q*gen_rate - q*Recombination_Combined)
    eqp = (0.00 == -TransientTerm(coeff=q, var=plocal) + DiffusionTerm(coeff=q * D * pmob.harmonicFaceValue, var=plocal) + ExponentialConvectionTerm(coeff=q * pmob.harmonicFaceValue * (HOMO - D*LogNvCell).faceGrad, var=plocal) + q*gen_rate - q*Recombination_Combined)
    eqa = (0.00 == -TransientTerm(coeff=q, var=alocal) + DiffusionTerm(coeff=q * D * anionmob.harmonicFaceValue, var=alocal) - ExponentialConvectionTerm(coeff=q * anionmob.harmonicFaceValue * LUMO_a.faceGrad, var=alocal))
    eqc = (0.00 == -TransientTerm(coeff=q, var=clocal) + DiffusionTerm(coeff=q * D * cationmob.harmonicFaceValue, var=clocal) + ExponentialConvectionTerm(coeff=q * cationmob.harmonicFaceValue * LUMO_c.faceGrad, var=clocal))
    eqpoisson = (0.00 == -TransientTerm(var=philocal) + DiffusionTerm(coeff=epsilon, var=philocal) + (q/epsilon_0) * (plocal - nlocal + clocal - alocal + NdCell - NaCell))

    dt, MaxTimeStep, desired_residual, DampingFactor, NumberofSweeps, max_iterations = 1e-9, 1e-6, 1e-10, 0.05, 1, 2000
    residual, residual_old, dt_old, TotalTime, SweepCounter = 1., 1e10, dt, 0.0, 0

    while SweepCounter < max_iterations and residual > desired_residual:

        t0 = time.time()

        for i in range(NumberofSweeps):
            eqpoisson.sweep(dt = dt, solver=solver)
            philocal.setValue(DampingFactor * philocal.value + (1 - DampingFactor) * philocal.old)  # The potential should be damped BEFORE passing to the continuity equations!

            residual = eqn.sweep(dt = dt, solver=solver) + eqp.sweep(dt = dt, solver=solver)
            nlocal.setValue(np.maximum(nlocal, 1.00e-30))
            plocal.setValue(np.maximum(plocal, 1.00e-30))
            nlocal.setValue(DampingFactor * nlocal.value + (1 - DampingFactor) * nlocal.old)
            plocal.setValue(DampingFactor * plocal.value + (1 - DampingFactor) * plocal.old)

        EnableIons = True
        if EnableIons:
            #Here the ionic continuity equations are solved
            residual += eqa.sweep(dt = dt, solver=solver) + eqc.sweep(dt = dt, solver=solver)
            alocal.setValue(DampingFactor * alocal.value + (1 - DampingFactor) * alocal.old)
            clocal.setValue(DampingFactor * clocal.value + (1 - DampingFactor) * clocal.old)

        PercentageImprovementPerSweep = (1 - (residual / residual_old) * dt_old / dt) * 100

        if residual > residual_old * 1.2:
            dt = max(1e-11, dt * 0.1)
        else:
            dt = min(MaxTimeStep, dt * 1.05)

        dt_old = dt
        residual_old = residual

        #Update old
        for v in (nlocal, plocal, alocal, clocal, philocal): v.updateOld()

        TotalTime = TotalTime + dt

        print("Sweep: ", SweepCounter, "TotalTime: ", TotalTime, "Residual: ", 1.00e-9*residual/dt, "Time for sweep: ", time.time() - t0, "dt: ", dt, "Percentage Improvement: ", PercentageImprovementPerSweep, "Damping: ", DampingFactor)
        SweepCounter += 1

    # Here the electron and hole quasi-fermi levels are calculated
    psinvar = LUMO - D * (numerix.log(nlocal) - LogNcCell)
    psipvar = HOMO + D * (numerix.log(plocal) - LogNvCell)

    def reshapefunction(FipyFlattenedArray):
        return [np.reshape(arr, (ny, nx)) for arr in FipyFlattenedArray]

    #Here the electron and hole current densities are calculated
    Jn = (q * nmob.globalValue * nlocal.globalValue * -psinvar.grad.globalValue) #Vector Quantity
    Jp = (q * pmob.globalValue * plocal.globalValue * -psipvar.grad.globalValue) #Vector Quantity
    E = -philocal.grad  #Vector Quantity

    Jn_Matrix = np.reshape(Jn, (Jn.shape[0], ny, nx))
    Jp_Matrix = np.reshape(Jp, (Jp.shape[0], ny, nx))
    Efield_matrix = np.reshape(E.globalValue, (E.shape[0], ny, nx))

    (PotentialMatrix, GenValues_Matrix, RecombinationMatrix, Recombination_Bimolecular_EQMatrix, NMatrix, PMatrix, chiMatrix, EgMatrix, psinvarmatrix, psipvarmatrix) = reshapefunction([philocal, gen_rate, Recombination_Combined, Recombination_Bimolecular_EQ, nlocal, plocal, ChiCell, EgCell, psinvar, psipvar])

    return {"NMatrix": NMatrix, "PMatrix": PMatrix, "RecombinationMatrix": RecombinationMatrix, "GenValues_Matrix": GenValues_Matrix, "PotentialMatrix": PotentialMatrix, "Efield_matrix": Efield_matrix, "n": nlocal.globalValue, "p": plocal.globalValue, "phi": philocal.globalValue, "ChiMatrix": chiMatrix, "EgMatrix": EgMatrix, "psinvarmatrix": psinvarmatrix, "psipvarmatrix": psipvarmatrix, "AnionDensityMatrix": alocal.globalValue, "CationDensityMatrix": clocal.globalValue, "ResidualMatrix": residual, "SweepCounterMatrix": SweepCounter, "Jn_Matrix": Jn_Matrix, "Jp_Matrix": Jp_Matrix, "Recombination_Bimolecular_EQMatrix": Recombination_Bimolecular_EQMatrix}

def simulate_device(output_dir):

    applied_voltages = np.arange(0.0, 1.3, 0.05)

    if len(applied_voltages) < multiprocessing.cpu_count() - 1:
        chunk_size = len(applied_voltages)
    else:
        chunk_size = multiprocessing.cpu_count() - 1

    n_values = 1.00e-30
    p_values = 1.00e-30
    a_values = a_initial_values.flatten()
    c_values = c_initial_values.flatten()
    phi_values = 1.00e-30

    def append_to_npy(filename, new_data):
        new_data = np.expand_dims(new_data, axis=0)
        path = os.path.join(output_dir, filename)
        if os.path.isfile(path):
            fulldata = np.concatenate((np.load(path), new_data), axis=0)
        else:
            fulldata = new_data
        np.save(path, fulldata)

    # Process voltages in sequential chunks
    for start in range(0, len(applied_voltages), chunk_size):
        # Create a chunk of voltages to simulate in parallel
        chunk_voltages = applied_voltages[start:start + chunk_size]

        # Parallel computation within the chunk
        chunk_results = Parallel(n_jobs=chunk_size, backend="multiprocessing")(delayed(solve_for_voltage)(voltage, n_values, p_values, a_values, c_values, phi_values) for voltage in chunk_voltages)

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
        n_values, p_values = last_result["n"], last_result["p"]
        a_values, c_values = last_result["AnionDensityMatrix"], last_result["CationDensityMatrix"]
        phi_values = last_result["phi"]
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