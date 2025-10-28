# -*- coding: utf-8 -*-
#This code is a simulation of a HTL-free perovskite solar cell using the finite volume method with the FiPy library.
#Device architecture: FTO (Boundary)|TiO2 (50 nm)|MAPbI3 (1600 nm)|Carbon (Boundary)
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

Carbon_ID = name_to_code_EL["Carbon"]
PS_ID = name_to_code_SC["PS"]
TiO2_ID = name_to_code_SC["mTiO2"]
FTO_ID = name_to_code_EL["FTO"]

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
DeviceArchitechture = np.empty((1650, 1))
DeviceArchitechture[0:1600,:] = PS_ID #1600nm PS Absorber
DeviceArchitechture[1600:1650,:] = TiO2_ID #50nm TiO2 ETL

TopElectrode = FTO_ID
TopLocationSC = DeviceArchitechture[-1,:].flatten() #Semiconducting materials adjacent to the top electrode
BottomLocationSC = DeviceArchitechture[0,:].flatten() #Semiconducting materials adjacent to the bottom electrode
BottomElectrode = Carbon_ID

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
def ohmic(sc_slice, electrode_id):
    nboundary = map_semiconductor_property(sc_slice, 'Nc') * np.exp((map_semiconductor_property(sc_slice, 'chi') - map_electrode_property(electrode_id, "WF")) / D)
    pboundary = map_semiconductor_property(sc_slice, 'Nv') * np.exp((map_electrode_property(electrode_id, "WF") - (map_semiconductor_property(sc_slice, 'chi') + map_semiconductor_property(sc_slice, 'Eg'))) / D)
    return nboundary, pboundary

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

Etrap = map_semiconductor_property(PS_ID, "chi") + map_semiconductor_property(PS_ID, "Eg")/2 #Mid-bandgap trap energy level in eV
Etrap_interface = map_semiconductor_property(TiO2_ID, "chi") + ((map_semiconductor_property(PS_ID, "chi") + map_semiconductor_property(PS_ID, "Eg"))-map_semiconductor_property(TiO2_ID, 'chi'))/2

#Here we define the mid-bandgap SRH trap energy level
n_hat = map_semiconductor_property(PS_ID, 'Nc') * np.exp((map_semiconductor_property(PS_ID, "chi") - Etrap) / D)
p_hat = map_semiconductor_property(PS_ID, 'Nv') * np.exp((Etrap - map_semiconductor_property(PS_ID, "chi") - map_semiconductor_property(PS_ID, "Eg")) / D)

#Here we define the mixed band PS-HOMO/TiO2-LUMO SRH trap level
n_hat_mixed = map_semiconductor_property(PS_ID, 'Nc') * np.exp((map_semiconductor_property(TiO2_ID, "chi") - Etrap_interface) / D)
p_hat_mixed = map_semiconductor_property(PS_ID, 'Nv') * np.exp((Etrap_interface - map_semiconductor_property(PS_ID, "chi") - map_semiconductor_property(PS_ID, "Eg")) / D)

niPS = np.sqrt(Nc * Nv * np.exp(-Eg / D))

def solve_for_voltage(voltage, n_values, p_values, a_values, c_values, phi_values):

    solver = fipy.solvers.LinearLUSolver(precon=None, iterations=1) #Works out of the box with fipy installation

    philocal = CellVariable(name="electrostatic potential", mesh=mesh, value=phi_values, hasOld=True)
    nlocal = CellVariable(name="electron density", mesh=mesh, value=n_values, hasOld=True)
    plocal = CellVariable(name="hole density", mesh=mesh, value=p_values, hasOld=True)
    alocal = CellVariable(name="anion density", mesh=mesh, value=a_values, hasOld=True)
    clocal = CellVariable(name="cation density", mesh=mesh, value=c_values, hasOld=True)

    contact_bcs = [
        {'boundary': mesh.facesTop, 'n': nTop, 'p': pTop, 'phi': 0},
        {'boundary': mesh.facesBottom, 'n': nBottom, 'p': pBottom, 'phi': -(Vbi - voltage)}
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

    Recombination_Combined = (Recombination_Bimolecular_EQ + Recombination_SRH_Bulk_EQ + Recombination_SRH_Interfacial_Mixed_EQ) #Include more recombination mechanisms by adding them to this line

    LUMO = philocal + ChiCell
    HOMO = philocal + ChiCell + EgCell

    LUMO_a = philocal + ChiCell_a
    LUMO_c = philocal + ChiCell_c

    eqn = (0.00 == -TransientTerm(coeff=q, var=nlocal) + DiffusionTerm(coeff=q * D * nmob.harmonicFaceValue, var=nlocal) - ExponentialConvectionTerm(coeff=q * nmob.harmonicFaceValue * (LUMO + D*LogNcCell).faceGrad, var=nlocal) + q*gen_rate - q*Recombination_Combined)
    eqp = (0.00 == -TransientTerm(coeff=q, var=plocal) + DiffusionTerm(coeff=q * D * pmob.harmonicFaceValue, var=plocal) + ExponentialConvectionTerm(coeff=q * pmob.harmonicFaceValue * (HOMO - D*LogNvCell).faceGrad, var=plocal) + q*gen_rate - q*Recombination_Combined)
    eqa = (0.00 == -TransientTerm(coeff=q, var=alocal) + DiffusionTerm(coeff=q * D * anionmob.harmonicFaceValue, var=alocal) - ExponentialConvectionTerm(coeff=q * anionmob.harmonicFaceValue * LUMO_a.faceGrad, var=alocal))
    eqc = (0.00 == -TransientTerm(coeff=q, var=clocal) + DiffusionTerm(coeff=q * D * cationmob.harmonicFaceValue, var=clocal) + ExponentialConvectionTerm(coeff=q * cationmob.harmonicFaceValue * LUMO_c.faceGrad, var=clocal))
    eqpoisson = (0.00 == -TransientTerm(var=philocal) + DiffusionTerm(coeff=epsilon, var=philocal) + (q/epsilon_0) * (plocal - nlocal + clocal - alocal + NdCell - NaCell))

    dt, MaxTimeStep, desired_residual, DampingFactor, NumberofSweeps, max_timesteps = 1e-9, 1e-6, 1e-10, 0.05, 1, 2000
    residual, residual_old, dt_old, TotalTime, SweepCounter = 1., 1e10, dt, 0.0, 0

    while SweepCounter < max_timesteps and residual > desired_residual:

        t0 = time.time()

        for i in range(NumberofSweeps):
            eqpoisson.sweep(dt = dt, solver=solver)
            philocal.setValue(DampingFactor * philocal + (1 - DampingFactor) * philocal.old) # The potential should be damped BEFORE passing to the continuity equations!

            residual = eqn.sweep(dt = dt, solver=solver) + eqp.sweep(dt = dt, solver=solver)
            nlocal.setValue(DampingFactor * np.maximum(nlocal, 1.00e-30) + (1 - DampingFactor) * nlocal.old)
            plocal.setValue(DampingFactor * np.maximum(plocal, 1.00e-30) + (1 - DampingFactor) * plocal.old)

        EnableIons = True
        if EnableIons:
            #Here the ionic continuity equations are solved
            residual += eqa.sweep(dt = dt, solver=solver) + eqc.sweep(dt = dt, solver=solver)
            alocal.setValue(DampingFactor * alocal + (1 - DampingFactor) * alocal.old)
            clocal.setValue(DampingFactor * clocal + (1 - DampingFactor) * clocal.old)

        PercentageImprovementPerSweep = (1 - (residual / residual_old) * dt_old / dt) * 100

        if residual > residual_old * 1.2:
            dt = max(1e-11, dt * 0.1)
        else:
            dt = min(MaxTimeStep, dt * 1.05)

        dt_old, residual_old = dt, residual

        #Update old
        for v in (nlocal, plocal, alocal, clocal, philocal): v.updateOld()

        TotalTime += dt

        print("Sweep: ", SweepCounter, "TotalTime: ", TotalTime, "Residual: ", residual, "Time for sweep: ", time.time() - t0, "dt: ", dt, "Percentage Improvement: ", PercentageImprovementPerSweep, "Damping: ", DampingFactor)
        SweepCounter += 1

    # Here the electron and hole quasi-fermi levels are calculated
    psinvar = LUMO - D * (numerix.log(nlocal) - LogNcCell)
    psipvar = HOMO + D * (numerix.log(plocal) - LogNvCell)

    def reshapefunction(FipyFlattenedArray):
        return [np.reshape(arr, (ny, nx)) for arr in FipyFlattenedArray]

    #Here the electron and hole current densities are calculated
    E = -philocal.grad.globalValue
    Jn = (q * nmob.globalValue * nlocal.globalValue * -psinvar.grad.globalValue)
    Jp = (q * pmob.globalValue * plocal.globalValue * -psipvar.grad.globalValue)
    Jn_Matrix, Jp_Matrix, Efield_matrix = [np.reshape(X, (X.shape[0], ny, nx)) for X in (Jn, Jp, E)]

    (PotentialMatrix, GenValues_Matrix, RecombinationMatrix, Recombination_Bimolecular_EQMatrix, NMatrix, PMatrix, chiMatrix, EgMatrix, psinvarmatrix, psipvarmatrix) = reshapefunction([philocal, gen_rate, Recombination_Combined, Recombination_Bimolecular_EQ, nlocal, plocal, ChiCell, EgCell, psinvar, psipvar])

    return {"NMatrix": NMatrix, "PMatrix": PMatrix, "RecombinationMatrix": RecombinationMatrix, "GenValues_Matrix": GenValues_Matrix, "PotentialMatrix": PotentialMatrix, "Efield_matrix": Efield_matrix, "n": nlocal.globalValue, "p": plocal.globalValue, "phi": philocal.globalValue, "ChiMatrix": chiMatrix, "EgMatrix": EgMatrix, "psinvarmatrix": psinvarmatrix, "psipvarmatrix": psipvarmatrix, "AnionDensityMatrix": alocal.globalValue, "CationDensityMatrix": clocal.globalValue, "ResidualMatrix": residual, "SweepCounterMatrix": SweepCounter, "Jn_Matrix": Jn_Matrix, "Jp_Matrix": Jp_Matrix, "Recombination_Bimolecular_EQMatrix": Recombination_Bimolecular_EQMatrix}

def simulate_device(output_dir):

    applied_voltages = np.arange(0.0, 1.15, 0.05)

    chunk_size = min(len(applied_voltages), max(1, multiprocessing.cpu_count() - 1))

    n_values = 1.00e-30
    p_values = 1.00e-30
    a_values = a_initial_values.flatten()
    c_values = c_initial_values.flatten()
    phi_values = 1.00e-30

    def append_to_npy(filename, new_data):
        new_data = np.expand_dims(new_data, axis=0)
        path = os.path.join(output_dir, filename)
        np.save(path, np.concatenate((np.load(path), new_data), axis=0) if os.path.isfile(path) else new_data)

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