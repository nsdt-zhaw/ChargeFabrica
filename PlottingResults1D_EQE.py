import numpy as np
import os
from workflow_utils import load_results
import matplotlib.pyplot as plt

# Requires a completed EQE simulation folder
Simulation_folder = "./Outputs/1D_NIP_EQE/WavelengthSweep/"

NumberOfSuns = 1.00
ScalingFactor = 1.00

# Load one ordered snapshot, including legacy per-field files.
results = load_results(Simulation_folder)
if results['applied_wavelengths'][0] != 0.:
    raise ValueError('The baseline EQE point has not completed yet.')
Jn_Y = results["ConservativeJnInternal"]
Jp_Y = results["ConservativeJpInternal"]
JTotal_Y = Jn_Y + Jp_Y
PhotonFluxMatrix = results["PhotonFluxArrayFinal"]
PhotonFluxArrayOriginal = results["PhotonFluxArrayOriginal"]
PhotonFluxArrayOriginalSplit = results["PhotonFluxArrayOriginalSplit"]
if os.path.isdir(os.path.join(Simulation_folder, "points")):
    PhotonFluxArrayOriginal = PhotonFluxArrayOriginal[:1]
    PhotonFluxArrayOriginalSplit = PhotonFluxArrayOriginalSplit[1:]
PhotonFluxPerturbation = PhotonFluxMatrix - PhotonFluxArrayOriginal
applied_wavelengths = results["applied_wavelengths"]

JTotal_Y_mean = -np.mean(JTotal_Y, axis=(1, 2))
Jsc1Sun = JTotal_Y_mean[0]

print("Jsc1Sun: ", Jsc1Sun, "A/m^2")

EQE = (JTotal_Y_mean[1:]-JTotal_Y_mean[0])*(1.00/(1.602e-19))/((PhotonFluxPerturbation[1:,-1,-1])*1.00e-9)
IntegratedEQE = EQE*PhotonFluxArrayOriginalSplit[0:,-1,0]*1.00e-9
IntegratedEQE = np.sum(IntegratedEQE)*1.602e-19

print("EQE Integrated Jsc", IntegratedEQE, "A/m^2 (Sanity check: Should be close to Jsc1Sun if simulation converged well)")

#Create the EQE plot
plt.plot(applied_wavelengths[1:], EQE, label="EQE")
plt.legend()
plt.ylim(0, 1)
plt.ylabel("EQE")
plt.xlabel("Wavelength (nm)")
plt.show()
