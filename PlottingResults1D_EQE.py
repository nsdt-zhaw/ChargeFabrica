import numpy as np
import matplotlib.pyplot as plt

# Requires a completed EQE simulation folder
Simulation_folder = "./Outputs/1D_NIP_EQE_2Bird/WavelengthSweep/"

NumberOfSuns = 1.00
ScalingFactor = 1.00

# Load matrices as before
Jn_Y = np.load(Simulation_folder + "ConservativeJnInternal.npy")
Jp_Y = np.load(Simulation_folder + "ConservativeJpInternal.npy")
JTotal_Y = Jn_Y + Jp_Y
PhotonFluxMatrix = np.load(Simulation_folder + "PhotonFluxArrayFinal.npy")
PhotonFluxArrayOriginal = np.load(Simulation_folder + "PhotonFluxArrayOriginal.npy")
PhotonFluxArrayOriginalSplit = np.load(Simulation_folder + "PhotonFluxArrayOriginalSplit.npy")
PhotonFluxPerturbation = PhotonFluxMatrix - PhotonFluxArrayOriginal
applied_wavelengths = np.load(Simulation_folder + "applied_wavelengths.npy")

JTotal_Y_mean = -np.mean(JTotal_Y[:,20:80,:], axis=(1, 2))
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
