import numpy as np
import matplotlib.pyplot as plt

# Requires a completed EQE simulation folder
Simulation_folder = "D:/Backup/Defect_Paper/1D_NIP_EQE/WavelengthSweep/"

NumberOfSuns = 1.00
ScalingFactor = 1.00

# Load matrices as before
Jn_Y = np.load(Simulation_folder + "Jn_Matrix.npy")[:,1,:,:]
Jp_Y = np.load(Simulation_folder + "Jp_Matrix.npy")[:,1,:,:]
JTotal_Y = Jn_Y + Jp_Y
PhotonFluxMatrix = np.load(Simulation_folder + "PhotonFluxArrayFinal.npy")
PhotonFluxArrayOriginal = np.load(Simulation_folder + "PhotonFluxArrayOriginal.npy")
PhotonFluxArrayOriginalSplit = np.load(Simulation_folder + "PhotonFluxArrayOriginalSplit.npy")
PhotonFluxPerturbation = PhotonFluxMatrix - PhotonFluxArrayOriginal
applied_wavelengths = np.load(Simulation_folder + "applied_wavelengths.npy")

def medfilt(x, k):
    """Apply a length-k median filter to a 1D array x. Boundaries are extended by repeating endpoints."""
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median(y, axis=1)

for i in range(JTotal_Y.shape[0]):
    JTotal_Y_Flattened = medfilt(JTotal_Y[i].flatten(), 5)
    JTotal_Y[i] = np.expand_dims(JTotal_Y_Flattened, axis=1)
for i in range(Jn_Y.shape[0]):
    Jn_Y_Flattened = medfilt(Jn_Y[i].flatten(), 5)
    Jn_Y[i] = np.expand_dims(Jn_Y_Flattened, axis=1)
for i in range(Jp_Y.shape[0]):
    Jp_Y_Flattened = medfilt(Jp_Y[i].flatten(), 5)
    Jp_Y[i] = np.expand_dims(Jp_Y_Flattened, axis=1)

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