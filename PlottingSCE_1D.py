import numpy as np
import matplotlib.pyplot as plt

#Plotting code for Spatial Collection Efficiency (SCE) calculation from 1D drift-diffusion simulation data
Simulation_folder = "./Outputs/1D_NIP_SCE_Example/VoltageSweep/"

GenValues_Matrix_default = np.load(Simulation_folder + "GenValues_Matrix.npy")[0]
GenValues_Matrix_SCE = np.load(Simulation_folder + "GenValues_Matrix.npy")[1:]

Position_SCE = np.load(Simulation_folder + "excitation_indices.npy")[0:GenValues_Matrix_SCE.shape[0]]

Jn_Matrix = np.load(Simulation_folder + "Jn_Matrix.npy")[:,1,:,:]
Jp_Matrix = np.load(Simulation_folder + "Jp_Matrix.npy")[:,1,:,:]
JTotal_Y = (Jn_Matrix + Jp_Matrix)

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
for i in range(Jn_Matrix.shape[0]):
    Jn_Matrix_Flattened = medfilt(Jn_Matrix[i].flatten(), 5)
    Jn_Matrix[i] = np.expand_dims(Jn_Matrix_Flattened, axis=1)
for i in range(Jp_Matrix.shape[0]):
    Jp_Matrix_Flattened = medfilt(Jp_Matrix[i].flatten(), 5)
    Jp_Matrix[i] = np.expand_dims(Jp_Matrix_Flattened, axis=1)

JTotal_Y_default = JTotal_Y[0]
JTotal_Y_SCE = JTotal_Y[1:]

JTotal_Y_mean_default = np.median(JTotal_Y[:,100:-100,:], axis=(1, 2))
JTotal_Y_mean_default_Jsc = JTotal_Y_mean_default[0]

JTotal_Y_mean_SCE = np.median(JTotal_Y_SCE[:,100:-100,:], axis=(1, 2))

DeltaJsc = JTotal_Y_mean_SCE - JTotal_Y_mean_default_Jsc #Should be negative due to extra generation in SCE

GenValues_Matrix_mean_default = np.mean(GenValues_Matrix_default, axis=1)

GenValues_Matrix_mean_SCE = np.mean(GenValues_Matrix_SCE, axis=2)

#Calculate the integrated photocurrent dynamically so that we get a 1D array, not a 0D point.
IntegratedPhotocurrentArray_default = np.zeros(GenValues_Matrix_mean_default.shape[0])
IntegratedPhotocurrentArray_SCE = np.zeros_like(GenValues_Matrix_mean_SCE)

#Calculate the integrated photocurrent for each thickness
for i in range(GenValues_Matrix_mean_default.shape[0]):
    IntegratedPhotocurrentArray_default[i] = np.trapz(GenValues_Matrix_mean_default[:i+1], dx=1e-9)

for j in range(GenValues_Matrix_mean_SCE.shape[0]):
    for i in range(GenValues_Matrix_mean_SCE.shape[1]):
        IntegratedPhotocurrentArray_SCE[j,i] = np.trapz(GenValues_Matrix_mean_SCE[j,:i+1], dx=1e-9)


#Calculate the integrated photocurrent for each thickness
DeltaIntegratedPhotocurrentArray = IntegratedPhotocurrentArray_SCE - IntegratedPhotocurrentArray_default

q = 1.602176634e-19  # Elementary charge in Coulombs

FinalSCE = -DeltaJsc/(DeltaIntegratedPhotocurrentArray[0:,-1]*q)

print("Jsc from simulation:", JTotal_Y_mean_default_Jsc)
print("Jsc calculated from SCE (Sanity Check: Should be close to Jsc):", -np.sum(FinalSCE * GenValues_Matrix_mean_default[0:GenValues_Matrix_SCE.shape[0]] * q * 1.00e-9))

#Plot FinalSCE against Position_SCE
plt.figure(figsize=(10, 6))
plt.plot(Position_SCE, FinalSCE, label='Final SCE', color='blue')
plt.xlabel('Position (nm)')
plt.ylabel('Final SCE')
plt.title('Final SCE vs Position')
#Set limit between 0 and 1
plt.ylim(-0.005, 1.0)
plt.twinx()
plt.plot(np.arange(GenValues_Matrix_default.shape[0]), GenValues_Matrix_mean_default, label='Generation', color='orange', linestyle='--')
plt.ylabel('Generation Rate (1/$m^3/s$)')
plt.ylim(0, np.max(GenValues_Matrix_mean_default)*1.1)
plt.legend(loc='upper right')
#Set dpi to 300
plt.savefig("Final_SCE_vs_Position.pdf", dpi=300, bbox_inches='tight')


plt.show()
