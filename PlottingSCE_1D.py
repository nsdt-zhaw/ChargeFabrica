import numpy as np
from workflow_utils import load_results
import matplotlib.pyplot as plt

#Plotting code for Spatial Collection Efficiency (SCE) calculation from 1D drift-diffusion simulation data
Simulation_folder = "./Outputs/1D_NIP_SCE_Example/VoltageSweep/"

results = load_results(Simulation_folder)
if results['excitation_indices'][0] != -1:
    raise ValueError('The baseline SCE point has not completed yet.')

GenValues_Matrix_default = results["GenValues_Matrix"][0]
GenValues_Matrix_SCE = results["GenValues_Matrix"][1:]

Position_SCE = results["excitation_indices"][1:1 + GenValues_Matrix_SCE.shape[0]]

Jn_Matrix = results["ConservativeJnInternal"]
Jp_Matrix = results["ConservativeJpInternal"]
JTotal_Y = (Jn_Matrix + Jp_Matrix)

JTotal_Y_default = JTotal_Y[0]
JTotal_Y_SCE = JTotal_Y[1:]

JTotal_Y_mean_default = np.median(JTotal_Y, axis=(1, 2))
JTotal_Y_mean_default_Jsc = JTotal_Y_mean_default[0]

JTotal_Y_mean_SCE = np.median(JTotal_Y_SCE, axis=(1, 2))

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
