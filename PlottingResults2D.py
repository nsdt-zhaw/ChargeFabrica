import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider
from scipy.interpolate import interp1d
from scipy import signal


Simulation_folder = "./Outputs/Drift_Diffusion_2D_IV_Simple_HTL_Free_Carbon_Device_IONS/VoltageSweep/"

NumberOfSuns = 1.00

class PercentileNormalizer(Normalize):
    def __init__(self, vmin=None, vmax=None, clip=False, lower_percentile=3, upper_percentile=97):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        Normalize.__init__(self, vmin, vmax, clip)

    def autoscale_None(self, A):
        if self.vmin is None and np.size(A) > 0:
            self.vmin = np.percentile(A, self.lower_percentile)
        if self.vmax is None and np.size(A) > 0:
            self.vmax = np.percentile(A, self.upper_percentile)

# Load matrices as before
GenerationMatrix = np.load(Simulation_folder + "GenValues_Matrix.npy")
RecombinationMatrix = np.load(Simulation_folder + "RecombinationMatrix.npy")
RadiativeRecombinationMatrix = np.load(Simulation_folder + "Recombination_Bimolecular_EQMatrix.npy")
PMatrix = np.load(Simulation_folder + "PMatrix.npy")
NMatrix = np.load(Simulation_folder + "NMatrix.npy")
Jn_Matrix = np.load(Simulation_folder + "Jn_Matrix.npy")[:,1,:,:]
Jp_Matrix = np.load(Simulation_folder + "Jp_Matrix.npy")[:,1,:,:]
JTotal_Y = (Jn_Matrix + Jp_Matrix)

for i in range(Jn_Matrix.shape[0]):
    Jn_Matrix[i] = signal.medfilt2d(Jn_Matrix[i], kernel_size=5)

for i in range(Jp_Matrix.shape[0]):
    Jp_Matrix[i] = signal.medfilt2d(Jp_Matrix[i], kernel_size=5)

for i in range(JTotal_Y.shape[0]):
    JTotal_Y[i] = signal.medfilt2d(JTotal_Y[i], kernel_size=5)

JTotal_Y_mean = np.mean(JTotal_Y[:,20:80,:], axis=(1, 2))
VocLocation = np.unravel_index(np.argmin(np.abs(JTotal_Y_mean)), JTotal_Y_mean.shape)

PLYield = 100*RadiativeRecombinationMatrix / (GenerationMatrix+1)

PotentialMatrix = np.load(Simulation_folder + "PotentialMatrix.npy")
EField_matrix = -np.load(Simulation_folder + "Efield_matrix.npy")[:,1,:,:]
psinvarmatrix = np.load(Simulation_folder + "psinvarmatrix.npy")
psipvarmatrix = np.load(Simulation_folder + "psipvarmatrix.npy")
ChiMatrix = np.load(Simulation_folder + "ChiMatrix.npy")
EgMatrix = np.load(Simulation_folder + "EgMatrix.npy")
AnionDensityMatrix = np.load(Simulation_folder + "AnionDensityMatrix.npy")
CationDensityMatrix = np.load(Simulation_folder + "CationDensityMatrix.npy")
AnionDensityMatrix = AnionDensityMatrix.reshape((AnionDensityMatrix.shape[0], NMatrix.shape[1], NMatrix.shape[2]))
CationDensityMatrix = CationDensityMatrix.reshape((CationDensityMatrix.shape[0], NMatrix.shape[1], NMatrix.shape[2]))
ResidualMatrix = np.load(Simulation_folder + "ResidualMatrix.npy")

SweepCounterMatrix = np.load(Simulation_folder + "SweepCounterMatrix.npy")
applied_voltages = np.load(Simulation_folder + "applied_voltages.npy")

import copy
my_cmap = copy.copy(plt.cm.get_cmap('hot')) # copy the default cmap
my_cmap.set_bad((0,0,0))

titles = ['EField_Y (V/m)', 'Electron Density (1/m3)', 'Hole Density (1/m3)', "Anion Density (1/m3)",
          "Cation Density (1/m3)", 'Generation Rate (1/m3)', 'JTotal_Y (A/m2)', 'Potential (V)',
         'Recombination Rate (1/m3)', 'PLYield (-)',
          "Electron Q.F.P. (eV)", "Hole Q.F.P. (eV)", "Jn_Y (A/m2)", "Jp_Y (A/m2)"]

data_matrices = [EField_matrix, NMatrix, PMatrix, AnionDensityMatrix, CationDensityMatrix, GenerationMatrix, JTotal_Y, PotentialMatrix,
                  RecombinationMatrix, PLYield,
                 psinvarmatrix, psipvarmatrix,
                 Jn_Matrix, Jp_Matrix]

num_plots = len(data_matrices)

# Set up subplots
fig, axs = plt.subplots(3, 5, figsize=(12, 10))
axs = axs.ravel()

# Initial plot setup
cax_list = []
colorbars = []

norms = []
for i, matrix in enumerate(data_matrices):
    norm = PercentileNormalizer(lower_percentile=3, upper_percentile=97)
    norms.append(norm)
    norm.autoscale_None(matrix[0])
    cax = axs[i].imshow(matrix[0], cmap='viridis', norm=norm)
    cbar = fig.colorbar(cax, ax=axs[i])
    axs[i].set_title("{} at {:.3f} V".format(titles[i], applied_voltages[0]))
    cax_list.append(cax)
    colorbars.append(cbar)

# Adjust subplot parameters for a better fit
fig.subplots_adjust(left=0.12, right=0.88, top=0.93, bottom=0.12, wspace=0.2, hspace=0.4)

# Slider axis (center slider position to avoid left gap)
slider_ax = plt.axes([0.2, 0.03, 0.6, 0.025])
slider = Slider(ax=slider_ax, label='Voltage [V]', valmin=applied_voltages.min(), valmax=applied_voltages.max(), valinit=applied_voltages[0])

# Find an interpolation function from voltage to indices
interp_indices = interp1d(applied_voltages, np.arange(len(applied_voltages)), bounds_error=False,fill_value="extrapolate")

JTotal_Y_mean = np.mean(JTotal_Y[:,20:80,:], axis=(1, 2))

if len(applied_voltages) > 3:
    # Interpolate JTotal_Y_mean over a larger set of voltages
    f_interp = interp1d(applied_voltages, JTotal_Y_mean, kind='linear')
    voltage_fine = np.linspace(applied_voltages[0], applied_voltages[-1], 1000)
    JTotal_Y_fine = f_interp(voltage_fine)

    # Compute the zero crossing for Voc
    Voc = voltage_fine[np.argmin(np.abs(JTotal_Y_fine))]
    # Compute Jsc as the interpolated value at 0 voltage
    Jsc = -f_interp(0)

    #Compute efficiency by determining the maximum power point
    max_power_index = np.argmin(JTotal_Y_fine * voltage_fine)
    max_power_voltage = voltage_fine[max_power_index]
    max_power_current = JTotal_Y_fine[max_power_index]

    #Efficiency is the ratio of maximum power to the product of the open circuit voltage and short circuit current
    MaxPowerOut = max_power_voltage * max_power_current * -1 #Units of W/m^2
    MaxPowerIn = NumberOfSuns * 1000 #Units of W/m^2

    Efficiency = (MaxPowerOut/MaxPowerIn)*100

    fig2, ax2 = plt.subplots()

    ax2.plot(voltage_fine, JTotal_Y_fine)
    ax2.set_xlabel("Applied Voltage (V)")
    ax2.set_ylabel("Current Density (A/m^2)")
    ax2.set_title("IV Curve")
    # Set y axis to between -100 and 100
    # Add line for maximum power point
    ax2.axvline(max_power_voltage, color='r', linestyle='--')
    ax2.axhline(max_power_current, color='r', linestyle='--')
    # Add a text label for the maximum power point, fill factor, and efficiency
    ax2.text(0.05, 0.70, "Efficiency: {:.3f}%".format(Efficiency), fontsize=8, transform=ax2.transAxes)
    #ax2.text(0.05, 0.65, "Fill Factor: {:.3f}%".format((MaxPowerOut/(Voc*Jsc))*100), fontsize=8, transform=ax2.transAxes)
    ax2.text(0.05, 0.60, "MPP: {:.3f} V, {:.3f} A/m^2".format(max_power_voltage, max_power_current), fontsize=8, transform=ax2.transAxes)

    #ax2.set_ylim(-Jsc-10, Jsc+10)
    ax2.set_ylim(-250 - 10, 250 + 10)
    ax2.set_xlim(-0.3, 1.4)
    #Twin y axis
    ax3 = ax2.twinx()
    #ax3.plot(applied_voltages, SweepCounterMatrix)
    ax3.plot(applied_voltages, ResidualMatrix)
    #Make y axis log
    ax3.set_yscale("log")
    ax3.set_ylabel("Residual")

    axs[-1].plot(voltage_fine, JTotal_Y_fine)
    axs[-1].set_xlabel("Applied Voltage (V)")
    axs[-1].set_title("JV Curve (A/m2)")
    axs[-1].set_ylim(-200 - 10, 250 + 10)
    axs[-1].set_xlim(-0.1, 1.2)

# Slider update function
def update(val):
    voltage = slider.val
    # find closest index for the given voltage
    frame = int(np.clip(np.round(interp_indices(voltage)), 0, len(applied_voltages) - 1))

    for i, matrix in enumerate(data_matrices):
        cax_list[i].set_data(matrix[frame])
        norms[i].autoscale_None(matrix[frame])  # re-autoscale for each frame, optional
        cax_list[i].set_norm(norms[i])
        axs[i].set_title("{} at {:.3f} V".format(titles[i], applied_voltages[frame]))
        cax_list[i].autoscale()

    fig.canvas.draw_idle()

update(0) # Initial call to display the first frame
slider.on_changed(update)
plt.show()