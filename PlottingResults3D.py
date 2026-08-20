import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider
from scipy.interpolate import interp1d
import copy

Simulation_folder = "D:\Backup\Massive_Finger_SRH_Comparison\Drift_Diffusion_3D_IV_Simple_HTL_Free_Carbon_Device_IONS/VoltageSweep/"
NumberOfSuns = 1.00

class PercentileNormalizer(Normalize):
    def init(self, vmin=None, vmax=None, clip=False, lower_percentile=3, upper_percentile=97):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        Normalize.init(self, vmin, vmax, clip)

def autoscale_None(self, A):
    if self.vmin is None and np.size(A) > 0:
        self.vmin = np.percentile(A, self.lower_percentile)
    if self.vmax is None and np.size(A) > 0:
        self.vmax = np.percentile(A, self.upper_percentile)

#Load full 4D scalar fields (voltage, y, x, z)
GenerationMatrix = np.load(Simulation_folder + "GenValues_Matrix.npy")        # (V, Y, X, Z)
RecombinationMatrix = np.load(Simulation_folder + "RecombinationMatrix.npy")  # (V, Y, X, Z)
RadiativeRecombinationMatrix = np.load(Simulation_folder + "Recombination_Bimolecular_EQMatrix.npy")  # (V, Y, X, Z)
PMatrix = np.load(Simulation_folder + "PMatrix.npy")                          # (V, Y, X, Z)
NMatrix = np.load(Simulation_folder + "NMatrix.npy")                          # (V, Y, X, Z)
NMatrixShapeFull = NMatrix.shape                                              # (V, Y, X, Z)

# Load conservative currents on internal faces (voltage, y-face, x, z).
Jn_Matrix = np.load(Simulation_folder + "ConservativeJnInternal.npy")
Jp_Matrix = np.load(Simulation_folder + "ConservativeJpInternal.npy")

#Total y-component
JTotal_Y = (Jn_Matrix + Jp_Matrix)  # (V, Y, X, Z)

#Quick 1D debug plot at initial z=0 (optional)
plt.figure()
plt.plot(np.mean(JTotal_Y[0, :, :, 0], axis=1))
plt.xlabel("X position (pixels)")
plt.title("Mean JTotal_Y at V=0, z=0")
plt.show()

#For JV analysis (initially at z=0)
JTotal_Y_mean = np.mean(JTotal_Y[:, 200:400, :, 0], axis=(1, 2))
VocLocation = np.unravel_index(np.argmin(np.abs(JTotal_Y_mean)), JTotal_Y_mean.shape)

#PL yield
PLYield = 100 * RadiativeRecombinationMatrix / (GenerationMatrix + 1)

#More fields (full 4D)
PotentialMatrix = np.load(Simulation_folder + "PotentialMatrix.npy")  # (V, Y, X, Z)
EField_matrix = -np.load(Simulation_folder + "Efield_matrix.npy")[:, 2, :, :, :]  # y-component, keep Z
psinvarmatrix = np.load(Simulation_folder + "psinvarmatrix.npy")
psipvarmatrix = np.load(Simulation_folder + "psipvarmatrix.npy")
ChiMatrix = np.load(Simulation_folder + "ChiMatrix.npy")
EgMatrix = np.load(Simulation_folder + "EgMatrix.npy")

#Ions (reshape to match (V, Y, X, Z))
AnionDensityMatrix = np.load(Simulation_folder + "AnionDensityMatrix.npy")
CationDensityMatrix = np.load(Simulation_folder + "CationDensityMatrix.npy")
AnionDensityMatrix = AnionDensityMatrix.reshape((AnionDensityMatrix.shape[0], NMatrixShapeFull[1], NMatrixShapeFull[2], NMatrixShapeFull[3]))
CationDensityMatrix = CationDensityMatrix.reshape((CationDensityMatrix.shape[0], NMatrixShapeFull[1], NMatrixShapeFull[2], NMatrixShapeFull[3]))

ResidualMatrix = np.load(Simulation_folder + "ResidualMatrix.npy")
SweepCounterMatrix = np.load(Simulation_folder + "SweepCounterMatrix.npy")
applied_voltages = np.load(Simulation_folder + "applied_voltages.npy")

my_cmap = copy.copy(plt.cm.get_cmap('hot'))
my_cmap.set_bad((0, 0, 0))

titles = [
'EField_Y (V/m)', 'Electron Density (1/m3)', 'Hole Density (1/m3)', "Anion Density (1/m3)",
"Cation Density (1/m3)", 'Generation Rate (1/m3)', 'JTotal_Y (A/m2)', 'Potential (V)',
'Recombination Rate (1/m3)', 'PLYield (-)',
"Electron Q.F.P. (eV)", "Hole Q.F.P. (eV)", "Jn_Y (A/m2)", "Jp_Y (A/m2)"
]

#All entries must be (V, Y, X, Z)
data_matrices = [
EField_matrix, NMatrix, PMatrix, AnionDensityMatrix, CationDensityMatrix, GenerationMatrix, JTotal_Y, PotentialMatrix,
RecombinationMatrix, PLYield, psinvarmatrix, psipvarmatrix, Jn_Matrix, Jp_Matrix
]
num_plots = len(data_matrices)

#Dimensions
V, NY, NX, NZ = NMatrix.shape

#Set up subplots
fig, axs = plt.subplots(3, 5, figsize=(12, 10))
axs = axs.ravel()

#Initial z-slice
z0 = 0

#Initial plot setup
images = []
colorbars = []
for i, matrix in enumerate(data_matrices):
    data0 = matrix[0, :, :, z0]
    vmin, vmax = np.nanpercentile(data0, 5), np.nanpercentile(data0, 95)
    im = axs[i].imshow(data0, cmap='viridis', vmin=vmin, vmax=vmax, origin='upper')
    cb = fig.colorbar(im, ax=axs[i])
    axs[i].set_title("{} at {:.3f} V, z={}".format(titles[i], applied_voltages[0], z0))
    images.append(im)
    colorbars.append(cb)

#The last (15th) axes is unused by images; we later plot JV there.
fig.subplots_adjust(left=0.12, right=0.88, top=0.93, bottom=0.17, wspace=0.2, hspace=0.4)

#Sliders
slider_ax = plt.axes([0.2, 0.08, 0.6, 0.025])
slider = Slider(ax=slider_ax, label='Voltage [V]', valmin=applied_voltages.min(), valmax=applied_voltages.max(), valinit=applied_voltages[0])

slider_z_ax = plt.axes([0.2, 0.03, 0.6, 0.025])
slider_z = Slider(ax=slider_z_ax, label='z index', valmin=0, valmax=NZ - 1, valinit=z0, valstep=1)

#Interpolate voltage to array index
interp_indices = interp1d(applied_voltages, np.arange(len(applied_voltages)), bounds_error=False, fill_value="extrapolate")

#Optional: plot IV curve in a separate figure for the initial z-slice
if len(applied_voltages) > 3:
    f_interp = interp1d(applied_voltages, JTotal_Y_mean, kind='linear')
    voltage_fine = np.linspace(applied_voltages[0], applied_voltages[-1], 1000)
    JTotal_Y_fine = f_interp(voltage_fine)

    Voc = voltage_fine[np.argmin(np.abs(JTotal_Y_fine))]
    Jsc = -f_interp(0)

    max_power_index = np.argmin(JTotal_Y_fine * voltage_fine)
    max_power_voltage = voltage_fine[max_power_index]
    max_power_current = JTotal_Y_fine[max_power_index]

    MaxPowerOut = max_power_voltage * max_power_current * -1  # W/m^2
    MaxPowerIn = NumberOfSuns * 1000  # W/m^2
    Efficiency = (MaxPowerOut / MaxPowerIn) * 100

    fig2, ax2 = plt.subplots()
    ax2.plot(voltage_fine, JTotal_Y_fine)
    ax2.set_xlabel("Applied Voltage (V)")
    ax2.set_ylabel("Current Density (A/m^2)")
    ax2.set_title("IV Curve (z=0)")
    ax2.axvline(max_power_voltage, color='r', linestyle='--')
    ax2.axhline(max_power_current, color='r', linestyle='--')
    ax2.text(0.05, 0.70, "Efficiency: {:.3f}%".format(Efficiency), fontsize=8, transform=ax2.transAxes)
    ax2.text(0.05, 0.60, "MPP: {:.3f} V, {:.3f} A/m^2".format(max_power_voltage, max_power_current), fontsize=8, transform=ax2.transAxes)
    ax2.set_ylim(-260, 260)
    ax2.set_xlim(0, 2.4)

    ax3 = ax2.twinx()
    ax3.plot(applied_voltages, ResidualMatrix)
    ax3.set_yscale("log")
    ax3.set_ylabel("Residual")

    axs[-1].plot(voltage_fine, JTotal_Y_fine)
    axs[-1].set_xlabel("Applied Voltage (V)")
    axs[-1].set_title("JV Curve (A/m2)")
    axs[-1].set_ylim(-210, 260)
    axs[-1].set_xlim(-0.1, 2.4)

def update(val):
    voltage = slider.val
    frame = int(np.clip(np.round(interp_indices(voltage)), 0, len(applied_voltages) - 1))

    z_idx = int(slider_z.val)

    # Update images and colorbars for the selected frame and z-slice
    for i, matrix in enumerate(data_matrices):
        data = matrix[frame, :, :, z_idx]
        images[i].set_data(data)
        vmin, vmax = np.nanpercentile(data, 1), np.nanpercentile(data, 99)
        images[i].set_clim(vmin, vmax)
        colorbars[i].update_normal(images[i])
        axs[i].set_title("{} at {:.3f} V, z={}".format(titles[i], applied_voltages[frame], z_idx))

fig.canvas.draw_idle()
#Initial draw
update(None)
slider.on_changed(update)
slider_z.on_changed(update)
plt.show()
