import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider
from scipy.interpolate import interp1d


# Load data
Simulation_folder = "./Outputs/Drift_Diffusion_1D_IV_Simple_HTL_Free_Carbon_Device_IONS/VoltageSweep/"
NumberOfSuns = 1.00

GenerationMatrix = np.load(Simulation_folder + "GenValues_Matrix.npy")[:]
RecombinationMatrix = np.load(Simulation_folder + "RecombinationMatrix.npy")[:]
PMatrix = np.load(Simulation_folder + "p.npy")[:]
NMatrix = np.load(Simulation_folder + "n.npy")[:]
JTotal_Y = np.load(Simulation_folder + "J_Total_Y.npy")[:]
Jn_Matrix = np.load(Simulation_folder + "Jn_Matrix.npy")[:]
Jp_Matrix = np.load(Simulation_folder + "Jp_Matrix.npy")[:]
PotentialMatrix = np.load(Simulation_folder + "phi.npy")[:]
EField_matrix = np.load(Simulation_folder + "EField_matrix.npy")[:]
applied_voltages = np.load(Simulation_folder + "applied_voltages.npy")[:]
psinvarmatrix = np.load(Simulation_folder + "psinvarmatrix.npy")[:]
psipvarmatrix = np.load(Simulation_folder + "psipvarmatrix.npy")[:]
ChiMatrix = np.load(Simulation_folder + "ChiMatrix.npy")[:]
EgMatrix = np.load(Simulation_folder + "EgMatrix.npy")[:]
ResidualMatrix = np.load(Simulation_folder + "ResidualMatrix.npy")[:]
RadiativeRecombinationMatrix = np.load(Simulation_folder + "Recombination_Bimolecular_EQ_ReLUMatrix.npy")[:]
PLYield = 100*RadiativeRecombinationMatrix / (GenerationMatrix+1)

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

AnionDensityMatrix = np.load(Simulation_folder + "AnionDensityMatrix.npy")[:]
CationDensityMatrix = np.load(Simulation_folder + "CationDensityMatrix.npy")[:]

# Initial calculations
JTotal_Y_mean = np.median(JTotal_Y[:,100:-100,:], axis=(1, 2))

titles = ['EField_matrix', 'PMatrix', 'Generation Rate', 'JTotal_Y', 'PotentialMatrix', 'NMatrix', 'RecombinationMatrix', "Joule Heating", "Anion Density", "Cation Density"]
print "JTotal_Y_mean shape: ", JTotal_Y_mean.shape
print "applied_voltages shape: ", applied_voltages.shape

if len(applied_voltages) > 3:
    # Interpolate JTotal_Y_mean over a larger set of voltages
    f_interp = interp1d(applied_voltages, JTotal_Y_mean, kind='linear')
    voltage_fine = np.linspace(applied_voltages[0], applied_voltages[-1], 1000)
    JTotal_Y_fine = f_interp(voltage_fine)
    Voc = voltage_fine[np.argmin(np.abs(JTotal_Y_fine))]
    Jsc = -f_interp(0)
    max_power_index = np.argmin(JTotal_Y_fine * voltage_fine)
    max_power_voltage = voltage_fine[max_power_index]
    max_power_current = JTotal_Y_fine[max_power_index]
    MaxPowerOut = max_power_voltage * max_power_current * -1
    MaxPowerIn = NumberOfSuns * 1000
    Efficiency = (MaxPowerOut/MaxPowerIn)*100

    print "Maximum power point: {:.3f} V, {:.3f} A/m^2".format(max_power_voltage, max_power_current)
    print "Efficiency: {:.3f}".format(Efficiency)
    print "Voc: {:.3f} V".format(Voc)
    print "Jsc: {:.3f} A/m^2".format(Jsc)
    print "Fill Factor: ", (MaxPowerOut/(Voc*Jsc))*100

    fig2, ax2 = plt.subplots()
    ax2.plot(voltage_fine, JTotal_Y_fine)
    ax2.set_xlabel("Applied Voltage (V)")
    ax2.set_ylabel("Current Density (A/m^2)")
    ax2.set_title("IV Curve")
    ax2.axvline(max_power_voltage, color='r', linestyle='--')
    ax2.axhline(max_power_current, color='r', linestyle='--')
    ax2.text(0.05, 0.70, "Efficiency: {:.3f}%".format(Efficiency), fontsize=8, transform=ax2.transAxes)
    ax2.text(0.05, 0.65, "Fill Factor: {:.3f}%".format((MaxPowerOut/(Voc*Jsc))*100), fontsize=8, transform=ax2.transAxes)
    ax2.text(0.05, 0.60, "MPP: {:.3f} V, {:.3f} A/m^2".format(max_power_voltage, max_power_current), fontsize=8, transform=ax2.transAxes)
    ax2.set_ylim(-300, 300)
    ax2.set_xlim(-0.3, 1.4)
    plt.show(block=False)

PotentialMatrix = np.expand_dims(PotentialMatrix, axis=2)
LUMO = PotentialMatrix + ChiMatrix
HOMO = PotentialMatrix + ChiMatrix + EgMatrix

titles = ['EField Strength (V/m)', 'Generation Rate (1/m3)', 'Potential (V)', 'Recombination (1/m3)', 'PLYield (-)']
data_matrices = [EField_matrix, GenerationMatrix, PotentialMatrix, RecombinationMatrix, PLYield]
num_plots = len(data_matrices)

fig, axs = plt.subplots(2, 4, figsize=(12, 10))
axs = axs.ravel()
cax_list = []
colorbars = []
norms = []

for i, matrix in enumerate(data_matrices):
    cax, = axs[i].plot(matrix[0]) # the comma unpacks the first element
    axs[i].set_title("{} at {:.3f} V".format(titles[i], applied_voltages[0]))
    cax_list.append(cax)

fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.20, wspace=0.2, hspace=0.4)
slider_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax=slider_ax, label='Voltage [V]', valmin=applied_voltages.min(), valmax=applied_voltages.max(), valinit=applied_voltages[0])
interp_indices = interp1d(applied_voltages, np.arange(len(applied_voltages)), bounds_error=False, fill_value="extrapolate")

def update(val):
    voltage = slider.val
    # find closest index for the given voltage
    frame = int(np.clip(np.round(interp_indices(voltage)), 0, len(applied_voltages) - 1))
    for i, matrix in enumerate(data_matrices):
        axs[i].clear()
        axs[i].plot(matrix[frame])
        axs[i].set_title("{} at {:.3f} V".format(titles[i], applied_voltages[frame]))
    axs[-1].clear()
    axs[-1].plot(LUMO[frame][:], "r")
    axs[-1].plot(HOMO[frame][:], "b")
    axs[-1].plot(psinvarmatrix[frame][:], "g")
    axs[-1].plot(psipvarmatrix[frame][:], color="purple")
    axs[-1].set_ylim(8, 0)
    axs[-1].legend(["LUMO", "HOMO", "psinvar", "psipvar"], fontsize=8)
    axs[-1].set_title("Band diagram (eV)")
    axs[-2].clear()
    axs[-2].plot(Jn_Matrix[frame][5:-5], "r")
    axs[-2].plot(Jp_Matrix[frame][5:-5], "b")
    axs[-2].plot(JTotal_Y[frame][5:-5], "g")
    axs[-2].set_ylim(-300, 500)
    axs[-2].legend(["Electron Current", "Hole Current", "Total Current"])
    axs[-2].set_title("Current Density (A/m2)")
    axs[-3].clear()
    axs[-3].plot(NMatrix[frame][:], "r")
    axs[-3].plot(PMatrix[frame][:], "b")
    axs[-3].plot(AnionDensityMatrix[frame][:], "g")
    axs[-3].plot(CationDensityMatrix[frame][:], "y")
    axs[-3].set_yscale("log")
    axs[-3].set_ylim(1.00e15,1.00e25)
    axs[-3].legend(["Electron", "Hole", "Anion", "Cation"])
    axs[-3].set_title("Charge Carrier Distribution (1/m3)")
    fig.canvas.draw_idle()

update(0) # Initial call to display the first frame
slider.on_changed(update)
plt.show()