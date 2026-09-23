import numpy as np
from workflow_utils import load_results
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider
from scipy.interpolate import interp1d
from plotting_utils import median_filter_1d

# Load data
Simulation_folder = "./Outputs/1D_IONS_NIP_Example/VoltageSweep/"
results = load_results(Simulation_folder)
NumberOfSuns = 1.00

GenerationMatrix = results["GenValues_Matrix"][:]
RecombinationMatrix = results["RecombinationMatrix"][:]
PMatrix = results["p"][:]
NMatrix = results["n"][:]

Jn_Matrix = results["ConservativeJnInternal"]
Jp_Matrix = results["ConservativeJpInternal"]

JTotal_Y = (Jn_Matrix + Jp_Matrix)
PotentialMatrix = results["phi"][:]
EField_matrix = -results["Efield_matrix"][:,1,:,:]

applied_voltages = results["applied_voltages"][:]
print("applied_voltages ", applied_voltages)
psinvarmatrix = results["psinvarmatrix"][:]
psipvarmatrix = results["psipvarmatrix"][:]
ChiMatrix = results["ChiMatrix"][:]
EgMatrix = results["EgMatrix"][:]
ResidualMatrix = results["ResidualMatrix"][:]
ResidualArray = results["ResidualArray"][:]
RadiativeRecombinationMatrix = results["Recombination_Bimolecular_EQMatrix"][:]
PLYield = 100*RadiativeRecombinationMatrix / (GenerationMatrix+1)
SweepCounterMatrix = results["SweepCounterMatrix"][:]

#Create a plot where residualarray over time is shown for different applied voltages
fig3, ax4 = plt.subplots()
for i in range(ResidualArray.shape[0]):
    ax4.plot(ResidualArray[i,:], label="V={:.2f}V".format(applied_voltages[i]))
ax4.set_yscale("log")
ax4.set_xlabel("Time Step")
ax4.set_ylabel("Residual")
ax4.set_title("Residual over Time Steps for Different Applied Voltages")
ax4.legend()
plt.show(block=False)

print("SweepCounterMatrix", SweepCounterMatrix)

AnionDensityMatrix = results["AnionDensityMatrix"][:]
CationDensityMatrix = results["CationDensityMatrix"][:]

# Initial calculations
JTotal_Y_mean = np.mean(JTotal_Y, axis=(1, 2))

titles = ['EField_matrix', 'PMatrix', 'Generation Rate', 'JTotal_Y', 'PotentialMatrix', 'NMatrix', 'RecombinationMatrix', "Joule Heating", "Anion Density", "Cation Density"]
print("JTotal_Y_mean shape: ", JTotal_Y_mean.shape)
print("applied_voltages shape: ", applied_voltages.shape)

if len(applied_voltages) > 3 and applied_voltages.min() <= 0 <= applied_voltages.max():
    # Interpolate JTotal_Y_mean over a larger set of voltages
    f_interp = interp1d(applied_voltages, JTotal_Y_mean, kind='linear')
    voltage_fine = np.linspace(applied_voltages[0], applied_voltages[-1], 1000)
    JTotal_Y_fine = f_interp(voltage_fine)
    Voc = applied_voltages[np.argmin(np.abs(JTotal_Y_mean))]
    Jsc = f_interp(0)
    PINCHECK = 1.00
    if Jsc > 0:
        PINCHECK = -1.00
    max_power_index = np.argmin(PINCHECK * JTotal_Y_mean * applied_voltages)
    max_power_voltage = applied_voltages[max_power_index]
    max_power_current = JTotal_Y_mean[max_power_index]
    MaxPowerOut = max_power_voltage * max_power_current * -1
    MaxPowerIn = NumberOfSuns * 1000
    Efficiency = (PINCHECK * MaxPowerOut/MaxPowerIn)*100

    print("Maximum power point: {:.3f} V, {:.3f} A/m^2".format(max_power_voltage, max_power_current))
    print("Efficiency: {:.3f}".format(Efficiency))
    print("Voc: {:.3f} V".format(Voc))
    #print("Jsc: {:.3f} A/m^2".format(Jsc))
    #print("Fill Factor: ", (MaxPowerOut/(Voc*Jsc))*100)

    fig2, ax2 = plt.subplots()
    ax2.plot(applied_voltages, JTotal_Y_mean)
    ax2.set_xlabel("Applied Voltage (V)")
    ax2.set_ylabel("Current Density (A/$\mathrm{m^2}$)")
    ax2.set_title("IV Curve")
    ax2.axvline(max_power_voltage, color='r', linestyle='--')
    ax2.axhline(max_power_current, color='r', linestyle='--')
    ax2.text(0.05, 0.70, "Efficiency: {:.3f}%".format(Efficiency), fontsize=8, transform=ax2.transAxes)
    #ax2.text(0.05, 0.65, "Fill Factor: {:.3f}%".format((MaxPowerOut/(Voc*Jsc))*100), fontsize=8, transform=ax2.transAxes)
    ax2.text(0.05, 0.60, "MPP: {:.3f} V, {:.3f} A/m^2".format(max_power_voltage, max_power_current), fontsize=8, transform=ax2.transAxes)
    ax2.set_ylim(-300, 300)
    ax2.set_xlim(0, 2.6)
    #Add twin y axis
    ax3 = ax2.twinx()
    ax3.plot(applied_voltages, ResidualMatrix)
    ax3.set_yscale("log")
    plt.show(block=False)

PotentialMatrix = np.expand_dims(PotentialMatrix, axis=2)

titles = ['EField Strength (V/m)', 'Generation Rate (1/$\mathrm{m^3}$)', 'Potential (V)', 'Recombination (1/$\mathrm{m^3}$)', 'PLYield (-)']
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
    #Flip the x axis
    axs[i].invert_xaxis()


fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.20, wspace=0.2, hspace=0.4)
slider_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax=slider_ax, label='Voltage [V]', valmin=applied_voltages.min(), valmax=max(applied_voltages.max(), applied_voltages.min() + 1e-12), valinit=applied_voltages[0])

def update(val):
    voltage = slider.val
    # find closest index for the given voltage
    frame = int(np.argmin(np.abs(applied_voltages - voltage)))
    for i, matrix in enumerate(data_matrices):
        axs[i].clear()
        axs[i].plot(matrix[frame])
        axs[i].set_title("{} at {:.3f} V".format(titles[i], applied_voltages[frame]))
        if titles[i] == "Recombination (1/$\mathrm{m^3}$)":
            axs[i].set_yscale("log")
            axs[i].set_ylim(1.00e24, 1.00e29)
        axs[i].invert_xaxis()

    axs[-1].clear()
    axs[-1].plot((PotentialMatrix + ChiMatrix)[frame][:], "r")
    axs[-1].plot((PotentialMatrix + ChiMatrix + EgMatrix)[frame][:], "b")
    axs[-1].plot(psinvarmatrix[frame][:], "g")
    axs[-1].plot(psipvarmatrix[frame][:], color="purple")
    axs[-1].set_ylim(8, 0)
    axs[-1].legend(["LUMO", "HOMO", "psinvar", "psipvar"], fontsize=8)
    axs[-1].set_title("Band diagram (eV)")
    axs[-1].invert_xaxis()
    axs[-2].clear()
    axs[-2].plot(Jn_Matrix[frame][5:-5], "r")
    axs[-2].plot(Jp_Matrix[frame][5:-5], "b")
    axs[-2].plot(JTotal_Y[frame][5:-5], "g")
    axs[-2].set_ylim(-300, 500)
    axs[-2].legend(["Electron Current", "Hole Current", "Total Current"])
    axs[-2].set_title("Current Density (A/$\mathrm{m^2}$)")
    axs[-2].invert_xaxis()
    axs[-3].clear()
    axs[-3].plot(NMatrix[frame][:], "r")
    axs[-3].plot(PMatrix[frame][:], "b")
    axs[-3].plot(AnionDensityMatrix[frame][:], "g")
    axs[-3].plot(CationDensityMatrix[frame][:], "y")
    axs[-3].set_yscale("log")

    #Find maximum of AnionDensityMatrix[frame][:] or CationDensityMatrix[frame][:] and set y limit accordingly
    max_anion = np.max(AnionDensityMatrix[frame][:])
    max_cation = np.max(CationDensityMatrix[frame][:])
    max_electron = np.max(NMatrix[frame][:])
    max_hole = np.max(PMatrix[frame][:])
    max_y = max(max_anion, max_cation, max_electron, max_hole)
    axs[-3].set_ylim(1.00e5,max_y*10)
    axs[-3].legend(["Electron", "Hole", "Anion", "Cation"])
    axs[-3].set_title("Charge Carrier Distribution (1/$\mathrm{m^3}$)")
    axs[-3].invert_xaxis()
    fig.canvas.draw_idle()

update(0) # Initial call to display the first frame
slider.on_changed(update)
plt.show()
