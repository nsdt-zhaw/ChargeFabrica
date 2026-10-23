import numpy as np
from workflow_utils import load_results
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.interpolate import interp1d

Simulation_folder = sys.argv[1] if len(sys.argv) > 1 else "./Outputs/2D_HTL_Free_Carbon_Device_IONS_Newton_Example/VoltageSweep/"
results = load_results(Simulation_folder, keys=[
    'AnionDensityMatrix', 'CationDensityMatrix',
    'JnY', 'JpY', 'Efield_matrix',
    'GenValues_Matrix', 'NMatrix',
    'PMatrix', 'PotentialMatrix', 'RecombinationMatrix',
    'Recombination_Bimolecular_EQMatrix', 'ResidualArray', 'ResidualMatrix',
    'applied_voltages', 'psinvarmatrix',
    'psipvarmatrix'])

NumberOfSuns = 1.00

# Load matrices as before
GenerationMatrix = results["GenValues_Matrix"]
RecombinationMatrix = results["RecombinationMatrix"]
RadiativeRecombinationMatrix = results["Recombination_Bimolecular_EQMatrix"]
PMatrix = results["PMatrix"]
NMatrix = results["NMatrix"]
Jn_Matrix = results["JnY"][:, 1:-1]
Jp_Matrix = results["JpY"][:, 1:-1]
JTotal_Y = (Jn_Matrix + Jp_Matrix)

JTotal_Y_mean = (results["JnY"][:, -1] + results["JpY"][:, -1]).mean(axis=1)

PLYield = 100*RadiativeRecombinationMatrix / (GenerationMatrix+1)

PotentialMatrix = results["PotentialMatrix"]
EField_matrix = -results["Efield_matrix"][:,1,:,:]
psinvarmatrix = results["psinvarmatrix"]
psipvarmatrix = results["psipvarmatrix"]
AnionDensityMatrix = results["AnionDensityMatrix"]
CationDensityMatrix = results["CationDensityMatrix"]
ResidualMatrix = results["ResidualMatrix"]
ResidualArray = results["ResidualArray"][:]

applied_voltages = results["applied_voltages"]

#Create a plot where residualarray over time is shown for different applied voltages
fig3, ax4 = plt.subplots()
for i in range(len(ResidualArray)):
    ax4.plot(ResidualArray[i], label="V={:.2f}V".format(applied_voltages[i]))
ax4.set_yscale("log")
ax4.set_xlabel("Time Step")
ax4.set_ylabel("Residual")
ax4.set_title("Residual over Time Steps for Different Applied Voltages")
ax4.legend()
plt.show(block=False)



titles = ['EField_Y (V/m)', 'Electron Density (1/m3)', 'Hole Density (1/m3)', "Anion Density (1/m3)",
          "Cation Density (1/m3)", 'Generation Rate (1/m3)', 'JTotal_Y (A/m2)', 'Potential (V)',
         'Recombination Rate (1/m3)', 'PLYield (-)',
          "Electron Q.F.P. (eV)", "Hole Q.F.P. (eV)", "Jn_Y (A/m2)", "Jp_Y (A/m2)"]

data_matrices = [EField_matrix, NMatrix, PMatrix, AnionDensityMatrix, CationDensityMatrix, GenerationMatrix, JTotal_Y, PotentialMatrix,
                  RecombinationMatrix, PLYield,
                 psinvarmatrix, psipvarmatrix,
                 Jn_Matrix, Jp_Matrix]


# Set up subplots
fig, axs = plt.subplots(3, 5, figsize=(16, 10))
axs = axs.ravel()

# Initial plot setup
cax_list = []
for i, matrix in enumerate(data_matrices):
    vmin, vmax = np.percentile(matrix[0], [3, 97])
    cax = axs[i].imshow(matrix[0], cmap='viridis', vmin=vmin, vmax=vmax)
    fig.colorbar(cax, ax=axs[i])
    axs[i].set_title("{}\n{:.3f} V".format(titles[i], applied_voltages[0]), fontsize=9)
    cax_list.append(cax)

# Adjust subplot parameters for a better fit
fig.subplots_adjust(left=0.06, right=0.96, top=0.93, bottom=0.12, wspace=0.6, hspace=0.5)

# Slider axis (center slider position to avoid left gap)
slider_ax = plt.axes([0.2, 0.03, 0.6, 0.025])
slider = Slider(ax=slider_ax, label='Voltage [V]', valmin=applied_voltages.min(), valmax=max(applied_voltages.max(), applied_voltages.min() + 1e-12), valinit=applied_voltages[0])

# Find an interpolation function from voltage to indices


if len(applied_voltages) > 3 and applied_voltages.min() <= 0 <= applied_voltages.max():
    # Interpolate JTotal_Y_mean over a larger set of voltages
    f_interp = interp1d(applied_voltages, JTotal_Y_mean, kind='linear')
    voltage_fine = np.linspace(applied_voltages[0], applied_voltages[-1], 1000)
    JTotal_Y_fine = f_interp(voltage_fine)

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

# Slider update function
def update(val):
    voltage = slider.val
    # find closest index for the given voltage
    frame = int(np.argmin(np.abs(applied_voltages - voltage)))

    for i, matrix in enumerate(data_matrices):
        cax_list[i].set_data(matrix[frame])
        cax_list[i].set_clim(*np.percentile(matrix[frame], [3, 97]))
        axs[i].set_title("{}\n{:.3f} V".format(titles[i], applied_voltages[frame]), fontsize=9)

    fig.canvas.draw_idle()

update(0) # Initial call to display the first frame
slider.on_changed(update)
plt.show()
