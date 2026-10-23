import numpy as np
from workflow_utils import load_results
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.interpolate import interp1d

Simulation_folder = sys.argv[1] if len(sys.argv) > 1 else "./Outputs/3D_HTL_Free_Carbon_Device_IONS/VoltageSweep/"
results = load_results(Simulation_folder, keys=[
    'AnionDensityMatrix', 'CationDensityMatrix',
    'JnZ', 'JpZ', 'Efield_matrix',
    'GenValues_Matrix', 'NMatrix',
    'PMatrix', 'PotentialMatrix', 'RecombinationMatrix',
    'Recombination_Bimolecular_EQMatrix', 'ResidualMatrix',
    'applied_voltages', 'psinvarmatrix', 'psipvarmatrix'])
NumberOfSuns = 1.00

#Load full 4D scalar fields (voltage, y, x, z)
GenerationMatrix = results["GenValues_Matrix"]        # (V, Y, X, Z)
RecombinationMatrix = results["RecombinationMatrix"]  # (V, Y, X, Z)
RadiativeRecombinationMatrix = results["Recombination_Bimolecular_EQMatrix"]  # (V, Y, X, Z)
PMatrix = results["PMatrix"]                          # (V, Y, X, Z)
NMatrix = results["NMatrix"]                          # (V, Y, X, Z)

# Load conservative currents on internal faces (voltage, y-face, x, z).
Jn_Matrix = results["JnZ"][:, 1:-1]
Jp_Matrix = results["JpZ"][:, 1:-1]

#Total y-component
JTotal_Y = (Jn_Matrix + Jp_Matrix)  # (V, Y, X, Z)

# Average the full electrode, including every lateral slice.
JTotal_Y_mean = (results["JnZ"][:, -1] + results["JpZ"][:, -1]).mean(axis=(1, 2))

#PL yield
PLYield = 100 * RadiativeRecombinationMatrix / (GenerationMatrix + 1)

#More fields (full 4D)
PotentialMatrix = results["PotentialMatrix"]  # (V, Y, X, Z)
EField_matrix = -results["Efield_matrix"][:, 2, :, :, :]  # y-component, keep Z
psinvarmatrix = results["psinvarmatrix"]
psipvarmatrix = results["psipvarmatrix"]

#Ions (reshape to match (V, Y, X, Z))
AnionDensityMatrix = results["AnionDensityMatrix"]
CationDensityMatrix = results["CationDensityMatrix"]

ResidualMatrix = results["ResidualMatrix"]
applied_voltages = results["applied_voltages"]


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

#Dimensions
V, NY, NX, NZ = NMatrix.shape

#Set up subplots
fig, axs = plt.subplots(3, 5, figsize=(16, 10))
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
    axs[i].set_title("{}\n{:.3f} V, z={}".format(titles[i], applied_voltages[0], z0), fontsize=9)
    images.append(im)
    colorbars.append(cb)

#The last (15th) axes is unused by images; we later plot JV there.
fig.subplots_adjust(left=0.06, right=0.96, top=0.93, bottom=0.17, wspace=0.6, hspace=0.5)

#Sliders
slider_ax = plt.axes([0.2, 0.08, 0.6, 0.025])
slider = Slider(ax=slider_ax, label='Voltage [V]', valmin=applied_voltages.min(), valmax=max(applied_voltages.max(), applied_voltages.min() + 1e-12), valinit=applied_voltages[0])

slider_z_ax = plt.axes([0.2, 0.03, 0.6, 0.025])
slider_z = Slider(ax=slider_z_ax, label='z index', valmin=0, valmax=NZ - 1, valinit=z0, valstep=1)

#Interpolate voltage to array index

#Optional: plot IV curve in a separate figure for the initial z-slice
if len(applied_voltages) > 3 and applied_voltages.min() <= 0 <= applied_voltages.max():
    f_interp = interp1d(applied_voltages, JTotal_Y_mean, kind='linear')
    voltage_fine = np.linspace(applied_voltages[0], applied_voltages[-1], 1000)
    JTotal_Y_fine = f_interp(voltage_fine)

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
    ax2.set_title("IV Curve (full electrode)")
    ax2.axvline(max_power_voltage, color='r', linestyle='--')
    ax2.axhline(max_power_current, color='r', linestyle='--')
    ax2.text(0.05, 0.70, "Efficiency: {:.3f}%".format(Efficiency), fontsize=8, transform=ax2.transAxes)
    ax2.text(0.05, 0.60, "MPP: {:.3f} V, {:.3f} A/m^2".format(max_power_voltage, max_power_current), fontsize=8, transform=ax2.transAxes)

    ax3 = ax2.twinx()
    ax3.plot(applied_voltages, ResidualMatrix)
    ax3.set_yscale("log")
    ax3.set_ylabel("Residual")

    axs[-1].plot(voltage_fine, JTotal_Y_fine)
    axs[-1].set_xlabel("Applied Voltage (V)")
    axs[-1].set_title("JV Curve (A/m2)")

def update(val):
    voltage = slider.val
    frame = int(np.argmin(np.abs(applied_voltages - voltage)))

    z_idx = int(slider_z.val)

    # Update images and colorbars for the selected frame and z-slice
    for i, matrix in enumerate(data_matrices):
        data = matrix[frame, :, :, z_idx]
        images[i].set_data(data)
        vmin, vmax = np.nanpercentile(data, 1), np.nanpercentile(data, 99)
        images[i].set_clim(vmin, vmax)
        colorbars[i].update_normal(images[i])
        axs[i].set_title("{}\n{:.3f} V, z={}".format(titles[i], applied_voltages[frame], z_idx), fontsize=9)

    fig.canvas.draw_idle()
#Initial draw
update(None)
slider.on_changed(update)
slider_z.on_changed(update)
plt.show()
