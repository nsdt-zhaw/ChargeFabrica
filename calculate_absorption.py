from scipy.interpolate import interp1d
from scipy.constants import c, h
import numpy as np
import matplotlib.pyplot as plt


def calculate_absorption_above_bandgap(SolarSpectrumWavelength, SolarSpectrumIrradiance, AbsorptionWavelength, AbsorptionCoefficient, DeviceArchitecture2D, PixelWidth, bandgap_eV):
    # Define constants
    nm_to_m = 1e-9  # Convert nm to m
    eV_to_J = 1.60218e-19  # Convert eV to J
    kT = 4.11e-21  # Thermal energy at room temperature (25 degrees Celsius) in Joules

    # Interpolate the absorption coefficient data to match the solar spectrum wavelengths
    interp_abs_coeff = interp1d(AbsorptionWavelength, AbsorptionCoefficient, kind='linear', bounds_error=False, fill_value="extrapolate")

    # Convert the interpolated absorption coefficient to the solar spectrum wavelength grid
    absorption_coeff_at_solar_wavelengths = interp_abs_coeff(SolarSpectrumWavelength)

    # Calculate the energy of a photon at each wavelength
    photon_energy = (h * c) / (SolarSpectrumWavelength * nm_to_m)  # photon energy in joules

    # Filter out photons with energy below the bandgap
    above_bandgap_indices = np.where(photon_energy >= bandgap_eV * eV_to_J)[0]

    all_energy_indices = np.where(photon_energy > 0)[0]

    #print(SolarSpectrumWavelength[above_bandgap_indices])

    # Calculate the number of photons at each wavelength above the bandgap
    photon_flux_above_bandgap = SolarSpectrumIrradiance[above_bandgap_indices] / photon_energy[above_bandgap_indices]  # photons/(m^2 s)

    photon_flux_total = SolarSpectrumIrradiance / photon_energy

    #print("photon flux above bandgap: ", sum(photon_flux_above_bandgap))

    # Initialize a hyperspectral photon flux array
    PhotonFluxArray = np.zeros((SolarSpectrumWavelength.shape[0], DeviceArchitecture2D.shape[0], DeviceArchitecture2D.shape[1]))
    GenerationArray = np.zeros((SolarSpectrumWavelength.shape[0], DeviceArchitecture2D.shape[0], DeviceArchitecture2D.shape[1]))
    ThermalisationArray = np.zeros((SolarSpectrumWavelength.shape[0], DeviceArchitecture2D.shape[0], DeviceArchitecture2D.shape[1]))
    PhotonEnergyArray = np.zeros((SolarSpectrumWavelength.shape[0], DeviceArchitecture2D.shape[0], DeviceArchitecture2D.shape[1]))

    # Set the PhotonFluxArray to the photon flux above the bandgap for the last row (the bottom of the device)
    for idx in above_bandgap_indices:
        absorption_coeff = absorption_coeff_at_solar_wavelengths[idx]
        attenuation = np.exp(-DeviceArchitecture2D[-1, :] * absorption_coeff * PixelWidth)
        PhotonFluxArray[idx, -1, :] = photon_flux_above_bandgap[idx] * attenuation / PixelWidth
        GenerationArray[idx, -1, :] = (photon_flux_above_bandgap[idx] - photon_flux_above_bandgap[idx] * attenuation) * DeviceArchitecture2D[-1, :] / PixelWidth
        PhotonEnergyArray[idx, -1, :] = PhotonFluxArray[idx, -1, :] * photon_energy[idx] * PixelWidth

    for idx in all_energy_indices:
        PhotonFluxArray[idx, -1, :] = photon_flux_total[idx] / PixelWidth

    for row in range(DeviceArchitecture2D.shape[0] - 2, -1, -1):  # Start from the second-to-last row and move upwards
        for idx in all_energy_indices:
            PhotonFluxArray[idx, row, :] = PhotonFluxArray[idx, row + 1, :]

    # Calculate photon flux from the bottom to the top
    for row in range(DeviceArchitecture2D.shape[0] - 2, -1, -1):  # Start from the second-to-last row and move upwards
        for idx in above_bandgap_indices:
            absorption_coeff = absorption_coeff_at_solar_wavelengths[idx]
            attenuation = np.exp(-DeviceArchitecture2D[row, :] * absorption_coeff * PixelWidth)
            PhotonFluxArray[idx, row, :] = PhotonFluxArray[idx, row + 1, :] * attenuation
            GenerationArray[idx, row, :] = PhotonFluxArray[idx, row + 1, :] - PhotonFluxArray[idx, row, :]
            ThermalisationArray[idx, row, :] = GenerationArray[idx, row, :] * (photon_energy[idx] - bandgap_eV * eV_to_J - 3*kT)

    for row in range(DeviceArchitecture2D.shape[0] - 2, -1, -1):  # Start from the second-to-last row and move upwards
        for idx in all_energy_indices:
            PhotonEnergyArray[idx, row, :] = PhotonFluxArray[idx, row + 1, :] * photon_energy[idx] * PixelWidth

    ThermalisationArray = -np.sum(ThermalisationArray, axis=0)
    # Calculate the generation rate by summing the photon flux at each wavelength
    GenerationArray = np.sum(GenerationArray, axis=0)

    PhotonFluxArray = np.sum(PhotonFluxArray, axis=0)

    PhotonEnergyArray = np.sum(PhotonEnergyArray, axis=0)

    #print("Number of photons absorbed: ", np.sum(GenerationArray, axis=0))
    return GenerationArray, ThermalisationArray, PhotonFluxArray, PhotonEnergyArray
