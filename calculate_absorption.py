from scipy.interpolate import interp1d
from scipy.constants import c, h
import numpy as np

def calculate_absorption_above_bandgap(SolarSpectrumWavelength,SolarSpectrumIrradiance,AbsorptionWavelength,AbsorptionCoefficient, DeviceArchitecture,PixelWidth,bandgap_eV):

    # Constants
    nm_to_m = 1e-9
    eV_to_J = 1.60218e-19
    kT = 4.11e-21  # ~25 C

    # Ensure arrays
    SolarSpectrumWavelength = np.asarray(SolarSpectrumWavelength)
    SolarSpectrumIrradiance = np.asarray(SolarSpectrumIrradiance)
    AbsorptionWavelength = np.asarray(AbsorptionWavelength)
    AbsorptionCoefficient = np.asarray(AbsorptionCoefficient)
    DeviceArchitecture = np.asarray(DeviceArchitecture)

    if DeviceArchitecture.ndim < 2:
        raise ValueError("DeviceArchitecture must have at least 2 dimensions: (depth, lateral...).")

    n_lambda = SolarSpectrumWavelength.shape[0]
    spatial_shape = DeviceArchitecture.shape  # (Nz, ...)

    # Interpolate absorption coefficient to solar spectrum grid
    interp_abs_coeff = interp1d(
        AbsorptionWavelength, AbsorptionCoefficient,
        kind='linear', bounds_error=False, fill_value="extrapolate"
    )
    absorption_coeff_at_solar_wavelengths = interp_abs_coeff(SolarSpectrumWavelength)  # shape (n_lambda,)

    # Photon energy at each wavelength
    photon_energy = (h * c) / (SolarSpectrumWavelength * nm_to_m)  # J

    # Indices
    Eg = bandgap_eV * eV_to_J
    above_bandgap = np.where(photon_energy >= Eg)[0]
    all_energy = np.where(photon_energy > 0)[0]

    # Photon flux (spectral)
    photon_flux_above = SolarSpectrumIrradiance[above_bandgap] / photon_energy[above_bandgap]  # (n_above,)
    photon_flux_total = SolarSpectrumIrradiance / photon_energy  # (n_lambda,)

    # Allocate hyperspectral arrays: (n_lambda, Nz, ...)
    out_shape = (n_lambda,) + spatial_shape
    PhotonFluxArray = np.zeros(out_shape)
    GenerationArray = np.zeros(out_shape)
    ThermalisationArray = np.zeros(out_shape)
    PhotonEnergyArray = np.zeros(out_shape)
    PhotonFluxArrayTotal = np.zeros(out_shape)

    # Bottom boundary (last depth slice = -1)
    for k, idx in enumerate(above_bandgap):
        mu = absorption_coeff_at_solar_wavelengths[idx]
        attenuation_bottom = np.exp(-DeviceArchitecture[-1, ...] * mu * PixelWidth)
        # Flux that emerges from bottom slice (per unit length)
        PhotonFluxArray[idx, -1, ...] = photon_flux_above[k] * attenuation_bottom / PixelWidth
        # Absorbed in bottom slice
        GenerationArray[idx, -1, ...] = (photon_flux_above[k] * (1.0 - attenuation_bottom)) * DeviceArchitecture[-1, ...] / PixelWidth
        # Energy carried (use the per-length flux at bottom)
        PhotonEnergyArray[idx, -1, ...] = PhotonFluxArray[idx, -1, ...] * photon_energy[idx] * PixelWidth

    # Initialize total photon flux at bottom for all energies
    for idx in all_energy:
        PhotonFluxArrayTotal[idx, -1, ...] = photon_flux_total[idx] / PixelWidth

    # Copy total flux upward (no attenuation yet; will be applied for above-bandgap wavelengths below)
    Nz = spatial_shape[0]
    for row in range(Nz - 2, -1, -1):
        for idx in all_energy:
            PhotonFluxArrayTotal[idx, row, ...] = PhotonFluxArrayTotal[idx, row + 1, ...]

    # Propagate from bottom to top for above-bandgap wavelengths
    for row in range(Nz - 2, -1, -1):
        layer = DeviceArchitecture[row, ...]
        for idx in above_bandgap:
            mu = absorption_coeff_at_solar_wavelengths[idx]
            attenuation = np.exp(-layer * mu * PixelWidth)
            # Flux that reaches the top of this layer
            PhotonFluxArray[idx, row, ...] = PhotonFluxArray[idx, row + 1, ...] * attenuation
            # Total flux used for energy tally (same attenuation for above-bandgap)
            PhotonFluxArrayTotal[idx, row, ...] = PhotonFluxArrayTotal[idx, row + 1, ...] * attenuation
            # Absorption in this layer
            GenerationArray[idx, row, ...] = PhotonFluxArray[idx, row + 1, ...] - PhotonFluxArray[idx, row, ...]
            # Thermalisation (excess energy above Eg minus ~3kT)
            ThermalisationArray[idx, row, ...] = GenerationArray[idx, row, ...] * (photon_energy[idx] - Eg - 3.0 * kT)

    # Energy carried per layer (using total flux in the layer above)
    for row in range(Nz - 2, -1, -1):
        for idx in all_energy:
            PhotonEnergyArray[idx, row, ...] = PhotonFluxArrayTotal[idx, row + 1, ...] * photon_energy[idx] * PixelWidth

    # Collapse spectral axis
    ThermalisationArray = -np.sum(ThermalisationArray, axis=0)
    GenerationArray = np.sum(GenerationArray, axis=0)
    PhotonFluxArray = np.sum(PhotonFluxArray, axis=0)
    PhotonEnergyArray = np.sum(PhotonEnergyArray, axis=0)

    # Shapes returned:
    #   GenerationArray: (Nz, ...)
    #   ThermalisationArray: (Nz, ...)
    #   PhotonFluxArray: (Nz, ...)
    #   PhotonEnergyArray: (Nz, ...)
    return GenerationArray, ThermalisationArray, PhotonFluxArray, PhotonEnergyArray