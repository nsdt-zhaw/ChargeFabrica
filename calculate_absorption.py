"""Shared Beer-Lambert optical generation and lattice thermalisation."""
import numpy as np
from scipy.constants import Boltzmann, c, h, electron_volt
from scipy.interpolate import interp1d


def calculate_absorption_above_bandgap(SolarSpectrumWavelength, SolarSpectrumIrradiance,
                                      AbsorptionWavelength, AbsorptionCoefficient,
                                      DeviceArchitecture, PixelWidth, bandgap_eV,
                                      temperature=298.15):
    """Return generation, thermalisation, outgoing photon flux and optical power.

    Illuminate the last row of a (depth, lateral...) nonnegative absorption
    mask, not a material-ID array. PixelWidth is depth-cell thickness in m;
    wavelengths are nm and absorption coefficients are 1/m. Spectral inputs
    use the repository's implicit 1 nm bins and are summed without new weights.
    Temperature is in K, either scalar or broadcastable to the device shape.

    All four outputs have the device shape. Generation is in 1/(m3 s),
    thermalisation in W/m3, outgoing above-bandgap photon flux divided by
    PixelWidth in 1/(m3 s), and outgoing all-spectrum optical power in W/m2.
    Exit faces lie toward decreasing depth indices. Heat is absorbed power
    minus G * (Eg + 3 kT); it can be negative close to the band edge.
    Below-bandgap light passes through unchanged. Negative interpolated
    absorption coefficients are clipped to zero. One wavelength is accumulated
    at a time, avoiding wavelength-by-device arrays.
    """
    wavelength, irradiance = SolarSpectrumWavelength, SolarSpectrumIrradiance
    absorption_wavelength, absorption_coefficient = AbsorptionWavelength, AbsorptionCoefficient
    spacing = PixelWidth
    mask = np.asarray(DeviceArchitecture, dtype=float)
    temperature = np.broadcast_to(np.asarray(temperature, dtype=float), mask.shape)
    wavelength = np.asarray(wavelength, dtype=float)
    irradiance = np.asarray(irradiance, dtype=float)
    alpha = interp1d(absorption_wavelength, absorption_coefficient,
                     bounds_error=False, fill_value="extrapolate")(wavelength)
    if (mask.ndim < 2 or spacing <= 0 or np.any(temperature <= 0) or np.any(~np.isfinite(temperature))
            or np.any(mask < 0) or np.any(~np.isfinite(alpha))):
        raise ValueError("Require a nonnegative device mask and positive spacing/temperature")
    generation, absorbed_power, photon_flux, transmitted = [np.zeros_like(mask) for _ in range(4)]
    # Optical depth between the illuminated surface and each cell's entrance.
    depth = np.cumsum(mask[::-1], axis=0)[::-1] - mask
    for energy, intensity, coefficient in zip(h*c/(wavelength*1e-9), irradiance, alpha):
        optical_depth = max(float(coefficient), 0.) * spacing
        if energy < bandgap_eV * electron_volt:
            transmitted += intensity
            continue
        incoming = intensity * np.exp(-optical_depth * depth)
        absorbed = incoming * (-np.expm1(-optical_depth * mask))
        outgoing = incoming - absorbed
        generation += absorbed / (energy * spacing)
        absorbed_power += absorbed / spacing
        photon_flux += outgoing / (energy * spacing)
        transmitted += outgoing
    heat = absorbed_power - generation * (bandgap_eV * electron_volt + 3*Boltzmann*temperature)
    return generation, heat, photon_flux, transmitted
