import os
import numpy as np

SolarSpectrumWavelength, SolarSpectrumIrradiance = np.loadtxt(
    os.path.join(os.path.dirname(__file__), 'Solar_Spectrum.csv'),
    delimiter=',', skiprows=1, unpack=True)
