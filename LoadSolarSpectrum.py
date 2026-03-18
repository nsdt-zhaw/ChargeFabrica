import pandas as pd

data = pd.read_excel('./Solar_Spectrum.xls', skiprows=2)
grouped = data.groupby(data.iloc[:,0].round().astype(int)).mean()
SolarSpectrumWavelength = grouped.index.values
SolarSpectrumIrradiance = grouped.iloc[:, 2].values