from material_maps import Semiconductors, Electrodes
import numpy as np
from constantsfile import D

def map_semiconductor_property(devarray, prop):
    return np.vectorize(lambda x: getattr(Semiconductors[x], prop))(devarray)

def map_electrode_property(devarray, prop):
    return np.vectorize(lambda x: getattr(Electrodes[x], prop))(devarray)

#Here we define the Ohmic boundary conditions
def ohmic(sc_slice, electrode_id):
    nboundary = map_semiconductor_property(sc_slice, 'Nc') * np.exp((map_semiconductor_property(sc_slice, 'chi') - map_electrode_property(electrode_id, "WF")) / D)
    pboundary = map_semiconductor_property(sc_slice, 'Nv') * np.exp((map_electrode_property(electrode_id, "WF") - (map_semiconductor_property(sc_slice, 'chi') + map_semiconductor_property(sc_slice, 'Eg'))) / D)
    return nboundary, pboundary