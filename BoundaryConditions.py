import numpy as np
from constantsfile import D
from material_maps import map_electrode_property, map_semiconductor_property

#Here we define the Ohmic boundary conditions
def ohmic(sc_slice, electrode_id):
    nboundary = map_semiconductor_property(sc_slice, 'Nc') * np.exp((map_semiconductor_property(sc_slice, 'chi') - map_electrode_property(electrode_id, "WF")) / D)
    pboundary = map_semiconductor_property(sc_slice, 'Nv') * np.exp((map_electrode_property(electrode_id, "WF") - (map_semiconductor_property(sc_slice, 'chi') + map_semiconductor_property(sc_slice, 'Eg'))) / D)
    return nboundary, pboundary