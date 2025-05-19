perov_LUMO = 3.9 # eV
perov_HOMO = 5.5 # eV

PS_WF = 0.00 # (We use chi and Eg for semiconductors)
TiO2_WF = 0.00 # (We use chi and Eg for semiconductors)
FTO_WF = 4.4 # eV
Spiro_WF = 0.00 # (We use chi and Eg for semiconductors)
Carbon_WF = 5.20 # eV
ZrO2_WF = 0.00 # (We use chi and Eg for semiconductors)

Spiro_HOMO = 5.4  # eV
Spiro_LUMO = 2.2  # eV

TiO2_HOMO = 7.3  # eV
TiO2_LUMO = 4.1

ZrO2_HOMO = 7.4
ZrO2_LUMO = 2.9

E_gap = perov_HOMO - perov_LUMO

GenerationRate_PS = 1.00 #In Suns. Doesnt need to be scaled by the time step since the equation is dn/dt = ... G-R where G and R are in units of m^-3s^-1
GenerationRate_Voids = 0.00
GenerationRate_mTiO2 = 0.00
GenerationRate_FTO = 0.00
GenerationRate_Spiro = 0.00
GenerationRate_Carbon = 0.00
GenerationRate_ZrO2 = 0.00

epsilon_PS = 35
epsilon_Voids = 1.00
epsilon_mTiO2 = 35
epsilon_FTO = 1.00e3
epsilon_Spiro = 5
epsilon_Carbon = 1.00e3
epsilon_ZrO2 = 35

pmob_PS = 1.00e-4 #Units are m^2/(V.s)
pmob_Voids = 1.00e-8
pmob_mTiO2 = 1.00e-6
pmob_FTO = 1.00e-4
pmob_Spiro = 1.0e-7
pmob_Carbon = 1.00e-4
pmob_ZrO2 = 1.00e-7

nmob_PS = 1.00e-4
nmob_Voids = 1.00e-8
nmob_mTiO2 = 1.00e-6
nmob_FTO = 1.00e-4
nmob_Spiro = 1.0e-7
nmob_Carbon = 1.00e-4
nmob_ZrO2 = 1.00e-7

Eg_PS = E_gap
Eg_Voids = TiO2_HOMO - TiO2_LUMO
Eg_mTiO2 = TiO2_HOMO - TiO2_LUMO
Eg_FTO = TiO2_HOMO - TiO2_LUMO
Eg_Spiro = Spiro_HOMO - Spiro_LUMO
Eg_Carbon = perov_HOMO - perov_LUMO
Eg_ZrO2 = ZrO2_HOMO - ZrO2_LUMO

Chi_PS = perov_LUMO
Chi_Voids = TiO2_LUMO
Chi_mTiO2 = TiO2_LUMO
Chi_FTO = TiO2_LUMO
Chi_Spiro = Spiro_LUMO
Chi_Carbon = perov_LUMO
Chi_ZrO2 = ZrO2_LUMO

cationmob_PS = 1.00e-6
cationmob_Voids = 0.00
cationmob_mTiO2 = 0.00
cationmob_FTO = 0.00
cationmob_Spiro = 0.00
cationmob_Carbon = 0.00
cationmob_ZrO2 = 0.00

anionmob_PS = 1.00e-6
anionmob_Voids = 0.00
anionmob_mTiO2 = 0.00
anionmob_FTO = 0.00
anionmob_Spiro = 0.00
anionmob_Carbon = 0.00
anionmob_ZrO2 = 0.00

Recombination_Langevin_PS = 1.00 #Langevin Efficiency
Recombination_Langevin_Voids = 0.00
Recombination_Langevin_mTiO2 = 0.00
Recombination_Langevin_FTO = 0.00
Recombination_Langevin_Spiro = 0.00
Recombination_Langevin_Carbon = 0.00
Recombination_Langevin_ZrO2 = 0.00

Recombination_Bimolecular_PS = 1.00e-18 #Bimolecular Rate constant
Recombination_Bimolecular_Voids = 0.00
Recombination_Bimolecular_mTiO2 = 0.00
Recombination_Bimolecular_FTO = 0.00
Recombination_Bimolecular_Spiro = 0.00
Recombination_Bimolecular_Carbon = 0.00
Recombination_Bimolecular_ZrO2 = 0.00

min_dense = 1.00e27

Nc_PS = 1.00e25
Nc_Voids = min_dense
Nc_mTiO2 = min_dense
Nc_FTO = 1.00e27
Nc_Spiro = min_dense
Nc_Carbon = 1.00e27
Nc_ZrO2 = 1.00e27

Nv_PS = 1.00e25
Nv_Voids = min_dense
Nv_mTiO2 = min_dense
Nv_FTO = 1.00e27
Nv_Spiro = min_dense
Nv_Carbon = 1.00e27
Nv_ZrO2 = 1.00e27

####################################
#Experimental Injection Barriers for ANIONS
Chi_PS_a = 0.00
Chi_Voids_a = 0.00
Chi_mTiO2_a = 0.00
Chi_FTO_a = 0.00
Chi_Spiro_a = 0.00
Chi_Carbon_a = 0.00
Chi_ZrO2_a = 0.00
#Experimental Injection Barriers for CATIONS
Chi_PS_c = 0.00
Chi_Voids_c = 0.00
Chi_mTiO2_c = 0.00
Chi_FTO_c = 0.00
Chi_Spiro_c = 0.00
Chi_Carbon_c = 0.00
Chi_ZrO2_c = 0.00

a_initial_level_PS = 1.00e23
a_initial_level_Voids = 1.00e23
a_initial_level_mTiO2 = 1.00e23
a_initial_level_FTO = 1.00e23
a_initial_level_Spiro = 1.00e23
a_initial_level_Carbon = 1.00e23
a_initial_level_ZrO2 = 1.00e23

c_initial_level_PS = 1.00e23
c_initial_level_Voids = 1.00e23
c_initial_level_mTiO2 = 1.00e23
c_initial_level_FTO = 1.00e23
c_initial_level_Spiro = 1.00e23
c_initial_level_Carbon = 1.00e23
c_initial_level_ZrO2 = 1.00e23

class Material(object):
    def __init__(self, name, code, GenRate, epsilon, pmob, nmob, Eg, chi,
                 cationmob, anionmob, Recombination_Langevin, Recombination_Bimolecular,
                 Nc, Nv, Chi_a, Chi_c, a_initial_level, c_initial_level, WF):
        self.name = name
        self.code = code
        self.GenRate = GenRate
        self.epsilon = epsilon
        self.pmob = pmob
        self.nmob = nmob
        self.Eg = Eg
        self.chi = chi
        self.cationmob = cationmob
        self.anionmob = anionmob
        self.Recombination_Langevin = Recombination_Langevin
        self.Recombination_Bimolecular = Recombination_Bimolecular
        self.Nc = Nc
        self.Nv = Nv
        self.Chi_a = Chi_a
        self.Chi_c = Chi_c
        self.a_initial_level = a_initial_level
        self.c_initial_level = c_initial_level
        self.WF = WF

# Collect all materials in a dict by code for fast lookup:
MATERIALS = {
    0: Material("Carbon", 0, GenerationRate_Carbon, epsilon_Carbon, pmob_Carbon, nmob_Carbon, Eg_Carbon, Chi_Carbon, cationmob_Carbon, anionmob_Carbon, Recombination_Langevin_Carbon, Recombination_Bimolecular_Carbon, Nc_Carbon, Nv_Carbon, Chi_Carbon_a, Chi_Carbon_c, a_initial_level_Carbon, c_initial_level_Carbon, Carbon_WF),
    100: Material("PS", 100, GenerationRate_PS, epsilon_PS, pmob_PS, nmob_PS, Eg_PS, Chi_PS, cationmob_PS, anionmob_PS, Recombination_Langevin_PS, Recombination_Bimolecular_PS, Nc_PS, Nv_PS, Chi_PS_a, Chi_PS_c, a_initial_level_PS, c_initial_level_PS, PS_WF),
    150: Material("mTiO2", 150, GenerationRate_mTiO2, epsilon_mTiO2, pmob_mTiO2, nmob_mTiO2, Eg_mTiO2, Chi_mTiO2, cationmob_mTiO2, anionmob_mTiO2, Recombination_Langevin_mTiO2, Recombination_Bimolecular_mTiO2, Nc_mTiO2, Nv_mTiO2, Chi_mTiO2_a, Chi_mTiO2_c, a_initial_level_mTiO2, c_initial_level_mTiO2, TiO2_WF),
    200: Material("FTO", 200, GenerationRate_FTO, epsilon_FTO, pmob_FTO, nmob_FTO, Eg_FTO, Chi_FTO, cationmob_FTO, anionmob_FTO, Recombination_Langevin_FTO, Recombination_Bimolecular_FTO, Nc_FTO, Nv_FTO, Chi_FTO_a, Chi_FTO_c, a_initial_level_FTO, c_initial_level_FTO, FTO_WF),
    225: Material("ZrO2", 225, GenerationRate_ZrO2,epsilon_ZrO2, pmob_ZrO2, nmob_ZrO2, Eg_ZrO2, Chi_ZrO2, cationmob_ZrO2, anionmob_ZrO2, Recombination_Langevin_ZrO2, Recombination_Bimolecular_ZrO2, Nc_ZrO2, Nv_ZrO2, Chi_ZrO2_a, Chi_ZrO2_c, a_initial_level_ZrO2, c_initial_level_ZrO2, ZrO2_WF),
}