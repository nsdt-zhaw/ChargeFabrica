import os

class Material(object):
    def __init__(self, name, code, GenRate, epsilon, pmob, nmob, Eg, chi, cationmob, anionmob, Recombination_Langevin,
                 Recombination_Bimolecular, Nc, Nv, Chi_a, Chi_c, a_initial_level, c_initial_level, WF):
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

def load_material_from_txt(path):
    fields = {}
    with open(path) as f:
        for line in f:
            if ":" not in line or line.strip().startswith("#"):
                continue  # Skip comments or invalid lines
            key, val = [i.strip() for i in line.split(":", 1)]
            try:
                val = float(val)
            except Exception:
                pass  # Keep as string if not a float
            fields[key] = val
    return Material(**fields)

def load_all_materials(folder):
    materials = {}
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            mat = load_material_from_txt(os.path.join(folder, fname))
            materials[int(mat.code)] = mat
    return materials

# Usage:
MATERIALS = load_all_materials('materials')