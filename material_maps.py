import os
import hashlib

class Semiconductor(object):
    def __init__(self, name, GenRate, epsilon, pmob, nmob, Eg, chi, cationmob, anionmob, Recombination_Langevin,
                 Recombination_Bimolecular, Nc, Nv, Chi_a, Chi_c, a_initial_level, c_initial_level, Nd, Na):
        self.name = name
        self.code = self.generate_code(name)
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
        self.Nd = Nd
        self.Na = Na

    @staticmethod
    def generate_code(name):
        # Generate stable integer code from name string using MD5 hash
        md5_hash = hashlib.md5(name.encode('utf-8')).hexdigest()
        # Take first 8 characters and convert to int base 16 (32 bits)
        code = int(md5_hash[:8], 16)
        return code
def load_semiconductor_from_txt(path):
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
    return Semiconductor(**fields)
def load_all_semiconductors(folder):
    materials = {}
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            mat = load_semiconductor_from_txt(os.path.join(folder, fname))
            materials[int(mat.code)] = mat
    return materials

class Electrode(object):
    def __init__(self, name, WF):
        self.name = name
        self.code = self.generate_code(name)
        self.WF = WF

    @staticmethod
    def generate_code(name):
        # Generate stable integer code from name string using MD5 hash
        md5_hash = hashlib.md5(name.encode('utf-8')).hexdigest()
        # Take first 8 characters and convert to int base 16 (32 bits)
        code = int(md5_hash[:8], 16)
        return code

def load_electrode_from_txt(path):
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
    return Electrode(**fields)

def load_all_electrodes(folder):
    materials = {}
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            mat = load_electrode_from_txt(os.path.join(folder, fname))
            materials[int(mat.code)] = mat
    return materials

# Usage:
Semiconductors = load_all_semiconductors('Semiconductors')
Electrodes = load_all_electrodes('Electrodes')