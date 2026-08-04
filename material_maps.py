import hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

class Material:
    """Material properties loaded from one text file. Hashing is used to generate a suitable integer code for each material based on its name."""
    def __init__(self, **properties):
        self.__dict__.update(properties)
        self.code = int(hashlib.md5(self.name.encode("utf-8")).hexdigest()[:8], 16)

def _read_fields(path):
    fields = {}
    with path.open(encoding="utf-8") as source:
        for raw_line in source:
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            try:
                value = float(value)
            except ValueError:
                pass
            fields[key] = value
    return fields

def _load_materials(folder):
    paths = (ROOT / folder).glob("*.txt")
    materials = [Material(**_read_fields(path)) for path in paths]
    return {material.code: material for material in materials}

def _map_property(values, prop, table):
    return np.vectorize(lambda material_id: getattr(table[material_id], prop))(values)

def map_semiconductor_property(values, prop):
    return _map_property(values, prop, Semiconductors)

def map_electrode_property(values, prop):
    return _map_property(values, prop, Electrodes)

def map_props(values, props, table):
    return [_map_property(values, prop, table) for prop in props]

Semiconductors = _load_materials("Semiconductors")
Electrodes = _load_materials("Electrodes")
name_to_code_SC = {material.name: code for code, material in Semiconductors.items()}
name_to_code_EL = {material.name: code for code, material in Electrodes.items()}