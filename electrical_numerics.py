import numpy as np
from constantsfile import q

def cell_variable(mesh, name, value=0.0, has_old=False):
    """Create a :class:`CellVariable` with the repository's common defaults."""
    from fipy import CellVariable
    return CellVariable(name=name, mesh=mesh, value=value, hasOld=has_old)

def device_field(mesh, architecture, name, values=None, smoothing=0.0, logarithm=False):
    """Map a material property or supplied array onto a FiPy mesh.

    Smoothing is a Gaussian width in architecture pixels. Take logarithms
    before smoothing, as required for the density-of-states energy gradients.
    All device settings are supplied by the caller; no example globals are used.
    """
    from SmoothingFunction import flatten_and_smooth_all
    if values is None:
        from material_maps import map_semiconductor_property
        values = map_semiconductor_property(architecture, name)
    values = np.log(values) if logarithm else np.asarray(values)
    return cell_variable(mesh, name, flatten_and_smooth_all([values], smoothing)[0])

def srh_rate_and_carrier_derivatives(zone, n, p, ni_squared,
                                     tau_p, tau_n, n1, p1):
    """Return an SRH rate and its exact carrier derivatives.

    Trap populations and intrinsic density are constant during an isothermal
    Newton step.  The denominator derivatives are retained, which keeps the
    Newton matrix consistent with the nonlinear residual at high injection.
    """
    excess_product = n * p - ni_squared
    denominator = tau_p * (n + n1) + tau_n * (p + p1)
    rate = zone * excess_product / denominator
    denominator_squared = denominator * denominator
    derivative_n = zone * (p * denominator - tau_p * excess_product) / denominator_squared
    derivative_p = zone * (n * denominator - tau_n * excess_product) / denominator_squared
    return rate, derivative_n, derivative_p


def fitted_face_currents(n, p, electron_drift, hole_drift, nmob, pmob,
                         thermal_voltage):
    """Return live Scharfetter-Gummel conventional-current vectors (A/m2).

    Pass the same drift gradients, mobilities and thermal voltage as the
    continuity equations. Thermal voltage can be a scalar or a CellVariable;
    cell fields use harmonic face values. For split electrothermal coupling,
    pass the electrical temperature snapshot, not the live thermal field.

    Exact neighbouring densities retain tiny minority populations. Exterior
    faces use the density's constrained face value and FiPy's half-cell
    distance. Insulating faces with zero normal drift have zero flux. Components
    follow positive mesh coordinates, including at the lower contacts.
    """
    from fipy import FaceVariable
    mesh = n.mesh
    cell1_ids = np.asarray(mesh.faceCellIDs[0])
    cell2_ids = mesh.faceCellIDs[1].filled(cell1_ids)
    distance = mesh.scaledCellDistances
    voltage = getattr(thermal_voltage, 'harmonicFaceValue', thermal_voltage)

    def neighbours(density):
        cell1, cell2 = FaceVariable(mesh=mesh), FaceVariable(mesh=mesh)
        cell1.constrain(density[cell1_ids])
        cell2.constrain(density[cell2_ids])
        cell2.constrain(density.faceValue, where=mesh.exteriorFaces)
        return cell1, cell2

    n1, n2 = neighbours(n)
    p1, p2 = neighbours(p)
    electron_step = (electron_drift * mesh.faceNormals).sum(axis=0) * distance / voltage
    hole_step = (hole_drift * mesh.faceNormals).sum(axis=0) * distance / voltage
    jn = (mesh.faceNormals * q * nmob.harmonicFaceValue * voltage / distance
          * (n2 * bernoulli_function(electron_step) - n1 * bernoulli_function(-electron_step)))
    jp = (-mesh.faceNormals * q * pmob.harmonicFaceValue * voltage / distance
          * (p2 * bernoulli_function(-hole_step) - p1 * bernoulli_function(hole_step)))
    return jn, jp


def face_current_arrays(jn, jp):
    """Export Cartesian face grids as JnX/JpX, JnY/JpY, and JnZ/JpZ.

    Array axes are reversed mesh-coordinate order, just like FiPy cell arrays:
    (y, x) in 2D and (z, y, x) in 3D. The component's own axis has one extra
    entry, including both boundary faces. No terminal-current copies are saved.
    Only serial, nonperiodic Cartesian grids are supported by this exporter.
    """
    mesh = jn.mesh
    if mesh.communicator.Nproc != 1 or not hasattr(mesh, 'shape'):
        raise ValueError('Face-current export requires a serial Cartesian grid')
    result = {}
    for component, label in enumerate('XYZ'[:mesh.dim]):
        shape = list(mesh.shape[::-1])
        shape[mesh.dim - 1 - component] += 1
        faces = mesh.faceNormals[component] != 0
        if int(np.sum(faces)) != int(np.prod(shape)):
            raise ValueError('Face-current export requires a nonperiodic Cartesian grid')
        for carrier, current in (('n', jn), ('p', jp)):
            result['J' + carrier + label] = current[component, faces].value.reshape(shape)
    return result

def bernoulli_function(value):
    """Evaluate B(x) = x / (exp(x) - 1) for NumPy or live FiPy inputs.

    Negative exponential arguments avoid overflow for large finite x; expm1
    retains accuracy near zero. Numerix preserves FiPy dependencies, including
    temperature-dependent Peclet numbers in electrothermal simulations.
    """
    from fipy.tools import numerix
    z = numerix.maximum(numerix.absolute(value), 1e-20)
    return z * numerix.exp(-z) / (-numerix.expm1(-z)) - numerix.minimum(value, 0.)
