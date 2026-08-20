import numpy as np
from constantsfile import q

def cell_variable(mesh, name, value=0.0, has_old=False):
    """Create a :class:`CellVariable` with the repository's common defaults."""
    from fipy import CellVariable
    return CellVariable(name=name, mesh=mesh, value=value, hasOld=has_old)

def as_cell_array(value, shape):
    """Return a FiPy or NumPy cell field in device-array ordering."""
    raw_value = getattr(value, "value", value)
    return np.asarray(raw_value).reshape(shape)

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

def bernoulli_function(value):
    """Evaluate the Bernoulli function stably near zero."""
    value = np.asarray(value, dtype=float)
    answer = np.empty_like(value)
    small = np.abs(value) < 1.0e-7
    answer[small] = (1.0 - value[small] / 2.0 + value[small] * value[small] / 12.0)
    with np.errstate(over="ignore", invalid="ignore"):
        answer[~small] = value[~small] / np.expm1(value[~small])
    return answer

def _harmonic_internal_faces(cell_value, axis):
    lower = [slice(None)] * cell_value.ndim
    upper = [slice(None)] * cell_value.ndim
    lower[axis] = slice(None, -1)
    upper[axis] = slice(1, None)
    lower = tuple(lower)
    upper = tuple(upper)
    denominator = cell_value[lower] + cell_value[upper]
    return np.divide(
        2.0 * cell_value[lower] * cell_value[upper],
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0.0,
    )

def conservative_internal_face_currents(n_value, p_value, phi_value, chi_value, bandgap_value, log_nc_value, log_nv_value, nmob_value, pmob_value, axis, spacing, thermal_voltage):
    """Reconstruct the exponentially fitted conventional carrier currents.

    The returned arrays live on internal faces and therefore have one fewer
    entry along ``axis`` than the cell-centred input fields.
    """
    fields = [np.asarray(value, dtype=float) for value in (n_value, p_value, phi_value, chi_value, bandgap_value, log_nc_value, log_nv_value, nmob_value, pmob_value)]
    (n_value, p_value, phi_value, chi_value, bandgap_value, log_nc_value, log_nv_value, nmob_value, pmob_value) = fields
    thermal_voltage = np.broadcast_to(np.asarray(thermal_voltage, dtype=float), n_value.shape)

    lower = [slice(None)] * n_value.ndim
    upper = [slice(None)] * n_value.ndim
    lower[axis] = slice(None, -1)
    upper[axis] = slice(1, None)
    lower = tuple(lower)
    upper = tuple(upper)

    voltage_face = _harmonic_internal_faces(thermal_voltage, axis)
    nmob_face = _harmonic_internal_faces(nmob_value, axis)
    pmob_face = _harmonic_internal_faces(pmob_value, axis)
    electron_energy = phi_value + chi_value + thermal_voltage * log_nc_value
    hole_energy = (phi_value + chi_value + bandgap_value - thermal_voltage * log_nv_value)
    electron_step = np.diff(electron_energy, axis=axis) / voltage_face
    hole_step = np.diff(hole_energy, axis=axis) / voltage_face

    jn_face = q * nmob_face * voltage_face / spacing * (n_value[upper] * bernoulli_function(electron_step) - n_value[lower] * bernoulli_function(-electron_step))
    jp_face = -q * pmob_face * voltage_face / spacing * (p_value[upper] * bernoulli_function(-hole_step) - p_value[lower] * bernoulli_function(hole_step))
    return jn_face, jp_face

def terminal_current_densities(jn_internal, jp_internal, axis=0):
    """Return bottom, top, and top-terminal total current densities."""
    total = jn_internal + jp_internal
    bottom = np.take(total, 0, axis=axis).copy()
    top = np.take(total, -1, axis=axis).copy()
    return bottom, top, top.copy()