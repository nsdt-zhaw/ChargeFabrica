# -*- coding: utf-8 -*-
"""FiPy Newton drivers: steady pseudo-time relaxation and physical time steps.
Device equations stay in the examples; FiPy assembles residuals and matrices.
"""
from __future__ import division, print_function

import time
import numpy as np
import fipy
from fipy import ImplicitSourceTerm, ExponentialConvectionTerm
from fipy.solvers.scipy import LinearLUSolver
from scipy.sparse.linalg import splu


class MeshOrderedLinearLUSolver(LinearLUSolver):
    """SciPy LU with geometric separators for serial Cartesian FiPy grids.

    Prefer a FiPy Grid1D/Grid2D/Grid3D mesh: infer dimensions and cell ordering
    from it. A shape tuple remains supported in NumPy array order (ny, nx) or
    (nz, ny, nx), with x varying fastest; its layout is the caller's responsibility.
    FiPy orders unknowns by field. Reorder cells by nested dissection and place
    their field unknowns together to reduce fill. Reversible row/column scaling
    permits unpivoted LU; check backward error and fall back to FiPy's pivoted
    LU if that factorization is singular or inaccurate.
    """
    def __init__(self, mesh_shape, **kwargs):
        super(MeshOrderedLinearLUSolver, self).__init__(**kwargs)
        mesh = mesh_shape if hasattr(mesh_shape, 'cellCenters') else None
        if mesh is not None:
            if mesh.communicator.Nproc != 1 or not hasattr(mesh, 'shape'):
                raise ValueError('Mesh ordering requires a serial Cartesian grid')
            mesh_shape = tuple(mesh.shape)[::-1]
        shape = tuple(mesh_shape)
        if (not 1 <= len(shape) <= 3 or
                any(not isinstance(n, (int, np.integer)) or n < 1 for n in shape)):
            raise ValueError('Require a positive integer Cartesian shape in 1D, 2D or 3D')
        grid = np.arange(int(np.prod(shape))).reshape(shape)
        if mesh is not None:
            if mesh.dim != len(shape) or mesh.numberOfCells != grid.size:
                raise ValueError('Mesh dimensions do not match its Cartesian shape')
            for axis, coordinates in enumerate(np.asarray(mesh.cellCenters)):
                axis = len(shape)-1-axis
                index = [0]*len(shape)
                index[axis] = slice(None)
                line_shape = [1]*len(shape)
                line_shape[axis] = shape[axis]
                coordinates = coordinates.reshape(shape)
                if not np.all(coordinates == coordinates[tuple(index)].reshape(line_shape)):
                    raise ValueError('Mesh cells must follow Cartesian, x-fastest ordering')
            face_cells = np.ma.asarray(mesh.faceCellIDs)
            internal = ~np.any(np.ma.getmaskarray(face_cells), axis=0)
            pairs = np.asarray(face_cells[:, internal], dtype=int)
            left = np.asarray(np.unravel_index(pairs[0], shape))
            right = np.asarray(np.unravel_index(pairs[1], shape))
            if np.any(np.sum(abs(left-right), axis=0) != 1):
                raise ValueError('Periodic or nonlocal mesh connections require graph ordering')

        def order(region):
            lengths = [s.stop-s.start for s in region]
            if max(lengths) <= 8:
                return grid[tuple(region)].ravel()
            axis = int(np.argmax(lengths))
            mid = (region[axis].start+region[axis].stop)//2
            pieces = []
            for start, stop in ((region[axis].start, mid),
                                (mid+1, region[axis].stop), (mid, mid+1)):
                part = list(region)
                part[axis] = slice(start, stop)
                pieces.append(order(part))
            return np.concatenate(pieces)
        self.cell_order = order([slice(0, n) for n in shape])
        self.fallback_count = 0
        self.last_backward_error = np.nan

    def _solve_(self, L, x, b):
        original_matrix = L
        # FiPy 3 passes its matrix wrapper; FiPy 4 passes SciPy CSR directly.
        if hasattr(L, 'matrix'):
            L = L.matrix
        cells = len(self.cell_order)
        if L.shape[0] % cells:
            raise ValueError('Linear system does not match the supplied Cartesian mesh')
        permutation = (self.cell_order[:, None] + np.arange(L.shape[0]//cells)*cells).ravel()
        matrix = L[permutation, :][:, permutation]
        col_scale = 1./np.maximum(abs(matrix).max(axis=0).toarray().ravel(), 1e-300)
        matrix = matrix.multiply(col_scale).tocsr()
        row_scale = 1./np.maximum(abs(matrix).max(axis=1).toarray().ravel(), 1e-300)
        matrix = matrix.multiply(row_scale[:, None]).tocsc()
        rhs = row_scale*b[permutation]
        try:
            lu = splu(matrix, permc_spec='NATURAL', diag_pivot_thresh=0., relax=1, panel_size=10)
            solution = lu.solve(rhs)
            for iteration in range(3):
                defect = rhs-matrix.dot(solution)
                denominator = abs(matrix).dot(abs(solution))+abs(rhs)
                error = float(np.max(abs(defect)/np.maximum(denominator, 1e-300)))
                if np.isfinite(error) and error <= self.tolerance:
                    break
                if iteration < 2:
                    solution += lu.solve(defect)
            else:
                raise RuntimeError('Mesh-ordered LU failed its backward-error check')
        except RuntimeError:
            self.fallback_count += 1
            return super(MeshOrderedLinearLUSolver, self)._solve_(original_matrix, x, b)
        x[permutation] = col_scale*solution
        self.last_backward_error = error
        # FiPy 4 records convergence explicitly; FiPy 3 has no such API.
        if hasattr(self, '_setConvergence'):
            self._setConvergence(suite='scipy', code=0, iterations=iteration+1,
                                 residual=float(np.linalg.norm(L.dot(x)-b)))
        return x


solver = fipy.solvers.LinearLUSolver(precon=None, iterations=1, tolerance=1e-12)
def _newton_limits(fields, equations, corrections, correction_equations, dt,
                   tolerance, damping, max_iterations, equation_tolerances):
    """Validate controls before changing any current or old field values."""
    if (not np.isfinite(dt) or dt <= 0 or not np.isfinite(tolerance) or tolerance <= 0
            or not np.isfinite(damping) or not 0 < damping <= 1
            or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1):
        raise ValueError('Require finite positive dt/tolerance, 0 < damping <= 1 and positive integer max_iterations')
    if not fields or not len(fields) == len(equations) == len(corrections) == len(correction_equations):
        raise ValueError('Fields, physical equations and corrections must have equal nonzero lengths')
    limits = np.asarray(equation_tolerances if equation_tolerances is not None else [tolerance]*len(fields))
    if limits.shape != (len(fields),) or np.any(~np.isfinite(limits)) or np.any(limits <= 0):
        raise ValueError('Require one positive finite tolerance per equation')
    return limits

def _solve_coupled(fields, equations, corrections, physical_equations, dt,
                   tolerance, limits, damping, max_iterations, linear_solver,
                   update_transport_response, steady=False, min_dt=None,
                   max_dt=None, verbose=False, line_search=True, potential_limit=.08):
    """Common FiPy block iteration; only steady mode advances the old state.

    Zero correction values make the cached pre-solve residual equal to the
    physical residual. Independently assemble physical equations at acceptance.
    """
    block = equations[0]
    for equation in equations[1:]:
        block = block & equation
    if len(equations) > 1:
        # Use FiPy's public ordering API. Algebraic rows have no transient or
        # diffusion term to guide its heuristic, which otherwise uses set order.
        block(corrections)
    history, previous_merit = [], np.inf
    for iteration in range(max_iterations):
        for correction in corrections:
            correction.setValue(0.)
        if update_transport_response is not None:
            update_transport_response()
        block.sweep(dt=dt, solver=linear_solver, cacheResidual=True)
        norms = np.linalg.norm(np.asarray(block.residualVector).reshape(len(fields), -1), axis=1)
        residual = float(sum(norms))
        history.append(residual)
        merit = float(np.max(norms/limits))
        if residual <= tolerance and merit <= 1:
            for correction in corrections:
                correction.setValue(0.)
            verified = np.asarray([np.linalg.norm(eq.justResidualVector(dt=1e100 if steady else dt))
                                   for eq in physical_equations])
            if sum(verified) <= tolerance and np.all(verified <= limits):
                history[-1] = float(sum(verified))
                return history[-1], iteration, np.asarray(history)
        if not np.isfinite(residual):
            raise RuntimeError('Nonfinite coupled residual')
        alpha = damping
        for index, (field, correction) in enumerate(zip(fields, corrections)):
            delta = np.asarray(correction.value)
            if not np.all(np.isfinite(delta)):
                raise RuntimeError('Nonfinite coupled correction')
            if index == 0 and not steady:
                alpha = min(alpha, potential_limit/max(float(np.max(abs(delta))), potential_limit))
            elif index > 2 and not steady:
                falling = delta < 0
                if np.any(falling):
                    alpha = min(alpha, .9*float(np.min(field.value[falling]/(-delta[falling]))))
        if alpha <= 0:
            raise RuntimeError('Coupled positivity constraint admits no step')
        values = [field.value.copy() for field in fields]
        search = not steady and line_search
        # Use the raw merit far from convergence. Once the sum criterion
        # passes, component limits drive unfinished continuity rows without
        # demanding further gains in the Poisson roundoff residual.
        use_components = residual <= tolerance
        start_merit = max(residual/tolerance, merit) if use_components else residual/tolerance
        for backtrack in range(20 if search else 1):
            for index, (field, value, correction) in enumerate(zip(fields, values, corrections)):
                trial_value = value + (correction.value if steady else alpha*correction.value)
                if index in (1, 2):
                    trial_value = np.maximum(trial_value, 1e-30)
                if steady:
                    trial_value = alpha*trial_value + (1-alpha)*field.old.value
                field.setValue(trial_value)
            if not search:
                break
            trial = np.asarray([np.linalg.norm(eq.justResidualVector(dt=dt)) for eq in physical_equations])
            trial_merit = max(float(sum(trial))/tolerance, float(np.max(trial/limits)))
            search_merit = trial_merit if use_components else float(sum(trial))/tolerance
            if np.all(np.isfinite(trial)) and (trial_merit <= 1. or search_merit <= (1-1e-4*alpha)*start_merit):
                break
            alpha *= .5
        else:
            raise RuntimeError('Coupled Newton line search failed: dt=%g residual=%g component ratios=%s' % (dt, residual, norms/limits))
        if steady:
            dt = max(min_dt, dt*.1) if merit > previous_merit*1.2 else min(max_dt, dt*1.05)
            for field in fields:
                field.updateOld()
        previous_merit = merit
        if verbose and iteration % 5 == 0:
            print('iteration=%d residual=%.3g dt=%.3g alpha=%.3g' % (iteration+1, residual, dt, alpha))
    for correction in corrections:
        correction.setValue(0.)
    raise RuntimeError('Coupled Newton did not converge after %d iterations: dt=%g residual=%g component ratios=%s' % (max_iterations, dt, history[-1], norms/limits))

class LiveExponentialConvectionTerm(ExponentialConvectionTerm):
    """Refresh contact interpolation weights as the potential changes."""
    def _buildMatrix(self, *args, **kwargs):
        self.__dict__.pop('constraintL', None)
        self.__dict__.pop('constraintB', None)
        return super(LiveExponentialConvectionTerm, self)._buildMatrix(*args, **kwargs)

class CoupledChargeTerm(ImplicitSourceTerm):
    """Keep both signs of an off-diagonal linear charge term implicit."""
    def _getWeight(self, var, transientGeomCoeff=None, diffusionGeomCoeff=None):
        zero = np.zeros(var.shape)
        return {'diagonal': np.ones(var.shape, dtype=bool),
                'old value': zero, 'b vector': zero, 'new value': zero}

def exponential_flux_response(density, gradient, sign, thermal_voltage):
    """Density multiplying the potential-gradient derivative of an SG flux."""
    mesh = density.mesh
    peclet = sign * np.asarray((gradient * mesh.faceNormals).sum(axis=0)) * mesh._cellDistances / thermal_voltage
    left, right = mesh._adjacentCellIDs
    nleft, nright = density.value[left], density.value[right].copy()
    constrained = np.asarray(density.arithmeticFaceValue.constraintMask, dtype=bool)
    nright[constrained] = density.arithmeticFaceValue.value[constrained]

    def bernoulli_derivative(x):
        z = np.maximum(abs(x), 1e-3)
        t = np.exp(-z)
        positive = t * (1-z-t) / (-np.expm1(-z))**2
        result = np.where(x >= 0, positive, -1-positive)
        return np.where(abs(x) < 1e-3, -.5+x/6-x**3/180, result)

    response = -nleft*bernoulli_derivative(-peclet) - nright*bernoulli_derivative(peclet)
    # FiPy clamps P to eps and requires abs(P) > eps for the exponential
    # branch. Thus its small-P interpolation is exactly alpha = 0.5.
    response = np.where(abs(peclet) < 1e-3, .5*(nleft+nright), response)
    return np.where(peclet > 101, nleft, response)

def solve_newton_coupled(fields, equations, corrections, dt=1e-7, max_dt=1e-5,
                 tolerance=1e-10, damping=.15, sweeps=1, max_steps=2000,
                 enable_ions=True, min_dt=1e-7, verbose=True, physical_equations=None,
                 update_transport_response=None, equation_tolerances=None, linear_solver=None):

    """Steady pseudo-time wrapper around the common coupled Newton engine.

    All supplied fields are solved together. For frozen ions, supply electronic
    fields and their corresponding equations only; silently ignoring rows would
    change the physical system. Physical equations are required for verification.
    """
    if physical_equations is None:
        raise ValueError('physical_equations are required for coupled convergence verification')
    if (not enable_ions and len(fields) > 3) or sweeps != 1:
        raise ValueError('Coupled mode solves every supplied row in one block sweep')
    limits = _newton_limits(fields, physical_equations, corrections, equations,
                            dt, tolerance, damping, max_steps, equation_tolerances)
    if not np.isfinite(min_dt) or not np.isfinite(max_dt) or not 0 < min_dt <= dt <= max_dt:
        raise ValueError('Require 0 < min_dt <= dt <= max_dt, all finite')
    result, iterations, values = _solve_coupled(fields, equations, corrections,
        physical_equations, dt, tolerance, limits, damping, max_steps,
        solver if linear_solver is None else linear_solver,
        update_transport_response, steady=True, min_dt=min_dt, max_dt=max_dt,
        verbose=verbose)
    history = np.full(max_steps, np.nan)
    history[:len(values)] = values
    return result, iterations+1, history
