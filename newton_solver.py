# -*- coding: utf-8 -*-
"""FiPy Newton driver for steady pseudo-time relaxation.
Device equations stay in the examples; FiPy assembles residuals and matrices.
"""
from __future__ import division, print_function

import numpy as np
import time
import fipy
from fipy import ImplicitSourceTerm, ExponentialConvectionTerm, ResidualTerm
from fipy.terms.binaryTerm import _BinaryTerm
from fipy.solvers.scipy import LinearLUSolver
from scipy.sparse.linalg import splu
from fipy.terms.explicitSourceTerm import _ExplicitSourceTerm

_iteration_clock = getattr(time, 'perf_counter', time.time)

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
                cell_axis = len(shape)-1-axis
                # Each row holds all cells at one coordinate on this axis.
                lines = coordinates.reshape(shape).swapaxes(0, cell_axis).reshape(shape[cell_axis], -1)
                if not np.all(lines == lines[:, :1]):
                    raise ValueError('Mesh cells must follow Cartesian, x-fastest ordering')
            face_cells = np.ma.asarray(mesh.faceCellIDs)
            internal = ~np.any(np.ma.getmaskarray(face_cells), axis=0)
            pairs = np.asarray(face_cells[:, internal], dtype=int)
            left = np.asarray(np.unravel_index(pairs[0], shape))
            right = np.asarray(np.unravel_index(pairs[1], shape))
            if np.any(np.sum(abs(left-right), axis=0) != 1):
                raise ValueError('Periodic or nonlocal mesh connections require graph ordering')

        def order(cells):
            if max(cells.shape) <= 8:
                return cells.ravel()
            axis = int(np.argmax(cells.shape))
            middle = cells.shape[axis]//2
            left, separator, right = np.split(cells, [middle, middle+1], axis=axis)
            return np.concatenate([order(part) for part in (left, right, separator)])
        self.cell_order = order(grid)
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


def fresh_equation(equation):
    """Copy assembly state, retaining live fields and coefficients (FiPy 3/4)."""
    if isinstance(equation, _BinaryTerm):
        return fresh_equation(equation.term) + fresh_equation(equation.other)
    return equation.copy()

class IndependentResidualTerm(ResidualTerm):
    """Avoid sharing FiPy 4's cached coupled matrix with a scalar residual."""
    def __init__(self, equation, underRelaxation=1.):
        super(IndependentResidualTerm, self).__init__(
            equation=fresh_equation(equation), underRelaxation=underRelaxation)

    def _buildMatrix(self, var, SparseMatrix, boundaryConditions=(), dt=None,
                     transientGeomCoeff=None, diffusionGeomCoeff=None):
        vec = _residual_vector(self.equation,
                    boundaryConditions=boundaryConditions, dt=dt)
        self.coeff = fipy.CellVariable(mesh=var.mesh, value=vec*self.underRelaxation)
        self.geomCoeff = None
        self.coeffVectors = None
        return _ExplicitSourceTerm._buildMatrix(self, var=var,
            SparseMatrix=SparseMatrix, boundaryConditions=boundaryConditions,
            dt=dt, transientGeomCoeff=transientGeomCoeff,
            diffusionGeomCoeff=diffusionGeomCoeff)


def _residual_vector(equation, **kwargs):
    """Always assemble independent residuals with SciPy, like the Newton solve."""
    return np.asarray(equation.justResidualVector(
        solver=LinearLUSolver(), **kwargs)).copy()


def _newton_limits(fields, physical_equations, equations, dt,
                   tolerance, damping, max_iterations, equation_tolerances):
    """Validate controls before changing any current or old field values."""
    if (not np.isfinite(dt) or dt <= 0 or not np.isfinite(tolerance) or tolerance <= 0
            or not np.isfinite(damping) or not 0 < damping <= 1
            or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1):
        raise ValueError('Require finite positive dt/tolerance, 0 < damping <= 1 and positive integer max_iterations')
    if not fields or not len(fields) == len(physical_equations) == len(equations):
        raise ValueError('Fields, physical equations and linearized equations must have equal nonzero lengths')
    limits = np.asarray(equation_tolerances if equation_tolerances is not None else [tolerance]*len(fields))
    if limits.shape != (len(fields),) or np.any(~np.isfinite(limits)) or np.any(limits <= 0):
        raise ValueError('Require one positive finite tolerance per equation')
    return limits

def _solve_coupled(fields, equations, physical_equations, dt,
                   tolerance, limits, damping, max_iterations, linear_solver,
                   update_transport_response, steady=False, min_dt=None,
                   max_dt=None, verbose=False, line_search=True, potential_limit=.08):
    """Solve candidate fields, apply their updates, and verify physical residuals."""
    if not isinstance(linear_solver, LinearLUSolver):
        raise ValueError('Newton requires a SciPy LinearLUSolver or MeshOrderedLinearLUSolver')
    block = equations[0]
    for equation in equations[1:]:
        block = block & equation
    if len(equations) > 1:
        # Use FiPy's public ordering API. Algebraic rows have no transient or
        # diffusion term to guide its heuristic, which otherwise uses set order.
        block(fields)
    history, previous_merit = [], np.inf
    for iteration in range(max_iterations):
        iteration_start = _iteration_clock()
        values = np.array([field.value for field in fields])
        if update_transport_response is not None:
            update_transport_response()
        block.sweep(dt=dt, solver=linear_solver, cacheResidual=True)
        # Convert candidate fields to updates, then restore the current iterate.
        deltas = np.asarray([field.value for field in fields]) - values
        for field, value in zip(fields, values):
            field.setValue(value)
        norms = np.linalg.norm(np.asarray(block.residualVector).reshape(len(fields), -1), axis=1)
        residual = float(sum(norms))
        history.append(residual)
        merit = float(np.max(norms/limits))
        if residual <= tolerance and merit <= 1:
            verified = np.asarray([np.linalg.norm(_residual_vector(eq, dt=1e100 if steady else dt))
                                   for eq in physical_equations])
            if sum(verified) <= tolerance and np.all(verified <= limits):
                history[-1] = float(sum(verified))
                return history[-1], iteration, np.asarray(history)
        if not np.isfinite(residual):
            raise RuntimeError('Nonfinite coupled residual')
        if not np.all(np.isfinite(deltas)):
            raise RuntimeError('Nonfinite coupled field update')
        alpha = damping
        if not steady:
            alpha = min(alpha, potential_limit/max(float(np.max(abs(deltas[0]))), potential_limit))
            falling = deltas[3:] < 0
            if np.any(falling):
                alpha = min(alpha, .9*float(np.min(values[3:][falling]/(-deltas[3:][falling]))))
        if alpha <= 0:
            raise RuntimeError('Coupled positivity constraint admits no step')
        search = not steady and line_search
        # Use the raw merit far from convergence. Once the sum criterion
        # passes, component limits drive unfinished continuity rows without
        # demanding further gains in the Poisson roundoff residual.
        use_components = residual <= tolerance
        start_merit = max(residual/tolerance, merit) if use_components else residual/tolerance
        for backtrack in range(20 if search else 1):
            trial_values = values + (deltas if steady else alpha*deltas)
            trial_values[1:3] = np.maximum(trial_values[1:3], 1e-30)
            if steady:
                trial_values = alpha*trial_values + (1-alpha)*np.asarray([field.old.value for field in fields])
            for field, value in zip(fields, trial_values):
                field.setValue(value)
            if not search:
                break
            trial = np.asarray([np.linalg.norm(_residual_vector(eq, dt=dt)) for eq in physical_equations])
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
            print('iteration=%d residual=%.3g dt=%.3g alpha=%.3g time_per_iteration=%.3fs' %
                  (iteration+1, residual, dt, alpha, _iteration_clock()-iteration_start))
    raise RuntimeError('Coupled Newton did not converge after %d iterations: dt=%g residual=%g component ratios=%s' % (max_iterations, dt, history[-1], norms/limits))

class LiveExponentialConvectionTerm(ExponentialConvectionTerm):
    """Refresh contact interpolation weights as the potential changes.

    FiPy 3.4.1 only builds constraintL/constraintB when absent, even though
    exponential interpolation depends on the evolving drift coefficient.
    This private-API workaround must be checked when upgrading FiPy.
    """
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

    # B'(-P) = -1-B'(P): evaluate the exponential only at |P|.
    z = np.maximum(abs(peclet), 1e-3)
    t = np.exp(-z)
    derivative = t * (1-z-t) / (-np.expm1(-z))**2
    upstream = np.where(peclet >= 0, nleft, nright)
    downstream = np.where(peclet >= 0, nright, nleft)
    response = upstream + (upstream-downstream)*derivative
    # FiPy clamps P to eps and requires abs(P) > eps for the exponential
    # branch. Thus its small-P interpolation is exactly alpha = 0.5.
    response = np.where(abs(peclet) < 1e-3, .5*(nleft+nright), response)
    return np.where(peclet > 101, nleft, response)

def solve_newton_coupled(fields, equations, dt=1e-7, max_dt=1e-5,
                 tolerance=1e-10, damping=.15, sweeps=1, max_steps=2000,
                 enable_ions=True, min_dt=1e-7, verbose=True, physical_equations=None,
                 update_transport_response=None, equation_tolerances=None, linear_solver=None):

    """Steady pseudo-time wrapper around the common coupled Newton engine.

    All supplied fields are solved together. For frozen ions, supply electronic
    fields and their corresponding equations only; silently ignoring rows would
    change the physical system. Physical equations are required for verification.
    The linearized equations act on the physical fields and include Jacobian-only
    additions (term + ResidualTerm(equation=term, underRelaxation=-1.)).
    """
    if physical_equations is None:
        raise ValueError('physical_equations are required for coupled convergence verification')
    if (not enable_ions and len(fields) > 3) or sweeps != 1:
        raise ValueError('Coupled mode solves every supplied row in one block sweep')
    limits = _newton_limits(fields, physical_equations, equations,
                            dt, tolerance, damping, max_steps, equation_tolerances)
    if not np.isfinite(min_dt) or not np.isfinite(max_dt) or not 0 < min_dt <= dt <= max_dt:
        raise ValueError('Require 0 < min_dt <= dt <= max_dt, all finite')
    result, iterations, values = _solve_coupled(fields, equations,
        physical_equations, dt, tolerance, limits, damping, max_steps,
        MeshOrderedLinearLUSolver(fields[0].mesh, tolerance=1e-12, iterations=1) if linear_solver is None else linear_solver,
        update_transport_response, steady=True, min_dt=min_dt, max_dt=max_dt,
        verbose=verbose)
    history = np.full(max_steps, np.nan)
    history[:len(values)] = values
    return result, iterations+1, history
