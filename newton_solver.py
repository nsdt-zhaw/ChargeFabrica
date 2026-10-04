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

solver = LinearLUSolver(precon=None, iterations=1, tolerance=1e-12)

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
        if verbose and iteration % 25 == 0:
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
                 update_transport_response=None, equation_tolerances=None):

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
        physical_equations, dt, tolerance, limits, damping, max_steps, solver,
        update_transport_response, steady=True, min_dt=min_dt, max_dt=max_dt,
        verbose=verbose)
    history = np.full(max_steps, np.nan)
    history[:len(values)] = values
    return result, iterations+1, history
