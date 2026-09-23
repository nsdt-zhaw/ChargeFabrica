import time
import numpy as np
import fipy

def solve_newton(fields, equations, corrections, dt=1e-7, max_dt=1e-5, tolerance=1e-10, damping=0.1, sweeps=1, max_steps=2000, enable_ions=True, min_dt=1e-7, verbose=True):
    solver = fipy.solvers.LinearLUSolver(precon=None, iterations=1, tolerance=1e-12)

    phi, n, p, a, c = fields
    deqpoisson, deqn, deqp, deqa, deqc = equations
    dphi, dn, dp, da, dc = corrections
    residual, residual_old, dt_old, total_time, steps = 1., 1e10, dt, 0.0, 0
    residual_history = np.zeros(max_steps)

    while steps < max_steps and residual > tolerance:

        t0 = time.time()

        for i in range(sweeps):
            # Each outer sweep linearizes around the newly accepted state.
            # Newton increments from the previous linearization must not enter
            # the new off-diagonal Jacobian terms as explicit source values.
            for correction in (dphi, dn, dp, da, dc):
                correction.setValue(0.0)

            deqpoisson.sweep(dt=dt, solver=solver)
            phi.setValue(damping * (phi + dphi) + (1 - damping) * phi.old)  # The potential should be damped BEFORE passing to the continuity equations!

            residual = deqn.sweep(dt=dt, solver=solver) + deqp.sweep(dt=dt, solver=solver)
            n.setValue(damping * np.maximum(n + dn, 1.00e-30) + (1 - damping) * n.old)
            p.setValue(damping * np.maximum(p + dp, 1.00e-30) + (1 - damping) * p.old)

        if enable_ions:
            residual += deqa.sweep(dt=dt, solver=solver) + deqc.sweep(dt=dt, solver=solver)
            a.setValue(damping * (a + da) + (1 - damping) * a.old)
            c.setValue(damping * (c + dc) + (1 - damping) * c.old)

        residual_history[steps] = residual

        improvement = (1 - (residual / residual_old) * dt_old / dt) * 100

        if residual > residual_old * 1.2:
            dt = max(min_dt, dt * 0.1)
        else:
            dt = min(max_dt, dt * 1.05)

        dt_old, residual_old = dt, residual

        # Update old
        for v in (n, p, a, c, phi): v.updateOld()

        total_time += dt

        if verbose and (steps == 0 or steps % 25 == 0 or residual <= tolerance):
            print("Sweep: ", steps, "TotalTime: ", total_time, "Residual: ", residual, "Time for sweep: ", time.time() - t0, "dt: ", dt, "Percentage Improvement: ", improvement, "Damping: ", damping)
        steps += 1

    return residual, steps, residual_history
