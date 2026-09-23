import time
import numpy as np
import fipy

def solve_gummel(fields, equations, dt=1e-9, max_dt=1e-6, tolerance=1e-10, damping=0.01, sweeps=1, max_steps=2000, enable_ions=True, min_dt=1e-9, verbose=True, adaptive_damping=True, min_damping=0.001, min_steps=0, residual_growth_limit=1.2, timestep_growth=1.05, extra_ions=(), fixed_fields=()):
    solver = fipy.solvers.LinearLUSolver(precon=None, iterations=1, tolerance=1e-12)

    phi, n, p, a, c = fields
    eqpoisson, eqn, eqp, eqa, eqc = equations
    residual, residual_old, dt_old, total_time, steps = 1., 1e10, dt, 0.0, 0
    residual_history = np.zeros(max_steps)

    while steps < max_steps and (residual > tolerance or steps < min_steps):

        t0 = time.time()

        for i in range(sweeps):
            eqpoisson.sweep(dt = dt, solver=solver)
            phi.setValue(damping * phi + (1 - damping) * phi.old) # The potential should be damped BEFORE passing to the continuity equations!

            residual = eqn.sweep(dt = dt, solver=solver) + eqp.sweep(dt = dt, solver=solver)
            n.setValue(damping * np.maximum(n, 1.00e-30) + (1 - damping) * n.old)
            p.setValue(damping * np.maximum(p, 1.00e-30) + (1 - damping) * p.old)

        if enable_ions:
            #Here the ionic continuity equations are solved
            ion_residual = eqa.sweep(dt=dt, solver=solver) + eqc.sweep(dt=dt, solver=solver)
            for field, equation in extra_ions:
                ion_residual += equation.sweep(dt=dt, solver=solver)
            residual += ion_residual
            a.setValue(damping * a + (1 - damping) * a.old)
            c.setValue(damping * c + (1 - damping) * c.old)
            for field, equation in extra_ions:
                field.setValue(damping * field + (1 - damping) * field.old)

        residual_history[steps] = residual

        improvement = (1 - (residual / residual_old) * dt_old / dt) * 100

        if residual > residual_old * residual_growth_limit:
            dt = max(min_dt, dt * 0.1)
            if adaptive_damping:
                damping = max(min_damping, damping * 0.1)
        else:
            dt = min(max_dt, dt * timestep_growth)
            if adaptive_damping:
                damping = min(0.2, damping * 1.01)

        dt_old, residual_old = dt, residual

        #Update old
        for v in (n, p, a, c, phi): v.updateOld()
        for v in fixed_fields: v.updateOld()
        for field, equation in extra_ions: field.updateOld()

        total_time += dt

        if verbose and (steps == 0 or steps % 25 == 0 or residual <= tolerance):
            print("Sweep: ", steps, "TotalTime: ", total_time, "Residual: ", residual, "Time for sweep: ", time.time() - t0, "dt: ", dt, "Percentage Improvement: ", improvement, "Damping: ", damping)
        steps += 1

    return residual, steps, residual_history
