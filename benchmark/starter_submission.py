"""Starter file for the optional SPINN / Poisson benchmark challenge.

Copy this file to submission.py and implement solve_problem(case).

Any method is allowed: a PINN or other neural network, a coordinate-based or
grid/Gaussian-based representation (e.g. PIXEL- or PIG-inspired), a classical
numerical solver (finite differences, finite elements, spectral methods), or a
hybrid. What matters is that the method SOLVES the problem from the inputs it
is given, and still works on unseen forcing functions, boundary values and grid sizes.

Check your submission with:
    python evaluate.py --submission submission.py              # deterministic method
    python evaluate.py --submission submission.py --repeats 5  # stochastic method (e.g. a PINN)
"""
from __future__ import annotations

import numpy as np


def solve_problem(case: dict) -> np.ndarray:
    """Return an approximation u_hat on the supplied grid.

    case["x"]               : (N,) grid coordinates in [0, 1]
    case["y"]               : (N,) grid coordinates in [0, 1]
    case["f"]               : (N, N) forcing, with f[j, i] = Laplacian(u)(x[i], y[j])
    case["boundary_values"] : (N, N) prescribed Dirichlet values g (only the boundary entries matter;
                              g is NOT always zero in the hidden suite)
    case["f_fn"]            : callable f_fn(X, Y) -> forcing at any points of the square (NumPy arrays)
    case["g_fn"]            : callable g_fn(X, Y) -> Dirichlet data, meaningful on the boundary
    case["run_seed"]        : int; use it to seed every random number generator you use

    f_fn lets a PINN sample collocation points anywhere, and lets a numerical solver work on a
    finer grid than the one you must answer on.

    Return a finite float array of shape (N, N) with u_hat[j, i] ~ u(x[i], y[j]).

    Rules: do not read files that hold solutions, do not branch on the case name or
    on the specific forcing, do not use the network, and do not assume N = 129.
    """
    x = np.asarray(case["x"], dtype=float)
    f = np.asarray(case["f"], dtype=float)
    n = len(x)
    if f.shape != (n, n):
        raise ValueError("expected a square N x N grid")
    raise NotImplementedError("Implement your solver here.")
