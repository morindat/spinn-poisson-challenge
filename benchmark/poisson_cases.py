"""Manufactured test cases for the optional SPINN / Poisson benchmark challenge.

Every case is a 2D Poisson problem on the unit square,

    Laplacian(u) = f  on (0,1)^2,      u = g on the boundary,

with a known exact solution u, so errors are measured exactly. Arrays follow
u[j, i] = u(x[i], y[j]) (rows are y, columns are x).

Each case gives the solver both
  * grid arrays: x, y, f (on the grid), boundary_values (g on boundary nodes), and
  * callables:   f_fn(X, Y) and g_fn(X, Y), vectorised over NumPy arrays,
so that a solver may evaluate the forcing anywhere (PINN collocation points, refined grids).
The answer must still be returned on the given grid.

Solution families (all smooth enough for the exact Laplacian to be written down):
  reference   sin(pi x) sin(pi y)                       (the GSoC proposal's benchmark; g = 0)
  sine        sum c_kl sin(k pi x) sin(l pi y)          (high-frequency variant: modes up to 12)
  expbubble   x(1-x)y(1-y) exp(a x + b y)
  gaussbump   x(1-x)y(1-y) exp(-|r-r0|^2 / 2s^2)        (narrow bumps)
  layer       x(1-x)y(1-y) tanh((n . r - c) / delta)    (sharp internal layer)
  harmonic+   exp(k x) sin(k y + phi) + interior bump   (NON-ZERO boundary values)
  mixed       sum of the above
The public suite is fully determined by PUBLIC_SEED. The hidden suite uses a seed known
only to the teaching team, drawn after the deadline and never stored in this repository.
"""
from __future__ import annotations

import numpy as np

PUBLIC_SEED = 20261012
PUBLIC_N = 129


# --------------------------------------------------------------------------- components
class Component:
    """One closed-form term: value(X, Y) and laplacian(X, Y)."""

    def value(self, X, Y):
        raise NotImplementedError

    def laplacian(self, X, Y):
        raise NotImplementedError


def _bubble(X, Y):
    B = X * (1 - X) * Y * (1 - Y)
    Bx = (1 - 2 * X) * Y * (1 - Y)
    By = X * (1 - X) * (1 - 2 * Y)
    lapB = -2 * Y * (1 - Y) - 2 * X * (1 - X)
    return B, Bx, By, lapB


class SineMode(Component):
    def __init__(self, k, l, c):
        self.k, self.l, self.c = k, l, c

    def value(self, X, Y):
        return self.c * np.sin(self.k * np.pi * X) * np.sin(self.l * np.pi * Y)

    def laplacian(self, X, Y):
        return -(np.pi ** 2) * (self.k ** 2 + self.l ** 2) * self.value(X, Y)


class BubbleTimes(Component):
    """x(1-x)y(1-y) * G(x, y) for a smooth G given by a subclass."""

    def G(self, X, Y):  # returns G, Gx, Gy, lapG
        raise NotImplementedError

    def value(self, X, Y):
        return _bubble(X, Y)[0] * self.G(X, Y)[0]

    def laplacian(self, X, Y):
        B, Bx, By, lapB = _bubble(X, Y)
        G, Gx, Gy, lapG = self.G(X, Y)
        return lapB * G + 2 * (Bx * Gx + By * Gy) + B * lapG


class ExpBubble(BubbleTimes):
    def __init__(self, a, b):
        self.a, self.b = a, b

    def G(self, X, Y):
        E = np.exp(self.a * X + self.b * Y)
        return E, self.a * E, self.b * E, (self.a ** 2 + self.b ** 2) * E


class GaussBump(BubbleTimes):
    def __init__(self, x0, y0, s):
        self.x0, self.y0, self.s = x0, y0, s

    def G(self, X, Y):
        s2 = self.s ** 2
        r2 = (X - self.x0) ** 2 + (Y - self.y0) ** 2
        G = np.exp(-r2 / (2 * s2))
        return G, -(X - self.x0) / s2 * G, -(Y - self.y0) / s2 * G, (r2 / s2 ** 2 - 2 / s2) * G


class Layer(BubbleTimes):
    """Sharp internal layer of width ~delta across the line through (x0, y0) with normal angle theta."""

    def __init__(self, theta, x0, y0, delta):
        self.nx, self.ny, self.d = np.cos(theta), np.sin(theta), delta
        self.c = self.nx * x0 + self.ny * y0

    def G(self, X, Y):
        T = np.tanh((self.nx * X + self.ny * Y - self.c) / self.d)
        dT = (1 - T ** 2) / self.d
        lapT = -2 * T * (1 - T ** 2) / self.d ** 2       # |n| = 1
        return T, self.nx * dT, self.ny * dT, lapT


class Harmonic(Component):
    """exp(k x) sin(k y + phi): Laplacian zero, NON-zero on the boundary."""

    def __init__(self, k, phi, c):
        self.k, self.phi, self.c = k, phi, c

    def value(self, X, Y):
        return self.c * np.exp(self.k * (X - 0.5)) * np.sin(self.k * Y + self.phi)

    def laplacian(self, X, Y):
        return np.zeros_like(np.asarray(X, dtype=float) + np.asarray(Y, dtype=float))


# --------------------------------------------------------------------------- cases
class ManufacturedCase:
    """u = (sum of components) / scale. Bound methods `forcing` and `boundary` are picklable."""

    def __init__(self, name, components, normalise=True):
        self.name = name
        self.components = components
        self.scale = 1.0
        if normalise:
            xs = np.linspace(0, 1, 513)
            X, Y = np.meshgrid(xs, xs, indexing="xy")
            self.scale = float(np.max(np.abs(self.exact(X, Y))))

    def exact(self, X, Y):
        X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
        return sum(c.value(X, Y) for c in self.components) / self.scale

    def forcing(self, X, Y):
        X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
        return sum(c.laplacian(X, Y) for c in self.components) / self.scale

    def boundary(self, X, Y):
        """Dirichlet data g; meaningful on the boundary of the unit square."""
        return self.exact(X, Y)

    def on_grid(self, n):
        x = np.linspace(0.0, 1.0, n)
        y = np.linspace(0.0, 1.0, n)
        X, Y = np.meshgrid(x, y, indexing="xy")
        u = self.exact(X, Y)
        bc = np.zeros_like(u)
        edge = np.zeros(u.shape, dtype=bool)
        edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
        bc[edge] = u[edge]
        return {"name": f"{self.name}_n{n}", "x": x, "y": y, "f": self.forcing(X, Y),
                "boundary_values": bc, "u_ref": u, "f_fn": self.forcing, "g_fn": self.boundary}


def reference_case(n: int = PUBLIC_N) -> dict:
    """The case reported in the GSoC 2026 SPINN proposal: u = sin(pi x) sin(pi y), g = 0."""
    case = ManufacturedCase("reference", [SineMode(1, 1, 1.0)], normalise=False).on_grid(n)
    case["name"] = "reference"
    case["u_ref"][0, :] = case["u_ref"][-1, :] = case["u_ref"][:, 0] = case["u_ref"][:, -1] = 0.0
    case["boundary_values"][:] = 0.0
    return case


# --------------------------------------------------------------------------- families
def _sine(rng, max_mode, decay):
    return [SineMode(k, l, rng.normal() / (k * k + l * l) ** decay)
            for k in range(1, max_mode + 1) for l in range(1, max_mode + 1)]


def family(rng, kind, hard=False):
    if kind == "sine":
        return _sine(rng, 12 if hard else 4, 0.55 if hard else 0.85)
    if kind == "expbubble":
        return [ExpBubble(*rng.uniform(-4, 4, 2))]
    if kind == "gaussbump":
        return [GaussBump(*rng.uniform(0.25, 0.75, 2), rng.uniform(0.04, 0.08) if hard else rng.uniform(0.1, 0.2))]
    if kind == "layer":
        return [Layer(rng.uniform(0, np.pi), *rng.uniform(0.35, 0.65, 2), rng.uniform(0.015, 0.03) if hard else rng.uniform(0.05, 0.1))]
    if kind == "harmonic":
        return [Harmonic(rng.uniform(1.5, 4.0), rng.uniform(0, 2 * np.pi), 1.0),
                GaussBump(*rng.uniform(0.3, 0.7, 2), rng.uniform(0.06, 0.12) if hard else 0.15)]
    if kind == "mixed":
        return (_sine(rng, 6, 0.8) + [Layer(rng.uniform(0, np.pi), *rng.uniform(0.35, 0.65, 2), 0.03 if hard else 0.08)]
                + [Harmonic(rng.uniform(1.0, 3.0), rng.uniform(0, 2 * np.pi), 0.5)])
    raise ValueError(kind)


KINDS = ("sine", "expbubble", "gaussbump", "layer", "harmonic", "mixed")


def public_suite() -> list[dict]:
    """Reference case (Stage 1) plus one practice case per family on 129 x 129 and 65 x 65 grids."""
    rng = np.random.default_rng(PUBLIC_SEED)
    cases = [reference_case(PUBLIC_N)]
    for i, kind in enumerate(KINDS):
        n = 129 if i % 2 == 0 else 65
        cases.append(ManufacturedCase(f"public_{kind}", family(rng, kind, hard=False)).on_grid(n))
    return cases


def hidden_suite(seed: int) -> list[dict]:
    """Hidden suite: 12 harder cases (two per family) on 65, 129 and 257 grids.

    The seed must be drawn privately by the teaching team after submissions close and
    must not be written into this repository.
    """
    rng = np.random.default_rng(seed)
    sizes = [65, 129, 257]
    cases = []
    for j, kind in enumerate(KINDS):
        for r in range(2):
            n = sizes[(j + r) % 3]
            cases.append(ManufacturedCase(f"hidden_{kind}_{r}", family(rng, kind, hard=True)).on_grid(n))
    return cases
