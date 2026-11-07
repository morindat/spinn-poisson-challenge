# Beating the SPINN Poisson benchmark with a Chebyshev spectral solver

Optional advanced challenge for **CS-3410 / MAT-3211-1, Introduction to Machine
Learning, Monsoon 2026** (Ashoka University).

We solve

$$
\nabla^2 u(x,y) = f(x,y), \quad (x,y)\in(0,1)^2,
\qquad u = g \ \text{on}\ \partial(0,1)^2,
$$

for **arbitrary** forcing `f` and Dirichlet data `g`, on any requested `N x N`
grid, using a Chebyshev--Gauss--Lobatto spectral collocation solver with a
tensor-product direct solve. There is no training and no stored answer: the
solver computes `u` from `f_fn`, `g_fn` and the boundary data on every call.

## Headline result (Stage 1, public reference case, 129 x 129, five runs)

| metric | reported in the proposal | Stage 1 target | **this submission** | margin |
|---|---|---|---|---|
| MAE | 2.39e-4 ± 1.2e-5 | < 2.39e-4 | **2.93e-13** | 8.1e8x |
| max abs error | 2.85e-3 | < 2.85e-3 | **7.95e-13** | 3.6e9x |
| relative L2 | ~1e-3 | < 1e-3 | **6.88e-13** | 1.5e9x |
| boundary MSE | ~1e-6 | < 1e-6 | **0.0** | exact |

All four Stage-1 gates pass by more than eight orders of magnitude. On 240
independently drawn hidden-like cases (20 seeds x 12 cases, the public
`hidden_suite` generator with seeds we chose ourselves) the worst case over all
six families is `relative L2 = 1.21e-11` and `BC MSE = 0`, and every case
saturates the Stage-3 score floor (composite `S = -7.50`, the lowest value the
scoring rule permits).

## Method

`submission.py` implements the following pipeline.

1. **Discretise on a Chebyshev--Gauss--Lobatto grid.** The 2-D operator is
   separable: `(A_y U + U A_x^T)[j,i] = f(x_i, y_j)` with `A_x`, `A_y` the 1-D
   CGL second-derivative matrices scaled to `[0,1]`.
2. **Solve directly by matrix diagonalisation.** With `A = V diag(λ) V^-1`, two
   triangular solves and an elementwise division `W_jk = G_jk / (λ_j + μ_k)`
   give the interior solution in `O(m^3)`; no iteration, no time stepping, no
   mesh generation.
3. **Impose Dirichlet data exactly** through the boundary columns of the 1-D
   operators, then overwrite the returned boundary rows/columns with the
   supplied `boundary_values`. `MSE_BC` is therefore exactly zero.
4. **Interpolate** the CGL solution to the requested grid with the barycentric
   formula (stable, spectrally accurate, no assumption that the grid is the
   same as the CGL grid or that `N = 129`).
5. **Choose the resolution adaptively:** solve on `m = 128, 192, ..., 640` and
   stop once two successive solutions agree on the *requested* grid to
   `3e-12` relative; if the differences start to grow, roundoff has taken over
   and the previous solution is kept. Smooth cases stop at `m = 192`; sharp
   layers reach `m = 384--512`.
6. **Fall back safely** to a second-order five-point finite-difference solve on
   the supplied grid if the spectral path ever raises or returns a non-finite
   value. The fallback alone already clears the Stage-1 gate.

## Reproduce

```bash
# Python 3.14; only numpy is required at solve time
pip install -r requirements.txt

# Stage 1 + the six public practice cases, five runs each (official evaluator)
python benchmark/evaluate.py --submission submission.py --repeats 5 \
    --json results/public_5run.json

# hidden-like generalisation sweep: 20 seeds x 12 cases (about 6 minutes)
python experiments/run_stress.py --seeds 20 --first-seed 20260101 \
    --out results/stress.json

# convergence + stability study (error vs m, residual, cond(V))
python experiments/convergence_study.py

# robustness checks: fallback, descending grids, N != 129
python experiments/check_robustness.py

# figures for the results note
python experiments/make_figures.py
```

The official evaluator is the one shipped in the student kit; `benchmark/`
holds an unmodified copy of `evaluate.py`, `poisson_cases.py` and
`starter_submission.py`.

## Hardware and runtime

Measured on a 4-core x86-64 laptop CPU (AMD/Intel, 11 GB RAM), Python 3.14.8,
numpy 2.5.4, BLAS = reference OpenBLAS 3.12, no GPU.

| workload | wall-clock per case |
|---|---|
| Stage 1 reference case, 129 x 129 | 0.13 s |
| public practice cases, 65--129 | 0.13 s |
| 12-case hidden-like suite (fresh process per case) | 1.5 s mean, 0.15 s median, 6.2 s max |
| 240-case sweep | 0.18 s median |

The official budget is 20 minutes per case on CPU, so the solver uses well
under 1% of the allowance. Runtime is dominated by the eigen-decomposition of
the 1-D operator, which is `O(m^3)` and independent of `N`.

## Repository layout

```
submission.py              the deliverable solver (self-contained, numpy only)
requirements.txt           pinned runtime dependency
README.md                  this file
results_note.pdf           the 2--3 page results note
benchmark/                 unmodified public evaluator + case generator
experiments/               scripts that produced every number in the note
results/                   JSON/CSV outputs, figures, logs
report/                    LaTeX source of the results note
```

## Trained weights

**None.** The method is a deterministic numerical solver; it trains nothing, so
there are no weights to ship. The submission is fully described by
`submission.py` and its docstring.

## Integrity

* The solver never reads a file, never uses the network, and never branches on a
  case's name, family or index.
* Nothing is stored between calls except diagnostics (`LAST_RUN`), which do not
  influence the answer.
* `N` is never assumed to be 129 (tested for `N = 33 ... 513`, including
  descending grids).
* The solver is deterministic, so `run_seed` is not needed; repeated runs give
  bit-identical output and zero-variance five-run statistics.
* The hidden-like suites in `experiments/run_stress.py` are generated by the
  public `poisson_cases.hidden_suite` with seeds we chose; the private seed was
  never accessed.

## Citations

1. S. Malik. *Physics-Informed Neural Network Shape Optimization.* GSoC 2026
   proposal, ML4SCI, 2026 (the benchmark and its reported values).
2. M. Raissi, P. Perdikaris, G. E. Karniadakis. Physics-informed neural
   networks. *J. Comput. Phys.* 378 (2019) 686--707.
3. Z. Zhang, C. Lin, B. Wang. Physics-informed shape optimization using
   coordinate projection. *Sci. Rep.* 14 (2024) 6537.
4. L. N. Trefethen. *Spectral Methods in MATLAB.* SIAM, 2000.
5. D. B. Haidvogel, T. A. Zang. The accurate solution of Poisson's equation by
   expansion in Chebyshev polynomials. *J. Comput. Phys.* 30 (1979) 167--180.
6. B. Fornberg. *A Practical Guide to Pseudospectral Methods.* CUP, 1996.
7. R. J. LeVeque. *Finite Difference Methods for Ordinary and Partial
   Differential Equations.* SIAM, 2007.
8. N. Kang et al. PIXEL: Physics-informed cell representations for fast and
   accurate PDE solvers. *AAAI* 37 (2023) 8186--8194.
9. N. Kang et al. PIG: Physics-informed Gaussians as adaptive parametric mesh
   representations. *ICLR* 2025.
