"""Convergence and stability study for the CGL spectral solver.

For a representative case from each family, solve on a sequence of CGL grids and
record (i) the error against the exact manufactured solution and (ii) the
relative residual of the discrete equations.  The residual is computable without
knowing the exact solution, so it is the certificate a real submission could
monitor; the error curve shows the spectral (exponential-then-roundoff) rate.

Usage
-----
    python experiments/convergence_study.py --out results/convergence.csv
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
import poisson_cases as pc  # noqa: E402

MS = (24, 32, 48, 64, 96, 128, 192, 256, 320, 384, 448, 512, 640)


def load_submission():
    spec = importlib.util.spec_from_file_location("submission", ROOT / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "convergence.csv"))
    ap.add_argument("--seed", type=int, default=20260101)
    args = ap.parse_args(argv)

    sub = load_submission()

    # one representative case per family, plus the Stage-1 reference
    cases = {"reference": pc.reference_case(129)}
    seen = set()
    for c in pc.hidden_suite(args.seed):
        fam = c["name"].split("_")[1]
        if fam not in seen:
            seen.add(fam)
            cases[fam] = c

    rows = []
    for label, case in cases.items():
        for m in MS:
            u, nodes, rel = sub._spectral_solution(m, case["f_fn"], case["g_fn"])
            pred = sub._interpolate(u, nodes, case["x"], case["y"])
            sub._set_boundary(pred, case["boundary_values"])
            err = pred - case["u_ref"]
            b = np.zeros(err.shape, bool)
            b[0, :] = b[-1, :] = b[:, 0] = b[:, -1] = True
            row = dict(
                case=label,
                family=label,
                n=int(len(case["x"])),
                m=int(m),
                mae=float(np.mean(np.abs(err))),
                max_abs_error=float(np.max(np.abs(err))),
                relative_l2=float(np.linalg.norm(err) / np.linalg.norm(case["u_ref"])),
                bc_mse=float(np.mean((pred[b] - case["boundary_values"][b]) ** 2)),
                residual=float(rel),
            )
            rows.append(row)
            print(f"{label:10s} m={m:4d} mae={row['mae']:.3e} max={row['max_abs_error']:.3e} "
                  f"resid={row['residual']:.2e}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("written:", out)

    # stability: eigenvector conditioning and eigenvalue spread versus m
    stab = []
    for m in MS:
        D2, _ = sub._cheb_cgl(m)
        T = 4.0 * D2[1:-1, 1:-1]
        lam, V = np.linalg.eig(T)
        stab.append(
            dict(m=int(m), cond_V=float(np.linalg.cond(V)),
                 lam_min=float(np.min(lam.real)), lam_max=float(np.max(lam.real)),
                 imag_max=float(np.max(np.abs(lam.imag))))
        )
        print(f"stability m={m:4d} cond(V)={stab[-1]['cond_V']:.3f} "
              f"|lam|max={abs(stab[-1]['lam_max']):.3e} imag={stab[-1]['imag_max']:.1e}")
    import json
    (out.with_suffix(".stability.json")).write_text(json.dumps(stab, indent=2))
    return rows


if __name__ == "__main__":
    main()
