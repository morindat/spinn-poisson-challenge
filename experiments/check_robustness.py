"""Sanity checks used during development (not part of the submission)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
import poisson_cases as pc  # noqa: E402
from evaluate import metrics  # noqa: E402

spec = importlib.util.spec_from_file_location("sub", ROOT / "submission.py")
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)

print("== fallback (2nd-order FD) accuracy ==")
for c in pc.public_suite():
    fb = sub._finite_difference_solution(c["x"], c["y"], c["f"], c["boundary_values"])
    m = metrics(fb, c["u_ref"], c["boundary_values"])
    print(f"{c['name']:26s} mae={m['mae']:.2e} rel={m['relative_l2']:.2e} bc={m['bc_mse']:.1e}")

print("\n== transposed / descending grids ==")
c = pc.reference_case(129)
for label, case in [
    ("ascending", c),
    ("descending x,y", {**c, "x": c["x"][::-1], "y": c["y"][::-1],
                        "f": c["f"][::-1, ::-1], "boundary_values": c["boundary_values"][::-1, ::-1],
                        "u_ref": c["u_ref"][::-1, ::-1]}),
]:
    out = sub.solve_problem(case)
    m = metrics(out, case["u_ref"], case["boundary_values"])
    print(f"{label:16s} shape={out.shape} mae={m['mae']:.2e} rel={m['relative_l2']:.2e} "
          f"bc={m['bc_mse']:.1e} diag={sub.LAST_RUN}")

print("\n== N is not assumed to be 129 ==")
for n in (33, 65, 129, 257, 513):
    case = pc.reference_case(n)
    out = sub.solve_problem(case)
    m = metrics(out, case["u_ref"], case["boundary_values"])
    print(f"N={n:4d} mae={m['mae']:.2e} rel={m['relative_l2']:.2e} m={sub.LAST_RUN['m']} "
          f"t={sub.LAST_RUN['time_s']:.2f}s")
