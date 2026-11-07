import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
import poisson_cases as pc  # noqa: E402

spec = importlib.util.spec_from_file_location("sub", ROOT / "submission.py")
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)

case = [c for c in pc.hidden_suite(20260101) if "layer_0" in c["name"]][0]
for m in (128, 192, 256, 384, 512, 640):
    t0 = time.perf_counter()
    u, nodes, rel = sub._spectral_solution(m, case["f_fn"], case["g_fn"])
    dt = time.perf_counter() - t0
    print(f"m={m:4d} solve {dt:.3f}s")

t0 = time.perf_counter()
out = sub.solve_problem({**{k: case[k] for k in ("x", "y", "f", "boundary_values", "f_fn", "g_fn")}, "run_seed": 0})
print(f"full solve {time.perf_counter()-t0:.2f}s diag={sub.LAST_RUN}")
