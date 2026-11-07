"""Stress-test the submission on many hidden-like suites drawn from public seeds.

The official hidden suite is ``poisson_cases.hidden_suite(<private seed>)``.  We
cannot know that seed, so we draw our own seeds and measure the distribution of
metrics over many independent draws.  This is a generalisation check, not a
peek at the private suite.

Usage
-----
    python experiments/run_stress.py --seeds 20 --first-seed 900000 \
        --out results/stress.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))

import poisson_cases as pc  # noqa: E402


def _load_submission():
    spec = importlib.util.spec_from_file_location("submission", ROOT / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _metrics(pred, truth, bc):
    """Same definitions as the official evaluator."""
    err = np.asarray(pred, float) - truth
    b = np.zeros(err.shape, bool)
    b[0, :] = b[-1, :] = b[:, 0] = b[:, -1] = True
    return {
        "mae": float(np.mean(np.abs(err))),
        "max_abs_error": float(np.max(np.abs(err))),
        "relative_l2": float(np.linalg.norm(err) / np.linalg.norm(truth)),
        "bc_mse": float(np.mean((np.asarray(pred)[b] - bc[b]) ** 2)),
    }


TAU = {"mae": 2.39e-4, "max_abs_error": 2.85e-3, "relative_l2": 1.0e-3, "bc_mse": 1.0e-6}
W = {"mae": 0.25, "max_abs_error": 0.25, "relative_l2": 0.35, "bc_mse": 0.15}
FLOOR, WORST = -6.0, 0.25


def case_score(m):
    return sum(W[k] * max(np.log10(max(m[k], 1e-300) / TAU[k]), FLOOR) for k in W)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--first-seed", type=int, default=900000)
    ap.add_argument("--out", default=str(ROOT / "results" / "stress.json"))
    args = ap.parse_args(argv)

    sub = _load_submission()
    rows = []
    t_all = time.perf_counter()

    for s in range(args.first_seed, args.first_seed + args.seeds):
        for case in pc.hidden_suite(s):
            t0 = time.perf_counter()
            pred = sub.solve_problem(
                {k: case[k] for k in ("x", "y", "f", "boundary_values", "f_fn", "g_fn")}
                | {"run_seed": 0}
            )
            dt = time.perf_counter() - t0
            m = _metrics(pred, case["u_ref"], case["boundary_values"])
            m.update(
                seed=int(s),
                name=case["name"],
                family=case["name"].split("_")[1],
                n=int(len(case["x"])),
                runtime_s=float(dt),
                **{f"diag_{k}": v for k, v in sub.LAST_RUN.items() if k in
                   ("m", "delta", "residual", "solves", "fallback")},
            )
            m["case_score"] = case_score(m)
            rows.append(m)
        scores = [r["case_score"] for r in rows if r["seed"] == s]
        print(f"seed {s}: S={np.mean(scores) + WORST * max(scores):+.3f} "
              f"worst rel_l2={max(r['relative_l2'] for r in rows if r['seed'] == s):.2e}")

    elapsed = time.perf_counter() - t_all

    per_seed = defaultdict(list)
    for r in rows:
        per_seed[r["seed"]].append(r["case_score"])
    composites = [float(np.mean(v) + WORST * max(v)) for v in per_seed.values()]

    fam = defaultdict(list)
    for r in rows:
        fam[r["family"]].append(r)

    summary = {
        "seeds": args.seeds,
        "first_seed": args.first_seed,
        "cases": len(rows),
        "elapsed_s": elapsed,
        "grids": sorted({r["n"] for r in rows}),
        "worst_case": {k: max(r[k] for r in rows) for k in TAU},
        "best_case": {k: min(r[k] for r in rows) for k in TAU},
        "stage2_pass": bool(
            max(r["relative_l2"] for r in rows) < 5e-2
            and max(r["bc_mse"] for r in rows) < 1e-4
        ),
        "runtime_s": {
            "mean": statistics.fmean(r["runtime_s"] for r in rows),
            "median": statistics.median(r["runtime_s"] for r in rows),
            "max": max(r["runtime_s"] for r in rows),
        },
        "composite": {
            "mean": statistics.fmean(composites),
            "min": min(composites),
            "max": max(composites),
            "ideal_floor": -6.0 * (1.0 + WORST),
        },
        "chosen_m": sorted({r["diag_m"] for r in rows}),
        "fallback_used": any(r["diag_fallback"] for r in rows),
        "by_family": {
            k: {
                "n_cases": len(v),
                "mae_max": max(x["mae"] for x in v),
                "max_abs_error_max": max(x["max_abs_error"] for x in v),
                "relative_l2_max": max(x["relative_l2"] for x in v),
                "bc_mse_max": max(x["bc_mse"] for x in v),
                "mae_median": statistics.median(x["mae"] for x in v),
                "relative_l2_median": statistics.median(x["relative_l2"] for x in v),
                "case_score_mean": statistics.fmean(x["case_score"] for x in v),
            }
            for k, v in sorted(fam.items())
        },
        "rows": rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print("\n--- summary -------------------------------------------------")
    print(f"cases: {len(rows)} over {args.seeds} seeds, wall {elapsed:.1f} s")
    print(f"worst-case metrics: " + ", ".join(f"{k}={v:.3e}" for k, v in summary["worst_case"].items()))
    print(f"Stage 2 pass: {summary['stage2_pass']}")
    print(f"composite S: mean {summary['composite']['mean']:+.3f} "
          f"(min {summary['composite']['min']:+.3f}, max {summary['composite']['max']:+.3f}, "
          f"floor {summary['composite']['ideal_floor']:+.2f})")
    print(f"runtime: mean {summary['runtime_s']['mean']:.2f} s, max {summary['runtime_s']['max']:.2f} s")
    print(f"chosen m: {summary['chosen_m']}   fallback used: {summary['fallback_used']}")
    print(f"written: {out}")
    return summary


if __name__ == "__main__":
    main()
