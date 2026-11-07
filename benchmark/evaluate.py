"""Reproducible evaluator for the optional SPINN / Poisson benchmark challenge.

Usage (students, public suite only):
    python evaluate.py --submission path/to/submission.py
    python evaluate.py --submission path/to/submission.py --repeats 5   # stochastic methods

Usage (teaching team, hidden suite; the seed is private and never committed):
    IML_SPINN_HIDDEN_SEED=<private integer> python evaluate.py --submission ... --suite hidden --json out.json

The submission must define

    solve_problem(case: dict) -> numpy.ndarray of shape (N, N)

where case has keys x (N,), y (N,), f (N, N), boundary_values (N, N), run_seed (int), and
the callables f_fn(X, Y) and g_fn(X, Y) (forcing and Dirichlet data, evaluable anywhere in
the square). The returned array follows u[j, i] = u(x[i], y[j]). The evaluator never
passes u_ref to the submission.

Stage 1 thresholds are the values reported in the GSoC 2026 SPINN proposal (p. 6).
Stage 2 thresholds, the composite score, its weights and the per-term floor are
ORGANISER RULES (proposed), not results from the proposal.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import os
import queue as queue_mod
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from poisson_cases import hidden_suite, public_suite  # noqa: E402

# --- Stage 1: values reported in the proposal (strict "less than" required) -------------
STAGE1 = {"mae": 2.39e-4, "max_abs_error": 2.85e-3, "relative_l2": 1.0e-3, "bc_mse": 1.0e-6}
# --- Stage 2: organiser-proposed eligibility (loose sanity limits, worst hidden case) -----
STAGE2 = {"relative_l2": 5.0e-2, "bc_mse": 1.0e-4}
# --- Stage 3: organiser-proposed ranking score (lower is better) --------------------------
# Per hidden case i: c_i = sum_k w_k * max(log10(e_ik / tau_k), TERM_FLOOR), tau = Stage 1 values.
# S = mean_i c_i + WORST_WEIGHT * max_i c_i  -> rewards a solver that is good on every case.
WEIGHTS = {"mae": 0.25, "max_abs_error": 0.25, "relative_l2": 0.35, "bc_mse": 0.15}
TERM_FLOOR = -6.0          # gains beyond 10^6 x below a threshold are not rewarded
WORST_WEIGHT = 0.25
TIE_TOLERANCE = 0.01       # scores closer than this are tied -> mean runtime decides
METRICS = ("mae", "max_abs_error", "relative_l2", "bc_mse")


def metrics(pred, truth, bc) -> dict:
    pred = np.asarray(pred, dtype=float)
    if pred.shape != truth.shape:
        raise ValueError(f"prediction shape {pred.shape} != expected {truth.shape}")
    if not np.isfinite(pred).all():
        raise ValueError("prediction contains NaN or inf")
    err = pred - truth
    boundary = np.zeros(pred.shape, dtype=bool)
    boundary[0, :] = boundary[-1, :] = boundary[:, 0] = boundary[:, -1] = True
    return {
        "mae": float(np.mean(np.abs(err))),
        "max_abs_error": float(np.max(np.abs(err))),
        "relative_l2": float(np.linalg.norm(err) / np.linalg.norm(truth)),
        "bc_mse": float(np.mean((pred[boundary] - bc[boundary]) ** 2)),
    }


def _worker(sub_path, case, queue):
    try:
        spec = importlib.util.spec_from_file_location("student_submission", sub_path)
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(Path(sub_path).resolve().parent))
        spec.loader.exec_module(module)
        t0 = time.perf_counter()
        pred = module.solve_problem(case)
        queue.put(("ok", np.asarray(pred, dtype=float), time.perf_counter() - t0))
    except Exception as exc:  # report, never crash the evaluator
        queue.put(("error", f"{type(exc).__name__}: {exc}", None))


def run_case(sub_path, case, run_seed, timeout):
    public = {k: case[k].copy() for k in ("x", "y", "f", "boundary_values")}
    public["f_fn"], public["g_fn"] = case["f_fn"], case["g_fn"]
    public["run_seed"] = int(run_seed)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(sub_path, public, q))
    p.start()
    # Read before join: a child holding a large array in the queue cannot exit
    # until the parent has consumed it.
    try:
        status, payload, runtime = q.get(timeout=timeout)
    except queue_mod.Empty:
        status, payload, runtime = ("timeout", None, None)
    p.join(5)
    if p.is_alive():
        p.terminate(); p.join()
    if status == "timeout":
        return {"status": "timeout", "runtime_s": timeout}
    if status != "ok":
        return {"status": "error", "message": payload}
    try:
        m = metrics(payload, case["u_ref"], case["boundary_values"])
    except ValueError as exc:
        return {"status": "invalid", "message": str(exc)}
    return {"status": "ok", "runtime_s": runtime, **m}


def case_score(row: dict) -> float:
    return sum(w * max(math.log10(max(row[k], 1e-300) / STAGE1[k]), TERM_FLOOR) for k, w in WEIGHTS.items())


def composite(rows: list[dict]) -> float:
    scores = [case_score(r) for r in rows]
    return float(np.mean(scores) + WORST_WEIGHT * np.max(scores))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", default="submission.py")
    ap.add_argument("--suite", choices=["public", "hidden"], default="public")
    ap.add_argument("--hidden-seed", type=int, default=None, help="or set IML_SPINN_HIDDEN_SEED")
    ap.add_argument("--repeats", type=int, default=1, help="independent runs (use 5 for stochastic methods)")
    ap.add_argument("--timeout", type=float, default=1200.0, help="seconds per case (official budget: 20 min)")
    ap.add_argument("--json", default=None, help="write full results to this file")
    a = ap.parse_args(argv)
    sub = str(Path(a.submission).resolve())
    if not Path(sub).exists():
        sys.exit(f"submission not found: {sub}")

    if a.suite == "public":
        cases = public_suite()
    else:
        seed = a.hidden_seed if a.hidden_seed is not None else os.environ.get("IML_SPINN_HIDDEN_SEED")
        if seed is None:
            sys.exit("hidden suite needs --hidden-seed or IML_SPINN_HIDDEN_SEED (teaching team only)")
        cases = hidden_suite(int(seed))

    results = []
    for case in cases:
        runs = [run_case(sub, case, r, a.timeout) for r in range(a.repeats)]
        row = {"case": case["name"], "grid": int(len(case["x"])), "runs": runs}
        ok = [r for r in runs if r["status"] == "ok"]
        row["status"] = "ok" if len(ok) == len(runs) else runs[0]["status"] if not ok else "partial"
        if row["status"] == "ok":
            for k in METRICS + ("runtime_s",):
                vals = np.array([r[k] for r in ok])
                row[k] = float(vals.mean())
                row[k + "_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        results.append(row)

    print(f"Submission: {sub}\nSuite: {a.suite} ({len(cases)} cases, {a.repeats} run(s) each)\n")
    print(f"{'case':28s} {'N':>4s} {'MAE':>11s} {'max|err|':>11s} {'rel L2':>11s} {'BC MSE':>11s} {'time s':>8s}")
    for r in results:
        if r["status"] == "ok":
            print(f"{r['case']:28s} {r['grid']:4d} {r['mae']:11.3e} {r['max_abs_error']:11.3e} "
                  f"{r['relative_l2']:11.3e} {r['bc_mse']:11.3e} {r['runtime_s']:8.2f}")
        else:
            print(f"{r['case']:28s} {r['grid']:4d}  {r['status'].upper()}: "
                  f"{r['runs'][0].get('message', '')[:60]}")

    summary = {"suite": a.suite, "repeats": a.repeats, "cases": results}
    all_ok = all(r["status"] == "ok" for r in results)
    if a.suite == "public":
        ref = next(r for r in results if r["case"] == "reference")
        gate = {k: (ref["status"] == "ok" and ref[k] < v) for k, v in STAGE1.items()}
        summary["stage1"] = {"thresholds_from_proposal": STAGE1, "pass_by_metric": gate,
                             "passed": all(gate.values())}
        print("\nStage 1 (reference case vs values reported in the proposal; strict <, mean over runs):")
        for k, v in STAGE1.items():
            val = ref.get(k, float("nan"))
            print(f"  {k:14s} {val:11.4e} < {v:.3g} : {'PASS' if gate[k] else 'FAIL'}")
        print("  STAGE 1:", "PASSED" if summary["stage1"]["passed"] else "NOT PASSED")
        print("\nPublic practice cases are for self-checking only; official ranking uses the hidden suite.")
    else:
        if all_ok:
            worst = {k: max(r[k] for r in results) for k in METRICS}
            gate = {k: worst[k] < v for k, v in STAGE2.items()}
            summary["stage2"] = {"worst_case": worst, "organiser_thresholds": STAGE2,
                                 "pass_by_metric": gate, "passed": all(gate.values())}
            for r in results:
                r["case_score"] = case_score(r)
            summary["stage3"] = {"composite_score": composite(results), "weights": WEIGHTS,
                                 "term_floor": TERM_FLOOR, "worst_weight": WORST_WEIGHT,
                                 "mean_case_score": float(np.mean([r["case_score"] for r in results])),
                                 "worst_case_score": float(np.max([r["case_score"] for r in results])),
                                 "mean_runtime_s": float(np.mean([r["runtime_s"] for r in results]))}
            print("\nStage 2 (worst case over hidden cases; organiser-proposed thresholds):")
            for k, v in STAGE2.items():
                print(f"  {k:14s} {worst[k]:11.4e} < {v:.3g} : {'PASS' if gate[k] else 'FAIL'}")
            label = "" if summary["stage2"]["passed"] else "  (NOT ELIGIBLE: Stage 2 failed; shown for diagnosis only)"
            summary["stage3"]["eligible"] = summary["stage2"]["passed"]
            print(f"Stage 3 composite score (lower is better): {summary['stage3']['composite_score']:.4f}{label}")
        else:
            summary["stage2"] = {"passed": False, "reason": "one or more hidden cases failed, timed out or were invalid"}
            print("\nStage 2: NOT PASSED (a hidden case failed, timed out, or returned an invalid array)")
    if a.json:
        Path(a.json).write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
