"""
experiments/batch_runner.py
===========================
Parallel 30-run executor for the coevolutionary recommender study.

Workflow
--------
1. Load MovieLens-100K via ``core.data_loader`` (or accept arrays directly).
2. For each configuration in the experiment grid:
   a. Generate 30 deterministic seeds via ``experiments.config.make_run_seeds``.
   b. Launch up to ``n_jobs`` parallel workers (joblib / loky backend).
   c. Each worker constructs an independent ``CoevolutionaryEngine`` with its
      own seed, runs it, and returns a compact result dict.
3. Aggregate results: mean±std RMSE, Wilcoxon rank-sum pairwise tests.
4. Persist to disk:
   - ``results/seeds.json``     — all seeds used (for exact reproduction)
   - ``results/runs.csv``       — one row per (config, run)
   - ``results/summary.csv``    — one row per config (mean/std/median RMSE)
   - ``results/wilcoxon.csv``   — pairwise significance table
   - ``results/boxplots.png``   — RMSE distribution box plots (matplotlib)

Public API
----------
run_single(config, R_train, R_test_pairs, R_test_ratings, seed) -> dict
    Execute one independent run and return a compact result dict.

run_experiment(name, config, R_train, R_test_pairs, R_test_ratings,
               n_runs, n_jobs) -> dict
    Run all ``n_runs`` replications of one configuration in parallel.

run_all_experiments(R_train, R_test_pairs, R_test_ratings,
                    configs, n_runs, n_jobs, output_dir) -> dict
    Top-level entry point: runs every configuration, saves all outputs.

save_seeds(seed_map, path)
load_seeds(path) -> dict

summarize_results(all_results) -> pd.DataFrame
wilcoxon_pairwise(all_results, names) -> pd.DataFrame
plot_boxplots(all_results, names, output_path, show)

Statistical notes
-----------------
Wilcoxon rank-sum (= Mann-Whitney U) is used — appropriate for comparing
two independent samples of 30 RMSE values.  A two-sided test at α=0.05
is reported; p-values are Bonferroni-corrected for multiple comparisons.
"""

from __future__ import annotations

import json
import os
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from core.coevo_engine import CoevolutionaryEngine
from experiments.config import (
    N_RUNS,
    RESULTS_DIR,
    get_experiment_grid,
    make_run_seeds,
    describe_config,
    list_configs,
    get_config,
)


# ---------------------------------------------------------------------------
# Single-run worker (must be a module-level function for joblib/loky pickling)
# ---------------------------------------------------------------------------

def run_single(
    config:         Dict,
    R_train:        np.ndarray,
    R_test_pairs:   np.ndarray,
    R_test_ratings: np.ndarray,
    seed:           int,
) -> Dict:
    """
    Execute ONE independent evolutionary run.

    Parameters
    ----------
    config : dict
        Fully merged configuration dict (from ``get_config``).
    R_train : np.ndarray, shape (n_users, n_items)
        Training rating matrix (0 = unobserved).
    R_test_pairs : np.ndarray, shape (n_test, 2)
        Test interactions as (user_idx, item_idx) pairs.
    R_test_ratings : np.ndarray, shape (n_test,)
        Corresponding ground-truth ratings.
    seed : int
        RNG seed for this run.  Stored in results for future reproduction.

    Returns
    -------
    dict with keys
        'seed'            : int
        'final_test_rmse' : float   — primary metric
        'best_fit_U'      : float   — best population-U fitness (= −best_RMSE_U)
        'best_fit_V'      : float   — best population-V fitness
        'final_gen'       : int     — last completed generation
        'total_wall_sec'  : float
        'convergence'     : list[float] — test RMSE at every logged generation
                            (compact: only stores logged values, not every gen)
        'error'           : str or None — exception message if run failed
    """
    run_cfg = {**config, "seed": seed}
    try:
        eng = CoevolutionaryEngine(
            run_cfg, R_train, R_test_pairs, R_test_ratings
        )
        results = eng.run()
        log     = results["log"]
        conv    = [row["test_rmse"] for row in log if "test_rmse" in row]
        return {
            "seed":            seed,
            "final_test_rmse": float(results["final_test_rmse"] or np.nan),
            "best_fit_U":      float(results["final_fit_U"].max()),
            "best_fit_V":      float(results["final_fit_V"].max()),
            "final_gen":       int(log[-1]["gen"]) if log else 0,
            "total_wall_sec":  float(results["total_wall_sec"]),
            "convergence":     [round(v, 6) for v in conv],
            "error":           None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "seed":            seed,
            "final_test_rmse": float("nan"),
            "best_fit_U":      float("nan"),
            "best_fit_V":      float("nan"),
            "final_gen":       0,
            "total_wall_sec":  0.0,
            "convergence":     [],
            "error":           traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Per-configuration runner
# ---------------------------------------------------------------------------

def run_experiment(
    name:           str,
    config:         Dict,
    R_train:        np.ndarray,
    R_test_pairs:   np.ndarray,
    R_test_ratings: np.ndarray,
    n_runs:         int  = N_RUNS,
    n_jobs:         int  = -1,
    verbose:        int  = 5,
) -> Dict:
    """
    Run all ``n_runs`` replications of one configuration in parallel.

    Parameters
    ----------
    name : str
        Configuration name (used for labelling and seed generation).
    config : dict
        Fully merged configuration dict.
    R_train, R_test_pairs, R_test_ratings : np.ndarray
        Dataset arrays (shared read-only across workers).
    n_runs : int
        Number of independent replications (default 30).
    n_jobs : int
        joblib parallelism.  -1 = all cores, 1 = sequential (no fork).
    verbose : int
        joblib verbosity level (0 = silent, 10 = noisy).

    Returns
    -------
    dict with keys
        'name'     : str
        'config'   : dict
        'seeds'    : list[int]  — seeds used (length n_runs)
        'runs'     : list[dict] — one compact result dict per run
        'n_failed' : int        — number of runs that raised exceptions
        'summary'  : dict       — mean/std/median/min/max of final_test_rmse
        'wall_sec' : float      — total elapsed time for this configuration
    """
    seeds = make_run_seeds(name, n_runs=n_runs)

    t0 = time.perf_counter()
    run_results: List[Dict] = Parallel(
        n_jobs=n_jobs, verbose=verbose, backend="loky",
        prefer=None,
    )(
        delayed(run_single)(
            config, R_train, R_test_pairs, R_test_ratings, s
        )
        for s in seeds
    )
    wall = time.perf_counter() - t0

    rmses     = np.array([r["final_test_rmse"] for r in run_results],
                          dtype=float)
    valid     = rmses[~np.isnan(rmses)]
    n_failed  = int(np.isnan(rmses).sum())

    summary: Dict = {}
    if len(valid) > 0:
        summary = {
            "mean_rmse":   float(np.mean(valid)),
            "std_rmse":    float(np.std(valid, ddof=1) if len(valid) > 1 else 0.0),
            "median_rmse": float(np.median(valid)),
            "min_rmse":    float(np.min(valid)),
            "max_rmse":    float(np.max(valid)),
            "n_valid":     int(len(valid)),
        }
    else:
        summary = {k: float("nan") for k in
                   ("mean_rmse","std_rmse","median_rmse","min_rmse","max_rmse")}
        summary["n_valid"] = 0

    return {
        "name":     name,
        "config":   config,
        "seeds":    seeds,
        "runs":     run_results,
        "n_failed": n_failed,
        "summary":  summary,
        "wall_sec": round(wall, 2),
    }


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_seeds(seed_map: Dict[str, List[int]], path: str | Path) -> None:
    """
    Persist the seed mapping to a JSON file.

    Parameters
    ----------
    seed_map : dict
        ``{config_name: [seed0, seed1, ..., seed_{n-1}]}``
    path : str or Path
        Destination file path (parent directories created automatically).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(seed_map, fh, indent=2)
    print(f"  Seeds saved -> {path}")


def load_seeds(path: str | Path) -> Dict[str, List[int]]:
    """
    Load a seeds JSON file produced by ``save_seeds``.

    Returns
    -------
    dict
        ``{config_name: [seed0, ..., seed_{n-1}]}``
    """
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_runs_csv(all_results: Dict[str, Dict], path: str | Path) -> pd.DataFrame:
    """
    Write one row per (config, run) to a CSV file.

    Columns: config_name, seed, run_idx, final_test_rmse,
             best_fit_U, best_fit_V, final_gen, total_wall_sec, error

    Returns
    -------
    pd.DataFrame
        The dataframe that was written.
    """
    rows = []
    for name, exp in all_results.items():
        for run_idx, run in enumerate(exp["runs"]):
            rows.append({
                "config_name":    name,
                "seed":           run["seed"],
                "run_idx":        run_idx,
                "final_test_rmse": run["final_test_rmse"],
                "best_fit_U":     run["best_fit_U"],
                "best_fit_V":     run["best_fit_V"],
                "final_gen":      run["final_gen"],
                "total_wall_sec": run["total_wall_sec"],
                "error":          run.get("error") or "",
            })
    df = pd.DataFrame(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Runs CSV saved -> {path}  ({len(df)} rows)")
    return df


def save_summary_csv(
    all_results: Dict[str, Dict],
    path: str | Path,
) -> pd.DataFrame:
    """
    Write one row per configuration summarising mean±std RMSE.

    Columns: config_name, description, mean_rmse, std_rmse, median_rmse,
             min_rmse, max_rmse, n_valid, n_failed, wall_sec
    """
    rows = []
    for name, exp in all_results.items():
        s = exp["summary"]
        rows.append({
            "config_name": name,
            "description": describe_config(name),
            **s,
            "n_failed":    exp["n_failed"],
            "wall_sec":    exp["wall_sec"],
        })
    df = pd.DataFrame(rows).sort_values("mean_rmse", ignore_index=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"  Summary CSV saved -> {path}  ({len(df)} configs)")
    return df


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def summarize_results(all_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Return a formatted summary DataFrame (mean ± std RMSE per config).

    The human-readable ``mean_std`` column is formatted as
    ``"1.2345 ± 0.0123"`` for direct inclusion in papers.

    Parameters
    ----------
    all_results : dict
        Output of ``run_all_experiments`` or a union of ``run_experiment`` dicts.

    Returns
    -------
    pd.DataFrame sorted by mean_rmse ascending.
    """
    rows = []
    for name, exp in all_results.items():
        s = exp["summary"]
        mean = s.get("mean_rmse", float("nan"))
        std  = s.get("std_rmse",  float("nan"))
        rows.append({
            "config_name": name,
            "description": describe_config(name),
            "mean_rmse":   round(mean, 4),
            "std_rmse":    round(std,  4),
            "mean_std":    f"{mean:.4f} +/- {std:.4f}",
            "median_rmse": round(s.get("median_rmse", float("nan")), 4),
            "min_rmse":    round(s.get("min_rmse",    float("nan")), 4),
            "max_rmse":    round(s.get("max_rmse",    float("nan")), 4),
            "n_valid":     s.get("n_valid", 0),
        })
    return pd.DataFrame(rows).sort_values("mean_rmse", ignore_index=True)


def wilcoxon_pairwise(
    all_results: Dict[str, Dict],
    names:       Optional[List[str]] = None,
    alpha:       float = 0.05,
) -> pd.DataFrame:
    """
    Compute Wilcoxon rank-sum (Mann-Whitney U) pairwise significance table.

    For every ordered pair (A, B), the two-sided rank-sum test is applied to
    the 30 final RMSE values.  p-values are Bonferroni-corrected for the
    number of pairs.

    Parameters
    ----------
    all_results : dict
        Output of ``run_all_experiments``.
    names : list[str] or None
        Subset of configuration names to compare.  If None, all configs are
        included.
    alpha : float
        Significance level BEFORE Bonferroni correction (default 0.05).

    Returns
    -------
    pd.DataFrame
        Columns: config_A, config_B, stat, p_raw, p_bonferroni,
                 significant, better  (which config has lower mean RMSE)
    """
    if names is None:
        names = sorted(all_results.keys())

    # Collect RMSE arrays per config (filter NaN)
    rmse_arrays: Dict[str, np.ndarray] = {}
    for name in names:
        vals = np.array(
            [r["final_test_rmse"] for r in all_results[name]["runs"]],
            dtype=float,
        )
        rmse_arrays[name] = vals[~np.isnan(vals)]

    pairs = list(itertools.combinations(names, 2))
    n_pairs = len(pairs)
    rows = []
    for (a, b) in pairs:
        xa, xb = rmse_arrays.get(a, np.array([])), rmse_arrays.get(b, np.array([]))
        if len(xa) < 3 or len(xb) < 3:
            stat, p_raw = float("nan"), float("nan")
        else:
            stat, p_raw = stats.ranksums(xa, xb)
            stat, p_raw = float(stat), float(p_raw)
        p_bonf     = min(float(p_raw) * n_pairs, 1.0) if not np.isnan(p_raw) else float("nan")
        significant = bool(p_bonf < alpha) if not np.isnan(p_bonf) else False
        mean_a = float(np.mean(xa)) if len(xa) > 0 else float("nan")
        mean_b = float(np.mean(xb)) if len(xb) > 0 else float("nan")
        better = a if mean_a < mean_b else b
        rows.append({
            "config_A":     a,
            "config_B":     b,
            "mean_rmse_A":  round(mean_a, 4),
            "mean_rmse_B":  round(mean_b, 4),
            "stat":         round(stat, 4) if not np.isnan(stat) else float("nan"),
            "p_raw":        round(p_raw, 6) if not np.isnan(p_raw) else float("nan"),
            "p_bonferroni": round(p_bonf, 6) if not np.isnan(p_bonf) else float("nan"),
            "significant":  significant,
            "better":       better,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_boxplots(
    all_results:  Dict[str, Dict],
    names:        Optional[List[str]] = None,
    output_path:  str | Path = "results/boxplots.png",
    show:         bool = False,
    figsize:      tuple = (16, 6),
    title:        str  = "Test RMSE Distribution across 30 Runs",
) -> None:
    """
    Produce a box plot of final test RMSE distributions for each configuration.

    Parameters
    ----------
    all_results : dict
        Output of ``run_all_experiments``.
    names : list[str] or None
        Subset of configs to plot.  If None, all are included.
    output_path : str or Path
        Where to save the PNG (parent dirs created automatically).
    show : bool
        If True, call ``plt.show()`` (blocks; use False in batch mode).
    figsize : tuple
        Matplotlib figure size (width, height) in inches.
    title : str
        Figure title.
    """
    import matplotlib
    matplotlib.use("Agg")           # non-interactive backend for batch mode
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    if names is None:
        names = sorted(all_results.keys())

    # Collect data grouped by config (sorted by median RMSE ascending)
    data   = []
    labels = []
    for name in names:
        vals = np.array(
            [r["final_test_rmse"] for r in all_results[name]["runs"]],
            dtype=float,
        )
        vals = vals[~np.isnan(vals)]
        data.append(vals)
        # Shorten label for readability
        labels.append(name.replace("_", "\n"))

    # Sort by median
    medians   = [float(np.median(d)) if len(d) > 0 else float("inf") for d in data]
    order     = sorted(range(len(data)), key=lambda i: medians[i])
    data      = [data[i]   for i in order]
    labels    = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(
        data, patch_artist=True, notch=False, vert=True,
        medianprops=dict(color="crimson", linewidth=2.0),
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
        boxprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )

    # Colour boxes with a gradient
    cmap = plt.get_cmap("viridis", len(data))
    for patch, c_idx in zip(bp["boxes"], range(len(data))):
        patch.set_facecolor(cmap(c_idx))
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Test RMSE", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Box plot saved -> {out}")
    if show:
        plt.show()
    plt.close(fig)


def plot_convergence(
    all_results:  Dict[str, Dict],
    names:        Optional[List[str]] = None,
    output_path:  str | Path = "results/convergence.png",
    show:         bool = False,
    figsize:      tuple = (12, 5),
    title:        str  = "Mean Convergence Curves (Test RMSE vs Generation)",
) -> None:
    """
    Plot mean convergence curves (test RMSE per generation, averaged across runs).

    Parameters
    ----------
    all_results : dict
    names       : list[str] or None — configs to include
    output_path : str or Path
    show        : bool
    figsize, title : matplotlib settings
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if names is None:
        names = sorted(all_results.keys())

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("tab10", len(names))

    for ci, name in enumerate(names):
        curves = [r["convergence"] for r in all_results[name]["runs"]
                  if r["convergence"]]
        if not curves:
            continue
        # Pad/truncate all curves to the shortest length
        min_len = min(len(c) for c in curves)
        arr = np.array([c[:min_len] for c in curves], dtype=float)
        mean_curve = arr.mean(axis=0)
        std_curve  = arr.std(axis=0, ddof=1) if len(arr) > 1 else np.zeros_like(mean_curve)
        xs = np.arange(len(mean_curve))

        ax.plot(xs, mean_curve,
                label=name[:30], color=cmap(ci), linewidth=1.8, alpha=0.9)
        ax.fill_between(xs,
                         mean_curve - std_curve,
                         mean_curve + std_curve,
                         color=cmap(ci), alpha=0.12)

    ax.set_xlabel("Logged generation", fontsize=11)
    ax.set_ylabel("Mean Test RMSE", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Convergence plot saved -> {out}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_all_experiments(
    R_train:        np.ndarray,
    R_test_pairs:   np.ndarray,
    R_test_ratings: np.ndarray,
    configs:        Optional[List[Dict]] = None,
    n_runs:         int  = N_RUNS,
    n_jobs:         int  = -1,
    output_dir:     str  = RESULTS_DIR,
    save_outputs:   bool = True,
    verbose:        int  = 5,
) -> Dict[str, Dict]:
    """
    Run all experiments and optionally persist all outputs.

    Parameters
    ----------
    R_train, R_test_pairs, R_test_ratings : np.ndarray
        Dataset arrays from ``core.data_loader``.
    configs : list[dict] or None
        Each dict must have keys ``'name'`` and ``'config'``.
        If None, ``get_experiment_grid()`` is used (all registered configs).
    n_runs : int
        Replications per configuration (default 30).
    n_jobs : int
        joblib workers (-1 = all CPU cores).
    output_dir : str
        Directory for all output files.
    save_outputs : bool
        If False, skip disk writes (useful for quick debugging).
    verbose : int
        joblib / print verbosity.

    Returns
    -------
    dict
        ``{config_name: experiment_result_dict}``
        where each value is the output of ``run_experiment``.
    """
    if configs is None:
        configs = get_experiment_grid()

    all_results: Dict[str, Dict] = {}
    n_total = len(configs)
    t_global = time.perf_counter()

    for exp_idx, entry in enumerate(configs, 1):
        name   = entry["name"]
        config = entry["config"]
        desc   = describe_config(name)
        print(f"\n[{exp_idx}/{n_total}] Running '{name}'")
        print(f"            {desc}")
        print(f"            {n_runs} runs x {n_jobs} workers")

        exp_result = run_experiment(
            name, config, R_train, R_test_pairs, R_test_ratings,
            n_runs=n_runs, n_jobs=n_jobs, verbose=verbose,
        )
        all_results[name] = exp_result
        s = exp_result["summary"]
        print(
            f"            RMSE: {s.get('mean_rmse', float('nan')):.4f}"
            f" +/- {s.get('std_rmse', float('nan')):.4f}"
            f"   failed={exp_result['n_failed']}/{n_runs}"
            f"   wall={exp_result['wall_sec']:.1f}s"
        )

    total_wall = round(time.perf_counter() - t_global, 1)
    print(f"\n=== All {n_total} experiments done in {total_wall}s ===")

    if save_outputs:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1. Seeds JSON
        seed_map = {name: exp["seeds"] for name, exp in all_results.items()}
        save_seeds(seed_map, out / "seeds.json")

        # 2. Per-run CSV
        save_runs_csv(all_results, out / "runs.csv")

        # 3. Summary CSV
        save_summary_csv(all_results, out / "summary.csv")

        # 4. Wilcoxon table (only if enough data)
        names_with_data = [n for n, e in all_results.items()
                           if e["summary"].get("n_valid", 0) >= 3]
        if len(names_with_data) >= 2:
            try:
                wdf = wilcoxon_pairwise(all_results, names_with_data)
                wdf.to_csv(out / "wilcoxon.csv", index=False, float_format="%.6f")
                print(f"  Wilcoxon table saved -> {out / 'wilcoxon.csv'}"
                      f"  ({len(wdf)} pairs)")
            except Exception as exc:
                warnings.warn(f"Wilcoxon analysis failed: {exc}")

        # 5. Box plots
        try:
            plot_boxplots(all_results, output_path=out / "boxplots.png")
            plot_convergence(all_results, output_path=out / "convergence.png")
        except Exception as exc:
            warnings.warn(f"Plotting failed: {exc}")

        print(f"\nAll outputs saved to: {out.resolve()}")

    return all_results


# ---------------------------------------------------------------------------
# Missing import for wilcoxon_pairwise
# ---------------------------------------------------------------------------
import itertools  # noqa: E402  (placed after functions that reference it)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(
    data_path:  str  = "data/ml-100k/u.data",
    n_runs:     int  = N_RUNS,
    n_jobs:     int  = -1,
    output_dir: str  = RESULTS_DIR,
    config_names: Optional[List[str]] = None,
) -> None:
    """
    Command-line entry point.

    Loads MovieLens-100K, runs all (or specified) experiments, saves results.

    Parameters
    ----------
    data_path    : path to u.data (tab-separated)
    n_runs       : replications per config
    n_jobs       : parallel workers
    output_dir   : where to write results
    config_names : if given, only these configs are run
    """
    print("=" * 66)
    print("Coevolutionary Recommender — Batch Runner")
    print("=" * 66)

    # ---- Load data -------------------------------------------------------
    try:
        from core.data_loader import load_movielens, train_test_split_matrix  # type: ignore
        df, n_users, n_items = load_movielens(data_path)
        R_train, R_test, n_users, n_items = train_test_split_matrix(df, test_size=0.2)
        test_r, test_c = np.nonzero(R_test)
        test_pairs   = np.column_stack([test_r, test_c])
        test_ratings = R_test[test_r, test_c]
        print(f"[Data] Loaded MovieLens-100K from '{data_path}'")
        print(f"       R_train {R_train.shape}, {int((R_train > 0).sum())} obs")
        print(f"       test set: {len(test_ratings)} interactions")
    except (ImportError, FileNotFoundError) as exc:
        print(f"[WARNING] Could not load real data ({exc}).")
        print("[WARNING] Using SYNTHETIC data (40 users, 60 items) for demo.")
        rng      = np.random.default_rng(2024)
        R_train  = np.zeros((40, 60), dtype=np.float32)
        mask     = rng.random((40, 60)) < 0.20
        R_train[mask] = rng.uniform(1, 5, int(mask.sum())).astype(np.float32)
        obs_r, obs_c = np.nonzero(mask)
        idx      = rng.choice(len(obs_r), 60, replace=False)
        test_pairs   = np.column_stack([obs_r[idx], obs_c[idx]])
        test_ratings = R_train[obs_r[idx], obs_c[idx]]

    # ---- Select configs ---------------------------------------------------
    if config_names:
        configs = [{"name": n, "config": get_config(n)} for n in config_names]
    else:
        configs = get_experiment_grid()

    print(f"\n[Plan] {len(configs)} configs × {n_runs} runs "
          f"× {n_jobs} workers → {len(configs)*n_runs} total runs")

    # ---- Run all ----------------------------------------------------------
    all_results = run_all_experiments(
        R_train, test_pairs, test_ratings,
        configs=configs,
        n_runs=n_runs,
        n_jobs=n_jobs,
        output_dir=output_dir,
        save_outputs=True,
        verbose=5,
    )

    # ---- Print final summary ----------------------------------------------
    summary_df = summarize_results(all_results)
    print("\n=== Final Summary (sorted by mean RMSE) ===")
    print(summary_df[["config_name", "mean_std", "median_rmse",
                       "n_valid"]].to_string(index=False))


# ---------------------------------------------------------------------------
# Sanity check — python -X utf8 experiments/batch_runner.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 66)
    print("experiments/batch_runner.py — sanity check")
    print("=" * 66)

    # ---- Tiny synthetic dataset (fast) ------------------------------------
    SEED_D   = 1234
    NU, NI   = 20, 30      # very small for speed
    K_DIM    = 5
    N_GEN    = 4            # minimal generations
    N_RUNS_T = 3            # 3 runs instead of 30 for the test

    rng_d  = np.random.default_rng(SEED_D)
    R_tiny = np.zeros((NU, NI), dtype=np.float32)
    msk    = rng_d.random((NU, NI)) < 0.25
    R_tiny[msk] = rng_d.uniform(1, 5, int(msk.sum())).astype(np.float32)
    or_, oc_ = np.nonzero(msk)
    ti       = rng_d.choice(len(or_), min(20, len(or_)), replace=False)
    t_pairs  = np.column_stack([or_[ti], oc_[ti]])
    t_rates  = R_tiny[or_[ti], oc_[ti]]
    print(f"\n[Synthetic data] R={R_tiny.shape}, "
          f"obs={int(msk.sum())}, test={len(t_rates)}")

    # Minimal configs for fast testing
    TEST_CONFIG_NAMES = ["baseline", "baseline_sharing"]
    test_configs = [
        {
            "name": name,
            "config": {
                **get_config(name),
                "k": K_DIM,
                "n_generations": N_GEN,
                "log_every": 1,
            },
        }
        for name in TEST_CONFIG_NAMES
    ]

    # -------------------------------------------------------------------
    # Test 1: run_single
    # -------------------------------------------------------------------
    print("\n[Test 1] run_single")
    cfg_single = {**get_config("baseline"), "k": K_DIM,
                  "n_generations": N_GEN, "log_every": 1}
    r = run_single(cfg_single, R_tiny, t_pairs, t_rates, seed=42)
    assert r["error"] is None, f"run_single raised: {r['error']}"
    assert isinstance(r["final_test_rmse"], float)
    assert 0 < r["final_test_rmse"] < 100
    assert r["seed"] == 42
    print(f"  seed=42, final_test_rmse={r['final_test_rmse']:.4f}, "
          f"wall={r['total_wall_sec']:.3f}s  OK")

    # -------------------------------------------------------------------
    # Test 2: run_experiment (sequential, n_jobs=1)
    # -------------------------------------------------------------------
    print(f"\n[Test 2] run_experiment (n_runs={N_RUNS_T}, n_jobs=1)")
    exp = run_experiment(
        "baseline", test_configs[0]["config"],
        R_tiny, t_pairs, t_rates,
        n_runs=N_RUNS_T, n_jobs=1, verbose=0,
    )
    assert exp["name"]     == "baseline"
    assert len(exp["runs"]) == N_RUNS_T
    assert len(exp["seeds"]) == N_RUNS_T
    assert exp["n_failed"] == 0, f"{exp['n_failed']} runs failed"
    assert 0 < exp["summary"]["mean_rmse"] < 100
    print(f"  mean_rmse={exp['summary']['mean_rmse']:.4f} "
          f"+/- {exp['summary']['std_rmse']:.4f}  OK")
    print(f"  seeds={exp['seeds']}  OK")

    # -------------------------------------------------------------------
    # Test 3: save_seeds / load_seeds round-trip
    # -------------------------------------------------------------------
    print(f"\n[Test 3] save_seeds / load_seeds")
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmp:
        seed_path = pathlib.Path(tmp) / "test_seeds.json"
        seed_map  = {"baseline": exp["seeds"]}
        save_seeds(seed_map, seed_path)
        loaded    = load_seeds(seed_path)
        assert loaded == seed_map, "Round-trip failed"
    print(f"  JSON round-trip OK")

    # -------------------------------------------------------------------
    # Test 4: run_all_experiments with save_outputs=True
    # -------------------------------------------------------------------
    print(f"\n[Test 4] run_all_experiments (save_outputs=True)")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_out:
        all_res = run_all_experiments(
            R_tiny, t_pairs, t_rates,
            configs=test_configs,
            n_runs=N_RUNS_T,
            n_jobs=1,
            output_dir=tmp_out,
            save_outputs=True,
            verbose=0,
        )
        # Check files were created
        for fname in ["seeds.json", "runs.csv", "summary.csv"]:
            fpath = pathlib.Path(tmp_out) / fname
            assert fpath.exists(), f"Missing output: {fname}"
            print(f"  {fname}  ({fpath.stat().st_size} bytes)  OK")
        # Check box plot
        bp_path = pathlib.Path(tmp_out) / "boxplots.png"
        if bp_path.exists():
            print(f"  boxplots.png  ({bp_path.stat().st_size} bytes)  OK")

    assert set(all_res.keys()) == {"baseline", "baseline_sharing"}
    print(f"  all_results keys: {sorted(all_res.keys())}  OK")

    # -------------------------------------------------------------------
    # Test 5: summarize_results
    # -------------------------------------------------------------------
    print(f"\n[Test 5] summarize_results")
    df_sum = summarize_results(all_res)
    assert len(df_sum) == len(TEST_CONFIG_NAMES)
    assert "mean_std" in df_sum.columns
    print(df_sum[["config_name","mean_std","n_valid"]].to_string(index=False))
    print(f"  Summary DataFrame shape={df_sum.shape}  OK")

    # -------------------------------------------------------------------
    # Test 6: wilcoxon_pairwise (needs >= 3 valid runs per config)
    # -------------------------------------------------------------------
    print(f"\n[Test 6] wilcoxon_pairwise")
    wdf = wilcoxon_pairwise(all_res)
    assert len(wdf) == 1, f"Expected 1 pair, got {len(wdf)}"
    assert "p_bonferroni" in wdf.columns
    row = wdf.iloc[0]
    print(f"  baseline vs baseline_sharing: "
          f"stat={row['stat']:.4f}, p_raw={row['p_raw']:.4f}, "
          f"sig={row['significant']}  OK")

    # -------------------------------------------------------------------
    # Test 7: convergence curve
    # -------------------------------------------------------------------
    print(f"\n[Test 7] Convergence curves")
    conv = all_res["baseline"]["runs"][0]["convergence"]
    assert len(conv) == N_GEN + 1, \
        f"Expected {N_GEN+1} convergence points (gen 0 to {N_GEN}), got {len(conv)}"
    print(f"  baseline run-0 convergence ({len(conv)} points): "
          f"{[round(v,4) for v in conv[:4]]} ... {conv[-1]:.4f}  OK")

    # -------------------------------------------------------------------
    # Test 8: Reproducibility — same seed produces same RMSE
    # -------------------------------------------------------------------
    print(f"\n[Test 8] Reproducibility")
    cfg_r = {**get_config("baseline"), "k": K_DIM,
             "n_generations": N_GEN, "log_every": 1}
    r_a = run_single(cfg_r, R_tiny, t_pairs, t_rates, seed=777)
    r_b = run_single(cfg_r, R_tiny, t_pairs, t_rates, seed=777)
    assert abs(r_a["final_test_rmse"] - r_b["final_test_rmse"]) < 1e-6, \
        "Same seed produced different RMSE!"
    print(f"  seed=777 x2: RMSE={r_a['final_test_rmse']:.6f} == "
          f"{r_b['final_test_rmse']:.6f}  OK")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 66)
    print("All 8 batch_runner tests passed  OK")
    print("=" * 66)
    print("\nTo run the full 30-run study over all configs:")
    print("  python -X utf8 experiments/batch_runner.py")
    print("     (uses real data from data/ml-100k/u.data)")
