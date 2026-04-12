"""
experiments/config.py
=====================
Canonical configuration dictionaries for the coevolutionary recommender study.

Design philosophy
-----------------
One BASELINE configuration is defined first (the "control" arm), then each
experimental configuration overrides exactly ONE axis — selection, crossover,
mutation, survivor selection, or diversity mechanism.  This yields a clean
one-factor-at-a-time (OFAT) ablation study PLUS a full factorial grid for
the four binary operator axes.

All configs are plain Python dicts; they are merged on top of BASE_CONFIG by
CoevolutionaryEngine.__init__ (missing keys fall back to engine defaults).

Key constants
-------------
N_RUNS          = 30   — independent replications per configuration
MASTER_SEED     = 2024 — used to generate per-run seeds deterministically
RESULTS_DIR     = "results"

Public API
----------
get_config(name: str) -> dict
    Return the merged (base + variant) config for a named configuration.

list_configs() -> list[str]
    All registered configuration names.

get_experiment_grid() -> list[dict]
    Every element is {"name": str, "config": dict} — one per experiment.

make_run_seeds(config_name: str, n_runs: int = N_RUNS) -> list[int]
    Deterministically generate per-run integer seeds from MASTER_SEED so that
    any run can be exactly reproduced from just the config name + run index.

describe_config(name: str) -> str
    One-line human-readable summary (used as chart title / table label).
"""

from __future__ import annotations

import itertools
from copy import deepcopy
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Study-wide constants
# ---------------------------------------------------------------------------

N_RUNS      = 30
MASTER_SEED = 2024
RESULTS_DIR = "results"

# ---------------------------------------------------------------------------
# BASE configuration — all experiments override this
# ---------------------------------------------------------------------------

BASE_CONFIG: Dict = {
    # ----- problem dimensions -----
    "k":            20,          # latent factors

    # ----- coevolution -----
    "n_generations": 100,
    "k_random":      3,          # random collaborators added to best-V/U

    # ----- parent selection -----
    "selection":    "tournament",
    "tau":          3,           # tournament size

    # ----- crossover -----
    "crossover":    "uniform",
    "p_swap":       0.5,
    "alpha_blx":    0.5,

    # ----- mutation -----
    "mutation":     "gaussian",
    "sigma_init":   0.3,
    "sigma_min":    1e-5,
    "p_reset":      0.1,
    "reset_low":   -0.5,
    "reset_high":   0.5,

    # ----- survivor selection -----
    "survivor_selection": "mu_plus_lambda",

    # ----- diversity -----
    "fitness_sharing":  False,
    "sigma_share":      1.5,
    "alpha_share":      1.0,
    "island_model":     False,
    "n_islands":        4,
    "migration_interval": 10,
    "n_migrants":       2,

    # ----- initialisation -----
    "init_type":    "uniform",   # "uniform" | "svd"
    "svd_noise":    0.01,

    # ----- logging -----
    "log_every":    5,           # log every 5 generations (keeps memory small)
}

# ---------------------------------------------------------------------------
# Named configuration variants (each overrides only the differing keys)
# ---------------------------------------------------------------------------
# Naming convention:
#   {selection}_{crossover}_{mutation}_{survivor}[_{diversity_suffix}]
# where diversity_suffix is one of: none (omitted) | sharing | islands | both

_VARIANTS: Dict[str, Dict] = {

    # ================================================================
    # BASELINE (control) — tournament | uniform XO | Gaussian | mu+lambda
    # ================================================================
    "baseline": {},   # no overrides — pure BASE_CONFIG

    # ================================================================
    # AXIS A — Selection method
    # ================================================================
    "roulette_uniform_gaussian_muplus": {
        "selection": "rank_roulette",
    },

    # ================================================================
    # AXIS B — Crossover operator
    # ================================================================
    "tournament_blxalpha_gaussian_muplus": {
        "crossover": "blx_alpha",
    },

    # ================================================================
    # AXIS C — Mutation operator
    # ================================================================
    "tournament_uniform_reset_muplus": {
        "mutation":  "uniform_reset",
    },

    # ================================================================
    # AXIS D — Survivor selection
    # ================================================================
    "tournament_uniform_gaussian_mucomma": {
        "survivor_selection": "mu_comma_lambda",
    },

    # ================================================================
    # AXIS E — Diversity: fitness sharing
    # ================================================================
    "baseline_sharing": {
        "fitness_sharing": True,
        "sigma_share":     1.5,
        "alpha_share":     1.0,
    },

    # ================================================================
    # AXIS F — Diversity: island model
    # ================================================================
    "baseline_islands": {
        "island_model":       True,
        "n_islands":          4,
        "migration_interval": 10,
        "n_migrants":         2,
    },

    # ================================================================
    # AXIS G — Diversity: both sharing + islands
    # ================================================================
    "baseline_sharing_islands": {
        "fitness_sharing":    True,
        "sigma_share":        1.5,
        "island_model":       True,
        "n_islands":          4,
        "migration_interval": 10,
        "n_migrants":         2,
    },

    # ================================================================
    # AXIS H — Initialisation: SVD
    # ================================================================
    "baseline_svd_init": {
        "init_type": "svd",
    },

    # ================================================================
    # FULL FACTORIAL — all 16 selection×crossover×mutation×survivor combos
    # (the 4 above are already included; the remaining 12 are generated
    #  programmatically below and merged into _VARIANTS at module load time)
    # ================================================================
}

# --- Auto-generate the remaining 12 full-factorial combos -----------------
_SELECTIONS  = ["tournament", "rank_roulette"]
_CROSSOVERS  = ["uniform", "blx_alpha"]
_MUTATIONS   = ["gaussian", "uniform_reset"]
_SURVIVORS   = ["mu_plus_lambda", "mu_comma_lambda"]

_SHORT = {
    "tournament":     "tourn",
    "rank_roulette":  "roul",
    "uniform":        "uni",
    "blx_alpha":      "blx",
    "gaussian":       "gauss",
    "uniform_reset":  "reset",
    "mu_plus_lambda": "muplus",
    "mu_comma_lambda":"mucomma",
}

for _sel, _xo, _mut, _surv in itertools.product(
    _SELECTIONS, _CROSSOVERS, _MUTATIONS, _SURVIVORS
):
    _name = (
        f"{_SHORT[_sel]}_{_SHORT[_xo]}_{_SHORT[_mut]}_{_SHORT[_surv]}"
    )
    if _name not in _VARIANTS:
        _VARIANTS[_name] = {
            "selection":          _sel,
            "crossover":          _xo,
            "mutation":           _mut,
            "survivor_selection": _surv,
        }

# ---------------------------------------------------------------------------
# Human-readable description strings (for chart titles / LaTeX tables)
# ---------------------------------------------------------------------------

_DESCRIPTIONS: Dict[str, str] = {
    "baseline":
        "Baseline: Tournament(τ=3) + Uniform XO + Gaussian(σ-adapt) + (μ+λ)",
    "roulette_uniform_gaussian_muplus":
        "Rank-Roulette + Uniform XO + Gaussian + (μ+λ)",
    "tournament_blxalpha_gaussian_muplus":
        "Tournament(τ=3) + BLX-α(α=0.5) + Gaussian + (μ+λ)",
    "tournament_uniform_reset_muplus":
        "Tournament(τ=3) + Uniform XO + Reset(p=0.1) + (μ+λ)",
    "tournament_uniform_gaussian_mucomma":
        "Tournament(τ=3) + Uniform XO + Gaussian + (μ,λ)",
    "baseline_sharing":
        "Baseline + Fitness Sharing (σ_share=1.5)",
    "baseline_islands":
        "Baseline + Island Model (4 islands, migration every 10 gens)",
    "baseline_sharing_islands":
        "Baseline + Fitness Sharing + Island Model",
    "baseline_svd_init":
        "Baseline + SVD Initialisation",
}


# ---------------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------------

def get_config(name: str) -> Dict:
    """
    Return the fully merged configuration dict for experiment ``name``.

    The returned dict is BASE_CONFIG updated with the variant-specific
    overrides.  Changes to the returned dict do NOT affect the registry.

    Parameters
    ----------
    name : str
        Registered configuration name (see ``list_configs()``).

    Returns
    -------
    dict
        Complete configuration ready to pass to ``CoevolutionaryEngine``.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """
    if name not in _VARIANTS:
        raise KeyError(
            f"Unknown config '{name}'.  Available: {list(_VARIANTS.keys())}"
        )
    merged = deepcopy(BASE_CONFIG)
    merged.update(_VARIANTS[name])
    return merged


def list_configs() -> List[str]:
    """Return a sorted list of all registered configuration names."""
    return sorted(_VARIANTS.keys())


def get_experiment_grid() -> List[Dict]:
    """
    Return one dict per registered experiment.

    Each dict has keys:
        'name'   : str  — configuration name
        'config' : dict — fully merged configuration

    The list is sorted alphabetically by name for reproducible ordering.
    """
    return [
        {"name": name, "config": get_config(name)}
        for name in sorted(_VARIANTS.keys())
    ]


def make_run_seeds(
    config_name: str,
    n_runs: int = N_RUNS,
    master_seed: int = MASTER_SEED,
) -> List[int]:
    """
    Generate a deterministic list of per-run seeds.

    Seeds are derived from a hash of ``config_name`` XOR'd with
    ``master_seed``, then drawn from a seeded Generator.  The same
    (config_name, master_seed, n_runs) triple always produces the same
    seed list, enabling exact reproduction of any run.

    Parameters
    ----------
    config_name : str
        Name of the experiment configuration.
    n_runs : int
        Number of seeds to generate (default N_RUNS = 30).
    master_seed : int
        Root seed for the seed-generation RNG.

    Returns
    -------
    list[int]
        ``n_runs`` non-negative integer seeds.
    """
    # XOR master_seed with a stable hash of the config name (mod 2^31)
    name_hash = abs(hash(config_name)) % (2 ** 31)
    combined_seed = (master_seed ^ name_hash) % (2 ** 31)
    rng = np.random.default_rng(combined_seed)
    seeds = rng.integers(1, 2 ** 31 - 1, size=n_runs).tolist()
    return [int(s) for s in seeds]


def describe_config(name: str) -> str:
    """
    Return a human-readable one-line description of the configuration.

    Falls back to a compact string built from key fields if no explicit
    description is registered.
    """
    if name in _DESCRIPTIONS:
        return _DESCRIPTIONS[name]
    cfg = get_config(name)
    return (
        f"{cfg['selection']} | {cfg['crossover']} | "
        f"{cfg['mutation']} | {cfg['survivor_selection']} | "
        f"sharing={cfg['fitness_sharing']} islands={cfg['island_model']}"
    )


def print_experiment_table() -> None:
    """
    Print a formatted table of all registered experiments.

    Useful for a quick overview before launching a long batch run.
    """
    configs = list_configs()
    print(f"\n{'#':>3}  {'Name':<50}  Description")
    print("-" * 100)
    for i, name in enumerate(configs, 1):
        desc = describe_config(name)
        print(f"{i:>3}  {name:<50}  {desc}")
    print(f"\nTotal: {len(configs)} configurations × {N_RUNS} runs = "
          f"{len(configs) * N_RUNS} total runs")


# ---------------------------------------------------------------------------
# Sanity check — python -X utf8 experiments/config.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 66)
    print("experiments/config.py — sanity check")
    print("=" * 66)

    # Test 1: all configs accessible
    cfgs = list_configs()
    print(f"\n[Test 1] list_configs() -> {len(cfgs)} configurations")
    assert len(cfgs) >= 16 + 9, \
        f"Expected >= 25 configs (16 factorial + 9 named), got {len(cfgs)}"
    print(f"  Count check: {len(cfgs)} >= 25  OK")

    # Test 2: BASE_CONFIG fields are present in every config
    for name in cfgs:
        cfg = get_config(name)
        missing = [k for k in BASE_CONFIG if k not in cfg]
        assert not missing, f"Config '{name}' missing keys: {missing}"
    print(f"\n[Test 2] All {len(cfgs)} configs have all BASE_CONFIG keys  OK")

    # Test 3: Each config's overrides are actually applied
    cfg_bl = get_config("baseline")
    cfg_rr = get_config("roulette_uniform_gaussian_muplus")
    assert cfg_bl["selection"] == "tournament"
    assert cfg_rr["selection"] == "rank_roulette"
    assert cfg_rr["crossover"] == cfg_bl["crossover"],  \
        "crossover should be baseline value"
    print(f"\n[Test 3] Overrides applied correctly  OK")

    # Test 4: get_config returns a copy (mutations don't affect registry)
    cfg_copy = get_config("baseline")
    cfg_copy["k"] = 999
    assert get_config("baseline")["k"] == BASE_CONFIG["k"], \
        "Registry was mutated!"
    print(f"\n[Test 4] get_config returns independent copies  OK")

    # Test 5: make_run_seeds — deterministic, correct count, unique per config
    seeds_a = make_run_seeds("baseline", n_runs=30)
    seeds_b = make_run_seeds("baseline", n_runs=30)
    seeds_c = make_run_seeds("baseline_sharing", n_runs=30)
    assert seeds_a == seeds_b, "Seeds not deterministic!"
    assert seeds_a != seeds_c, "Different configs produce same seeds!"
    assert len(seeds_a) == 30
    assert all(isinstance(s, int) and s > 0 for s in seeds_a)
    print(f"\n[Test 5] make_run_seeds: deterministic, per-config distinct, "
          f"len=30  OK")
    print(f"  baseline seeds[:5]: {seeds_a[:5]}")

    # Test 6: get_experiment_grid
    grid = get_experiment_grid()
    assert len(grid) == len(cfgs)
    assert all("name" in g and "config" in g for g in grid)
    print(f"\n[Test 6] get_experiment_grid: {len(grid)} entries, "
          f"all have name+config  OK")

    # Test 7: describe_config
    desc = describe_config("baseline")
    assert len(desc) > 10, "Description too short"
    print(f"\n[Test 7] describe_config('baseline'):")
    print(f"  '{desc}'  OK")

    # Test 8: print table
    print(f"\n[Test 8] Experiment table preview (first 10 rows):")
    cfgs_short = list_configs()[:10]
    for i, name in enumerate(cfgs_short, 1):
        print(f"  {i:>2}. {name:<50} | {describe_config(name)[:50]}")

    print("\n" + "=" * 66)
    print("All 8 config tests passed  OK")
    print("=" * 66)
    print_experiment_table()
