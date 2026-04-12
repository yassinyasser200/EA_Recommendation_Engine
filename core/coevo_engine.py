"""
core/coevo_engine.py
====================
Full coevolutionary evolutionary loop for latent-factor collaborative filtering.

Architecture
------------
Two populations are co-evolved:

    Population U — shape (n_users, k) — one latent vector per MovieLens user
    Population V — shape (n_items, k) — one latent vector per MovieLens item

Because the index mapping is permanent (U[i] always represents user i, V[j]
always represents item j), each "individual" at position i is evolved IN-PLACE:

    offspring[i] = mutate( crossover( U[i], U[partner_i] ) )

The crossover partner is selected stochastically from the whole population
using the configured selection operator (tournament or rank-roulette).  This
allows genetic exchange while preserving the index-to-user/item mapping.

Operator pipeline (per generation, per population)
---------------------------------------------------
1. Select one crossover partner per individual (vectorised)
2. Apply crossover   (uniform or BLX-alpha, fully batched)
3. Apply mutation    (self-adaptive Gaussian or uniform reset, fully batched)
4. Evaluate offspring fitness (via fitness.py evaluate functions)
5. Survivor selection  ((mu+lambda) per-position or (mu,lambda) replace-all)
6. [Optional] Apply fitness sharing before the NEXT generation's selection

Island model (ring topology)
----------------------------
When enabled, U and V are each partitioned round-robin into n_islands
sub-populations.  Each island evolves its local U-slice and V-slice
independently.  Fitness evaluation always uses the full opposing-population
matrix (otherwise we would evaluate user i against items that never co-occur).  
Every ``migration_interval`` generations the best ``n_migrants`` individuals
from each island are copied to the clockwise neighbour.

Public API
----------
CoevolutionaryEngine(config, R_train, R_test_pairs=None, R_test_ratings=None)
    .run()                 -> results dict
    .get_best_solution()   -> (U_real, V_real)
    .get_logs()            -> list-of-dicts, one per logged generation

Configuration dict keys (all keys with defaults)
-------------------------------------------------
See DEFAULT_CONFIG below for the full annotated reference.
"""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np

# Internal modules (same package)
from core.fitness import (
    select_collaborators,
    evaluate_population_U,
    evaluate_population_V,
    compute_test_rmse,
)
from core.operators import (
    tournament_selection,
    rank_roulette_selection,
    mu_plus_lambda,
    mu_comma_lambda,
)
from core.diversity import apply_fitness_sharing, IslandModel


# ---------------------------------------------------------------------------
# Default configuration — serves as authoritative reference for all keys
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict = {
    # ---- latent dimension ------------------------------------------------
    "k": 20,

    # ---- coevolution -----------------------------------------------------
    "n_generations": 100,       # total EA generations
    "k_random": 3,              # random collaborators added to best

    # ---- parent selection -----------------------------------------------
    "selection": "tournament",  # "tournament" | "rank_roulette"
    "tau": 3,                   # tournament size (only for tournament)

    # ---- crossover -------------------------------------------------------
    "crossover": "uniform",     # "uniform" | "blx_alpha"
    "p_swap": 0.5,              # per-gene swap prob  (uniform only)
    "alpha_blx": 0.5,           # blend factor        (blx_alpha only)

    # ---- mutation --------------------------------------------------------
    "mutation": "gaussian",     # "gaussian" | "uniform_reset"
    "sigma_init": 0.3,          # initial self-adaptive step size
    "sigma_min": 1e-5,          # lower bound on sigma
    "p_reset": 0.1,             # per-gene reset prob (uniform_reset only)
    "reset_low": -0.5,          # reset domain lower bound
    "reset_high": 0.5,          # reset domain upper bound

    # ---- survivor selection ----------------------------------------------
    "survivor_selection": "mu_plus_lambda",  # "mu_plus_lambda" | "mu_comma_lambda"

    # ---- fitness sharing -------------------------------------------------
    "fitness_sharing": False,
    "sigma_share": 1.5,         # niche radius
    "alpha_share": 1.0,         # sharing-function exponent

    # ---- island model ----------------------------------------------------
    "island_model": False,
    "n_islands": 4,             # 3-5 recommended
    "migration_interval": 10,   # migrate every N generations
    "n_migrants": 2,            # individuals copied per island per round

    # ---- initialisation --------------------------------------------------
    "init_type": "uniform",     # "uniform" (U(-0.5,0.5)) | "svd" (SVD + noise)
    "svd_noise": 0.01,          # Gaussian noise std added on top of SVD init

    # ---- logging ---------------------------------------------------------
    "log_every": 1,             # log stats every N generations (1 = every gen)

    # ---- reproducibility -------------------------------------------------
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Main engine class
# ---------------------------------------------------------------------------

class CoevolutionaryEngine:
    """
    Full coevolutionary engine for latent-factor collaborative filtering.

    Parameters
    ----------
    config : dict
        Operator choices and hyperparameters.  Any key missing from ``config``
        falls back to ``DEFAULT_CONFIG``.  Use ``experiments/config.py`` to
        build experiment-specific dicts.
    R_train : np.ndarray, shape (n_users, n_items), dtype float32
        Training rating matrix (0 = unobserved).
    R_test_pairs : np.ndarray or None, shape (n_test, 2), dtype int
        Each row (user_idx, item_idx) — 0-indexed.  If provided, test RMSE
        is computed at every logged generation.
    R_test_ratings : np.ndarray or None, shape (n_test,)
        Ground-truth ratings corresponding to ``R_test_pairs``.
    """

    def __init__(
        self,
        config: Dict,
        R_train: np.ndarray,
        R_test_pairs:   Optional[np.ndarray] = None,
        R_test_ratings: Optional[np.ndarray] = None,
    ) -> None:
        # Merge caller config on top of defaults
        self.cfg = {**DEFAULT_CONFIG, **config}
        self.R_train         = R_train.astype(np.float32)
        self.R_test_pairs    = R_test_pairs
        self.R_test_ratings  = (R_test_ratings.astype(np.float32)
                                if R_test_ratings is not None else None)

        n_users, n_items = R_train.shape
        self.n_users = n_users
        self.n_items = n_items
        self.k       = int(self.cfg["k"])

        self.rng = np.random.default_rng(int(self.cfg["seed"]))

        # ---- Initialise population matrices ----------------------------
        self.U, self.V = self._init_matrices()

        # ---- Self-adaptive step sizes (per individual, per gene) -------
        sigma_init = float(self.cfg["sigma_init"])
        self.sig_U = np.full((n_users, self.k), sigma_init, dtype=np.float32)
        self.sig_V = np.full((n_items, self.k), sigma_init, dtype=np.float32)

        # ---- Fitness arrays (higher = better; range (-inf, 0]) ---------
        self.fit_U = np.full(n_users, -np.inf, dtype=np.float32)
        self.fit_V = np.full(n_items, -np.inf, dtype=np.float32)

        # ---- Logging and best-solution tracking -----------------------
        self._generation_log: List[Dict] = []
        self.best_U:          np.ndarray  = self.U.copy()
        self.best_V:          np.ndarray  = self.V.copy()
        self.best_mean_fit_U: float       = -np.inf
        self.best_test_rmse:  float       = np.inf

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialise U and V according to ``config['init_type']``.

        Returns
        -------
        U : np.ndarray, shape (n_users, k), dtype float32
        V : np.ndarray, shape (n_items, k), dtype float32
        """
        k   = self.k
        rng = self.rng
        init_type = self.cfg["init_type"]

        if init_type == "svd":
            # Truncated SVD — fill unobserved entries with row means first
            R      = self.R_train.copy()
            obs    = R > 0
            row_means = np.where(obs, R, 0).sum(axis=1) / np.maximum(obs.sum(axis=1), 1)
            # Fill zeros row-by-row (avoids 2-D boolean-index assignment)
            R_filled = R.copy()
            for i in range(self.n_users):
                zero_cols = R_filled[i] == 0
                if zero_cols.any():
                    R_filled[i, zero_cols] = float(row_means[i])

            # Economy SVD, keep top-k components
            try:
                U_sv, s_sv, Vt_sv = np.linalg.svd(R_filled, full_matrices=False)
                kk = min(k, len(s_sv))
                U_init = (U_sv[:, :kk] * np.sqrt(s_sv[:kk])).astype(np.float32)
                V_init = (Vt_sv[:kk, :].T * np.sqrt(s_sv[:kk])).astype(np.float32)
                # Pad to k if truncated SVD gave fewer than k features
                if kk < k:
                    pad_u = np.zeros((self.n_users, k - kk), dtype=np.float32)
                    pad_v = np.zeros((self.n_items, k - kk), dtype=np.float32)
                    U_init = np.concatenate([U_init, pad_u], axis=1)
                    V_init = np.concatenate([V_init, pad_v], axis=1)
            except np.linalg.LinAlgError:
                # Fallback to uniform if SVD fails
                U_init = rng.uniform(-0.5, 0.5, (self.n_users, k)).astype(np.float32)
                V_init = rng.uniform(-0.5, 0.5, (self.n_items, k)).astype(np.float32)

            noise = float(self.cfg.get("svd_noise", 0.01))
            U_out = (U_init + rng.normal(0, noise, U_init.shape)).astype(np.float32)
            V_out = (V_init + rng.normal(0, noise, V_init.shape)).astype(np.float32)

        else:  # "uniform"
            U_out = rng.uniform(-0.5, 0.5, (self.n_users, k)).astype(np.float32)
            V_out = rng.uniform(-0.5, 0.5, (self.n_items, k)).astype(np.float32)

        return U_out, V_out

    # ------------------------------------------------------------------
    # Fitness evaluation helpers
    # ------------------------------------------------------------------

    def _eval_U(
        self,
        U_mat: np.ndarray,
        V_mat: np.ndarray,
        fit_V_ref: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate all rows of ``U_mat`` against ``V_mat``.

        Collaborator set = best in V + k_random random V individuals.
        """
        best_v = int(np.argmax(fit_V_ref))
        collabs = select_collaborators(
            len(V_mat), best_v, int(self.cfg["k_random"]), self.rng
        )
        return evaluate_population_U(U_mat, V_mat, self.R_train, collabs)

    def _eval_V(
        self,
        U_mat: np.ndarray,
        V_mat: np.ndarray,
        fit_U_ref: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate all rows of ``V_mat`` against ``U_mat``.
        """
        best_u = int(np.argmax(fit_U_ref))
        collabs = select_collaborators(
            len(U_mat), best_u, int(self.cfg["k_random"]), self.rng
        )
        return evaluate_population_V(U_mat, V_mat, self.R_train, collabs)

    # ------------------------------------------------------------------
    # Batch evolutionary operators
    # ------------------------------------------------------------------

    def _select_partners(self, fitnesses: np.ndarray) -> np.ndarray:
        """
        Select one crossover partner index per individual (vectorised).

        Returns
        -------
        partner_idx : np.ndarray, shape (N,), dtype int
        """
        N = len(fitnesses)
        sel = self.cfg["selection"]
        if sel == "tournament":
            return tournament_selection(
                fitnesses, N, int(self.cfg["tau"]), self.rng
            )
        else:  # rank_roulette
            return rank_roulette_selection(fitnesses, N, self.rng)

    def _batch_crossover(
        self,
        inds: np.ndarray,       # (N, k)
        sigs: np.ndarray,       # (N, k)
        partner_idx: np.ndarray,  # (N,) int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply configured crossover to all N individuals simultaneously.

        Each individual ``inds[i]`` is crossed with ``inds[partner_idx[i]]``.
        Returns two arrays (offspring, offspring_sigmas).

        For sigma recombination, arithmetic mean of parent sigmas is used —
        the standard ES recombination for strategy parameters.
        """
        N, k    = inds.shape
        partners     = inds[partner_idx]           # (N, k)
        partner_sigs = sigs[partner_idx]           # (N, k)

        xover = self.cfg["crossover"]

        if xover == "uniform":
            p_swap    = float(self.cfg.get("p_swap", 0.5))
            swap_mask = self.rng.random((N, k)) < p_swap   # (N, k)
            offspring = np.where(swap_mask, partners, inds).astype(np.float32)

        else:  # blx_alpha
            alpha   = float(self.cfg.get("alpha_blx", 0.5))
            lo      = np.minimum(inds, partners)            # (N, k)
            hi      = np.maximum(inds, partners)            # (N, k)
            extent  = hi - lo                               # I_d
            lo_ext  = lo - alpha * extent
            hi_ext  = hi + alpha * extent
            u       = self.rng.random((N, k)).astype(np.float32)
            offspring = (lo_ext + u * (hi_ext - lo_ext)).astype(np.float32)

        # Sigma: arithmetic mean of the two parents' strategy parameters
        offspring_sigs = (0.5 * (sigs + partner_sigs)).astype(np.float32)

        return offspring, offspring_sigs

    def _batch_mutate(
        self,
        offspring: np.ndarray,       # (N, k)
        offspring_sigs: np.ndarray,  # (N, k)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply configured mutation to all N offspring simultaneously.

        Returns
        -------
        mutated      : np.ndarray (N, k) float32
        mutated_sigs : np.ndarray (N, k) float32
        """
        N, k   = offspring.shape
        mut    = self.cfg["mutation"]

        if mut == "gaussian":
            # Self-adaptive: sigma adapts BEFORE object variables
            tau        = 1.0 / np.sqrt(float(k))
            sigma_min  = float(self.cfg.get("sigma_min", 1e-5))
            noise_sig  = self.rng.standard_normal((N, k)).astype(np.float32)
            new_sigs   = offspring_sigs * np.exp(tau * noise_sig)
            new_sigs   = np.maximum(new_sigs, sigma_min)

            noise_x    = self.rng.standard_normal((N, k)).astype(np.float32)
            mutated    = (offspring + new_sigs * noise_x).astype(np.float32)
            return mutated, new_sigs

        else:  # uniform_reset
            p_reset   = float(self.cfg.get("p_reset", 0.1))
            low       = float(self.cfg.get("reset_low",  -0.5))
            high      = float(self.cfg.get("reset_high",  0.5))
            reset     = self.rng.random((N, k)) < p_reset
            new_vals  = self.rng.uniform(low, high, (N, k)).astype(np.float32)
            mutated   = np.where(reset, new_vals, offspring).astype(np.float32)
            return mutated, offspring_sigs   # sigma unchanged for reset mutation

    # ------------------------------------------------------------------
    # One-generation evolution of a single population (panmictic)
    # ------------------------------------------------------------------

    def _evolve_pop(
        self,
        inds:     np.ndarray,   # (N, k) — modified in-place
        fits:     np.ndarray,   # (N,)   — modified in-place
        sigs:     np.ndarray,   # (N, k) — modified in-place
        eval_fn,                # callable(inds:(N,k)) -> fits:(N,)
    ) -> None:
        """
        Run one complete generation on a single population.

        Steps
        -----
        1. Select crossover partners (one per individual)
        2. Batch crossover → offspring
        3. Batch mutation  → mutated offspring
        4. Evaluate offspring fitness
        5. Survivor selection: update inds/fits/sigs in-place

        With (mu+lambda): position i keeps better of {ind[i], offspring[i]}.
        With (mu,lambda): position i is always replaced by offspring[i].

        Parameters
        ----------
        inds, fits, sigs : modified in-place
        eval_fn          : fitness evaluator for offspring
        """
        # Optionally apply fitness sharing to steer partner selection
        if self.cfg["fitness_sharing"]:
            sel_fits = apply_fitness_sharing(
                fits, inds,
                float(self.cfg["sigma_share"]),
                float(self.cfg["alpha_share"]),
            )
        else:
            sel_fits = fits

        partner_idx   = self._select_partners(sel_fits)
        offspring, offspring_sigs = self._batch_crossover(inds, sigs, partner_idx)
        offspring, offspring_sigs = self._batch_mutate(offspring, offspring_sigs)
        offspring_fits            = eval_fn(offspring)

        surv = self.cfg["survivor_selection"]
        if surv == "mu_plus_lambda":
            # Per-position elitist: keep parent if better than its offspring
            improve_mask = offspring_fits > fits
            inds[improve_mask] = offspring[improve_mask]
            sigs[improve_mask] = offspring_sigs[improve_mask]
            fits[improve_mask] = offspring_fits[improve_mask]
        else:  # mu_comma_lambda — always replace (generational)
            inds[:] = offspring
            sigs[:] = offspring_sigs
            fits[:] = offspring_fits

    # ------------------------------------------------------------------
    # Diversity metrics
    # ------------------------------------------------------------------

    def _diversity(self, inds: np.ndarray, sample: int = 50) -> float:
        """
        Mean pairwise Euclidean distance on a random sample.

        Matches the metric in Population.diversity() from population.py.

        Parameters
        ----------
        inds   : np.ndarray, shape (N, k)
        sample : int — max individuals to sample (default 50)
        """
        N = len(inds)
        s = min(sample, N)
        idx = self.rng.choice(N, size=s, replace=False)
        subset = inds[idx]
        sq  = np.sum(subset ** 2, axis=1)
        D2  = np.maximum(
            sq[:, None] + sq[None, :] - 2.0 * subset @ subset.T, 0.0
        )
        tri = np.triu(np.ones((s, s), dtype=bool), k=1)
        return float(np.sqrt(D2[tri]).mean()) if tri.any() else 0.0

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _make_log_entry(self, gen: int, wall_sec: float) -> Dict:
        """
        Assemble a per-generation log dict from current population state.

        Keys
        ----
        gen, wall_sec,
        best_fit_U, avg_fit_U, diversity_U,
        best_fit_V, avg_fit_V, diversity_V,
        test_rmse (only if test data provided)
        """
        entry: Dict = {
            "gen":         gen,
            "wall_sec":    round(wall_sec, 3),
            # --- User population ---
            "best_fit_U":  float(self.fit_U.max()),
            "avg_fit_U":   float(self.fit_U.mean()),
            "best_rmse_U": float(-self.fit_U.max()),
            "div_U":       self._diversity(self.U),
            # --- Item population ---
            "best_fit_V":  float(self.fit_V.max()),
            "avg_fit_V":   float(self.fit_V.mean()),
            "best_rmse_V": float(-self.fit_V.max()),
            "div_V":       self._diversity(self.V),
        }
        if self.R_test_pairs is not None and self.R_test_ratings is not None:
            entry["test_rmse"] = compute_test_rmse(
                self.U, self.V, self.R_test_pairs, self.R_test_ratings
            )
        return entry

    def _update_best(self) -> None:
        """Snapshot U and V if this is the best generation so far."""
        mean_U = float(self.fit_U.mean())
        if mean_U > self.best_mean_fit_U:
            self.best_mean_fit_U = mean_U
            self.best_U = self.U.copy()
            self.best_V = self.V.copy()

    # ------------------------------------------------------------------
    # Panmictic main loop
    # ------------------------------------------------------------------

    def _run_panmictic(self) -> None:
        """
        Evolve without island model.

        Evaluation order each generation:
            evaluate U -> evolve U -> evaluate V -> evolve V

        This interleaved scheme means U's evolution already uses the latest
        V matrix, and V's evolution sees the freshly-updated U.
        """
        n_gen     = int(self.cfg["n_generations"])
        log_every = int(self.cfg["log_every"])
        t0        = time.perf_counter()

        # --- Initial evaluation ---
        # Bootstrap: fitnesses start at -inf; set to 0 for collab selection
        _fake_fits_V = np.zeros(self.n_items, dtype=np.float32)
        _fake_fits_U = np.zeros(self.n_users, dtype=np.float32)
        self.fit_U = self._eval_U(self.U, self.V, _fake_fits_V)
        self.fit_V = self._eval_V(self.U, self.V, _fake_fits_U)
        self._update_best()
        if 0 % log_every == 0:
            self._generation_log.append(
                self._make_log_entry(0, time.perf_counter() - t0)
            )

        for gen in range(1, n_gen + 1):
            # --- Evolve U ---
            self._evolve_pop(
                self.U, self.fit_U, self.sig_U,
                lambda off: self._eval_U(off, self.V, self.fit_V),
            )

            # --- Re-evaluate V with updated U (collaborator landscape shifts) ---
            self.fit_V = self._eval_V(self.U, self.V, self.fit_U)

            # --- Evolve V ---
            self._evolve_pop(
                self.V, self.fit_V, self.sig_V,
                lambda off: self._eval_V(self.U, off, self.fit_U),
            )

            # --- Re-evaluate U with updated V ---
            self.fit_U = self._eval_U(self.U, self.V, self.fit_V)

            self._update_best()
            if gen % log_every == 0:
                self._generation_log.append(
                    self._make_log_entry(gen, time.perf_counter() - t0)
                )

    # ------------------------------------------------------------------
    # Island model main loop
    # ------------------------------------------------------------------

    def _run_island(self) -> None:
        """
        Evolve with ring-topology island model.

        Both U and V are partitioned identically (round-robin) across
        n_islands sub-populations.  Fitness evaluation always uses the
        *full* opposing matrix for predictions (so all observed ratings
        are covered), but collaborators are drawn exclusively from the
        island-local subset of the other population.

        Every ``migration_interval`` generations, migrants are copied
        between adjacent islands in both U and V island models.
        """
        n_gen     = int(self.cfg["n_generations"])
        n_islands = int(self.cfg["n_islands"])
        mig_int   = int(self.cfg["migration_interval"])
        n_mig     = int(self.cfg["n_migrants"])
        log_every = int(self.cfg["log_every"])
        t0        = time.perf_counter()

        # --- Create island models for U and V ---
        im_U = IslandModel(self.U, self.fit_U, n_islands, sigmas=self.sig_U)
        im_V = IslandModel(self.V, self.fit_V, n_islands, sigmas=self.sig_V)

        # --- Initial evaluation (full populations) ---
        _fake_V = np.zeros(self.n_items, dtype=np.float32)
        _fake_U = np.zeros(self.n_users, dtype=np.float32)
        init_fit_U = self._eval_U(self.U, self.V, _fake_V)
        init_fit_V = self._eval_V(self.U, self.V, _fake_U)
        # Distribute fitnesses to islands
        for isl_idx in range(n_islands):
            orig_u = im_U.get_island(isl_idx)["_orig_indices"]
            orig_v = im_V.get_island(isl_idx)["_orig_indices"]
            im_U.get_island(isl_idx)["fitnesses"][:] = init_fit_U[orig_u]
            im_V.get_island(isl_idx)["fitnesses"][:] = init_fit_V[orig_v]

        # Merge so self.U, self.V, self.fit_U, self.fit_V are current
        self.U, self.fit_U, self.sig_U = im_U.merge()
        self.V, self.fit_V, self.sig_V = im_V.merge()
        self._update_best()
        if 0 % log_every == 0:
            self._generation_log.append(
                self._make_log_entry(0, time.perf_counter() - t0)
            )

        for gen in range(1, n_gen + 1):

            # ---- Evolve each island independently ----------------------
            for isl_idx in range(n_islands):
                isl_u = im_U.get_island(isl_idx)
                isl_v = im_V.get_island(isl_idx)

                U_isl  = isl_u["individuals"]   # (island_size_u, k)
                V_isl  = isl_v["individuals"]   # (island_size_v, k)
                fU_isl = isl_u["fitnesses"]
                fV_isl = isl_v["fitnesses"]
                sU_isl = isl_u["sigmas"] if isl_u["sigmas"] is not None \
                         else np.full_like(U_isl, float(self.cfg["sigma_init"]))
                sV_isl = isl_v["sigmas"] if isl_v["sigmas"] is not None \
                         else np.full_like(V_isl, float(self.cfg["sigma_init"]))

                # Evolve U island — predict against FULL V (all items)
                # but select collaborators from this island's V subset
                def make_eval_U_island(V_isl_ref, fV_isl_ref):
                    def _eval(off):
                        best_v = int(np.argmax(fV_isl_ref))
                        collabs = select_collaborators(
                            len(V_isl_ref), best_v,
                            int(self.cfg["k_random"]), self.rng
                        )
                        return evaluate_population_U(
                            off, self.V, self.R_train, collabs
                        )
                    return _eval

                def make_eval_V_island(U_isl_ref, fU_isl_ref):
                    def _eval(off):
                        best_u = int(np.argmax(fU_isl_ref))
                        collabs = select_collaborators(
                            len(U_isl_ref), best_u,
                            int(self.cfg["k_random"]), self.rng
                        )
                        return evaluate_population_V(
                            self.U, off, self.R_train, collabs
                        )
                    return _eval

                self._evolve_pop(
                    U_isl, fU_isl, sU_isl,
                    make_eval_U_island(V_isl, fV_isl.copy()),
                )
                fV_isl[:] = make_eval_V_island(U_isl, fU_isl)(V_isl)
                self._evolve_pop(
                    V_isl, fV_isl, sV_isl,
                    make_eval_V_island(U_isl, fU_isl.copy()),
                )

                # Write evolved data back into island models
                im_U.set_island(isl_idx, U_isl, fU_isl, sU_isl)
                im_V.set_island(isl_idx, V_isl, fV_isl, sV_isl)

            # ---- Migration step -----------------------------------------
            if gen % mig_int == 0:
                im_U.migrate(n_mig, self.rng)
                im_V.migrate(n_mig, self.rng)

            # ---- Merge islands → update master arrays -------------------
            self.U, self.fit_U, self.sig_U = im_U.merge()
            self.V, self.fit_V, self.sig_V = im_V.merge()

            self._update_best()
            if gen % log_every == 0:
                self._generation_log.append(
                    self._make_log_entry(gen, time.perf_counter() - t0)
                )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """
        Run the full coevolutionary loop and return a results dict.

        Returns
        -------
        dict with keys
            'log'           : list of per-generation dicts (see _make_log_entry)
            'best_U'        : np.ndarray (n_users, k) — best snapshot of U
            'best_V'        : np.ndarray (n_items, k) — best snapshot of V
            'final_fit_U'   : np.ndarray (n_users,)   — final raw fitnesses
            'final_fit_V'   : np.ndarray (n_items,)   — final raw fitnesses
            'final_test_rmse' : float or None
            'config'        : copy of the merged config dict used
            'total_wall_sec': float
        """
        t_start = time.perf_counter()

        if self.cfg["island_model"]:
            self._run_island()
        else:
            self._run_panmictic()

        total_sec = time.perf_counter() - t_start

        final_test_rmse: Optional[float] = None
        if self.R_test_pairs is not None and self.R_test_ratings is not None:
            final_test_rmse = compute_test_rmse(
                self.best_U, self.best_V,
                self.R_test_pairs, self.R_test_ratings,
            )
            self.best_test_rmse = final_test_rmse

        return {
            "log":              self._generation_log,
            "best_U":          self.best_U,
            "best_V":          self.best_V,
            "final_fit_U":     self.fit_U,
            "final_fit_V":     self.fit_V,
            "final_test_rmse": final_test_rmse,
            "config":          deepcopy(self.cfg),
            "total_wall_sec":  round(total_sec, 3),
        }

    def get_best_solution(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the best latent matrices observed during the run.

        Returns
        -------
        U : np.ndarray, shape (n_users, k), dtype float32
        V : np.ndarray, shape (n_items, k), dtype float32
        """
        return self.best_U.copy(), self.best_V.copy()

    def get_logs(self) -> List[Dict]:
        """
        Return the per-generation log list.

        Each dict contains: gen, wall_sec, best_fit_U, avg_fit_U,
        best_rmse_U, div_U, best_fit_V, avg_fit_V, best_rmse_V, div_V,
        and optionally test_rmse.
        """
        return list(self._generation_log)

    def __repr__(self) -> str:
        return (
            f"CoevolutionaryEngine("
            f"n_users={self.n_users}, n_items={self.n_items}, k={self.k}, "
            f"generations={self.cfg['n_generations']}, "
            f"selection={self.cfg['selection']}, "
            f"crossover={self.cfg['crossover']}, "
            f"mutation={self.cfg['mutation']}, "
            f"survivor={self.cfg['survivor_selection']}, "
            f"sharing={self.cfg['fitness_sharing']}, "
            f"islands={self.cfg['island_model']})"
        )


# ---------------------------------------------------------------------------
# Convenience factory — used by batch_runner.py
# ---------------------------------------------------------------------------

def build_engine(
    config: Dict,
    R_train: np.ndarray,
    R_test_pairs:   Optional[np.ndarray] = None,
    R_test_ratings: Optional[np.ndarray] = None,
) -> CoevolutionaryEngine:
    """
    Construct a CoevolutionaryEngine with ``config`` merged over defaults.

    Parameters
    ----------
    config : dict
        Partial or full configuration dict.
    R_train : np.ndarray, shape (n_users, n_items)
    R_test_pairs : np.ndarray or None
    R_test_ratings : np.ndarray or None

    Returns
    -------
    CoevolutionaryEngine
    """
    return CoevolutionaryEngine(config, R_train, R_test_pairs, R_test_ratings)


# ---------------------------------------------------------------------------
# Sanity check — python -X utf8 core/coevo_engine.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 66)
    print("core/coevo_engine.py — sanity check")
    print("=" * 66)

    SEED     = 99
    N_USERS  = 40       # small synthetic problem for speed
    N_ITEMS  = 60
    K        = 8
    DENSITY  = 0.2
    N_GEN    = 8        # short run — just verify the loop works

    rng_data = np.random.default_rng(SEED)

    # ---- Synthetic rating matrix ----------------------------------------
    R = np.zeros((N_USERS, N_ITEMS), dtype=np.float32)
    mask = rng_data.random((N_USERS, N_ITEMS)) < DENSITY
    R[mask] = rng_data.uniform(1, 5, int(mask.sum())).astype(np.float32)
    n_obs = int(mask.sum())
    print(f"\n[Data]  R {R.shape}, observed={n_obs} ({100*n_obs/R.size:.1f}%)")

    # ---- Synthetic test set (15% of observed) ---------------------------
    obs_r, obs_c = np.nonzero(mask)
    n_test  = max(1, int(len(obs_r) * 0.15))
    t_idx   = rng_data.choice(len(obs_r), n_test, replace=False)
    test_pairs   = np.column_stack([obs_r[t_idx], obs_c[t_idx]])
    test_ratings = R[obs_r[t_idx], obs_c[t_idx]]

    base_cfg = dict(k=K, n_generations=N_GEN, k_random=2,
                    seed=SEED, log_every=1, init_type="uniform")

    # -------------------------------------------------------------------
    # Test 1: Panmictic — tournament + uniform XO + gaussian mutation + mu+lambda
    # -------------------------------------------------------------------
    print(f"\n[Test 1] Panmictic | tournament | uniform XO | gaussian | mu+lambda")
    cfg1 = {**base_cfg,
            "selection": "tournament", "tau": 3,
            "crossover": "uniform",    "p_swap": 0.5,
            "mutation":  "gaussian",   "sigma_init": 0.3,
            "survivor_selection": "mu_plus_lambda",
            "fitness_sharing": False,
            "island_model": False}

    eng1 = CoevolutionaryEngine(cfg1, R, test_pairs, test_ratings)
    print(f"  {eng1}")
    res1 = eng1.run()

    log1 = res1["log"]
    assert len(log1) == N_GEN + 1, f"Expected {N_GEN+1} log entries, got {len(log1)}"
    # Fitness should be non-trivially negative
    last = log1[-1]
    assert last["best_fit_U"] <= 0, "Fitness must be <= 0 (negated RMSE)"
    assert last["best_fit_V"] <= 0, "Fitness must be <= 0 (negated RMSE)"
    assert "test_rmse" in last, "test_rmse missing from log"
    assert isinstance(res1["total_wall_sec"], float)

    first_rmse = log1[0].get("test_rmse", float("inf"))
    last_rmse  = log1[-1].get("test_rmse", float("inf"))
    print(f"  test RMSE: gen0={first_rmse:.4f}  ->  gen{N_GEN}={last_rmse:.4f}")
    print(f"  best_fit_U: {last['best_fit_U']:.4f}, avg_fit_U: {last['avg_fit_U']:.4f}")
    print(f"  div_U={last['div_U']:.4f}, div_V={last['div_V']:.4f}")
    print(f"  total_wall_sec={res1['total_wall_sec']:.3f}  OK")

    # best_U / best_V snapshots have correct shape
    bU, bV = eng1.get_best_solution()
    assert bU.shape == (N_USERS, K), f"best_U shape {bU.shape}"
    assert bV.shape == (N_ITEMS, K), f"best_V shape {bV.shape}"
    print(f"  get_best_solution: U={bU.shape}, V={bV.shape}  OK")

    # -------------------------------------------------------------------
    # Test 2: mu,lambda + rank_roulette + BLX-alpha + uniform reset
    # -------------------------------------------------------------------
    print(f"\n[Test 2] Panmictic | rank_roulette | BLX-alpha | uniform_reset | mu,lambda")
    cfg2 = {**base_cfg,
            "selection": "rank_roulette",
            "crossover": "blx_alpha",    "alpha_blx": 0.5,
            "mutation":  "uniform_reset","p_reset": 0.15,
            "reset_low": -1.0, "reset_high": 1.0,
            "survivor_selection": "mu_comma_lambda",
            "fitness_sharing": True,
            "sigma_share": 2.0, "alpha_share": 1.0,
            "island_model": False}

    eng2 = CoevolutionaryEngine(cfg2, R, test_pairs, test_ratings)
    res2 = eng2.run()
    log2 = res2["log"]
    assert len(log2) == N_GEN + 1
    print(f"  test RMSE: gen0={log2[0]['test_rmse']:.4f} -> gen{N_GEN}={log2[-1]['test_rmse']:.4f}")
    print(f"  all log entries have test_rmse: {all('test_rmse' in e for e in log2)}  OK")

    # -------------------------------------------------------------------
    # Test 3: Island model (n_islands=3, migration_interval=3)
    # -------------------------------------------------------------------
    print(f"\n[Test 3] Island model (n_islands=3, migration_interval=3)")
    cfg3 = {**base_cfg, "n_generations": 6,
            "selection": "tournament", "tau": 2,
            "crossover": "uniform",
            "mutation":  "gaussian",
            "survivor_selection": "mu_plus_lambda",
            "fitness_sharing": False,
            "island_model": True,
            "n_islands": 3, "migration_interval": 3, "n_migrants": 1}

    eng3 = CoevolutionaryEngine(cfg3, R, test_pairs, test_ratings)
    res3 = eng3.run()
    log3 = res3["log"]
    assert len(log3) == 7, f"Expected 7 log entries (gen 0..6), got {len(log3)}"
    print(f"  log entries: {len(log3)} (gen 0 to 6)  OK")
    print(f"  test RMSE: gen0={log3[0]['test_rmse']:.4f} -> gen6={log3[-1]['test_rmse']:.4f}")
    print(f"  total_wall_sec={res3['total_wall_sec']:.3f}  OK")

    # -------------------------------------------------------------------
    # Test 4: SVD initialisation
    # -------------------------------------------------------------------
    print(f"\n[Test 4] SVD initialisation")
    cfg4 = {**base_cfg, "init_type": "svd", "n_generations": 3,
            "island_model": False, "fitness_sharing": False,
            "selection": "tournament", "tau": 2,
            "crossover": "uniform", "mutation": "gaussian",
            "survivor_selection": "mu_plus_lambda"}
    eng4 = CoevolutionaryEngine(cfg4, R)
    res4 = eng4.run()
    assert res4["best_U"].shape == (N_USERS, K)
    print(f"  SVD init completed, best_U shape={res4['best_U'].shape}  OK")

    # -------------------------------------------------------------------
    # Test 5: Reproducibility — same seed → same final result
    # -------------------------------------------------------------------
    print(f"\n[Test 5] Reproducibility (same seed)")
    cfg5 = {**base_cfg, "n_generations": 5,
            "island_model": False, "fitness_sharing": False}
    eng5a = CoevolutionaryEngine(cfg5, R)
    eng5b = CoevolutionaryEngine(cfg5, R)
    res5a = eng5a.run()
    res5b = eng5b.run()
    assert np.allclose(res5a["best_U"], res5b["best_U"]), \
        "Different best_U from same seed!"
    assert np.allclose(res5a["best_V"], res5b["best_V"]), \
        "Different best_V from same seed!"
    print(f"  Same seed -> identical best_U and best_V  OK")

    # -------------------------------------------------------------------
    # Test 6: build_engine factory
    # -------------------------------------------------------------------
    print(f"\n[Test 6] build_engine factory")
    eng6 = build_engine({"k": K, "n_generations": 2, "seed": 1}, R)
    res6 = eng6.run()
    assert "log" in res6 and "best_U" in res6
    print(f"  build_engine returned valid results dict  OK")

    # -------------------------------------------------------------------
    # Test 7: get_logs()
    # -------------------------------------------------------------------
    print(f"\n[Test 7] get_logs()")
    logs = eng1.get_logs()
    required_keys = {"gen","wall_sec","best_fit_U","avg_fit_U",
                     "best_rmse_U","div_U","best_fit_V","avg_fit_V",
                     "best_rmse_V","div_V","test_rmse"}
    for row in logs:
        missing = required_keys - row.keys()
        assert not missing, f"Missing keys in log: {missing}"
    print(f"  All {len(logs)} log rows have required keys  OK")
    print(f"  Log columns: {sorted(logs[0].keys())}")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 66)
    print("All 7 engine tests passed  OK")
    print("=" * 66)
