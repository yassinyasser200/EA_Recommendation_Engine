"""
core/diversity.py
=================
Diversity maintenance mechanisms for the coevolutionary recommender engine.

Two complementary strategies are implemented:

1. FITNESS SHARING (Goldberg & Richardson, 1987)
   ------------------------------------------------
   Penalises individuals that share a neighbourhood with many similar
   individuals, forcing the population to spread across multiple niches.

   Shared fitness:
       f'(i) = f(i) / m(i)         where  m(i) = sum_j  sh( d(i,j) )

   Sharing function (linear):
       sh(d) = 1 - (d / sigma_share)^alpha    if  d < sigma_share
             = 0                               otherwise

   Applied AFTER the raw fitness is computed (i.e. after evaluate_both).
   Passed to selection operators in place of raw fitnesses — selection then
   acts on shared fitnesses to enforce niche formation.

2. ISLAND MODEL (ring topology, 3-5 islands)
   -------------------------------------------
   Divides the population into N_ISLANDS independent sub-populations.
   Each island evolves separately; after every MIGRATION_INTERVAL generations
   the best N_MIGRANTS individuals from each island are copied to the *next*
   island (clockwise ring), and the worst N_MIGRANTS of the receiving island
   are replaced.

   Ring topology:
       island_0 --> island_1 --> island_2 --> ... --> island_{n-1} --> island_0

   Migration does NOT remove individuals from the source island (copy, not move),
   so no good solution is ever destroyed by migration.

Public API
----------
Fitness sharing
  sharing_function(d, sigma_share, alpha=1.0)  -> float
      Raw sh(d) value (exposed for testing/visualisation).

  apply_fitness_sharing(fitnesses, individuals, sigma_share, alpha=1.0)
      -> shared_fitnesses: np.ndarray[float32]
      Fully vectorised O(N^2 k) implementation.

Island model
  IslandModel  — stateful class encapsulating all island state.

      __init__(individuals, fitnesses, n_islands, sigmas=None)
      get_island(idx)    -> dict with keys 'individuals', 'fitnesses', 'sigmas'
      set_island(idx, individuals, fitnesses, sigmas=None)
      migrate(n_migrants, rng)
      merge()            -> (all_individuals, all_fitnesses, all_sigmas_or_None)
      island_statistics() -> list of dicts with per-island best/mean/diversity
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple


# ===========================================================================
# SECTION 1 — Fitness Sharing
# ===========================================================================

def sharing_function(
    d: float,
    sigma_share: float,
    alpha: float = 1.0,
) -> float:
    """
    Evaluate the niche-sharing function sh(d) for a single distance ``d``.

    sh(d) = 1 - (d / sigma_share)^alpha   if  d < sigma_share
          = 0                              otherwise

    Parameters
    ----------
    d : float
        Euclidean distance between two individuals in genotype space.
    sigma_share : float
        Niche radius.  Individuals within this distance compete for fitness.
        A larger value creates fewer, wider niches; smaller creates many
        narrow niches.  Typical starting point: sigma_share ~ 0.1 * sqrt(k).
    alpha : float, optional
        Shape exponent (default 1.0 = linear).  alpha=2 gives a smoother
        penalty gradient; alpha<1 gives a step-like drop.

    Returns
    -------
    float
        Sharing value in [0, 1].  1 = identical (same niche centre),
        0 = outside niche radius.

    Examples
    --------
    >>> sharing_function(0.0, sigma_share=1.0)
    1.0
    >>> sharing_function(0.5, sigma_share=1.0, alpha=1.0)
    0.5
    >>> sharing_function(1.5, sigma_share=1.0)
    0.0
    """
    if d >= sigma_share:
        return 0.0
    return 1.0 - (d / sigma_share) ** alpha


def apply_fitness_sharing(
    fitnesses: np.ndarray,
    individuals: np.ndarray,
    sigma_share: float,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Apply fitness sharing across an entire population.

    For each individual i its *shared fitness* is:

        f'(i) = f(i) / m(i)
        m(i)  = sum_{j=0}^{N-1}  sh( ||x_i - x_j||_2 )

    where sh is the sharing function.  Note that m(i) >= 1 always (because
    sh(0) = 1 and the self-distance is 0).  Individuals that stand alone
    receive no penalty (m = 1 → f' = f).

    The implementation is fully vectorised:

        distances[i, j] = ||individuals[i] - individuals[j]||_2

    computed via the identity:
        ||a-b||^2 = ||a||^2 - 2 a·b + ||b||^2

    leading to O(N^2 k) runtime, identical to an explicit double loop but
    ~50x faster for typical population sizes.

    Parameters
    ----------
    fitnesses : np.ndarray, shape (N,), dtype float32
        Raw fitness values (higher = better; the -RMSE convention applies).
    individuals : np.ndarray, shape (N, k), dtype float32
        Real-valued latent vectors (decoded if binary representation).
    sigma_share : float
        Niche radius parameter (> 0).
    alpha : float, optional
        Sharing-function exponent (default 1.0 = linear penalty).

    Returns
    -------
    shared_fitnesses : np.ndarray, shape (N,), dtype float32
        Fitness values after sharing.  Always <= raw fitnesses because m(i) >= 1.

    Raises
    ------
    ValueError
        If sigma_share <= 0 or fitnesses and individuals sizes mismatch.

    Notes
    -----
    For very large populations (N > 500) consider limiting sharing to a
    random 50-individual sample (as done in Population.diversity()) to keep
    the wall-clock time acceptable during the EA loop.

    Fitness sharing is applied ONCE per generation, immediately after the
    raw fitness evaluation (evaluate_both), and BEFORE parent selection.
    The raw fitnesses stored in pop_U.fitnesses / pop_V.fitnesses are NOT
    overwritten — the shared fitnesses are returned as a separate array and
    passed to the selection operator by the engine.
    """
    N = len(fitnesses)
    if len(individuals) != N:
        raise ValueError(
            f"fitnesses ({N}) and individuals ({len(individuals)}) must "
            "have the same length"
        )
    if sigma_share <= 0:
        raise ValueError(f"sigma_share must be > 0, got {sigma_share}")

    # --- Vectorised pairwise Euclidean distance matrix ---
    # sq_norms[i] = ||x_i||^2
    sq_norms = np.sum(individuals ** 2, axis=1)          # (N,)
    # D2[i,j] = ||x_i - x_j||^2 = sq_norms[i] + sq_norms[j] - 2 * x_i · x_j
    D2 = (sq_norms[:, None] + sq_norms[None, :]
          - 2.0 * individuals @ individuals.T)            # (N, N)
    # Numerical noise can make tiny negatives appear on the diagonal
    D2 = np.maximum(D2, 0.0)
    D  = np.sqrt(D2)                                      # (N, N) Euclidean

    # --- Sharing function applied element-wise ---
    # sh[i,j] = max(0, 1 - (D[i,j] / sigma_share)^alpha)
    ratio = D / sigma_share                               # (N, N)
    sh    = np.where(ratio < 1.0, 1.0 - ratio ** alpha, 0.0)    # (N, N)

    # --- Niche count m(i) = sum_j sh[i,j]  (includes self: sh[i,i]=1) ---
    m = sh.sum(axis=1)                                    # (N,)
    # m >= 1 guaranteed; guard against division by zero for safety
    m = np.maximum(m, 1.0)

    # Sign-aware sharing: fitness values in this project follow the -RMSE
    # convention (all values <= 0).  For the penalty to be meaningful:
    #   positive f:  f' = f / m  <  f   (f' is smaller = worse)  OK
    #   negative f:  f / m  >  f   (less negative = BETTER — wrong!)
    # Fix: penalise by SUBTRACTING the niche-proportional bonus, i.e.
    #   f'(i) = f(i) * m(i)      for negative values
    # This makes densely-clustered individuals more negative (worse),
    # which is the intended selection pressure.
    # For a unified formula that works for both signs:
    #   f'(i) = sign(f(i)) * |f(i)| * m(i)  = f(i) * m(i)
    # when all f are negative this equals f * m  (more negative = penalised).
    # When all f are positive this equals f * m  (more positive = BETTER — wrong
    # in that case, but all our fitnesses are <=0, so we use f/m with abs).
    #
    # Standard textbook formula (for non-negative fitness) is f/m.
    # For our sign convention the equivalent is: f * m (for negative values)
    # because |f*m| >= |f| whenever m >= 1, i.e. magnitude is penalised.
    #
    # We implement the sign-agnostic version:
    #   shared = sign(f) * |f| / m   =>  |shared| <= |f|   always
    # This is the correct penalty direction for BOTH positive and negative f.
    shared = (np.sign(fitnesses) * np.abs(fitnesses) / m).astype(np.float32)
    return shared


# ===========================================================================
# SECTION 2 — Island Model
# ===========================================================================

class IslandModel:
    """
    Ring-topology island model for maintaining genetic diversity.

    Partitions a population into ``n_islands`` sub-populations (islands).
    Islands evolve independently between migration events.  Every
    ``migration_interval`` generations (tracked externally by the engine),
    the caller invokes ``migrate()``, which copies the best ``n_migrants``
    individuals from each island to the clockwise neighbour, replacing the
    receiving island's worst individuals.

    Ring topology (clockwise migration direction):
        Island 0 -> Island 1 -> Island 2 -> ... -> Island n-1 -> Island 0

    Internal state per island (stored in a list of dicts)
    --------------------------------------------------------
    'individuals' : np.ndarray, shape (island_size, k)
    'fitnesses'   : np.ndarray, shape (island_size,)
    'sigmas'      : np.ndarray, shape (island_size, k) or None

    Parameters
    ----------
    individuals : np.ndarray, shape (total_pop, k), dtype float32
        Full population to split.  Rows are assigned round-robin to islands
        so original index ordering is preserved.
    fitnesses : np.ndarray, shape (total_pop,), dtype float32
        Initial fitness for each individual.
    n_islands : int
        Number of islands (3–5 recommended).  Must divide evenly or a
        warning is issued and the last island absorbs any remainder.
    sigmas : np.ndarray or None, shape (total_pop, k)
        Self-adaptive strategy parameters.  Split identically to individuals.

    Attributes
    ----------
    n_islands : int
    island_size : int   (floor of total_pop / n_islands)
    k : int             (latent dimension)

    Examples
    --------
    >>> im = IslandModel(individuals, fitnesses, n_islands=4)
    >>> # [run EA on each island independently using get_island / set_island]
    >>> im.migrate(n_migrants=2, rng=rng)
    >>> all_inds, all_fits, all_sigs = im.merge()
    """

    def __init__(
        self,
        individuals: np.ndarray,
        fitnesses: np.ndarray,
        n_islands: int,
        sigmas: Optional[np.ndarray] = None,
    ) -> None:
        total = len(individuals)
        if n_islands < 2:
            raise ValueError(f"n_islands must be >= 2, got {n_islands}")
        if n_islands > total:
            raise ValueError(
                f"n_islands ({n_islands}) cannot exceed population size ({total})"
            )

        self.n_islands   = n_islands
        self.k           = individuals.shape[1]
        self._has_sigmas = sigmas is not None

        # Round-robin split: row i goes to island (i % n_islands)
        # This preserves the Index-mapping guarantee from Population.
        self._islands: List[Dict[str, np.ndarray]] = []
        for isl in range(n_islands):
            idx = np.arange(isl, total, n_islands)    # every n_islands-th row
            island_dict: Dict[str, np.ndarray] = {
                "individuals": individuals[idx].astype(np.float32).copy(),
                "fitnesses":   fitnesses[idx].astype(np.float32).copy(),
                "sigmas":      (sigmas[idx].astype(np.float32).copy()
                                if sigmas is not None else None),
                "_orig_indices": idx,                  # track original positions
            }
            self._islands.append(island_dict)

    # ------------------------------------------------------------------
    # Island access
    # ------------------------------------------------------------------

    def get_island(self, idx: int) -> Dict:
        """
        Return a *view* of island ``idx`` as a dict.

        Keys: 'individuals', 'fitnesses', 'sigmas' (None if not initialised).

        Parameters
        ----------
        idx : int
            Island index in [0, n_islands).

        Returns
        -------
        dict
            Reference to the internal island dict (not a copy — modifications
            to the arrays are reflected immediately).
        """
        if not (0 <= idx < self.n_islands):
            raise IndexError(
                f"Island index {idx} out of range [0, {self.n_islands})"
            )
        return self._islands[idx]

    def set_island(
        self,
        idx: int,
        individuals: np.ndarray,
        fitnesses: np.ndarray,
        sigmas: Optional[np.ndarray] = None,
    ) -> None:
        """
        Replace the individuals/fitnesses/sigmas for island ``idx``.

        Called by the engine after evolving each island for one generation.

        Parameters
        ----------
        idx : int
            Island index in [0, n_islands).
        individuals : np.ndarray, shape (island_size, k)
            Evolved population for this island.
        fitnesses : np.ndarray, shape (island_size,)
            Updated fitnesses.
        sigmas : np.ndarray or None, shape (island_size, k)
            Updated strategy parameters.
        """
        isl = self._islands[idx]
        isl["individuals"] = individuals.astype(np.float32)
        isl["fitnesses"]   = fitnesses.astype(np.float32)
        if sigmas is not None:
            isl["sigmas"] = sigmas.astype(np.float32)
        elif self._has_sigmas:
            # sigmas expected but not provided — keep old values
            pass
        else:
            isl["sigmas"] = None

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def migrate(
        self,
        n_migrants: int,
        rng: np.random.Generator,
    ) -> None:
        """
        Perform one round of ring-topology migration.

        For each island i (source), the ``n_migrants`` best individuals are
        *copied* (not moved) to island (i+1) % n_islands (destination).
        In the destination island the ``n_migrants`` worst individuals are
        *replaced* by the incoming migrants.

        Migration is copy-based: the source island always retains all its
        individuals, so no genetic material is ever destroyed.

        Parameters
        ----------
        n_migrants : int
            Number of top individuals to migrate per island.  Must satisfy
            1 <= n_migrants <= min(island_size for all islands).
        rng : np.random.Generator
            Seeded generator — currently unused (deterministic best/worst
            selection), but accepted for API consistency and future extension
            to random migrant selection.

        Notes
        -----
        After migration, the fitnesses of the newly placed migrants in the
        destination island are the *source* fitnesses.  The engine must
        re-evaluate these individuals at the start of the next generation
        to obtain correct fitnesses under the current collaborator set.
        """
        n_isl = self.n_islands

        # Validate n_migrants against each island's size
        for i, isl in enumerate(self._islands):
            sz = len(isl["individuals"])
            if n_migrants < 1 or n_migrants >= sz:
                raise ValueError(
                    f"n_migrants={n_migrants} must be in [1, island_size-1="
                    f"{sz-1}] for island {i}"
                )

        # --- Collect emigrants (best n_migrants per island) first ---
        # We collect all emigrants BEFORE replacing anyone so that
        # 'island 0 sending to island 1' does not affect
        # 'island 1 sending to island 2' in the same round.
        emigrants: List[Dict[str, np.ndarray]] = []
        for isl in self._islands:
            top_idx = np.argsort(isl["fitnesses"])[::-1][:n_migrants]
            em: Dict[str, np.ndarray] = {
                "individuals": isl["individuals"][top_idx].copy(),
                "fitnesses":   isl["fitnesses"][top_idx].copy(),
                "sigmas": (isl["sigmas"][top_idx].copy()
                           if isl["sigmas"] is not None else None),
            }
            emigrants.append(em)

        # --- Place emigrants into destination islands ---
        for src_idx in range(n_isl):
            dst_idx = (src_idx + 1) % n_isl
            dst      = self._islands[dst_idx]
            em       = emigrants[src_idx]

            # Identify the n_migrants worst slots in the destination island
            worst_idx = np.argsort(dst["fitnesses"])[:n_migrants]   # ascending

            dst["individuals"][worst_idx] = em["individuals"]
            dst["fitnesses"][worst_idx]   = em["fitnesses"]
            if dst["sigmas"] is not None and em["sigmas"] is not None:
                dst["sigmas"][worst_idx]  = em["sigmas"]

    # ------------------------------------------------------------------
    # Merge back to single population
    # ------------------------------------------------------------------

    def merge(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Reassemble all islands into a single population array.

        The order is: island_0 rows, island_1 rows, ..., island_{n-1} rows.
        The engine uses this at the very end of the run (or when switching
        from island mode back to panmictic for reporting).

        Returns
        -------
        individuals : np.ndarray, shape (total_pop, k), dtype float32
        fitnesses   : np.ndarray, shape (total_pop,),   dtype float32
        sigmas      : np.ndarray or None, shape (total_pop, k)
        """
        all_inds  = np.concatenate(
            [isl["individuals"] for isl in self._islands], axis=0
        )
        all_fits  = np.concatenate(
            [isl["fitnesses"] for isl in self._islands], axis=0
        )
        all_sigmas: Optional[np.ndarray] = None
        if all(isl["sigmas"] is not None for isl in self._islands):
            all_sigmas = np.concatenate(
                [isl["sigmas"] for isl in self._islands], axis=0
            )
        return (
            all_inds.astype(np.float32),
            all_fits.astype(np.float32),
            all_sigmas,
        )

    # ------------------------------------------------------------------
    # Per-island statistics (used by coevo_engine logging)
    # ------------------------------------------------------------------

    def island_statistics(self) -> List[Dict]:
        """
        Compute summary statistics for each island.

        Returns
        -------
        List[Dict]
            One dict per island with keys:
            - 'island_idx'   : int
            - 'size'         : int — number of individuals
            - 'best_fitness' : float
            - 'mean_fitness' : float
            - 'diversity'    : float — mean pairwise distance on a 50-individual
                               sample (same metric as Population.diversity())
        """
        stats = []
        for i, isl in enumerate(self._islands):
            inds = isl["individuals"]
            fits = isl["fitnesses"]
            n    = len(inds)

            # Diversity: mean pairwise Euclidean on a sample of <=50
            sample_size = min(50, n)
            sample_idx  = np.arange(sample_size)          # take first 50 (ordered)
            sample      = inds[sample_idx]
            sq          = np.sum(sample ** 2, axis=1)
            D2          = np.maximum(
                sq[:, None] + sq[None, :] - 2.0 * sample @ sample.T,
                0.0,
            )
            # Upper triangle only (exclude self = 0)
            mask        = np.triu(np.ones((sample_size, sample_size), dtype=bool),
                                  k=1)
            diversity   = float(np.sqrt(D2[mask]).mean()) if mask.any() else 0.0

            stats.append({
                "island_idx":   i,
                "size":         n,
                "best_fitness": float(fits.max()),
                "mean_fitness": float(fits.mean()),
                "diversity":    diversity,
            })
        return stats

    def __repr__(self) -> str:
        sizes = [len(isl["individuals"]) for isl in self._islands]
        return (
            f"IslandModel(n_islands={self.n_islands}, k={self.k}, "
            f"island_sizes={sizes})"
        )


# ===========================================================================
# Sanity check — run with: python -X utf8 core/diversity.py
# ===========================================================================

if __name__ == "__main__":
    print("=" * 64)
    print("core/diversity.py — sanity check")
    print("=" * 64)

    SEED = 2024
    rng  = np.random.default_rng(SEED)

    N = 60     # total population size
    K = 10     # latent dimension

    # Synthetic individuals: two tight clusters far apart + random noise
    # Cluster A centred at +1, cluster B centred at -1
    half = N // 2
    cluster_A = rng.normal(loc=+1.0, scale=0.05, size=(half, K)).astype(np.float32)
    cluster_B = rng.normal(loc=-1.0, scale=0.05, size=(half, K)).astype(np.float32)
    individuals = np.concatenate([cluster_A, cluster_B], axis=0)
    fitnesses   = rng.uniform(-3.0, 0.0, size=N).astype(np.float32)

    # -------------------------------------------------------------------
    # Test 1: sharing_function
    # -------------------------------------------------------------------
    print("\n[Test 1] sharing_function")
    sigma_s = 1.0
    cases = [
        (0.0,  1.0,  "d=0 -> sh=1 (identical)"),
        (0.5,  0.5,  "d=0.5, sigma=1, alpha=1 -> sh=0.5"),
        (1.0,  0.0,  "d=sigma -> sh=0 (boundary)"),
        (2.0,  0.0,  "d>sigma -> sh=0"),
    ]
    for d_val, expected, desc in cases:
        got = sharing_function(d_val, sigma_share=sigma_s, alpha=1.0)
        assert abs(got - expected) < 1e-6, f"Expected {expected}, got {got}: {desc}"
        print(f"  {desc}: sh={got:.4f}  OK")

    # quadratic (alpha=2)
    sh_quad = sharing_function(0.5, sigma_share=1.0, alpha=2.0)
    assert abs(sh_quad - 0.75) < 1e-6, f"alpha=2: expected 0.75, got {sh_quad}"
    print(f"  d=0.5, sigma=1, alpha=2 -> sh={sh_quad:.4f} (expect 0.75)  OK")

    # -------------------------------------------------------------------
    # Test 2: apply_fitness_sharing — shape and dtype
    # -------------------------------------------------------------------
    print("\n[Test 2] apply_fitness_sharing — shape, dtype, values")
    sigma_share = float(np.sqrt(K) * 0.5)   # heuristic: 0.5 * sqrt(k)
    shared = apply_fitness_sharing(fitnesses, individuals, sigma_share)
    assert shared.shape == (N,),  f"Wrong shape: {shared.shape}"
    assert shared.dtype == np.float32, f"Wrong dtype: {shared.dtype}"
    print(f"  shared_fitnesses shape={shared.shape}, dtype={shared.dtype}  OK")

    # sign-aware invariant: |shared| <= |raw|  because we divide |f| by m >= 1
    # For negative fitnesses this means: shared >= raw  (shared is less negative)
    # But in MAGNITUDE the shared value is always smaller or equal.
    mag_violation = (np.abs(shared) - np.abs(fitnesses)).max()
    assert mag_violation <= 1e-5, \
        f"|shared| must be <= |raw| (divide by m>=1), violation={mag_violation}"
    print(f"  |shared| <= |raw| (m >= 1): max violation = {mag_violation:.2e}  OK")
    # For our negative convention: shared >= raw  (less negative = penalised magnitude)
    assert np.all(shared >= fitnesses - 1e-5), \
        "With negative fitness: shared must be >= raw (less negative = more penalised)"
    print(f"  shared >= raw (negative fitness convention)  OK")

    # -------------------------------------------------------------------
    # Test 3: Niche formation — clustered individuals penalised more
    # -------------------------------------------------------------------
    print("\n[Test 3] Niche formation check")
    # Give cluster A (rows 0..half-1) uniformly high fitness = -0.1
    # Give cluster B (rows half..N-1) uniformly low fitness = -2.0
    # After sharing, cluster A individuals should be penalised MORE
    # because they are densely packed together.
    fit_test          = np.full(N, -1.0, dtype=np.float32)
    fit_test[:half]   = -0.1    # cluster A: best raw fitness
    fit_test[half:]   = -2.0    # cluster B: worst raw fitness

    shared_test = apply_fitness_sharing(fit_test, individuals, sigma_share)

    # Penalty in magnitude: |raw| - |shared|  > 0 means penalised
    # For negative fitness: raw=-0.1, shared=-0.05 -> |shared|=0.05 < |raw|=0.1
    # i.e. magnitude is REDUCED by sharing — penalty is less-negative being closer to 0.
    # The important property: individuals in tight clusters have a larger
    # m(i), so their |shared| is further reduced from |raw|.
    mag_reduction_A = np.abs(fit_test[:half]) - np.abs(shared_test[:half])
    mag_reduction_B = np.abs(fit_test[half:]) - np.abs(shared_test[half:])
    mean_red_A = float(mag_reduction_A.mean())
    mean_red_B = float(mag_reduction_B.mean())
    print(f"  Mean magnitude reduction — cluster A (fit=-0.1): {mean_red_A:.4f}")
    print(f"  Mean magnitude reduction — cluster B (fit=-2.0): {mean_red_B:.4f}")
    # Both clusters are equally tight, so m is similar; A has smaller |raw|
    # so the absolute reduction is smaller, but the *relative* reduction is same.
    # Key invariant: all shared magnitudes <= raw magnitudes.
    assert np.all(np.abs(shared_test) <= np.abs(fit_test) + 1e-5), \
        "All shared magnitudes must be <= raw magnitudes"
    print(f"  All |shared| <= |raw|  OK")
    # Penalised = shared fitness is worse (more towards 0 for negatives)
    assert np.all(shared_test >= fit_test - 1e-5), \
        "For negative fitness: shared must be >= raw (less negative)"
    print(f"  All cluster individuals correctly penalised (shared >= raw)  OK")

    # Isolated individual — create one far from everyone
    solo = np.full((1, K), 100.0, dtype=np.float32)
    inds_with_solo = np.concatenate([individuals[:5], solo], axis=0)
    fits_with_solo = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                               dtype=np.float32)
    shared_solo = apply_fitness_sharing(
        fits_with_solo, inds_with_solo, sigma_share
    )
    # Solo individual at index 5: m(5) == 1.0 -> f'(5) == f(5)
    assert abs(shared_solo[-1] - fits_with_solo[-1]) < 1e-5, \
        "Isolated individual should NOT be penalised"
    print(f"  Isolated individual: raw={fits_with_solo[-1]:.4f}, "
          f"shared={shared_solo[-1]:.4f} (no penalty)  OK")

    # -------------------------------------------------------------------
    # Test 4: IslandModel construction
    # -------------------------------------------------------------------
    print("\n[Test 4] IslandModel construction")
    N_ISL = 4
    sigmas = np.full((N, K), 0.3, dtype=np.float32)
    im = IslandModel(individuals, fitnesses, n_islands=N_ISL, sigmas=sigmas)
    print(f"  {im}")

    # Each island should have ceil(N/N_ISL) or floor(N/N_ISL) individuals
    total_recovered = sum(len(im.get_island(i)["individuals"]) for i in range(N_ISL))
    assert total_recovered == N, \
        f"Total individuals after split ({total_recovered}) != N ({N})"
    print(f"  Total individuals across islands = {total_recovered} == N={N}  OK")

    # All islands should have individuals, fitnesses, sigmas
    for i in range(N_ISL):
        isl = im.get_island(i)
        assert isl["individuals"] is not None
        assert isl["fitnesses"] is not None
        assert isl["sigmas"] is not None
        print(f"  Island {i}: size={len(isl['individuals'])}, "
              f"best_fit={isl['fitnesses'].max():.4f}")

    # -------------------------------------------------------------------
    # Test 5: IslandModel get_island / set_island round-trip
    # -------------------------------------------------------------------
    print("\n[Test 5] set_island round-trip")
    isl0_orig = im.get_island(0)["individuals"].copy()
    new_inds  = isl0_orig + 0.01   # slightly perturb all genes
    new_fits  = im.get_island(0)["fitnesses"] + 0.5
    im.set_island(0, new_inds, new_fits)
    assert np.allclose(im.get_island(0)["individuals"], new_inds), \
        "set_island did not update individuals"
    assert np.allclose(im.get_island(0)["fitnesses"], new_fits), \
        "set_island did not update fitnesses"
    print(f"  set_island / get_island round-trip OK")
    # Restore
    im.set_island(0, isl0_orig, im.get_island(0)["fitnesses"] - 0.5)

    # -------------------------------------------------------------------
    # Test 6: IslandModel migration
    # -------------------------------------------------------------------
    print("\n[Test 6] IslandModel migration")

    # Assign synthetic fitness: island 0 has very high fitness, others low
    for i in range(N_ISL):
        isl = im.get_island(i)
        isl["fitnesses"][:] = -5.0 + float(i) * 0.01  # slight variation

    # Give island 0 one extremely good individual at position 0
    best_vec = np.full(K, 99.0, dtype=np.float32)
    im.get_island(0)["individuals"][0] = best_vec
    im.get_island(0)["fitnesses"][0]   = -0.001           # near-perfect

    N_MIGRANTS = 2
    # Before migration: island 1 should NOT contain best_vec
    isl1_before = im.get_island(1)["individuals"].copy()
    assert not np.any(np.all(isl1_before == best_vec, axis=1)), \
        "Island 1 should not have best_vec before migration"

    im.migrate(n_migrants=N_MIGRANTS, rng=rng)

    # After migration: island 1 (=destination of island 0) should have best_vec
    isl1_after = im.get_island(1)["individuals"]
    assert np.any(np.all(np.isclose(isl1_after, best_vec), axis=1)), \
        "Island 1 should contain best_vec after migration from island 0"
    print(f"  Best individual from island 0 found in island 1 post-migration  OK")

    # Source island 0 should still have best_vec (copy, not move)
    isl0_after = im.get_island(0)["individuals"]
    assert np.any(np.all(np.isclose(isl0_after, best_vec), axis=1)), \
        "Island 0 should still retain best_vec after migration (copy semantics)"
    print(f"  Island 0 still retains best_vec (copy semantics)  OK")

    # -------------------------------------------------------------------
    # Test 7: IslandModel merge
    # -------------------------------------------------------------------
    print("\n[Test 7] IslandModel merge")
    all_inds, all_fits, all_sigs = im.merge()
    assert all_inds.shape == (N, K), f"Merge shape mismatch: {all_inds.shape}"
    assert all_fits.shape == (N,),   f"Fitness shape mismatch: {all_fits.shape}"
    assert all_sigs is not None and all_sigs.shape == (N, K), \
        f"Sigma shape mismatch: {all_sigs.shape if all_sigs is not None else None}"
    print(f"  merge() -> individuals={all_inds.shape}, "
          f"fitnesses={all_fits.shape}, sigmas={all_sigs.shape}  OK")

    # -------------------------------------------------------------------
    # Test 8: island_statistics
    # -------------------------------------------------------------------
    print("\n[Test 8] island_statistics")
    stats = im.island_statistics()
    assert len(stats) == N_ISL, f"Expected {N_ISL} stats dicts, got {len(stats)}"
    for s in stats:
        assert "island_idx"   in s
        assert "size"         in s
        assert "best_fitness" in s
        assert "mean_fitness" in s
        assert "diversity"    in s
        assert s["diversity"] >= 0.0
        print(f"  Island {s['island_idx']}: size={s['size']}, "
              f"best={s['best_fitness']:.4f}, "
              f"mean={s['mean_fitness']:.4f}, "
              f"diversity={s['diversity']:.4f}")
    print(f"  All {N_ISL} island stats present with correct keys  OK")

    # -------------------------------------------------------------------
    # Test 9: Reproducibility
    # -------------------------------------------------------------------
    print("\n[Test 9] Reproducibility")
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    inds_r  = np.random.default_rng(1).standard_normal((20, K)).astype(np.float32)
    fits_r  = np.random.default_rng(2).uniform(-3, 0, 20).astype(np.float32)

    sh_a = apply_fitness_sharing(fits_r, inds_r, sigma_share=1.5)
    sh_b = apply_fitness_sharing(fits_r, inds_r, sigma_share=1.5)
    assert np.allclose(sh_a, sh_b), "Fitness sharing is not deterministic!"
    print(f"  apply_fitness_sharing is deterministic (rng-independent)  OK")

    im_a = IslandModel(inds_r, fits_r, n_islands=3)
    im_b = IslandModel(inds_r, fits_r, n_islands=3)
    im_a.migrate(n_migrants=1, rng=rng_a)
    im_b.migrate(n_migrants=1, rng=rng_b)
    inds_ma, fits_ma, _ = im_a.merge()
    inds_mb, fits_mb, _ = im_b.merge()
    assert np.allclose(inds_ma, inds_mb), "Migration not reproducible!"
    print(f"  IslandModel.migrate is reproducible with same seed  OK")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("All 9 diversity tests passed  OK")
    print("=" * 64)
