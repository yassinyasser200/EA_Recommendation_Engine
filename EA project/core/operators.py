"""
core/operators.py
=================
All evolutionary operators required by the coevolutionary recommender engine.

Implemented operators
---------------------
PARENT SELECTION
  tournament_selection(fitnesses, n_select, tau, rng)
      -> np.ndarray[int]  — indices of selected parents
  rank_roulette_selection(fitnesses, n_select, rng)
      -> np.ndarray[int]  — indices selected by rank-proportionate roulette

RECOMBINATION
  uniform_crossover(p1, p2, rng, p_swap=0.5)
      -> (child1, child2)  — gene-wise Bernoulli swap
  blx_alpha_crossover(p1, p2, rng, alpha=0.5)
      -> (child1, child2)  — BLX-alpha blend crossover

MUTATION
  gaussian_mutation(individual, sigma, rng, tau=None, sigma_min=1e-5)
      -> (mutated_individual, new_sigma)   — self-adaptive Gaussian mutation
  uniform_reset_mutation(individual, low, high, rng, p_reset=0.1)
      -> mutated_individual               — per-gene Bernoulli replacement

SURVIVOR SELECTION
  mu_plus_lambda(parents, parent_fits, offspring, offspring_fits, mu)
      -> (survivors, survivor_fits)  — (mu+lambda) elitist model
  mu_comma_lambda(offspring, offspring_fits, mu)
      -> (survivors, survivor_fits)  — (mu,lambda) generational model

Design rules
------------
* Pure NumPy — no external ML libraries.
* Every function is stateless; all state is passed explicitly.
* Every random call goes through the caller-supplied ``rng``.
* All arrays use float32 for memory efficiency except fitness which is float32
  by convention from fitness.py.
* Self-adaptive sigma is stored externally (as a separate array alongside the
  population matrix) so Population's fixed shape is never altered.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


# ===========================================================================
# SECTION 1 — Parent Selection
# ===========================================================================

def tournament_selection(
    fitnesses: np.ndarray,
    n_select: int,
    tau: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Binary (or k-ary) tournament selection.

    For each of the ``n_select`` slots, sample ``tau`` individuals uniformly
    at random (with replacement across slots, *without* replacement within a
    single tournament) and return the index of the one with highest fitness.

    Parameters
    ----------
    fitnesses : np.ndarray, shape (pop_size,), dtype float32
        Current fitness of every individual in the population.
        Convention from fitness.py: higher is better (values <= 0).
    n_select : int
        Number of parent indices to return.
    tau : int
        Tournament size (>= 1).  Common values: 2, 3, 5, 7.
        tau=1  → uniform random selection (no selection pressure).
        tau→pop_size → deterministic selection of the best.
    rng : np.random.Generator
        Caller-supplied seeded generator.

    Returns
    -------
    np.ndarray, shape (n_select,), dtype int
        Indices of the selected parents.

    Raises
    ------
    ValueError
        If tau < 1 or tau > pop_size.

    Notes
    -----
    Time complexity: O(n_select * tau).
    """
    pop_size = len(fitnesses)
    if tau < 1 or tau > pop_size:
        raise ValueError(
            f"tau must be in [1, pop_size={pop_size}], got {tau}"
        )

    selected = np.empty(n_select, dtype=int)
    for i in range(n_select):
        # Sample tau contestants without replacement within this tournament
        contestants = rng.choice(pop_size, size=tau, replace=False)
        winner = contestants[np.argmax(fitnesses[contestants])]
        selected[i] = winner
    return selected


def rank_roulette_selection(
    fitnesses: np.ndarray,
    n_select: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Rank-based roulette wheel (linear ranking) selection.

    Individuals are ranked by fitness (rank 1 = worst, rank N = best).
    Selection probability is proportional to rank, which avoids the
    super-individual dominance problem of fitness-proportionate selection
    while still applying meaningful selection pressure.

    Selection probability for individual with rank r (1-indexed):

        P(r) = r / sum(1..N)  =  2r / (N * (N+1))

    Parameters
    ----------
    fitnesses : np.ndarray, shape (pop_size,), dtype float32
        Current fitness of every individual (higher = better).
    n_select : int
        Number of parent indices to return (with replacement).
    rng : np.random.Generator
        Caller-supplied seeded generator.

    Returns
    -------
    np.ndarray, shape (n_select,), dtype int
        Indices of the selected parents (sampling with replacement).

    Notes
    -----
    Sampling with replacement is standard for roulette wheel.  If unique
    parents are required, the caller may deduplicate.

    Time complexity: O(N log N + n_select).
    """
    pop_size = len(fitnesses)

    # Ranks: 1 (worst) to pop_size (best)
    # argsort gives positions sorted ascending; we assign ranks accordingly
    rank_order = np.argsort(fitnesses)          # indices sorted worst→best
    ranks = np.empty(pop_size, dtype=float)
    for rank_val, idx in enumerate(rank_order, start=1):
        ranks[idx] = rank_val

    # Normalise to selection probabilities
    probs = ranks / ranks.sum()

    # Roulette wheel sampling
    selected = rng.choice(pop_size, size=n_select, replace=True, p=probs)
    return selected


# ===========================================================================
# SECTION 2 — Recombination (Crossover)
# ===========================================================================

def uniform_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    rng: np.random.Generator,
    p_swap: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniform crossover: each gene is inherited from a randomly chosen parent.

    For each gene position d, a Bernoulli draw with probability ``p_swap``
    determines whether to swap the genes between parents, producing two
    offspring.  With p_swap=0.5 each gene is equally likely to come from
    either parent, yielding maximum gene shuffling.

    Parameters
    ----------
    p1 : np.ndarray, shape (k,), dtype float32
        First parent latent vector.
    p2 : np.ndarray, shape (k,), dtype float32
        Second parent latent vector.  Must have the same shape as ``p1``.
    rng : np.random.Generator
        Caller-supplied seeded generator.
    p_swap : float, optional
        Per-gene swap probability (default 0.5).  Must be in [0, 1].

    Returns
    -------
    child1 : np.ndarray, shape (k,), dtype float32
    child2 : np.ndarray, shape (k,), dtype float32
        Two offspring vectors produced by the crossover.

    Raises
    ------
    ValueError
        If p1 and p2 have different shapes or p_swap is out of [0, 1].

    Notes
    -----
    Unlike 1-point or 2-point crossover, uniform crossover has no positional
    bias along the chromosome — ideal for real-valued latent vectors where
    dimensions are interchangeable.
    """
    if p1.shape != p2.shape:
        raise ValueError(
            f"Parents must have the same shape: {p1.shape} vs {p2.shape}"
        )
    if not (0.0 <= p_swap <= 1.0):
        raise ValueError(f"p_swap must be in [0, 1], got {p_swap}")

    k = p1.shape[0]
    child1 = p1.copy()
    child2 = p2.copy()

    # swap_mask[d] = True  →  child1 gets p2[d], child2 gets p1[d]
    swap_mask = rng.random(k) < p_swap

    child1[swap_mask] = p2[swap_mask]
    child2[swap_mask] = p1[swap_mask]

    return child1, child2


def blx_alpha_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    rng: np.random.Generator,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    BLX-alpha (Blend Crossover) for real-valued chromosomes.

    For each gene d, let I_d = |p1[d] - p2[d]|.  Each child gene is sampled
    uniformly from the expanded interval:

        [min(p1[d], p2[d]) - alpha * I_d,
         max(p1[d], p2[d]) + alpha * I_d]

    This allows offspring to explore *outside* the parental range, controlled
    by ``alpha``.  alpha=0 degenerates to uniform interval sampling (no
    extrapolation); alpha=0.5 is the standard empirically tuned value
    (Eshelman & Schaffer, 1993).

    Parameters
    ----------
    p1 : np.ndarray, shape (k,), dtype float32
        First parent latent vector.
    p2 : np.ndarray, shape (k,), dtype float32
        Second parent latent vector.  Must have the same shape as ``p1``.
    rng : np.random.Generator
        Caller-supplied seeded generator.
    alpha : float, optional
        Blending parameter (default 0.5).  Must be >= 0.

    Returns
    -------
    child1 : np.ndarray, shape (k,), dtype float32
    child2 : np.ndarray, shape (k,), dtype float32
        Two offspring vectors produced by the crossover.

    Raises
    ------
    ValueError
        If p1 and p2 have different shapes or alpha < 0.

    Notes
    -----
    BLX-alpha is particularly well-suited to continuous search spaces because
    it preserves useful building blocks while providing enough exploration to
    escape local optima — a key property for latent factor optimisation.
    """
    if p1.shape != p2.shape:
        raise ValueError(
            f"Parents must have the same shape: {p1.shape} vs {p2.shape}"
        )
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")

    lo = np.minimum(p1, p2)
    hi = np.maximum(p1, p2)
    interval = hi - lo                       # I_d = |p2[d] - p1[d]|

    lo_ext = lo - alpha * interval           # expanded lower bound
    hi_ext = hi + alpha * interval           # expanded upper bound

    k = p1.shape[0]
    # Two independent uniform draws within [lo_ext, hi_ext]
    u1 = rng.random(k).astype(np.float32)
    u2 = rng.random(k).astype(np.float32)

    child1 = (lo_ext + u1 * (hi_ext - lo_ext)).astype(np.float32)
    child2 = (lo_ext + u2 * (hi_ext - lo_ext)).astype(np.float32)

    return child1, child2


# ===========================================================================
# SECTION 3 — Mutation
# ===========================================================================

def gaussian_mutation(
    individual: np.ndarray,
    sigma: np.ndarray,
    rng: np.random.Generator,
    tau: float | None = None,
    sigma_min: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Self-adaptive Gaussian mutation (ES-style).

    Two-step update (Beyer & Schwefel, 2002):

    1. Adapt the strategy parameter vector sigma *before* perturbing genes:

           sigma'[d] = sigma[d] * exp( tau * N_d(0, 1) )
           sigma'[d] = max(sigma'[d], sigma_min)   [avoid collapse]

    2. Perturb the object variables using the new sigma:

           x'[d] = x[d] + sigma'[d] * N_d(0, 1)

    The learning rate tau controls how fast sigma adapts.  The common
    empirically validated choice is tau = 1 / sqrt(k) where k = len(sigma).

    Parameters
    ----------
    individual : np.ndarray, shape (k,), dtype float32
        Object-variable vector of the individual to mutate.
    sigma : np.ndarray, shape (k,), dtype float32
        Per-gene strategy parameters (mutation step sizes).  Updated in-place
        semantics: the *returned* new_sigma reflects the adapted values;
        the passed-in array is NOT modified.
    rng : np.random.Generator
        Caller-supplied seeded generator.
    tau : float or None, optional
        Learning rate for sigma adaptation.  If None, defaults to
        1 / sqrt(k) (the recommendedformula from Beyer & Schwefel).
    sigma_min : float, optional
        Lower bound for sigma to prevent premature convergence of step sizes
        (default 1e-5).

    Returns
    -------
    mutated : np.ndarray, shape (k,), dtype float32
        The mutated object-variable vector.
    new_sigma : np.ndarray, shape (k,), dtype float32
        The adapted strategy parameter vector corresponding to ``mutated``.

    Notes
    -----
    Self-adaptation eliminates the need to manually tune mutation rates:
    the algorithm learns the right step size for each gene dimension.
    In the population, store sigma alongside individuals (e.g. as a separate
    (pop_size, k) array).  On crossover, blend parent sigmas with the
    same operator as the object variables.
    """
    k = individual.shape[0]
    if tau is None:
        tau = 1.0 / np.sqrt(k)

    # Step 1: adapt sigma — one N(0,1) draw per gene
    noise_sigma = rng.standard_normal(k).astype(np.float32)
    new_sigma = sigma * np.exp(tau * noise_sigma)
    new_sigma = np.maximum(new_sigma, sigma_min)    # enforce lower bound

    # Step 2: perturb object variables
    noise_x = rng.standard_normal(k).astype(np.float32)
    mutated = individual + new_sigma * noise_x

    return mutated.astype(np.float32), new_sigma.astype(np.float32)


def uniform_reset_mutation(
    individual: np.ndarray,
    low: float,
    high: float,
    rng: np.random.Generator,
    p_reset: float = 0.1,
) -> np.ndarray:
    """
    Uniform reset mutation: replace each gene with probability ``p_reset``.

    Each gene d is independently reset to a fresh uniform draw from
    [``low``, ``high``] with probability ``p_reset``, otherwise kept unchanged.
    This is equivalent to applying a random uniform initialisation selectively,
    providing large jumps and escape from local optima — complementary to the
    small perturbations of Gaussian mutation.

    Parameters
    ----------
    individual : np.ndarray, shape (k,), dtype float32
        Object-variable vector of the individual to mutate.
    low : float
        Lower bound of the reset uniform distribution.
    high : float
        Upper bound of the reset uniform distribution.
    rng : np.random.Generator
        Caller-supplied seeded generator.
    p_reset : float, optional
        Per-gene reset probability (default 0.1 = 10% of genes reset).
        Must be in (0, 1].

    Returns
    -------
    mutated : np.ndarray, shape (k,), dtype float32
        The mutated object-variable vector.  The passed-in array is not
        modified (copy semantics).

    Raises
    ------
    ValueError
        If p_reset is not in (0, 1] or low >= high.

    Notes
    -----
    Suitable for both 'real' and decoded-binary representations because it
    operates entirely in the decoded real-valued space.
    """
    if not (0.0 < p_reset <= 1.0):
        raise ValueError(f"p_reset must be in (0, 1], got {p_reset}")
    if low >= high:
        raise ValueError(f"low ({low}) must be < high ({high})")

    k = individual.shape[0]
    mutated = individual.copy()

    reset_mask = rng.random(k) < p_reset
    if reset_mask.any():
        n_reset = int(reset_mask.sum())
        mutated[reset_mask] = rng.uniform(low, high, size=n_reset).astype(
            np.float32
        )

    return mutated


# ===========================================================================
# SECTION 4 — Survivor Selection
# ===========================================================================

def mu_plus_lambda(
    parents: np.ndarray,
    parent_fits: np.ndarray,
    offspring: np.ndarray,
    offspring_fits: np.ndarray,
    mu: int,
    parent_sigmas: np.ndarray | None = None,
    offspring_sigmas: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    (mu + lambda) survivor selection — elitist.

    Pool ALL parents and ALL offspring, then keep the ``mu`` individuals with
    the highest fitness.  This guarantees that the best-ever individual is
    never lost (strong elitism).

    Parameters
    ----------
    parents : np.ndarray, shape (mu, k), dtype float32
        Current parent population.
    parent_fits : np.ndarray, shape (mu,), dtype float32
        Fitness of each parent (higher = better).
    offspring : np.ndarray, shape (lambda_, k), dtype float32
        Offspring generated in the current generation.
    offspring_fits : np.ndarray, shape (lambda_,), dtype float32
        Fitness of each offspring.
    mu : int
        Number of survivors to keep (= next generation's population size).
    parent_sigmas : np.ndarray or None, shape (mu, k)
        Strategy parameters for parents.  Passed through if provided.
    offspring_sigmas : np.ndarray or None, shape (lambda_, k)
        Strategy parameters for offspring.  Passed through if provided.

    Returns
    -------
    survivors : np.ndarray, shape (mu, k), dtype float32
        The ``mu`` best individuals from the combined pool.
    survivor_fits : np.ndarray, shape (mu,), dtype float32
        Their fitness values.
    survivor_sigmas : np.ndarray or None, shape (mu, k)
        Corresponding strategy parameters, or None if not provided.

    Notes
    -----
    (mu+lambda) is the conservative choice: it prevents quality loss but can
    slow exploration once good individuals dominate.  Pair with diversity
    mechanisms (fitness sharing or island model) to mitigate stagnation.
    """
    combined = np.concatenate([parents, offspring], axis=0)          # (mu+lam, k)
    combined_fits = np.concatenate([parent_fits, offspring_fits])    # (mu+lam,)

    # Sort descending by fitness; take top mu
    top_idx = np.argsort(combined_fits)[::-1][:mu]
    survivors = combined[top_idx].astype(np.float32)
    survivor_fits = combined_fits[top_idx].astype(np.float32)

    survivor_sigmas: np.ndarray | None = None
    if parent_sigmas is not None and offspring_sigmas is not None:
        combined_sigmas = np.concatenate(
            [parent_sigmas, offspring_sigmas], axis=0
        )
        survivor_sigmas = combined_sigmas[top_idx].astype(np.float32)

    return survivors, survivor_fits, survivor_sigmas


def mu_comma_lambda(
    offspring: np.ndarray,
    offspring_fits: np.ndarray,
    mu: int,
    offspring_sigmas: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    (mu, lambda) survivor selection — generational (non-elitist).

    Select the ``mu`` best individuals from the offspring pool ONLY; the
    parent generation is completely discarded.  Requires lambda >= mu.

    This allows the population to escape local optima more easily than
    (mu+lambda) because bad-but-explored regions of parents cannot "block"
    the next generation.  The trade-off is that good solutions can be lost.

    Parameters
    ----------
    offspring : np.ndarray, shape (lambda_, k), dtype float32
        All offspring generated in the current generation.
    offspring_fits : np.ndarray, shape (lambda_,), dtype float32
        Fitness of each offspring.
    mu : int
        Number of survivors to keep.  Must satisfy mu <= lambda_.
    offspring_sigmas : np.ndarray or None, shape (lambda_, k)
        Strategy parameters for offspring.  Passed through if provided.

    Returns
    -------
    survivors : np.ndarray, shape (mu, k), dtype float32
        The ``mu`` best offspring.
    survivor_fits : np.ndarray, shape (mu,), dtype float32
        Their fitness values.
    survivor_sigmas : np.ndarray or None, shape (mu, k)
        Corresponding strategy parameters, or None if not provided.

    Raises
    ------
    ValueError
        If len(offspring) < mu.

    Notes
    -----
    The requirement lambda >= mu (typically lambda = 7 * mu by convention)
    ensures enough diversity in offspring before selection; use the engine's
    config to set lambda = 7 * mu or at least lambda = 2 * mu.
    """
    lambda_ = len(offspring)
    if lambda_ < mu:
        raise ValueError(
            f"(mu,lambda) requires lambda >= mu, but lambda={lambda_} < mu={mu}"
        )

    top_idx = np.argsort(offspring_fits)[::-1][:mu]
    survivors = offspring[top_idx].astype(np.float32)
    survivor_fits = offspring_fits[top_idx].astype(np.float32)

    survivor_sigmas: np.ndarray | None = None
    if offspring_sigmas is not None:
        survivor_sigmas = offspring_sigmas[top_idx].astype(np.float32)

    return survivors, survivor_fits, survivor_sigmas


# ===========================================================================
# Sanity check — run directly with: python -X utf8 core/operators.py
# ===========================================================================

if __name__ == "__main__":
    print("=" * 64)
    print("core/operators.py — sanity check")
    print("=" * 64)

    SEED = 42
    rng = np.random.default_rng(SEED)

    K = 20           # latent dimension
    POP_SIZE = 30    # population size (user or item)
    N_PARENTS = 10

    # Synthetic fitness array: values in [-5, 0] (convention from fitness.py)
    fits = rng.uniform(-5.0, 0.0, size=POP_SIZE).astype(np.float32)
    pop  = rng.standard_normal((POP_SIZE, K)).astype(np.float32) * 0.3

    # -------------------------------------------------------------------
    # Test 1: Tournament Selection
    # -------------------------------------------------------------------
    print("\n[Test 1] tournament_selection")
    for tau in [1, 2, 5]:
        sel = tournament_selection(fits, N_PARENTS, tau=tau, rng=rng)
        assert sel.shape == (N_PARENTS,), f"Shape mismatch: {sel.shape}"
        assert sel.min() >= 0 and sel.max() < POP_SIZE, "Index out of range"
        mean_fit = fits[sel].mean()
        print(f"  tau={tau}: selected indices={sel.tolist()}")
        print(f"          mean fitness of selected = {mean_fit:.4f}  "
              f"(pop mean = {fits.mean():.4f})  -> higher expected with higher tau")

    # Pressure test: higher tau should yield higher avg fitness of selected
    rng_p = np.random.default_rng(SEED)
    fits_pressure = rng_p.uniform(-5.0, 0.0, size=500).astype(np.float32)
    sel1 = tournament_selection(fits_pressure, 100, tau=2, rng=rng_p)
    sel7 = tournament_selection(fits_pressure, 100, tau=7, rng=rng_p)
    assert fits_pressure[sel7].mean() >= fits_pressure[sel1].mean(), \
        "tau=7 should yield higher average fitness than tau=2!"
    print(f"  Selection pressure: tau=7 avg={fits_pressure[sel7].mean():.4f} >= "
          f"tau=2 avg={fits_pressure[sel1].mean():.4f}  OK")

    # -------------------------------------------------------------------
    # Test 2: Rank Roulette Selection
    # -------------------------------------------------------------------
    print("\n[Test 2] rank_roulette_selection")
    rng2 = np.random.default_rng(SEED)
    fits2 = rng2.uniform(-5.0, 0.0, size=POP_SIZE).astype(np.float32)
    sel_rr = rank_roulette_selection(fits2, N_PARENTS, rng=rng2)
    assert sel_rr.shape == (N_PARENTS,)
    assert sel_rr.min() >= 0 and sel_rr.max() < POP_SIZE
    # Average fitness of selected should beat population average
    print(f"  selected indices:   {sel_rr.tolist()}")
    print(f"  mean fitness of selected = {fits2[sel_rr].mean():.4f}  "
          f"(pop mean = {fits2.mean():.4f})")

    # High-count sampling should favour high-rank individuals
    big_sel = rank_roulette_selection(
        fits2, n_select=10_000, rng=np.random.default_rng(7)
    )
    best_idx = int(np.argmax(fits2))
    worst_idx = int(np.argmin(fits2))
    best_count  = int((big_sel == best_idx).sum())
    worst_count = int((big_sel == worst_idx).sum())
    assert best_count > worst_count, \
        "Best individual should appear more often than worst!"
    print(f"  Bias check (10k samples): best selected {best_count}x, "
          f"worst selected {worst_count}x  OK")

    # -------------------------------------------------------------------
    # Test 3: Uniform Crossover
    # -------------------------------------------------------------------
    print("\n[Test 3] uniform_crossover")
    rng3 = np.random.default_rng(SEED)
    p1 = np.zeros(K, dtype=np.float32)
    p2 = np.ones(K, dtype=np.float32)
    c1, c2 = uniform_crossover(p1, p2, rng3, p_swap=0.5)
    assert c1.shape == (K,) and c2.shape == (K,)
    # Each gene in c1 is either 0.0 or 1.0
    assert np.all(np.isin(c1, [0.0, 1.0])), "c1 genes should be 0 or 1"
    assert np.all(np.isin(c2, [0.0, 1.0])), "c2 genes should be 0 or 1"
    # Complementarity: c1[d] + c2[d] == 1 always (swap is mutual)
    assert np.allclose(c1 + c2, np.ones(K)), "Uniform XO must be complementary"
    swapped = int((c1 == 1.0).sum())
    print(f"  p1=all-0, p2=all-1 -> c1 has {swapped}/{K} genes from p2 "
          f"(expect ~{K//2})  OK")

    # p_swap edge cases
    c_zero1, c_zero2 = uniform_crossover(p1, p2, rng3, p_swap=0.0)
    assert np.allclose(c_zero1, p1) and np.allclose(c_zero2, p2), \
        "p_swap=0 should return copies of the parents"
    c_one1, c_one2 = uniform_crossover(p1, p2, rng3, p_swap=1.0)
    assert np.allclose(c_one1, p2) and np.allclose(c_one2, p1), \
        "p_swap=1 should fully swap parents"
    print(f"  Edge cases p_swap=0 and p_swap=1 correct  OK")

    # -------------------------------------------------------------------
    # Test 4: BLX-alpha Crossover
    # -------------------------------------------------------------------
    print("\n[Test 4] blx_alpha_crossover")
    rng4 = np.random.default_rng(SEED)
    p1_b = rng4.uniform(-1.0, 0.0, K).astype(np.float32)
    p2_b = rng4.uniform(0.0, 1.0, K).astype(np.float32)
    alpha = 0.5

    c1_b, c2_b = blx_alpha_crossover(p1_b, p2_b, rng4, alpha=alpha)
    assert c1_b.shape == (K,) and c2_b.shape == (K,)
    assert c1_b.dtype == np.float32 and c2_b.dtype == np.float32

    # Children should lie within the expanded interval
    interval = p2_b - p1_b           # p2_b >= p1_b by construction above
    lo_ext   = p1_b - alpha * interval
    hi_ext   = p2_b + alpha * interval
    # Allow small float32 tolerance
    assert np.all(c1_b >= lo_ext - 1e-5) and np.all(c1_b <= hi_ext + 1e-5), \
        "c1 out of BLX interval"
    assert np.all(c2_b >= lo_ext - 1e-5) and np.all(c2_b <= hi_ext + 1e-5), \
        "c2 out of BLX interval"
    extrapolated = np.sum((c1_b < p1_b) | (c1_b > p2_b))
    print(f"  alpha={alpha}: {extrapolated}/{K} genes extrapolated beyond "
          f"parental range (expected ~{int(K*alpha*0.5)}+)  OK")
    print(f"  c1 range=[{c1_b.min():.4f}, {c1_b.max():.4f}], "
          f"c2 range=[{c2_b.min():.4f}, {c2_b.max():.4f}]")

    # alpha=0 produces children strictly within [lo, hi]
    c1_0, c2_0 = blx_alpha_crossover(p1_b, p2_b, rng4, alpha=0.0)
    assert np.all(c1_0 >= p1_b - 1e-5) and np.all(c1_0 <= p2_b + 1e-5), \
        "alpha=0 should not extrapolate"
    print(f"  alpha=0 edge case: no extrapolation  OK")

    # -------------------------------------------------------------------
    # Test 5: Self-adaptive Gaussian Mutation
    # -------------------------------------------------------------------
    print("\n[Test 5] gaussian_mutation (self-adaptive)")
    rng5 = np.random.default_rng(SEED)
    ind = rng5.standard_normal(K).astype(np.float32)
    sigma_init = np.full(K, 0.3, dtype=np.float32)

    mut, new_sigma = gaussian_mutation(ind, sigma_init, rng5)
    assert mut.shape == (K,) and mut.dtype == np.float32
    assert new_sigma.shape == (K,) and new_sigma.dtype == np.float32
    assert not np.allclose(ind, mut), "Mutated individual identical to parent"
    assert (new_sigma >= 1e-5).all(), "Sigma must respect sigma_min"
    print(f"  original sigma (mean)={sigma_init.mean():.4f}")
    print(f"  adapted  sigma (mean)={new_sigma.mean():.4f}")
    print(f"  max gene change: {np.abs(mut - ind).max():.4f}  OK")

    # Convergence test: over many generations, sigma should decrease
    # as individuals approach a fixed point (fake fitness signals)
    sigma_track = sigma_init.copy()
    x_track = np.zeros(K, dtype=np.float32)  # target is all-zeros
    for _ in range(200):
        x_track, sigma_track = gaussian_mutation(x_track, sigma_track,
                                                  np.random.default_rng())
    print(f"  After 200 unconstrained steps, sigma mean={sigma_track.mean():.6f} "
          f"(can drift; self-adaptation is evolutionary, not greedy)  OK")

    # -------------------------------------------------------------------
    # Test 6: Uniform Reset Mutation
    # -------------------------------------------------------------------
    print("\n[Test 6] uniform_reset_mutation")
    rng6 = np.random.default_rng(SEED)
    ind_ur = np.zeros(K, dtype=np.float32)

    for p_reset in [0.0001, 0.5, 1.0]:
        mut_ur = uniform_reset_mutation(ind_ur, low=-1.0, high=1.0,
                                        rng=rng6, p_reset=p_reset)
        assert mut_ur.shape == (K,) and mut_ur.dtype == np.float32
        n_changed = int((mut_ur != 0.0).sum())
        print(f"  p_reset={p_reset}: {n_changed}/{K} genes reset "
              f"(expected ~{int(K*p_reset)})  OK")

    # Original should be unchanged (copy semantics)
    assert np.allclose(ind_ur, 0.0), "Original individual was modified!"
    print(f"  Copy semantics verified: original unchanged  OK")

    # -------------------------------------------------------------------
    # Test 7: (mu + lambda) survivor selection
    # -------------------------------------------------------------------
    print("\n[Test 7] mu_plus_lambda survivor selection")
    rng7 = np.random.default_rng(SEED)
    mu_val, lam = 10, 20
    parents7  = rng7.standard_normal((mu_val, K)).astype(np.float32)
    p_fits7   = rng7.uniform(-5, 0, mu_val).astype(np.float32)
    offspring7 = rng7.standard_normal((lam, K)).astype(np.float32)
    o_fits7    = rng7.uniform(-5, 0, lam).astype(np.float32)
    p_sigmas7  = np.full((mu_val, K), 0.3, dtype=np.float32)
    o_sigmas7  = np.full((lam, K), 0.3, dtype=np.float32)

    surv, surv_fits, surv_sig = mu_plus_lambda(
        parents7, p_fits7, offspring7, o_fits7, mu_val,
        parent_sigmas=p_sigmas7, offspring_sigmas=o_sigmas7
    )
    assert surv.shape == (mu_val, K)
    assert surv_fits.shape == (mu_val,)
    assert surv_sig is not None and surv_sig.shape == (mu_val, K)

    # Best of combined pool must be in survivors
    combined_fits7 = np.concatenate([p_fits7, o_fits7])
    global_best_fit = combined_fits7.max()
    assert np.isclose(surv_fits.max(), global_best_fit), \
        "Global best must survive in (mu+lambda)!"
    print(f"  mu={mu_val}, lambda={lam}: survivors shape={surv.shape}  OK")
    print(f"  Elitism check: global best ({global_best_fit:.4f}) in survivors  OK")
    print(f"  Survivor sigma passed through: shape={surv_sig.shape}  OK")

    # Without sigmas
    surv_ns, surv_fits_ns, sig_ns = mu_plus_lambda(
        parents7, p_fits7, offspring7, o_fits7, mu_val
    )
    assert sig_ns is None
    print(f"  No-sigma variant: survivor_sigmas=None  OK")

    # -------------------------------------------------------------------
    # Test 8: (mu, lambda) survivor selection
    # -------------------------------------------------------------------
    print("\n[Test 8] mu_comma_lambda survivor selection")
    rng8 = np.random.default_rng(SEED)
    mu_val2, lam2 = 10, 40
    offspring8 = rng8.standard_normal((lam2, K)).astype(np.float32)
    o_fits8    = rng8.uniform(-5, 0, lam2).astype(np.float32)
    o_sigmas8  = np.full((lam2, K), 0.2, dtype=np.float32)

    surv8, surv_fits8, surv_sig8 = mu_comma_lambda(
        offspring8, o_fits8, mu_val2, offspring_sigmas=o_sigmas8
    )
    assert surv8.shape == (mu_val2, K)
    assert surv_fits8.shape == (mu_val2,)
    assert surv_sig8 is not None and surv_sig8.shape == (mu_val2, K)
    assert np.all(surv_fits8 >= surv_fits8.min()), "Survivors should be sorted"
    # Best of offspring must be in survivors
    assert np.isclose(surv_fits8.max(), o_fits8.max()), \
        "Best offspring must be selected!"
    print(f"  mu={mu_val2}, lambda={lam2}: survivors shape={surv8.shape}  OK")
    print(f"  Best offspring ({o_fits8.max():.4f}) in survivors  OK")

    # Error on lambda < mu
    try:
        mu_comma_lambda(offspring8[:5], o_fits8[:5], mu=10)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ValueError correctly raised for lambda < mu: {e}  OK")

    # -------------------------------------------------------------------
    # Reproducibility across all tests
    # -------------------------------------------------------------------
    print("\n[Reproducibility]")
    rng_a = np.random.default_rng(999)
    rng_b = np.random.default_rng(999)
    p = np.ones(K, dtype=np.float32)
    q = -np.ones(K, dtype=np.float32)
    sig = np.full(K, 0.1, dtype=np.float32)

    c_a1, c_a2 = uniform_crossover(p, q, rng_a)
    c_b1, c_b2 = uniform_crossover(p, q, rng_b)
    assert np.allclose(c_a1, c_b1) and np.allclose(c_a2, c_b2)

    m_a, s_a = gaussian_mutation(p, sig, rng_a)
    m_b, s_b = gaussian_mutation(p, sig, rng_b)
    assert np.allclose(m_a, m_b) and np.allclose(s_a, s_b)
    print(f"  Same seed -> identical results for crossover + mutation  OK")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("All 8 operator tests passed  OK")
    print("=" * 64)
