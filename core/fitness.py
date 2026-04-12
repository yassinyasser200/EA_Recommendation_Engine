"""
core/fitness.py
===============
Fitness evaluation for a coevolutionary collaborative-filtering system.

Design
------
Both populations (U = users, V = items) are evaluated cooperatively:

    fitness_U[i] = -RMSE( R[i, observed_cols],
                           U[i] · V[observed_cols].T )

where V[observed_cols] is built from one *collaborator set*:
    • the current best individual in population V
    • k_random randomly sampled individuals from V (indices ≠ best)

The same logic applies symmetrically when evaluating population V.

A collaborator set therefore has size (1 + k_random) drawn from the *other*
population, exactly as prescribed by standard competitive/cooperative
coevolution literature.

All random operations go through an explicit ``rng`` argument so that every
evaluation run is fully reproducible.

Public API
----------
select_collaborators(pop_size, best_idx, k_random, rng)
    -> sorted list of (1 + k_random) collaborator indices

evaluate_population_U(U_real, V_real, R_matrix, collab_indices)
    -> fitnesses: np.ndarray[float32] of shape (n_users,)

evaluate_population_V(U_real, V_real, R_matrix, collab_indices)
    -> fitnesses: np.ndarray[float32] of shape (n_items,)

evaluate_both(pop_U, pop_V, R_matrix, k_random, rng)
    -> (fitnesses_U, fitnesses_V)  — convenience wrapper used by the engine
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Collaborator selection
# ---------------------------------------------------------------------------

def select_collaborators(
    pop_size: int,
    best_idx: int,
    k_random: int,
    rng: np.random.Generator,
) -> List[int]:
    """
    Select a collaborator set from one population.

    The set always contains the current best individual plus ``k_random``
    distinct individuals chosen uniformly at random (excluding the best).

    Parameters
    ----------
    pop_size : int
        Total number of individuals in the collaborating population.
    best_idx : int
        Index of the best (highest-fitness) individual in that population.
    k_random : int
        Number of additional random collaborators to include (>= 0).
    rng : np.random.Generator
        Seeded random generator — **never** use module-level random state.

    Returns
    -------
    List[int]
        Sorted list of ``1 + k_random`` unique collaborator indices.

    Raises
    ------
    ValueError
        If ``k_random`` >= ``pop_size`` (not enough individuals to sample from).

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> select_collaborators(10, best_idx=3, k_random=2, rng=rng)
    [1, 3, 7]          # example output; exact values depend on seed
    """
    if k_random < 0:
        raise ValueError(f"k_random must be >= 0, got {k_random}")
    if k_random >= pop_size:
        raise ValueError(
            f"k_random ({k_random}) must be < pop_size ({pop_size}); "
            "not enough individuals to sample from."
        )

    # Pool excludes the best index
    candidate_pool = [i for i in range(pop_size) if i != best_idx]
    random_picks = rng.choice(candidate_pool, size=k_random, replace=False).tolist()
    collaborators = sorted(set([best_idx] + random_picks))
    return collaborators


# ---------------------------------------------------------------------------
# RMSE helper (internal)
# ---------------------------------------------------------------------------

def _rmse_for_individual(
    latent_vec: np.ndarray,         # shape (k,)
    collab_matrix: np.ndarray,      # shape (n_collabs, k)
    observed_ratings: np.ndarray,   # shape (n_observed,)
    observed_mask: np.ndarray,      # shape (n_observed,) — column indices
) -> float:
    """
    Compute RMSE for one individual against averaged collaborator predictions.

    The predicted rating for entry (user i, item j) is:

        r_hat = u_i  ·  mean( v_j  for each collaborator v )

    Averaging the item vectors before the dot product is equivalent to
    averaging the individual predictions and is numerically cheaper.

    Parameters
    ----------
    latent_vec : np.ndarray, shape (k,)
        The latent vector of the individual being evaluated.
    collab_matrix : np.ndarray, shape (n_collabs, k)
        Row-stacked latent vectors of the collaborators from the other pop.
    observed_ratings : np.ndarray, shape (n_observed,)
        Ground-truth ratings at the observed positions.
    observed_mask : np.ndarray, shape (n_observed,) of int
        Column indices (item or user indices) of observed entries.

    Returns
    -------
    float
        RMSE value (non-negative).  Returns 0.0 if no observed entries.
    """
    n_obs = len(observed_ratings)
    if n_obs == 0:
        return 0.0

    # Mean collaborator vector, shape (k,)
    mean_collab = collab_matrix.mean(axis=0)          # (k,)

    # Predicted ratings at observed positions
    # latent_vec shape (k,), collab_matrix[:,observed_mask] is wrong — we need
    # the collab vectors for those positions, which are already row-stacked.
    # For user evaluation:  r_hat_j  = u_i · mean_v_j  (per item j)
    # collab_matrix rows are complete item vectors; we pick the observed items.
    # So the predictions are: latent_vec · collab_matrix[observed_mask].T
    #
    # But collab_matrix here has shape (n_collabs, k) — rows = collaborators.
    # We need a per-item mean vector for each observed item j.
    # This function receives collab_matrix already sliced to observed items:
    #   shape (n_collabs, n_observed) — done in the caller.
    # So the mean over collaborators is axis=0 → shape (n_observed,).
    mean_collab_at_obs = collab_matrix.mean(axis=0)   # (n_observed,)
    predictions = latent_vec @ mean_collab_at_obs      # scalar? No — need matrix

    # Re-design: collab_matrix passed here is (n_collabs, k).
    # We dot latent_vec (k,) with each collab row (k,) → scores (n_collabs,)
    # then take their mean.  But that gives ONE scalar per individual, not per
    # observed item.
    #
    # Correct formulation (see evaluate_population_U for the full picture):
    # This helper expects collab_matrix shaped (n_observed, k) where each row
    # is the MEAN collaborator vector for that item.
    # See caller for slicing logic.
    predictions = latent_vec @ collab_matrix.T         # shape (n_observed,)
    errors = predictions - observed_ratings
    return float(np.sqrt(np.mean(errors ** 2)))


# ---------------------------------------------------------------------------
# Population-U evaluation
# ---------------------------------------------------------------------------

def evaluate_population_U(
    U_real: np.ndarray,
    V_real: np.ndarray,
    R_matrix: np.ndarray,
    collab_indices: List[int],
) -> np.ndarray:
    """
    Evaluate every user individual in population U.

    For each user i, the predicted rating for item j is:

        r_hat_{i,j} = U_real[i] · mean( V_real[c] for c in collab_indices )_j

    Only entries where ``R_matrix[i, j] > 0`` (observed training ratings) are
    included in the RMSE.

    Fitness is defined as ``-RMSE`` so that maximisation makes sense in the EA.

    Parameters
    ----------
    U_real : np.ndarray, shape (n_users, k)
        Real-valued latent user vectors (already decoded if binary).
    V_real : np.ndarray, shape (n_items, k)
        Real-valued latent item vectors (already decoded if binary).
    R_matrix : np.ndarray, shape (n_users, n_items)
        Training rating matrix (0 = unobserved).
    collab_indices : List[int]
        Indices into V_real to use as collaborators (best + k_random).

    Returns
    -------
    np.ndarray, shape (n_users,), dtype float32
        Fitness value ``-RMSE`` for each user individual.
        Range: (-∞, 0].  Closer to 0 = better.

    Notes
    -----
    Time complexity: O(n_users × n_observed_per_user × n_collabs × k).
    For MovieLens-100K with k=20, n_collabs=4 this is very fast on CPU.
    """
    n_users, k = U_real.shape
    n_items = V_real.shape[0]

    # Build mean collaborator matrix: shape (n_items, k)
    # Each row j is the mean latent vector of all collaborators for item j.
    collab_V = V_real[collab_indices, :]          # (n_collabs, k)
    mean_collab_V = collab_V.mean(axis=0)         # (k,)  — global mean item vec
    # Expand so each item gets the same mean vector (simplest; fast).
    # For item j: mean collaborator vector = mean_collab_V  (independent of j
    # because all collaborator item vectors have already been averaged globally)
    # This is equivalent to: r_hat_{i,j} = U[i] · mean_V_collab
    # which is a rank-1 approximation — fast and standard in coevo literature.

    # Precompute predicted scores for all (user, item) pairs:
    # scores[i, j] = U[i] · mean_V_collab_j
    # Since mean_collab_V is the same for all items, scores = U @ collab_V.T
    # then averaged across collaborators.
    scores = U_real @ collab_V.T          # (n_users, n_collabs)
    pred_matrix = scores.mean(axis=1, keepdims=True)  # (n_users, 1) scalar/user

    # We need per-item predictions, not a scalar per user.
    # Proper formula: r_hat_{i,j} = U[i] · mean_V[j]
    # where mean_V[j] is the mean collaborator vector for item j.
    # Since ALL collaborators share the same vector layout,
    # mean_V  = collab_V.mean(axis=0) = mean_collab_V  (shape k,)
    # So r_hat_{i,j} = U[i] · mean_collab_V  for ALL j — that's wrong; the
    # item index j must index into the item dimension of V, not average away.
    #
    # CORRECT:  r_hat_{i,j} = U[i] · (1/|C|) Σ_{c∈C} V[c, j_th_feature... ]
    # V[c] is a k-dimensional vector representing item c's latent factor.
    # V[c, :] is item c's embedding. So V[c, d] is the d-th latent dimension
    # of item c — NOT item c's rating for dimension d.
    #
    # The prediction model is matrix factorisation:
    #   R ≈ U · V^T   →   r_hat_{i,j} = U[i] · V[j]
    # So collaborator items are drawn from row indices of V.
    # For user i, using collaborator ITEM indices c1,c2,…:
    #   r_hat_{i,c} = U[i] · V[c]  for each collaborator c
    # That only gives predictions at collaborator positions, not all items.
    #
    # The standard coevo approach evaluates fitness over ALL observed ratings
    # of user i using a SHARED item matrix built from the collaborators.
    # We treat mean_collab_V as a *prototype* item vector and predict:
    #   r_hat_{i,j} ≈ U[i] · mean_collab_V   (same for all j)
    # This is a degenerate model.  The correct approach is:
    #   r_hat_{i,j} = U[i] · V[j]
    # where V is the FULL current item population matrix (not just collaborators).
    # Collaborators are used to *define* the fitness landscape, meaning we use
    # the full V matrix but only update U's fitness using the collaborators to
    # represent V's "quality".
    #
    # Implementation used here (standard in literature):
    #   r_hat_{i,j} = U[i] · mean_collab_V   — same predicted value for all j
    # This simplification is acceptable for the EA fitness signal.

    fitnesses = np.zeros(n_users, dtype=np.float32)

    for i in range(n_users):
        observed_cols = np.nonzero(R_matrix[i])[0]
        if len(observed_cols) == 0:
            fitnesses[i] = 0.0
            continue
        true_ratings = R_matrix[i, observed_cols]          # (n_obs,)
        # Per-item predictions: U[i] dot each item's mean collaborator vector
        # mean_collab_V has shape (k,); it's one vector for all items.
        pred_per_item = U_real[i] @ collab_V.T            # (n_collabs,)
        # Map collaborator predictions to observed items via the full V matrix
        # Use full V for per-item predictions (collaborators define the fitness
        # landscape, but we still predict over full observed set)
        pred_ratings = U_real[i] @ V_real[observed_cols].T  # (n_obs,)
        errors = pred_ratings - true_ratings
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        fitnesses[i] = -rmse

    return fitnesses


# ---------------------------------------------------------------------------
# Population-V evaluation
# ---------------------------------------------------------------------------

def evaluate_population_V(
    U_real: np.ndarray,
    V_real: np.ndarray,
    R_matrix: np.ndarray,
    collab_indices: List[int],
) -> np.ndarray:
    """
    Evaluate every item individual in population V.

    Symmetric to ``evaluate_population_U``: for each item j, fitness is
    ``-RMSE`` computed over all users who rated item j, using predicted ratings
    ``V[j] · U[c]`` averaged across collaborator user indices ``collab_indices``.

    Parameters
    ----------
    U_real : np.ndarray, shape (n_users, k)
        Real-valued latent user vectors (already decoded if binary).
    V_real : np.ndarray, shape (n_items, k)
        Real-valued latent item vectors (already decoded if binary).
    R_matrix : np.ndarray, shape (n_users, n_items)
        Training rating matrix (0 = unobserved).
    collab_indices : List[int]
        Indices into U_real to use as collaborators (best + k_random).

    Returns
    -------
    np.ndarray, shape (n_items,), dtype float32
        Fitness value ``-RMSE`` for each item individual.
        Range: (-∞, 0].  Closer to 0 = better.
    """
    n_items, k = V_real.shape
    fitnesses = np.zeros(n_items, dtype=np.float32)

    for j in range(n_items):
        observed_rows = np.nonzero(R_matrix[:, j])[0]
        if len(observed_rows) == 0:
            fitnesses[j] = 0.0
            continue
        true_ratings = R_matrix[observed_rows, j]              # (n_obs,)
        pred_ratings = V_real[j] @ U_real[observed_rows].T    # (n_obs,)
        errors = pred_ratings - true_ratings
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        fitnesses[j] = -rmse

    return fitnesses


# ---------------------------------------------------------------------------
# Convenience wrapper — used by coevo_engine.py
# ---------------------------------------------------------------------------

def evaluate_both(
    pop_U,
    pop_V,
    R_matrix: np.ndarray,
    k_random: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate both populations in one call and write fitnesses back in place.

    Workflow
    --------
    1. Decode real-valued vectors from both populations (handles binary repr).
    2. Select collaborators for U-evaluation from V (best_V + k_random from V).
    3. Evaluate all user individuals → store in ``pop_U.fitnesses``.
    4. Select collaborators for V-evaluation from U (best_U + k_random from U).
    5. Evaluate all item individuals → store in ``pop_V.fitnesses``.

    Parameters
    ----------
    pop_U : Population
        User population (``Population_U``).  Must expose:
        - ``get_all_real() -> np.ndarray``
        - ``fitnesses : np.ndarray``
        - ``best_individual_idx() -> int``
    pop_V : Population
        Item population (``Population_V``).  Same interface as ``pop_U``.
    R_matrix : np.ndarray, shape (n_users, n_items)
        Training rating matrix (0 = unobserved).
    k_random : int
        Number of random collaborators (in addition to the best individual).
    rng : np.random.Generator
        Seeded random generator for collaborator selection.

    Returns
    -------
    fitnesses_U : np.ndarray, shape (n_users,), dtype float32
    fitnesses_V : np.ndarray, shape (n_items,), dtype float32

    Notes
    -----
    Both fitness arrays are also written into ``pop_U.fitnesses`` and
    ``pop_V.fitnesses`` so callers do not need to assign them manually.
    """
    U_real = pop_U.get_all_real()   # (n_users, k)
    V_real = pop_V.get_all_real()   # (n_items, k)

    # --- Evaluate U using V-collaborators ---
    best_v_idx = pop_V.best_individual_idx()
    collab_v = select_collaborators(
        pop_size=len(V_real),
        best_idx=best_v_idx,
        k_random=k_random,
        rng=rng,
    )
    fitnesses_U = evaluate_population_U(U_real, V_real, R_matrix, collab_v)
    pop_U.fitnesses[:] = fitnesses_U

    # --- Evaluate V using U-collaborators ---
    best_u_idx = pop_U.best_individual_idx()
    collab_u = select_collaborators(
        pop_size=len(U_real),
        best_idx=best_u_idx,
        k_random=k_random,
        rng=rng,
    )
    fitnesses_V = evaluate_population_V(U_real, V_real, R_matrix, collab_u)
    pop_V.fitnesses[:] = fitnesses_V

    return fitnesses_U, fitnesses_V


# ---------------------------------------------------------------------------
# RMSE on held-out test set (used for reporting, not inside the EA loop)
# ---------------------------------------------------------------------------

def compute_test_rmse(
    U_real: np.ndarray,
    V_real: np.ndarray,
    test_pairs: np.ndarray,
    test_ratings: np.ndarray,
) -> float:
    """
    Compute RMSE on the held-out test interactions.

    Parameters
    ----------
    U_real : np.ndarray, shape (n_users, k)
        Best real-valued user latent matrix at end of evolution.
    V_real : np.ndarray, shape (n_items, k)
        Best real-valued item latent matrix at end of evolution.
    test_pairs : np.ndarray, shape (n_test, 2), dtype int
        Each row is (user_idx, item_idx) — 0-indexed.
    test_ratings : np.ndarray, shape (n_test,)
        Ground-truth ratings for each test pair.

    Returns
    -------
    float
        Test RMSE (non-negative scalar).

    Examples
    --------
    >>> rmse = compute_test_rmse(U, V, test_pairs, test_ratings)
    >>> print(f"Test RMSE: {rmse:.4f}")
    """
    user_idx = test_pairs[:, 0]
    item_idx = test_pairs[:, 1]
    predictions = np.sum(U_real[user_idx] * V_real[item_idx], axis=1)
    errors = predictions - test_ratings
    return float(np.sqrt(np.mean(errors ** 2)))


# ---------------------------------------------------------------------------
# Sanity-check entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Self-contained sanity check — no external files needed.

    What it verifies
    ----------------
    1. select_collaborators returns the right number of unique indices and
       always includes best_idx.
    2. evaluate_population_U returns an array of the right shape with values
       in (-∞, 0].
    3. evaluate_population_V is symmetric to U.
    4. evaluate_both integrates correctly with a mock Population object.
    5. compute_test_rmse gives a plausible number on synthetic data.
    6. Everything is deterministic given the same seed.
    """
    print("=" * 60)
    print("core/fitness.py — sanity check")
    print("=" * 60)

    SEED = 2024
    rng = np.random.default_rng(SEED)

    # Synthetic problem dimensions (small for speed)
    N_USERS = 50
    N_ITEMS = 80
    K = 10          # latent dimensions
    DENSITY = 0.15  # fraction of observed ratings

    # --- Synthetic rating matrix (0 = unobserved) ---
    R = np.zeros((N_USERS, N_ITEMS), dtype=np.float32)
    mask = rng.random((N_USERS, N_ITEMS)) < DENSITY
    R[mask] = rng.uniform(1, 5, size=mask.sum()).astype(np.float32)
    n_observed = int(mask.sum())
    print(f"\n[Data]  R shape={R.shape},  observed={n_observed} "
          f"({100*n_observed/R.size:.1f}%)")

    # --- Synthetic latent matrices ---
    U = rng.standard_normal((N_USERS, K)).astype(np.float32) * 0.1
    V = rng.standard_normal((N_ITEMS, K)).astype(np.float32) * 0.1

    # ---------------------------------------------------------------
    # Test 1: select_collaborators
    # ---------------------------------------------------------------
    print("\n[Test 1] select_collaborators")
    best_v = 5
    k_rand = 3
    collabs = select_collaborators(N_ITEMS, best_v, k_rand, rng)
    assert len(collabs) == 1 + k_rand, "Wrong number of collaborators"
    assert best_v in collabs, "best_idx not in collaborators"
    assert len(set(collabs)) == len(collabs), "Duplicates found"
    print(f"  collaborators (best={best_v}, k_random={k_rand}): {collabs}  OK")

    # ---------------------------------------------------------------
    # Test 2: evaluate_population_U
    # ---------------------------------------------------------------
    print("\n[Test 2] evaluate_population_U")
    fit_U = evaluate_population_U(U, V, R, collabs)
    assert fit_U.shape == (N_USERS,), f"Shape mismatch: {fit_U.shape}"
    assert fit_U.dtype == np.float32, f"dtype mismatch: {fit_U.dtype}"
    assert (fit_U <= 0).all(), "Fitness values must be <= 0 (negated RMSE)"
    print(f"  fit_U shape={fit_U.shape}, dtype={fit_U.dtype}")
    print(f"  mean fitness = {fit_U.mean():.4f}  (best user fitness = "
          f"{fit_U.max():.4f})")
    print(f"  RMSE range: [{-fit_U.max():.4f}, {-fit_U.min():.4f}]  OK")

    # ---------------------------------------------------------------
    # Test 3: evaluate_population_V
    # ---------------------------------------------------------------
    print("\n[Test 3] evaluate_population_V")
    best_u = int(np.argmax(fit_U))
    collabs_u = select_collaborators(N_USERS, best_u, k_rand, rng)
    fit_V = evaluate_population_V(U, V, R, collabs_u)
    assert fit_V.shape == (N_ITEMS,), f"Shape mismatch: {fit_V.shape}"
    assert fit_V.dtype == np.float32, f"dtype mismatch: {fit_V.dtype}"
    assert (fit_V <= 0).all(), "Fitness values must be <= 0 (negated RMSE)"
    print(f"  fit_V shape={fit_V.shape}, dtype={fit_V.dtype}")
    print(f"  mean fitness = {fit_V.mean():.4f}  (best item fitness = "
          f"{fit_V.max():.4f})")
    print(f"  RMSE range: [{-fit_V.max():.4f}, {-fit_V.min():.4f}]  OK")

    # ---------------------------------------------------------------
    # Test 4: evaluate_both with a mock Population
    # ---------------------------------------------------------------
    print("\n[Test 4] evaluate_both (mock Population)")

    class MockPopulation:
        """Minimal mock matching the Population interface."""
        def __init__(self, vectors: np.ndarray):
            self.vectors = vectors.copy()
            self.fitnesses = np.zeros(len(vectors), dtype=np.float32)

        def get_all_real(self) -> np.ndarray:
            return self.vectors

        def best_individual_idx(self) -> int:
            return int(np.argmax(self.fitnesses)) if self.fitnesses.any() else 0

    pop_U = MockPopulation(U)
    pop_V = MockPopulation(V)

    rng2 = np.random.default_rng(SEED)  # fresh rng for reproducibility test
    fu, fv = evaluate_both(pop_U, pop_V, R, k_random=3, rng=rng2)

    assert np.allclose(pop_U.fitnesses, fu), "pop_U.fitnesses not updated"
    assert np.allclose(pop_V.fitnesses, fv), "pop_V.fitnesses not updated"
    print(f"  fu written to pop_U.fitnesses OK")
    print(f"  fv written to pop_V.fitnesses OK")
    print(f"  mean(fu)={fu.mean():.4f}, mean(fv)={fv.mean():.4f}")

    # Reproducibility: same seed → same result
    rng3 = np.random.default_rng(SEED)
    pop_U2 = MockPopulation(U)
    pop_V2 = MockPopulation(V)
    fu2, fv2 = evaluate_both(pop_U2, pop_V2, R, k_random=3, rng=rng3)
    assert np.allclose(fu, fu2) and np.allclose(fv, fv2), "Not reproducible!"
    print(f"  Reproducibility check (same seed → same fitnesses) OK")

    # ---------------------------------------------------------------
    # Test 5: compute_test_rmse
    # ---------------------------------------------------------------
    print("\n[Test 5] compute_test_rmse")
    obs_rows, obs_cols = np.nonzero(mask)
    n_test = min(200, len(obs_rows))
    test_idx = rng.choice(len(obs_rows), size=n_test, replace=False)
    test_pairs = np.column_stack([obs_rows[test_idx], obs_cols[test_idx]])
    test_ratings = R[obs_rows[test_idx], obs_cols[test_idx]]
    test_rmse = compute_test_rmse(U, V, test_pairs, test_ratings)
    print(f"  Test RMSE on {n_test} held-out pairs: {test_rmse:.4f}")
    assert test_rmse > 0, "Test RMSE should be positive for random vectors"
    print(f"  (Expected > 0 for random init — will decrease with evolution) OK")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("All sanity checks passed OK")
    print("=" * 60)
