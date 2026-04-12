"""
core/data_loader.py
===================
MovieLens-100K data-loading utilities.

All user and item IDs are converted to **0-indexed** integers so that the
permanent mapping rule is satisfied without any offset arithmetic downstream:

    Population_U[i]  ←→  user whose original ID was  i + 1
    Population_V[j]  ←→  item whose original ID was  j + 1

Public API
----------
load_movielens(path) -> (df, n_users, n_items)
    Load u.data, return a tidy DataFrame with 0-indexed IDs.

build_rating_matrix(df, n_users, n_items) -> R
    Dense float32 matrix, shape (n_users, n_items). 0 = unobserved.

train_test_split_matrix(df, test_size, random_state)
    -> (R_train, R_test, n_users, n_items)
    Interaction-level (not user-level) 80/20 split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core loading function
# ---------------------------------------------------------------------------

def load_movielens(
    path: str | Path = "data/ml-100k/u.data",
) -> Tuple[pd.DataFrame, int, int]:
    """
    Load the MovieLens-100K ratings file and convert IDs to 0-indexed.

    The file format is tab-separated with four columns and **no header**::

        user_id  item_id  rating  timestamp

    Original IDs are 1-indexed (users 1-943, items 1-1682).  This function
    subtracts 1 from both so that all downstream code can use 0-indexed
    arrays without any offset arithmetic.

    Parameters
    ----------
    path : str or Path
        Path to ``u.data``.  Default matches the project's ``data/`` layout.

    Returns
    -------
    df : pd.DataFrame
        Columns: ``user_id`` (int, 0-indexed), ``item_id`` (int, 0-indexed),
        ``rating`` (float32), ``timestamp`` (int).
        Sorted by ``user_id`` then ``item_id`` for reproducibility.
    n_users : int
        Number of distinct users (943 for the full ML-100K dataset).
    n_items : int
        Number of distinct items (1682 for the full ML-100K dataset).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"MovieLens data file not found: '{path}'.  "
            "Download from https://grouplens.org/datasets/movielens/100k/ "
            "and place u.data in data/ml-100k/."
        )

    df = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": np.int32, "item_id": np.int32,
               "rating": np.float32, "timestamp": np.int64},
        engine="python",
    )

    # Convert 1-indexed → 0-indexed
    df["user_id"] = (df["user_id"] - 1).astype(np.int32)
    df["item_id"] = (df["item_id"] - 1).astype(np.int32)

    n_users = int(df["user_id"].max()) + 1
    n_items = int(df["item_id"].max()) + 1

    # Stable sort for reproducibility
    df = df.sort_values(["user_id", "item_id"], ignore_index=True)

    return df, n_users, n_items


# ---------------------------------------------------------------------------
# Matrix builder
# ---------------------------------------------------------------------------

def build_rating_matrix(
    df: pd.DataFrame,
    n_users: int,
    n_items: int,
) -> np.ndarray:
    """
    Build a dense float32 rating matrix from a tidy interactions DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``user_id`` (0-indexed int), ``item_id``
        (0-indexed int), and ``rating`` (numeric).
    n_users : int
        Total number of users (determines number of rows).
    n_items : int
        Total number of items (determines number of columns).

    Returns
    -------
    R : np.ndarray, shape (n_users, n_items), dtype float32
        ``R[i, j] = rating``  if user i rated item j,  else  ``0.0``.

    Notes
    -----
    When a user rates an item more than once (rare in ML-100K), the
    **last** rating in ``df`` wins (standard NumPy advanced-indexing
    semantics for duplicate assignments).
    """
    R = np.zeros((n_users, n_items), dtype=np.float32)
    u_idx = df["user_id"].to_numpy(dtype=np.int32)
    i_idx = df["item_id"].to_numpy(dtype=np.int32)
    vals  = df["rating"].to_numpy(dtype=np.float32)
    R[u_idx, i_idx] = vals
    return R


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def train_test_split_matrix(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Split interactions (not users) into train and test sets and build matrices.

    ``test_size`` fraction of rows in ``df`` are withheld as the test set.
    This is an **interaction-level** split: every user may have both train
    and test interactions, which is the standard protocol for evaluating
    collaborative-filtering models without cold-start bias.

    Parameters
    ----------
    df : pd.DataFrame
        Full interactions DataFrame from :func:`load_movielens`.
    test_size : float
        Fraction of interactions to hold out (default 0.20 → 80/20 split).
    random_state : int
        Seed for the shuffle RNG (default 42).

    Returns
    -------
    R_train : np.ndarray, shape (n_users, n_items), dtype float32
        Training rating matrix (0 = unobserved or held-out).
    R_test  : np.ndarray, shape (n_users, n_items), dtype float32
        Test rating matrix   (0 = not in test set).
    n_users : int
    n_items : int

    Examples
    --------
    Extract test pairs and ratings from R_test for the engine::

        R_train, R_test, n_users, n_items = train_test_split_matrix(df)
        test_r, test_c = np.nonzero(R_test)
        test_pairs   = np.column_stack([test_r, test_c])   # (n_test, 2)
        test_ratings = R_test[test_r, test_c]              # (n_test,)
    """
    n_users = int(df["user_id"].max()) + 1
    n_items = int(df["item_id"].max()) + 1

    rng     = np.random.default_rng(random_state)
    n       = len(df)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_test     = max(1, int(n * test_size))
    test_idx   = indices[:n_test]
    train_idx  = indices[n_test:]

    df_train = df.iloc[train_idx]
    df_test  = df.iloc[test_idx]

    R_train = build_rating_matrix(df_train, n_users, n_items)
    R_test  = build_rating_matrix(df_test,  n_users, n_items)

    return R_train, R_test, n_users, n_items


# ---------------------------------------------------------------------------
# Sanity check — python -X utf8 core/data_loader.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("core/data_loader.py — sanity check")
    print("=" * 60)

    # 1. Load
    DATA_PATH = "data/ml-100k/u.data"
    try:
        df, n_users, n_items = load_movielens(DATA_PATH)
    except FileNotFoundError as e:
        print(f"\n[WARNING] {e}")
        print("Creating tiny SYNTHETIC dataset for self-test …\n")
        rng_s = np.random.default_rng(0)
        rows  = []
        for u in range(1, 21):         # 20 users (1-indexed)
            for it in range(1, 31):    # 30 items (1-indexed)
                if rng_s.random() < 0.3:
                    rows.append({"user_id": u, "item_id": it,
                                 "rating": float(rng_s.integers(1, 6)),
                                 "timestamp": 0})
        df_raw = pd.DataFrame(rows)
        # Save to a temp file and reload through the function
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv",
                                         delete=False) as f:
            df_raw.to_csv(f, sep="\t", header=False, index=False)
            tmp_path = f.name
        df, n_users, n_items = load_movielens(tmp_path)
        os.unlink(tmp_path)

    print(f"\n[Load]")
    print(f"  Total interactions : {len(df):,}")
    print(f"  n_users            : {n_users}")
    print(f"  n_items            : {n_items}")
    print(f"  Columns            : {list(df.columns)}")
    print(f"  user_id range      : [{df['user_id'].min()}, {df['user_id'].max()}]  (0-indexed)")
    print(f"  item_id range      : [{df['item_id'].min()}, {df['item_id'].max()}]  (0-indexed)")
    print(f"  rating range       : [{df['rating'].min():.1f}, {df['rating'].max():.1f}]")
    assert df["user_id"].min() == 0, "user_id not 0-indexed!"
    assert df["item_id"].min() == 0, "item_id not 0-indexed!"
    print(f"  0-indexed check    : OK")

    # 2. build_rating_matrix
    R_full = build_rating_matrix(df, n_users, n_items)
    n_obs  = int((R_full > 0).sum())
    spars  = 1.0 - n_obs / (n_users * n_items)
    print(f"\n[build_rating_matrix]")
    print(f"  Shape    : {R_full.shape}")
    print(f"  dtype    : {R_full.dtype}")
    print(f"  Observed : {n_obs:,}  ({n_obs / (n_users * n_items):.2%} density)")
    print(f"  Sparsity : {spars:.2%}")
    assert R_full.shape == (n_users, n_items), "Wrong shape!"
    assert R_full.dtype == np.float32,         "Wrong dtype!"
    assert np.all(R_full >= 0),                "Negative rating!"
    print(f"  All checks passed  : OK")

    # 3. train_test_split_matrix
    R_train, R_test, nu, ni = train_test_split_matrix(df, test_size=0.2,
                                                       random_state=42)
    obs_train = int((R_train > 0).sum())
    obs_test  = int((R_test  > 0).sum())
    print(f"\n[train_test_split_matrix]")
    print(f"  R_train : {R_train.shape}  dtype={R_train.dtype}  obs={obs_train:,}")
    print(f"  R_test  : {R_test.shape}   dtype={R_test.dtype}   obs={obs_test:,}")
    print(f"  Split ratio (test) : {obs_test / (obs_train + obs_test):.2%}  (target 20%)")
    assert R_train.shape == (n_users, n_items)
    assert R_test.shape  == (n_users, n_items)
    assert abs(obs_test / (obs_train + obs_test) - 0.20) < 0.01, "Split ratio off!"
    # Train and test should not overlap on the same cell
    overlap = int(((R_train > 0) & (R_test > 0)).sum())
    assert overlap == 0, f"Train/test overlap: {overlap} cells!"
    print(f"  Train/test overlap : {overlap} cells  OK")
    print(f"  Reproducible split : ", end="")
    R2, _, _, _ = train_test_split_matrix(df, test_size=0.2, random_state=42)
    assert np.array_equal(R_train, R2), "Split not reproducible!"
    print(f"OK")

    # 4. Extract test pairs (as used by CoevolutionaryEngine)
    test_r, test_c = np.nonzero(R_test)
    test_pairs   = np.column_stack([test_r, test_c])
    test_ratings = R_test[test_r, test_c]
    print(f"\n[Test pairs for CoevolutionaryEngine]")
    print(f"  test_pairs   : shape={test_pairs.shape}, dtype={test_pairs.dtype}")
    print(f"  test_ratings : shape={test_ratings.shape}, dtype={test_ratings.dtype}")
    print(f"  Sample: user={test_pairs[0,0]}, item={test_pairs[0,1]}, "
          f"rating={test_ratings[0]:.1f}")

    print("\n" + "=" * 60)
    print("All data_loader checks passed  OK")
    print("=" * 60)
