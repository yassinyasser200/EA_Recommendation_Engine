"""
core/population.py
==================
Population representation for the coevolutionary recommender engine.

Design constraints
------------------
1. **Permanent index mapping**: ``Population_U[i]`` always represents dataset
   user ``i``; ``Population_V[j]`` always represents item ``j``.  Nothing in
   this class rearranges individuals — that guarantee is upheld by the engine.

2. **Two representations** (selectable via ``repr_type``):
   - ``'real'``   — direct float32 latent vectors, shape ``(size, k)``.
   - ``'binary'`` — Gray-coded binary strings, shape ``(size, k * n_bits)``.
     Real values are decoded on demand via ``get_real`` / ``get_all_real``.

3. **Two initialisations** (selectable via ``init_type``):
   - ``'uniform'`` — each gene drawn from ``U(-0.5, 0.5)``.
   - ``'svd'``     — truncated SVD of the rating matrix ``R`` gives the
     population mean; Gaussian noise (std = 0.01) adds per-individual
     variation.  Requires ``R`` and ``role`` arguments.

4. **Fitness array**: ``self.fitnesses`` (shape ``(size,)``, float32)
   initialised to ``-np.inf`` (maximisation convention: fitness = -RMSE).

Public API
----------
Population(size, k, repr_type, n_bits, init_type, R, role)
    .get_real(idx)          → np.ndarray, shape (k,)
    .get_all_real()         → np.ndarray, shape (size, k)
    ._encode(real_vector)   → np.ndarray of bits, shape (k * n_bits,)
    ._decode(binary_vector) → np.ndarray, shape (k,)
    .best_individual_idx()  → int
    .best_fitness()         → float
    .average_fitness()      → float
    .diversity(sample_size) → float   (mean pairwise Euclidean distance)

Binary encoding details
-----------------------
Each real gene value ``v ∈ [ENCODE_LO, ENCODE_HI]`` is:
  1. Quantised to an unsigned integer in ``[0, 2^n_bits − 1]``.
  2. Reflected into the equivalent Gray code (XOR trick: ``g = n ^ (n >> 1)``).
  3. Stored as ``n_bits`` uint8 bits, MSB first.

Decoding reverses this: Gray to binary (prefix-XOR descent), binary to int,
int to float.  The Gray coding ensures that adjacent integers differ by
exactly one bit — a key property that makes BLX-α crossover on binary strings
biologically meaningful and prevents Hamming cliffs.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Encoding domain — all latent factors are expected to remain in this range.
# The range is intentionally wider than U(-0.5, 0.5) init so that SVD-
# initialised and BLX-α extended offspring are still representable exactly.
# ---------------------------------------------------------------------------

ENCODE_LO: float = -4.0
ENCODE_HI: float =  4.0


# ---------------------------------------------------------------------------
# Module-level Gray-code helpers (pure NumPy, work on one gene at a time)
# ---------------------------------------------------------------------------

def _quantise(val: float, n_bits: int) -> int:
    """Clamp and map ``val ∈ [ENCODE_LO, ENCODE_HI]`` to ``[0, 2^n_bits−1]``."""
    val     = float(np.clip(val, ENCODE_LO, ENCODE_HI))
    n_max   = (1 << n_bits) - 1
    normed  = (val - ENCODE_LO) / (ENCODE_HI - ENCODE_LO)
    return int(round(normed * n_max))


def _dequantise(int_val: int, n_bits: int) -> float:
    """Inverse of ``_quantise``."""
    n_max = (1 << n_bits) - 1
    return ENCODE_LO + (ENCODE_HI - ENCODE_LO) * int_val / n_max


def _int_to_gray_bits(n: int, n_bits: int) -> np.ndarray:
    """Return the Gray code of unsigned int ``n`` as a uint8 array (MSB first)."""
    gray = n ^ (n >> 1)
    bits = np.zeros(n_bits, dtype=np.uint8)
    for i in range(n_bits):                         # LSB→MSB
        bits[n_bits - 1 - i] = (gray >> i) & 1
    return bits


def _gray_bits_to_int(bits: np.ndarray) -> int:
    """Decode a Gray-code bit array (MSB first) to an unsigned int."""
    n_bits = len(bits)
    # Step 1: Gray → standard binary via prefix XOR
    binary = np.empty(n_bits, dtype=np.uint8)
    binary[0] = bits[0]
    for i in range(1, n_bits):
        binary[i] = binary[i - 1] ^ bits[i]
    # Step 2: binary → int
    result = 0
    for b in binary:
        result = (result << 1) | int(b)
    return result


# ---------------------------------------------------------------------------
# Population class
# ---------------------------------------------------------------------------

class Population:
    """
    A fixed-size population for one of the two coevolving species.

    Parameters
    ----------
    size : int
        Number of individuals.  For user populations this equals ``n_users``
        (943 for ML-100K); for item populations this equals ``n_items`` (1682).
    k : int
        Number of latent dimensions (genes per individual).
    repr_type : {'real', 'binary'}
        Internal storage representation.  Both types expose real-valued
        vectors through ``get_real`` / ``get_all_real`` — the difference is
        purely in how the chromosome is physically stored and what the
        genetic operators see.
    n_bits : int
        Bits per gene for the binary representation (default 8 → 256 levels).
        Ignored when ``repr_type='real'``.
    init_type : {'uniform', 'svd'}
        Initialisation strategy.  ``'svd'`` requires ``R`` and ``role``.
    R : np.ndarray or None
        Rating matrix, shape ``(n_users, n_items)``.  Required for SVD init.
    role : {'user', 'item'}
        Determines which factor matrix (U or V) to use for SVD init.
    seed : int
        RNG seed (default 0).  Passed to ``np.random.default_rng``.

    Attributes
    ----------
    size : int
    k : int
    repr_type : str
    n_bits : int
    fitnesses : np.ndarray, shape (size,), float32
        Initialised to ``-np.inf``.  The engine updates this after each
        call to ``evaluate_population_U / V``.
    """

    def __init__(
        self,
        size:      int,
        k:         int,
        repr_type: str            = "real",
        n_bits:    int            = 8,
        init_type: str            = "uniform",
        R:         Optional[np.ndarray] = None,
        role:      str            = "user",
        seed:      int            = 0,
    ) -> None:

        if repr_type not in ("real", "binary"):
            raise ValueError(f"repr_type must be 'real' or 'binary', got '{repr_type}'")
        if init_type not in ("uniform", "svd"):
            raise ValueError(f"init_type must be 'uniform' or 'svd', got '{init_type}'")
        if init_type == "svd" and R is None:
            raise ValueError("init_type='svd' requires R (rating matrix).")
        if role not in ("user", "item"):
            raise ValueError(f"role must be 'user' or 'item', got '{role}'")

        self.size      = size
        self.k         = k
        self.repr_type = repr_type
        self.n_bits    = n_bits
        self.role      = role
        self._rng      = np.random.default_rng(seed)

        # Fitness array — maximisation convention (fitness = −RMSE)
        self.fitnesses = np.full(size, -np.inf, dtype=np.float32)

        # Build initial real-valued matrix, then encode if needed
        real_init = self._build_real_init(init_type, R, size, k)

        if repr_type == "real":
            self._data = real_init                            # (size, k) float32
        else:
            # Encode each row to binary
            total_bits = k * n_bits
            self._data = np.zeros((size, total_bits), dtype=np.uint8)
            for i in range(size):
                self._data[i] = self._encode(real_init[i])

    # ------------------------------------------------------------------
    # Private: initialisation helpers
    # ------------------------------------------------------------------

    def _build_real_init(
        self,
        init_type: str,
        R:         Optional[np.ndarray],
        size:      int,
        k:         int,
    ) -> np.ndarray:
        """Return a (size, k) float32 array for the requested init strategy."""
        if init_type == "uniform":
            return self._rng.uniform(-0.5, 0.5, (size, k)).astype(np.float32)

        # --- SVD initialisation -----------------------------------------
        # Fill unobserved entries with the global mean so SVD is well-defined.
        R = R.astype(np.float32)
        global_mean  = float(R[R > 0].mean()) if (R > 0).any() else 0.0
        R_filled     = R.copy()
        R_filled[R_filled == 0] = global_mean

        # Truncated SVD
        try:
            U_sv, s_sv, Vt_sv = np.linalg.svd(R_filled, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fallback to uniform if SVD diverges (unlikely but safe)
            return self._rng.uniform(-0.5, 0.5, (size, k)).astype(np.float32)

        kk = min(k, len(s_sv))         # actual rank available

        if self.role == "user":
            # User latent matrix: U * sqrt(s), shape (n_users, kk)
            mean_matrix = (U_sv[:, :kk] * np.sqrt(s_sv[:kk])).astype(np.float32)
        else:
            # Item latent matrix: Vt.T * sqrt(s), shape (n_items, kk)
            mean_matrix = (Vt_sv[:kk, :].T * np.sqrt(s_sv[:kk])).astype(np.float32)

        # Pad to k if SVD rank < k
        if kk < k:
            pad = np.zeros((mean_matrix.shape[0], k - kk), dtype=np.float32)
            mean_matrix = np.concatenate([mean_matrix, pad], axis=1)

        # Ensure mean_matrix has exactly `size` rows (in case R dims ≠ size)
        if mean_matrix.shape[0] != size:
            mean_matrix = mean_matrix[:size] if mean_matrix.shape[0] > size \
                          else np.vstack([mean_matrix,
                                         np.zeros((size - mean_matrix.shape[0], k),
                                                  dtype=np.float32)])

        # Add small Gaussian noise so individuals are distinct
        noise = self._rng.normal(0, 0.01, (size, k)).astype(np.float32)
        return (mean_matrix + noise).astype(np.float32)

    # ------------------------------------------------------------------
    # Gray-code encode / decode (one individual at a time)
    # ------------------------------------------------------------------

    def _encode(self, real_vector: np.ndarray) -> np.ndarray:
        """
        Encode a real-valued latent vector to a flat Gray-code bit array.

        Parameters
        ----------
        real_vector : np.ndarray, shape (k,)

        Returns
        -------
        bits : np.ndarray, shape (k * n_bits,), dtype uint8
            Flat bit string: gene 0 occupies bits [0 : n_bits],
            gene d occupies bits [d*n_bits : (d+1)*n_bits], MSB first.
        """
        bits = np.empty(self.k * self.n_bits, dtype=np.uint8)
        for d in range(self.k):
            int_val = _quantise(float(real_vector[d]), self.n_bits)
            gray    = _int_to_gray_bits(int_val, self.n_bits)
            bits[d * self.n_bits: (d + 1) * self.n_bits] = gray
        return bits

    def _decode(self, binary_vector: np.ndarray) -> np.ndarray:
        """
        Decode a flat Gray-code bit array back to a real-valued latent vector.

        Parameters
        ----------
        binary_vector : np.ndarray, shape (k * n_bits,), dtype uint8

        Returns
        -------
        real_vector : np.ndarray, shape (k,), dtype float32
        """
        real = np.empty(self.k, dtype=np.float32)
        for d in range(self.k):
            bits    = binary_vector[d * self.n_bits: (d + 1) * self.n_bits]
            int_val = _gray_bits_to_int(bits)
            real[d] = _dequantise(int_val, self.n_bits)
        return real

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_real(self, idx: int) -> np.ndarray:
        """
        Return the real-valued latent vector for individual ``idx``.

        Always returns a float32 array of shape ``(k,)`` regardless of the
        internal representation.

        Parameters
        ----------
        idx : int
            Index in ``[0, size)``.  Corresponds to user ``idx`` (role='user')
            or item ``idx`` (role='item').
        """
        if not (0 <= idx < self.size):
            raise IndexError(f"idx={idx} out of range [0, {self.size})")
        if self.repr_type == "real":
            return self._data[idx].copy()
        else:
            return self._decode(self._data[idx])

    def get_all_real(self) -> np.ndarray:
        """
        Return the entire population as a real-valued matrix.

        Returns
        -------
        np.ndarray, shape (size, k), dtype float32
            A fresh copy — modifications do not affect the population.
        """
        if self.repr_type == "real":
            return self._data.copy()
        else:
            real = np.empty((self.size, self.k), dtype=np.float32)
            for i in range(self.size):
                real[i] = self._decode(self._data[i])
            return real

    # ------------------------------------------------------------------
    # Fitness convenience methods
    # ------------------------------------------------------------------

    def best_individual_idx(self) -> int:
        """
        Return the index of the individual with the highest fitness.

        If all fitnesses are ``-inf`` (not yet evaluated), returns 0.
        """
        if np.all(np.isinf(self.fitnesses) & (self.fitnesses < 0)):
            return 0
        return int(np.argmax(self.fitnesses))

    def best_fitness(self) -> float:
        """Return the maximum fitness value in the population."""
        return float(np.max(self.fitnesses))

    def average_fitness(self) -> float:
        """
        Return the mean fitness, ignoring ``-inf`` entries (unevaluated
        individuals).
        """
        valid = self.fitnesses[np.isfinite(self.fitnesses)]
        return float(valid.mean()) if len(valid) > 0 else float("-inf")

    def diversity(self, sample_size: int = 50) -> float:
        """
        Estimate population diversity as the mean pairwise Euclidean distance
        on a random subsample of up to ``sample_size`` individuals.

        Uses the squared-distance identity for O(s²k) complexity instead of
        an explicit O(s²) loop.

        Parameters
        ----------
        sample_size : int
            Maximum subsample size.  The actual sample is
            ``min(sample_size, self.size)``.

        Returns
        -------
        float
            Mean pairwise Euclidean distance.  Returns ``0.0`` for size ≤ 1.
        """
        if self.size <= 1:
            return 0.0

        real = self.get_all_real()               # (size, k)
        s    = min(sample_size, self.size)
        idx  = self._rng.choice(self.size, size=s, replace=False)
        sub  = real[idx]                         # (s, k)

        sq   = np.sum(sub ** 2, axis=1)          # (s,)
        D2   = np.maximum(
            sq[:, None] + sq[None, :] - 2.0 * (sub @ sub.T),
            0.0,
        )
        # Upper-triangle mask (exclude diagonal)
        tri  = np.triu(np.ones((s, s), dtype=bool), k=1)
        dists = np.sqrt(D2[tri])
        return float(dists.mean()) if len(dists) > 0 else 0.0

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"Population(size={self.size}, k={self.k}, "
            f"repr='{self.repr_type}', init='{self.role}', "
            f"best_fitness={self.best_fitness():.4f})"
        )


# ---------------------------------------------------------------------------
# Sanity check — python -X utf8 core/population.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 64)
    print("core/population.py — sanity check")
    print("=" * 64)

    SIZE = 60    # representative population size (small for speed)
    K    = 10    # latent dimensions
    SEED = 7

    # Synthetic R for SVD tests
    rng_d = np.random.default_rng(42)
    R_syn = np.zeros((SIZE, 80), dtype=np.float32)
    mask  = rng_d.random(R_syn.shape) < 0.20
    R_syn[mask] = rng_d.uniform(1, 5, int(mask.sum())).astype(np.float32)

    # ------------------------------------------------------------------
    # Test 1: real / uniform
    # ------------------------------------------------------------------
    print(f"\n[Test 1]  repr='real', init='uniform'")
    p1 = Population(SIZE, K, repr_type="real", init_type="uniform", seed=SEED)
    all1 = p1.get_all_real()
    assert all1.shape == (SIZE, K),  f"shape mismatch: {all1.shape}"
    assert all1.dtype == np.float32, f"dtype mismatch: {all1.dtype}"
    assert all1.min() >= -0.51 and all1.max() <= 0.51, "Values outside init range"
    single = p1.get_real(5)
    assert single.shape == (K,)
    assert np.allclose(single, all1[5]), "get_real inconsistent with get_all_real"
    assert len(p1) == SIZE
    assert p1.best_individual_idx() == 0,  "Should be 0 (all -inf)"
    assert np.isinf(p1.best_fitness()) and p1.best_fitness() < 0
    div1 = p1.diversity()
    assert div1 > 0, "Diversity should be > 0 for uniform init"
    print(f"  shape={all1.shape}, dtype={all1.dtype}")
    print(f"  sample vector (idx=5): {all1[5, :4].round(4)} …")
    print(f"  diversity score: {div1:.4f}")
    print(f"  fitnesses (uninit): all -inf  OK")
    print(f"  get_real / get_all_real consistent  OK")

    # Assign fake fitnesses and test accessors
    p1.fitnesses = rng_d.uniform(-5, -0.5, SIZE).astype(np.float32)
    bi  = p1.best_individual_idx()
    assert p1.fitnesses[bi] == p1.fitnesses.max()
    print(f"  best_individual_idx={bi}, best_fitness={p1.best_fitness():.4f}  OK")
    print(f"  average_fitness={p1.average_fitness():.4f}  OK")

    # ------------------------------------------------------------------
    # Test 2: real / svd  (user role)
    # ------------------------------------------------------------------
    print(f"\n[Test 2]  repr='real', init='svd', role='user'")
    p2 = Population(SIZE, K, repr_type="real", init_type="svd",
                    R=R_syn, role="user", seed=SEED)
    all2 = p2.get_all_real()
    assert all2.shape == (SIZE, K)
    assert all2.dtype == np.float32
    # SVD init should NOT be all zeros (unless R is all zeros)
    assert not np.allclose(all2, 0), "SVD init produced all-zero matrix!"
    div2 = p2.diversity()
    print(f"  shape={all2.shape}, dtype={all2.dtype}")
    print(f"  sample vector (idx=0): {all2[0, :4].round(4)} …")
    print(f"  diversity score: {div2:.4f}")
    print(f"  SVD init non-zero  OK")

    # ------------------------------------------------------------------
    # Test 3: real / svd  (item role)
    # ------------------------------------------------------------------
    print(f"\n[Test 3]  repr='real', init='svd', role='item'")
    p3 = Population(80, K, repr_type="real", init_type="svd",
                    R=R_syn, role="item", seed=SEED)
    all3 = p3.get_all_real()
    assert all3.shape == (80, K)
    print(f"  shape={all3.shape}  OK (item population, 80 items)")

    # ------------------------------------------------------------------
    # Test 4: binary / uniform  — encode/decode round-trip
    # ------------------------------------------------------------------
    print(f"\n[Test 4]  repr='binary', init='uniform', n_bits=8")
    N_BITS = 8
    p4 = Population(SIZE, K, repr_type="binary", n_bits=N_BITS,
                    init_type="uniform", seed=SEED)
    # Check shape
    assert p4._data.shape == (SIZE, K * N_BITS), \
        f"Binary data shape {p4._data.shape} != {(SIZE, K*N_BITS)}"
    assert p4._data.dtype == np.uint8

    # Round-trip test: encode then decode ≈ original (within quantisation error)
    rng_t  = np.random.default_rng(123)
    sample = rng_t.uniform(-0.5, 0.5, K).astype(np.float32)
    bits   = p4._encode(sample)
    back   = p4._decode(bits)
    max_err = float(np.abs(sample - back).max())
    quant_step = (ENCODE_HI - ENCODE_LO) / ((1 << N_BITS) - 1)
    assert max_err <= quant_step + 1e-6, \
        f"Encode→decode error {max_err:.6f} > quantisation step {quant_step:.6f}"
    print(f"  binary data shape: {p4._data.shape}, dtype={p4._data.dtype}  OK")
    print(f"  encode→decode max error: {max_err:.6f}  "
          f"(≤ quant step {quant_step:.6f})  OK")

    # get_all_real() on binary population
    all4 = p4.get_all_real()
    assert all4.shape == (SIZE, K)
    assert all4.dtype == np.float32
    div4 = p4.diversity()
    print(f"  get_all_real() shape={all4.shape}, dtype={all4.dtype}  OK")
    print(f"  diversity score: {div4:.4f}")

    # ------------------------------------------------------------------
    # Test 5: n_bits=16 — higher precision
    # ------------------------------------------------------------------
    print(f"\n[Test 5]  repr='binary', n_bits=16 — precision test")
    p5 = Population(10, K, repr_type="binary", n_bits=16, init_type="uniform")
    sample16 = rng_t.uniform(-3.0, 3.0, K).astype(np.float32)
    bits16   = p5._encode(sample16)
    back16   = p5._decode(bits16)
    max_err16 = float(np.abs(sample16 - back16).max())
    quant_step16 = (ENCODE_HI - ENCODE_LO) / ((1 << 16) - 1)
    assert max_err16 <= quant_step16 + 1e-6
    print(f"  encode→decode max error (16-bit): {max_err16:.8f}  OK")

    # ------------------------------------------------------------------
    # Test 6: Reproducibility
    # ------------------------------------------------------------------
    print(f"\n[Test 6]  Reproducibility (same seed)")
    pa = Population(20, K, repr_type="real", init_type="uniform", seed=99)
    pb = Population(20, K, repr_type="real", init_type="uniform", seed=99)
    assert np.array_equal(pa.get_all_real(), pb.get_all_real()), \
        "Same seed produced different populations!"
    print(f"  Same seed → identical populations  OK")

    # ------------------------------------------------------------------
    # Test 7: __repr__
    # ------------------------------------------------------------------
    print(f"\n[Test 7]  __repr__ and __len__")
    p1.fitnesses = np.array([-1.23] + [-np.inf] * (SIZE - 1), dtype=np.float32)
    p1.fitnesses[3] = -0.5
    print(f"  repr: {repr(p1)}")
    print(f"  len : {len(p1)}  OK")

    print("\n" + "=" * 64)
    print("All 7 population tests passed  OK")
    print("=" * 64)
    print()
    print("Combinations verified:")
    print("  ✓  real  / uniform")
    print("  ✓  real  / svd  (user role)")
    print("  ✓  real  / svd  (item role)")
    print("  ✓  binary / uniform  (n_bits=8)")
    print("  ✓  binary / uniform  (n_bits=16)")
