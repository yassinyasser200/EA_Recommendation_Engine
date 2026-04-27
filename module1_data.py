import numpy as np, pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATOR (MovieLens-like structure)
# ═══════════════════════════════════════════════════════════════════════════════
def gen_data(nu, ni, nr, seed=42):
    rng = np.random.default_rng(seed)
    u_lat, i_lat = rng.normal(0, 1, (nu, 3)), rng.normal(0, 1, (ni, 3))
    rc = np.array(list({(rng.integers(nu), rng.integers(ni)) for _ in range(nr*2)})[:nr])
    r, c = rc[:,0], rc[:,1]
    vals = np.clip(np.round((np.sum(u_lat[r]*i_lat[c], 1) + 3 + rng.normal(0,.5,len(r)))*2)/2, 1, 5)
    return r, c, vals

def parse_upload(file, max_ratings=2000):
    df = pd.read_csv(file, sep='\t', header=None, usecols=[0,1,2], names=['u','i','r'])
    # Sample FIRST so the subsequent remap covers only the retained rows
    if len(df) > max_ratings: df = df.sample(max_ratings, random_state=42)
    # Remap IDs to zero-based contiguous indices (handles non-contiguous / non-zero-based IDs)
    df['u'], _ = pd.factorize(df['u'], sort=True)
    df['i'], _ = pd.factorize(df['i'], sort=True)
    n_users, n_items = df['u'].nunique(), df['i'].nunique()
    return df['u'].values, df['i'].values, df['r'].values.astype(float), n_users, n_items

# ═══════════════════════════════════════════════════════════════════════════════
# REPRESENTATION: Real-valued vs Binary-encoded latent features
# Each individual = full embedding matrix (n_entities × latent_dim)
# ═══════════════════════════════════════════════════════════════════════════════
encode = lambda pop, rep: [((p > 0).astype(float)*2-1) if rep=="binary" else p for p in pop]

# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZATION: Random uniform vs Heuristic (rating-bias seeded)
# Population = list of candidate embedding matrices
# ═══════════════════════════════════════════════════════════════════════════════
def init_pop(ps, n_ent, dim, method, bounds, rng, idx_arr=None, val_arr=None):
    pop = [rng.uniform(bounds[0], bounds[1], (n_ent, dim)) for _ in range(ps)]
    if method == "heuristic" and idx_arr is not None:
        for p in pop:
            for i in range(n_ent):
                m = idx_arr == i
                if m.any(): p[i, 0] = (val_arr[m].mean() - 3) / 2
    return pop

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTRAINT HANDLING: Repair function — clip to valid bounds
# ═══════════════════════════════════════════════════════════════════════════════
repair = lambda ind, b: np.clip(ind, b[0], b[1])

# Exported: initialize_population (alias), evaluate_constraint (alias)
initialize_population = init_pop
evaluate_constraint = repair
