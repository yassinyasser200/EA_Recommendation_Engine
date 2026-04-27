import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# FITNESS: RMSE + regularisation penalty (vectorised)
# Constrained optimisation: maximise accuracy, minimise complexity
# ═══════════════════════════════════════════════════════════════════════════════
def calc_fitness(u_pop, i_pop, r, c, v, reg, coevo):
    # Evaluate each (u_individual, i_individual) pairing
    # For cooperative: pair each user candidate with best item candidate
    n_u, n_i = len(u_pop), len(i_pop)
    u_fit, i_fit = np.full(n_u, np.inf), np.full(n_i, np.inf)
    best_rmse = np.inf
    # Use representative item/user (index 0 = current best) for cross-evaluation
    ref_i, ref_u = i_pop[0], u_pop[0]
    for k in range(n_u):
        preds = np.einsum('j,ij->i', np.zeros(0), np.zeros((0,0))) if False else \
                np.sum(u_pop[k][r] * ref_i[c], axis=1) + 3
        rmse_k = np.sqrt(np.mean((preds - v)**2))
        u_fit[k] = rmse_k + reg * np.mean(u_pop[k]**2)
    for k in range(n_i):
        preds = np.sum(ref_u[r] * i_pop[k][c], axis=1) + 3
        rmse_k = np.sqrt(np.mean((preds - v)**2))
        i_fit[k] = rmse_k + reg * np.mean(i_pop[k]**2)
        if coevo == "competitive": i_fit[k] = -i_fit[k]
    # Global RMSE using best pair
    preds = np.sum(u_pop[np.argmin(u_fit)][r] * i_pop[np.argmin(np.abs(i_fit))][c], axis=1) + 3
    best_rmse = np.sqrt(np.mean((preds - v)**2))
    return u_fit, i_fit, best_rmse

# Alias for external import
compute_fitness = calc_fitness

# ═══════════════════════════════════════════════════════════════════════════════
# RECOMBINATION: Uniform, BLX-α, and Differential Evolution (SOTA)
# ═══════════════════════════════════════════════════════════════════════════════
def cx_uniform(p1, p2, pr, rng):
    if rng.random() > pr: return p1.copy(), p2.copy()
    m = rng.random(p1.shape) < 0.5
    return np.where(m,p1,p2), np.where(m,p2,p1)

def cx_blx(p1, p2, alpha, pr, rng):
    if rng.random() > pr: return p1.copy(), p2.copy()
    lo, hi = np.minimum(p1,p2), np.maximum(p1,p2); d = hi - lo
    return rng.uniform(lo-alpha*d, hi+alpha*d), rng.uniform(lo-alpha*d, hi+alpha*d)

# SOTA Variant: Differential Evolution crossover (DE/rand/1/bin)
def cx_de(target, pop, f, cr, rng):
    idxs = rng.choice(len(pop), 3, replace=False)
    mutant = pop[idxs[0]] + f * (pop[idxs[1]] - pop[idxs[2]])
    mask = rng.random(target.shape) < cr; mask[0] = True
    return np.where(mask, mutant, target), target.copy()

CX = {"uniform": lambda p1,p2,cfg,pop,rng: cx_uniform(p1,p2,cfg.cx_prob,rng),
      "blx_alpha": lambda p1,p2,cfg,pop,rng: cx_blx(p1,p2,cfg.blx_alpha,cfg.cx_prob,rng),
      "de": lambda p1,p2,cfg,pop,rng: cx_de(p1,pop,cfg.de_f,cfg.cx_prob,rng)}

# Alias for external import
def crossover(p1, p2, cfg, pop, rng):
    return CX[cfg.crossover](p1, p2, cfg, pop, rng)

# ═══════════════════════════════════════════════════════════════════════════════
# MUTATION: Gaussian vs Polynomial
# ═══════════════════════════════════════════════════════════════════════════════
def mut_gauss(ind, sigma, rate, rng):
    m = rng.random(ind.shape) < rate
    return ind + m * rng.normal(0, sigma, ind.shape)

def mut_poly(ind, eta, rate, brange, rng):
    m = rng.random(ind.shape) < rate; u = rng.random(ind.shape)
    delta = np.where(u<.5, (2*u)**(1/(eta+1))-1, 1-(2*(1-u))**(1/(eta+1)))
    return ind + m * delta * brange

MUT = {"gaussian": lambda ind,sig,cfg,rng: mut_gauss(ind,sig,cfg.mut_rate,rng),
       "polynomial": lambda ind,sig,cfg,rng: mut_poly(ind,cfg.poly_eta,cfg.mut_rate,cfg.bounds[1]-cfg.bounds[0],rng)}

# Alias for external import
def mutate(ind, sig, cfg, rng):
    return MUT[cfg.mutation](ind, sig, cfg, rng)

# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER CONTROL: Self-adaptive mutation step sizes (τ · N(0,1))
# ═══════════════════════════════════════════════════════════════════════════════
adapt_sigma = lambda sig, rng, tau=0.1: np.clip(sig*np.exp(tau*rng.normal(0,1,sig.shape)), 1e-4, 2.0)

# ═══════════════════════════════════════════════════════════════════════════════
# DIVERSITY: Fitness Sharing vs Crowding Distance
# ═══════════════════════════════════════════════════════════════════════════════
def fitness_sharing(fit, pop_list, sigma, alpha):
    # Flatten each individual to 1D for distance computation
    flat = np.array([p.ravel() for p in pop_list])
    d = np.sqrt(((flat[:,None]-flat[None,:])**2).sum(2))
    sh = np.where(d < sigma, 1-(d/sigma)**alpha, 0).sum(1)
    return fit * np.maximum(sh, 1)

def crowding_dist(fit):
    n = len(fit); cd = np.zeros(n); si = np.argsort(fit)
    if n < 3: return np.full(n, np.inf)
    cd[si[0]] = cd[si[-1]] = np.inf
    rng_f = fit[si[-1]] - fit[si[0]] + 1e-10
    for i in range(1, n-1): cd[si[i]] = (fit[si[i+1]]-fit[si[i-1]]) / rng_f
    return cd

# ═══════════════════════════════════════════════════════════════════════════════
# HYBRID: PSO refinement step on best individual each generation
# ═══════════════════════════════════════════════════════════════════════════════
def pso_step(pop, vel, pbest, gbest, w, c1, c2, rng):
    new_pop, new_vel = [], []
    for p, v, pb in zip(pop, vel, pbest):
        r1, r2 = rng.random(p.shape), rng.random(p.shape)
        nv = w*v + c1*r1*(pb-p) + c2*r2*(gbest-p)
        new_vel.append(nv); new_pop.append(p + nv)
    return new_pop, new_vel

# Alias for external import
hybrid_refine = pso_step
