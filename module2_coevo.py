import numpy as np
from module1_data import init_pop, encode, repair

# ═══════════════════════════════════════════════════════════════════════════════
# PARENT SELECTION: Tournament vs Roulette Wheel
# ═══════════════════════════════════════════════════════════════════════════════
def select_parents(fit, n, method, k, rng):
    if method == "tournament":
        cands = rng.integers(0, len(fit), (n, k))
        return cands[np.arange(n), np.argmin(fit[cands], axis=1)]
    inv = 1.0 / (fit - fit.min() + 1e-8)
    return rng.choice(len(fit), n, p=inv/inv.sum())

# Over-selection: 80% from top fraction, 20% from rest
def over_select(fit, n, top_f, rng):
    si = np.argsort(fit); tn = max(1, int(len(fit)*top_f))
    nt = int(n*0.8)
    return np.concatenate([rng.choice(si[:tn], nt), rng.choice(si[tn:], n-nt)])

# ═══════════════════════════════════════════════════════════════════════════════
# SURVIVOR SELECTION: Generational+Elitism vs (μ,λ)-selection
# ═══════════════════════════════════════════════════════════════════════════════
def surv_select(pop, fit, sig, ps):
    idx = np.argsort(fit)[:ps]
    return [pop[i] for i in idx], fit[idx], sig[idx]

# Alias for external import
replace_population = surv_select

# ═══════════════════════════════════════════════════════════════════════════════
# COEVOLUTIONARY ENGINE — Two sub-populations evolving in parallel
# User population: each individual = complete user embedding matrix (n_users × dim)
# Item population: each individual = complete item embedding matrix (n_items × dim)
# ═══════════════════════════════════════════════════════════════════════════════
def coevolve_step():
    """Marker — the full coevolution loop is run_coevo() below."""
    pass

def run_coevo(cfg, rows, cols, vals, seed=42, progress_cb=None):
    # Import variation/hybrid/fitness components from module3
    from module3_variation import (calc_fitness, CX, MUT, adapt_sigma,
                                   fitness_sharing, crowding_dist,
                                   pso_step)

    rng = np.random.default_rng(seed)
    nu, ni, dim, ps = cfg.n_users, cfg.n_items, cfg.latent_dim, cfg.pop_size
    # Initialisation (two sub-populations of embedding matrices)
    u_pop = encode(init_pop(ps,nu,dim,cfg.init_method,cfg.bounds,rng,rows,vals), cfg.representation)
    i_pop = encode(init_pop(ps,ni,dim,cfg.init_method,cfg.bounds,rng,cols,vals), cfg.representation)
    u_sig, i_sig = np.full(ps, cfg.mut_sigma), np.full(ps, cfg.mut_sigma)
    # PSO state (hybrid)
    if cfg.hybrid_pso:
        u_vel = [rng.normal(0,.1,(nu,dim)) for _ in range(ps)]
        i_vel = [rng.normal(0,.1,(ni,dim)) for _ in range(ps)]
        u_pb, i_pb = [p.copy() for p in u_pop], [p.copy() for p in i_pop]
        u_pf, i_pf = np.full(ps, np.inf), np.full(ps, np.inf)
    hist = {"rmse":[], "u_fit":[], "i_fit":[], "diversity":[]}
    best_rmse = np.inf; best_u, best_i = u_pop[0].copy(), i_pop[0].copy()

    for gen in range(cfg.n_gens):
        # Fitness: RMSE + penalty (coevolution: cooperative vs competitive)
        u_fit, i_fit, rmse = calc_fitness(u_pop, i_pop, rows, cols, vals, cfg.reg_lambda, cfg.coevo)
        if rmse < best_rmse:
            best_rmse = rmse
            best_u = u_pop[np.argmin(u_fit)].copy()
            best_i = i_pop[np.argmin(np.abs(i_fit))].copy()
        # Keep best individual at index 0 (reference for fitness eval)
        bi_u, bi_i = int(np.argmin(u_fit)), int(np.argmin(np.abs(i_fit)))
        if bi_u != 0: u_pop[0], u_pop[bi_u] = u_pop[bi_u], u_pop[0]; u_fit[0], u_fit[bi_u] = u_fit[bi_u], u_fit[0]
        if bi_i != 0: i_pop[0], i_pop[bi_i] = i_pop[bi_i], i_pop[0]; i_fit[0], i_fit[bi_i] = i_fit[bi_i], i_fit[0]
        # Diversity: Sharing or Crowding
        if cfg.diversity == "sharing":
            u_fit_s = fitness_sharing(u_fit, u_pop, cfg.share_sigma, cfg.share_alpha)
            i_fit_s = fitness_sharing(i_fit, i_pop, cfg.share_sigma, cfg.share_alpha)
            div = float(np.mean([np.std(p) for p in u_pop]))
        else:
            u_fit_s, i_fit_s = u_fit.copy(), i_fit.copy()
            cd = crowding_dist(u_fit)
            u_fit_s -= 0.1 * np.where(np.isfinite(cd), cd, 0)
            i_cd = crowding_dist(i_fit)
            i_fit_s -= 0.1 * np.where(np.isfinite(i_cd), i_cd, 0)
            div = float(np.mean(cd[np.isfinite(cd)])) if np.any(np.isfinite(cd)) else 0.0
        hist["rmse"].append(float(rmse)); hist["u_fit"].append(float(np.min(u_fit)))
        hist["i_fit"].append(float(np.min(np.abs(i_fit)))); hist["diversity"].append(div)
        if progress_cb: progress_cb(gen, rmse)
        # Parameter control: self-adaptive mutation sigmas
        if cfg.adaptive_mut:
            u_sig, i_sig = adapt_sigma(u_sig, rng), adapt_sigma(i_sig, rng)
        # Evolve each sub-population
        n_ch = int(ps * cfg.lambda_ratio) if cfg.survivor == "mu_lambda" else ps
        def evolve(pop, fit_s, sig, n_c):
            pidx = over_select(fit_s, n_c, cfg.over_top, rng) if cfg.over_selection \
                else select_parents(fit_s, n_c, cfg.parent_sel, cfg.tourn_k, rng)
            children, csig = [], []
            for k in range(0, n_c-1, 2):
                p1, p2 = pop[pidx[k]], pop[pidx[k+1]]
                s = (sig[pidx[k]] + sig[pidx[k+1]]) / 2
                c1, c2 = CX[cfg.crossover](p1, p2, cfg, pop, rng)
                c1 = repair(MUT[cfg.mutation](c1, s, cfg, rng), cfg.bounds)
                c2 = repair(MUT[cfg.mutation](c2, s, cfg, rng), cfg.bounds)
                children.extend([c1, c2]); csig.extend([s, s])
            if len(children) < n_c:
                children.append(pop[pidx[-1]].copy()); csig.append(sig[pidx[-1]])
            return children[:n_c], np.array(csig[:n_c])

        u_ch, u_cs = evolve(u_pop, u_fit_s, u_sig, n_ch)
        i_ch, i_cs = evolve(i_pop, i_fit_s, i_sig, n_ch)
        # Survivor selection
        if cfg.survivor == "elitism":
            au = u_pop + u_ch; asig_u = np.concatenate([u_sig, u_cs])
            ai = i_pop + i_ch; asig_i = np.concatenate([i_sig, i_cs])
            fu, fi, _ = calc_fitness(au, ai, rows, cols, vals, cfg.reg_lambda, cfg.coevo)
            u_pop, u_fit, u_sig = surv_select(au, fu, asig_u, ps)
            i_pop, i_fit, i_sig = surv_select(ai, fi, asig_i, ps)
        else:  # (μ,λ): best from children only
            fu, fi, _ = calc_fitness(u_ch, i_ch, rows, cols, vals, cfg.reg_lambda, cfg.coevo)
            u_pop, u_fit, u_sig = surv_select(u_ch, fu, u_cs, ps)
            i_pop, i_fit, i_sig = surv_select(i_ch, fi, i_cs, ps)
        # Hybrid PSO refinement
        if cfg.hybrid_pso:
            uf2, if2, _ = calc_fitness(u_pop, i_pop, rows, cols, vals, cfg.reg_lambda, cfg.coevo)
            gb_u, gb_i = u_pop[np.argmin(uf2)], i_pop[np.argmin(np.abs(if2))]
            for k in range(ps):
                if uf2[k] < u_pf[k]: u_pb[k] = u_pop[k].copy(); u_pf[k] = uf2[k]
                if np.abs(if2[k]) < i_pf[k]: i_pb[k] = i_pop[k].copy(); i_pf[k] = np.abs(if2[k])
            u_pop, u_vel = pso_step(u_pop, u_vel, u_pb, gb_u, cfg.pso_w, cfg.pso_c1, cfg.pso_c2, rng)
            i_pop, i_vel = pso_step(i_pop, i_vel, i_pb, gb_i, cfg.pso_w, cfg.pso_c1, cfg.pso_c2, rng)
            u_pop = [repair(p, cfg.bounds) for p in u_pop]
            i_pop = [repair(p, cfg.bounds) for p in i_pop]
    return best_u, best_i, best_rmse, hist, u_pop, i_pop
