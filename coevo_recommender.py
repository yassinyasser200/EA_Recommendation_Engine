import streamlit as st, numpy as np, matplotlib.pyplot as plt, pandas as pd
import json, time, io, copy
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG & HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class Cfg:
    n_users:int=50; n_items:int=30; n_ratings:int=500; latent_dim:int=8
    pop_size:int=30; n_gens:int=100; elite_frac:float=0.1; tourn_k:int=3
    mut_rate:float=0.15; mut_sigma:float=0.3; cx_prob:float=0.8; blx_alpha:float=0.5
    reg_lambda:float=0.01; bounds:tuple=(-3.0,3.0)
    coevo:str="cooperative"       # cooperative | competitive
    parent_sel:str="tournament"   # tournament | roulette
    crossover:str="blx_alpha"     # uniform | blx_alpha | de
    mutation:str="gaussian"       # gaussian | polynomial
    survivor:str="elitism"        # elitism | mu_lambda
    representation:str="real"     # real | binary
    init_method:str="random"      # random | heuristic
    diversity:str="sharing"       # sharing | crowding
    share_sigma:float=2.0; share_alpha:float=1.0; poly_eta:float=20.0
    de_f:float=0.8; lambda_ratio:float=2.0
    over_selection:bool=False; over_top:float=0.2
    hybrid_pso:bool=False; pso_w:float=0.7; pso_c1:float=1.5; pso_c2:float=1.5
    adaptive_mut:bool=True
SEEDS = list(range(1, 31))  # 30 fixed seeds for benchmarking

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
    df['u'], _ = pd.factorize(df['u']); df['i'], _ = pd.factorize(df['i'])
    if len(df) > max_ratings: df = df.sample(max_ratings, random_state=42)
    return df['u'].values, df['i'].values, df['r'].values.astype(float), df['u'].nunique(), df['i'].nunique()

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

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTRAINT HANDLING: Repair function — clip to valid bounds
# ═══════════════════════════════════════════════════════════════════════════════
repair = lambda ind, b: np.clip(ind, b[0], b[1])

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
# SURVIVOR SELECTION: Generational+Elitism vs (μ,λ)-selection
# ═══════════════════════════════════════════════════════════════════════════════
def surv_select(pop, fit, sig, ps):
    idx = np.argsort(fit)[:ps]
    return [pop[i] for i in idx], fit[idx], sig[idx]

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

# ═══════════════════════════════════════════════════════════════════════════════
# COEVOLUTIONARY ENGINE — Two sub-populations evolving in parallel
# User population: each individual = complete user embedding matrix (n_users × dim)
# Item population: each individual = complete item embedding matrix (n_items × dim)
# ═══════════════════════════════════════════════════════════════════════════════
def run_coevo(cfg, rows, cols, vals, seed=42, progress_cb=None):
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

# ═══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATIONS: predict top-N items for each user
# ═══════════════════════════════════════════════════════════════════════════════
def recommend(best_u, best_i, top_n=5):
    scores = best_u @ best_i.T + 3
    return [np.argsort(-scores[u])[:top_n] for u in range(len(best_u))]

# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARKING: 30 independent runs per setting, fixed seeds
# ═══════════════════════════════════════════════════════════════════════════════
BENCH_PARAMS = {
    "parent_sel": ["tournament","roulette"], "crossover": ["uniform","blx_alpha","de"],
    "mutation": ["gaussian","polynomial"], "survivor": ["elitism","mu_lambda"],
    "representation": ["real","binary"], "init_method": ["random","heuristic"],
    "diversity": ["sharing","crowding"]
}

def run_benchmarks(cfg, r, c, v, n_runs=5, progress_cb=None):
    results = {}; total = sum(len(vs) for vs in BENCH_PARAMS.values()); done = 0
    for param, values in BENCH_PARAMS.items():
        results[param] = {}; orig = getattr(cfg, param)
        for val in values:
            setattr(cfg, param, val)
            rmses = [run_coevo(cfg, r, c, v, s)[2] for s in SEEDS[:n_runs]]
            results[param][val] = {"mean":float(np.mean(rmses)), "std":float(np.std(rmses)), "best":float(np.min(rmses))}
            done += 1
            if progress_cb: progress_cb(done, total)
        setattr(cfg, param, orig)
    return results

def plot_benchmark(results):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8)); axes = axes.flatten()
    for idx, (param, data) in enumerate(results.items()):
        if idx >= 8: break
        ax = axes[idx]; names = list(data.keys())
        means = [data[n]["mean"] for n in names]; stds = [data[n]["std"] for n in names]
        bars = ax.bar(names, means, yerr=stds, capsize=4, color=plt.cm.Set2(np.linspace(0,1,len(names))), edgecolor='#333')
        ax.set_title(param.replace('_',' ').title(), fontweight='bold', fontsize=10)
        ax.set_ylabel("RMSE"); ax.tick_params(labelsize=8)
        for bar, m in zip(bars, means): ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{m:.3f}', ha='center', fontsize=7)
    if len(results) < 8: axes[-1].axis('off')
    plt.tight_layout(); return fig

# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="🧬 CoEvo Recommender", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""<style>
    .block-container{padding-top:1rem} .stTabs [data-baseweb="tab"]{font-size:14px;font-weight:600}
    div[data-testid="stMetric"]{background:#1a1a2e;border-radius:10px;padding:12px;border:1px solid #333}
    </style>""", unsafe_allow_html=True)
    st.title("🧬 Adaptive Recommendation Engine — Coevolutionary Algorithms")
    st.caption("Two sub-populations (Users & Items) co-evolve to learn latent factor recommendations")

    with st.sidebar:
        st.header("⚙️ Configuration")
        uploaded = st.file_uploader("📂 Upload ratings (tab-separated: user item rating)", type=["csv","tsv","txt","data"])
        st.subheader("Data")
        c1, c2 = st.columns(2)
        nu = c1.number_input("Users", 10, 500, 50); ni = c2.number_input("Items", 10, 300, 30)
        nr = st.number_input("Ratings", 50, 5000, 500)
        st.subheader("Algorithm")
        c1, c2 = st.columns(2)
        ps = c1.number_input("Pop Size", 5, 100, 20); ng = c2.number_input("Generations", 10, 500, 100)
        ld = st.number_input("Latent Dim", 2, 32, 8)
        coevo = st.selectbox("Coevolution Mode", ["cooperative","competitive"])
        st.subheader("Operators")
        psel = st.selectbox("Parent Selection", ["tournament","roulette"])
        cxop = st.selectbox("Crossover", ["blx_alpha","uniform","de"])
        mtop = st.selectbox("Mutation", ["gaussian","polynomial"])
        svop = st.selectbox("Survivor Selection", ["elitism","mu_lambda"])
        st.subheader("Representation & Init")
        rep = st.selectbox("Representation", ["real","binary"])
        ini = st.selectbox("Initialisation", ["random","heuristic"])
        div = st.selectbox("Diversity Method", ["sharing","crowding"])
        st.subheader("Bonus Features")
        over = st.checkbox("Over-selection", False)
        pso = st.checkbox("Hybrid PSO", False)
        adapt = st.checkbox("Adaptive Mutation σ", True)
        st.subheader("Hyperparams")
        mr = st.slider("Mutation Rate", 0.01, 0.5, 0.15)
        cp = st.slider("Crossover Prob", 0.1, 1.0, 0.8)
        rl = st.slider("Regularisation λ", 0.0, 0.1, 0.01, 0.001)

    cfg = Cfg(n_users=nu, n_items=ni, n_ratings=nr, latent_dim=ld, pop_size=ps, n_gens=ng,
              coevo=coevo, parent_sel=psel, crossover=cxop, mutation=mtop, survivor=svop,
              representation=rep, init_method=ini, diversity=div, over_selection=over,
              hybrid_pso=pso, adaptive_mut=adapt, mut_rate=mr, cx_prob=cp, reg_lambda=rl)

    if uploaded:
        rows, cols, vals, nu_a, ni_a = parse_upload(uploaded)
        cfg.n_users, cfg.n_items = nu_a, ni_a
    else:
        rows, cols, vals = gen_data(cfg.n_users, cfg.n_items, cfg.n_ratings)

    tab1, tab2, tab3 = st.tabs(["🚀 Run Evolution", "📊 Benchmarking", "🎓 Educational Mode"])

    with tab1:
        col_run, col_info = st.columns([3, 1])
        with col_info:
            st.metric("Ratings", len(vals)); st.metric("Users", cfg.n_users)
            st.metric("Items", cfg.n_items); st.metric("Mode", cfg.coevo.title())
        with col_run:
            if st.button("▶️ Run Coevolution", type="primary", use_container_width=True):
                pbar = st.progress(0, text="Evolving...")
                def cb(g, rmse): pbar.progress((g+1)/cfg.n_gens, text=f"Gen {g+1}/{cfg.n_gens} — RMSE: {rmse:.4f}")
                t0 = time.time()
                bu, bi, brmse, hist, uf, ifn = run_coevo(cfg, rows, cols, vals, seed=42, progress_cb=cb)
                elapsed = time.time() - t0
                pbar.progress(1.0, text="✅ Complete!")
                st.success(f"Best RMSE: **{brmse:.4f}** in {elapsed:.1f}s")
                st.session_state.update({"hist":hist, "best_u":bu, "best_i":bi, "cfg_s":copy.deepcopy(cfg)})

        if "hist" in st.session_state:
            h = st.session_state["hist"]
            st.subheader("📈 Fitness Curves")
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].plot(h["rmse"], color='#e74c3c', lw=2); axes[0].set_title("RMSE Convergence", fontweight='bold')
            axes[0].set_xlabel("Generation"); axes[0].set_ylabel("RMSE"); axes[0].grid(alpha=0.3)
            axes[1].plot(h["u_fit"], label="User", color='#3498db', lw=2)
            axes[1].plot(h["i_fit"], label="Item", color='#2ecc71', lw=2)
            axes[1].set_title("Best Individual Fitness", fontweight='bold'); axes[1].legend(); axes[1].grid(alpha=0.3)
            axes[2].plot(h["diversity"], color='#9b59b6', lw=2); axes[2].set_title("Population Diversity", fontweight='bold')
            axes[2].set_xlabel("Generation"); axes[2].grid(alpha=0.3)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.subheader("🎯 Top-5 Recommendations")
            recs = recommend(st.session_state["best_u"], st.session_state["best_i"])
            n_show = min(10, len(recs))
            rec_df = pd.DataFrame({f"User {u}": [f"Item {i}" for i in recs[u]] for u in range(n_show)})
            st.dataframe(rec_df, use_container_width=True)

    with tab2:
        st.subheader("🔬 Automated Benchmarking (30 seeds)")
        st.caption("Compares all operator/strategy variants across independent runs")
        c1, c2 = st.columns(2)
        n_br = c1.number_input("Runs per setting", 3, 30, 5)
        bg = c2.number_input("Generations (bench)", 10, 200, 30)
        if st.button("🏃 Run Full Benchmark", type="primary"):
            bcfg = copy.deepcopy(cfg); bcfg.n_gens = bg; bcfg.pop_size = min(cfg.pop_size, 15)
            pbar = st.progress(0, "Benchmarking...")
            def bcb(d, t): pbar.progress(d/t, f"Setting {d}/{t}")
            results = run_benchmarks(bcfg, rows, cols, vals, n_br, bcb)
            pbar.progress(1.0, "✅ Benchmark Complete!")
            st.session_state["bench"] = results
        if "bench" in st.session_state:
            res = st.session_state["bench"]
            st.subheader("📊 Comparison Plots")
            fig = plot_benchmark(res); st.pyplot(fig); plt.close()
            st.subheader("📋 Summary Table")
            tbl = [{"Parameter":p, "Value":v, "Mean RMSE":f'{s["mean"]:.4f}', "Std":f'{s["std"]:.4f}', "Best":f'{s["best"]:.4f}'}
                   for p, d in res.items() for v, s in d.items()]
            st.dataframe(pd.DataFrame(tbl), use_container_width=True)
            buf = io.StringIO(); json.dump(res, buf, indent=2)
            st.download_button("⬇️ Download Results (JSON)", buf.getvalue(), "benchmark_results.json", "application/json")

    with tab3:
        st.subheader("🎓 Educational Visualisation — Side-by-Side Comparison")
        st.caption("Compare how different strategies affect evolution in real-time")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Configuration A**")
            a_co = st.selectbox("Coevo A", ["cooperative","competitive"], key="a_co")
            a_cx = st.selectbox("Crossover A", ["blx_alpha","uniform","de"], key="a_cx")
            a_mt = st.selectbox("Mutation A", ["gaussian","polynomial"], key="a_mt")
            a_dv = st.selectbox("Diversity A", ["sharing","crowding"], key="a_dv")
        with c2:
            st.markdown("**Configuration B**")
            b_co = st.selectbox("Coevo B", ["competitive","cooperative"], key="b_co")
            b_cx = st.selectbox("Crossover B", ["uniform","blx_alpha","de"], key="b_cx")
            b_mt = st.selectbox("Mutation B", ["polynomial","gaussian"], key="b_mt")
            b_dv = st.selectbox("Diversity B", ["crowding","sharing"], key="b_dv")
        eg = st.number_input("Generations (edu)", 10, 200, 50, key="eg")

        if st.button("🎬 Run Side-by-Side Comparison", type="primary"):
            ca = copy.deepcopy(cfg); ca.n_gens = eg
            ca.coevo, ca.crossover, ca.mutation, ca.diversity = a_co, a_cx, a_mt, a_dv
            cb_ = copy.deepcopy(cfg); cb_.n_gens = eg
            cb_.coevo, cb_.crossover, cb_.mutation, cb_.diversity = b_co, b_cx, b_mt, b_dv
            pbar = st.progress(0, "Running A...")
            def cba(g, r): pbar.progress((g+1)/(eg*2), f"Config A — Gen {g+1}")
            _, _, ra, ha, _, _ = run_coevo(ca, rows, cols, vals, 42, cba)
            def cbb(g, r): pbar.progress(0.5+(g+1)/(eg*2), f"Config B — Gen {g+1}")
            _, _, rb, hb, _, _ = run_coevo(cb_, rows, cols, vals, 42, cbb)
            pbar.progress(1.0, "✅ Done!")

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            axes[0].plot(ha["rmse"], label=f"A ({a_co})", color='#e74c3c', lw=2)
            axes[0].plot(hb["rmse"], label=f"B ({b_co})", color='#3498db', lw=2, ls='--')
            axes[0].set_title("RMSE Convergence", fontweight='bold'); axes[0].legend(); axes[0].grid(alpha=0.3)
            axes[1].plot(ha["u_fit"], label="A User", color='#e74c3c', lw=2)
            axes[1].plot(hb["u_fit"], label="B User", color='#3498db', lw=2, ls='--')
            axes[1].plot(ha["i_fit"], label="A Item", color='#e67e22', lw=1.5, alpha=.7)
            axes[1].plot(hb["i_fit"], label="B Item", color='#1abc9c', lw=1.5, ls='--', alpha=.7)
            axes[1].set_title("Fitness Landscape", fontweight='bold'); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
            axes[2].plot(ha["diversity"], label="A", color='#e74c3c', lw=2)
            axes[2].plot(hb["diversity"], label="B", color='#3498db', lw=2, ls='--')
            axes[2].set_title("Diversity Dynamics", fontweight='bold'); axes[2].legend(); axes[2].grid(alpha=0.3)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            mc1, mc2 = st.columns(2)
            mc1.metric("Config A Final RMSE", f"{ra:.4f}")
            mc2.metric("Config B Final RMSE", f"{rb:.4f}")
            st.success(f"🏆 Configuration **{'A' if ra < rb else 'B'}** achieved lower RMSE!")

if __name__ == "__main__":
    main()
