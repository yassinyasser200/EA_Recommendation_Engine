import streamlit as st, numpy as np, matplotlib.pyplot as plt, pandas as pd
import json, time, io, copy, os
from dataclasses import dataclass

from module1_data import gen_data, parse_upload
from module2_coevo import run_coevo
from module3_variation import calc_fitness

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

# Alias for external import
run_experiment = run_benchmarks

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
def launch_ui():
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

    # Data loading: uploaded file > local u.data > synthetic
    default_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "u.data")
    if uploaded:
        rows, cols, vals, nu_a, ni_a = parse_upload(uploaded)
        cfg.n_users, cfg.n_items = nu_a, ni_a
    elif os.path.isfile(default_data):
        rows, cols, vals, nu_a, ni_a = parse_upload(default_data)
        cfg.n_users, cfg.n_items = nu_a, ni_a
        st.sidebar.success(f"📁 Auto-loaded u.data ({len(vals)} ratings, {nu_a} users, {ni_a} items)")
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
    launch_ui()
