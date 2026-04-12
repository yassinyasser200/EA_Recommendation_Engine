"""
ui/app.py
=========
Streamlit dashboard for the Adaptive Coevolutionary Recommender Engine.

Tabs
----
1. Run Evolution   — configure all EA parameters, run the engine, watch the
                     animated Plotly fitness curve, inspect per-generation metrics.
2. Compare Runs    — table + bar chart of every completed run stored in session
                     state; export as CSV.
3. Step-Through    — educational mode: advance one generation at a time on a
                     small population, visualise population movement in PCA space,
                     highlight parents / offspring / survivors.
4. Algorithm Guide — self-contained explanation of every EA concept used.

Run with:
    streamlit run ui/app.py
"""

from __future__ import annotations

import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Make sibling packages importable regardless of working directory
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from core.coevo_engine import CoevolutionaryEngine
from core.fitness import (
    select_collaborators,
    evaluate_population_U,
    evaluate_population_V,
    compute_test_rmse,
)
from core.operators import tournament_selection, rank_roulette_selection
from experiments.config import (
    BASE_CONFIG,
    get_config,
    list_configs,
    describe_config,
    make_run_seeds,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CoEvo Recommender Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Premium CSS Theme
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Page background ── */
    .stApp { background: linear-gradient(135deg, #0d1117 0%, #161b27 50%, #0d1117 100%); }

    /* ── Hero header ── */
    .hero-header {
        background: linear-gradient(120deg, #6c63ff22 0%, #ff658422 100%);
        border: 1px solid #6c63ff44;
        border-radius: 16px;
        padding: 2rem 2.5rem 1.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(8px);
    }
    .hero-header h1 {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #6c63ff, #ff6584, #00d4aa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0 0 0.3rem;
    }
    .hero-header p { color: #8892a4; font-size: 1rem; margin: 0; }

    /* ── Metric cards ── */
    div[data-testid="metric-container"] {
        background: #1a1f2e;
        border: 1px solid #2d3452;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 0 18px #6c63ff18;
        transition: box-shadow 0.3s;
    }
    div[data-testid="metric-container"]:hover { box-shadow: 0 0 28px #6c63ff35; }
    div[data-testid="metric-container"] label { color: #8892a4 !important; font-size: 0.78rem !important; }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e8eaf6 !important; font-size: 1.6rem !important; font-weight: 600;
    }

    /* ── Section labels ── */
    .section-label {
        color: #6c63ff; font-size: 0.75rem; font-weight: 600;
        letter-spacing: 0.12em; text-transform: uppercase;
        border-left: 3px solid #6c63ff; padding-left: 0.6rem;
        margin: 1.2rem 0 0.6rem;
    }

    /* ── Step badge ── */
    .step-badge {
        display: inline-block;
        background: linear-gradient(90deg, #6c63ff, #9c8dff);
        color: white; border-radius: 20px;
        padding: 0.25rem 0.8rem; font-size: 0.8rem; font-weight: 600;
        margin-right: 0.5rem;
    }
    .step-done  { background: linear-gradient(90deg, #00d4aa, #00b090) !important; }
    .step-next  { background: linear-gradient(90deg, #ff6584, #ff4066) !important; }

    /* ── Info card ── */
    .info-card {
        background: #1a1f2e; border: 1px solid #2d3452; border-radius: 12px;
        padding: 1.2rem 1.5rem; margin: 0.8rem 0;
    }
    .info-card h4 { color: #6c63ff; margin-top: 0; font-size: 1rem; }
    .info-card p,li { color: #c0c7d4; font-size: 0.9rem; line-height: 1.6; }
    .info-card code {
        background: #252b40; color: #ff6584;
        border-radius: 4px; padding: 0.1rem 0.35rem;
        font-family: 'JetBrains Mono', monospace; font-size: 0.85em;
    }

    /* ── Algorithm pseudocode ── */
    .pseudo {
        background: #0d1117; border: 1px solid #30363d;
        border-radius: 10px; padding: 1rem 1.4rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem; line-height: 1.8; color: #c9d1d9;
    }
    .pseudo .hl  { color: #6c63ff; font-weight: 700; }
    .pseudo .kw  { color: #ff7b72; }
    .pseudo .num { color: #79c0ff; }
    .pseudo .cmt { color: #8b949e; }
    .pseudo .cur { background: #6c63ff22; border-radius: 4px; padding: 0 4px; }

    /* ── Sidebar styling ── */
    section[data-testid="stSidebar"] { background: #111520 !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label { color: #c0c7d4 !important; font-size: 0.85rem !important; }

    /* ── Progress bar ── */
    .stProgress > div > div > div { background: linear-gradient(90deg, #6c63ff, #00d4aa); }

    /* ── Tabs ── */
    button[data-baseweb="tab"] { font-weight: 500; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #6c63ff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers: PCA projection (no sklearn needed — pure numpy SVD)
# ---------------------------------------------------------------------------

def pca_2d(matrix: np.ndarray) -> np.ndarray:
    """Project (N, k) to (N, 2) using truncated SVD."""
    if matrix.shape[1] <= 2:
        return matrix[:, :2]
    centered = matrix - matrix.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:2].T   # (N, 2)


# ---------------------------------------------------------------------------
# Data loading (cached so it runs once per session)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading dataset …")
def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Try to load MovieLens-100K; fall back to synthetic data if unavailable.

    Returns
    -------
    R_train      : (n_users, n_items) float32 training matrix
    test_pairs   : (n_test, 2) int
    test_ratings : (n_test,) float32
    source_label : human-readable description of what was loaded
    """
    data_path = "data/ml-100k/u.data"
    try:
        from core.data_loader import load_movielens  # type: ignore
        R_train, _, test_pairs, test_ratings = load_movielens(data_path)
        label = f"MovieLens-100K  ({R_train.shape[0]} users, {R_train.shape[1]} items)"
        return R_train, test_pairs, test_ratings, label
    except Exception:
        pass

    # Fallback: synthetic data (fast enough for interactive demo)
    rng  = np.random.default_rng(42)
    NU, NI = 120, 180
    R    = np.zeros((NU, NI), dtype=np.float32)
    mask = rng.random((NU, NI)) < 0.10
    R[mask] = rng.uniform(1, 5, int(mask.sum())).astype(np.float32)
    or_, oc_ = np.nonzero(mask)
    n_test   = max(20, int(len(or_) * 0.15))
    idx      = rng.choice(len(or_), n_test, replace=False)
    pairs    = np.column_stack([or_[idx], oc_[idx]])
    ratings  = R[or_[idx], oc_[idx]]
    label    = f"Synthetic data ({NU} users, {NI} items) — place u.data for real data"
    return R, pairs, ratings, label


# ---------------------------------------------------------------------------
# StepRecord — captures every intermediate array from one generation
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    gen:              int
    U_before:         np.ndarray
    fit_before:       np.ndarray
    partner_idx:      np.ndarray
    offspring_raw:    np.ndarray   # after crossover, before mutation
    offspring_mut:    np.ndarray   # after mutation
    offspring_fits:   np.ndarray
    U_after:          np.ndarray
    fit_after:        np.ndarray
    sigma_mean_before: float
    sigma_mean_after:  float
    n_improved:       int
    best_rmse_before: float
    best_rmse_after:  float
    test_rmse:        Optional[float] = None


# ---------------------------------------------------------------------------
# StepThroughEngine — thin, observable wrapper for educational stepping
# ---------------------------------------------------------------------------

class StepThroughEngine:
    """
    Tiny coevolutionary engine that exposes one-generation stepping and
    returns full intermediate arrays for visualisation.

    Uses only a subset of users/items for speed (configurable).
    """

    def __init__(
        self,
        R_train:        np.ndarray,
        R_test_pairs:   Optional[np.ndarray],
        R_test_ratings: Optional[np.ndarray],
        config:         Dict,
    ) -> None:
        self.cfg        = {**BASE_CONFIG, **config}
        self.R_train    = R_train.astype(np.float32)
        self.test_pairs = R_test_pairs
        self.test_rates = (R_test_ratings.astype(np.float32)
                           if R_test_ratings is not None else None)
        self.rng = np.random.default_rng(int(self.cfg.get("seed", 42)))
        k        = int(self.cfg["k"])

        # Initialise small populations
        n_u, n_i = R_train.shape
        self.U   = self.rng.uniform(-0.5, 0.5, (n_u, k)).astype(np.float32)
        self.V   = self.rng.uniform(-0.5, 0.5, (n_i, k)).astype(np.float32)
        sig_v    = float(self.cfg.get("sigma_init", 0.3))
        self.sig_U = np.full((n_u, k), sig_v, dtype=np.float32)
        self.sig_V = np.full((n_i, k), sig_v, dtype=np.float32)

        # Bootstrap fitness
        fake_fv = np.zeros(n_i, dtype=np.float32)
        fake_fu = np.zeros(n_u, dtype=np.float32)
        best_v  = 0
        cv = select_collaborators(n_i, best_v, int(self.cfg["k_random"]), self.rng)
        self.fit_U = evaluate_population_U(self.U, self.V, self.R_train, cv)
        cu = select_collaborators(n_u, 0, int(self.cfg["k_random"]), self.rng)
        self.fit_V = evaluate_population_V(self.U, self.V, self.R_train, cu)

        self.gen = 0

    # -- Batch crossover -----------------------------------------------
    def _crossover(self, inds, sigs, partner_idx):
        partners = inds[partner_idx]
        p_sigs   = sigs[partner_idx]
        xover    = self.cfg["crossover"]
        if xover == "uniform":
            pm = self.rng.random(inds.shape) < float(self.cfg.get("p_swap", 0.5))
            off = np.where(pm, partners, inds).astype(np.float32)
        else:
            alpha   = float(self.cfg.get("alpha_blx", 0.5))
            lo  = np.minimum(inds, partners);  hi  = np.maximum(inds, partners)
            ext = hi - lo
            u   = self.rng.random(inds.shape).astype(np.float32)
            off = (lo - alpha*ext + u * (hi - lo + 2*alpha*ext)).astype(np.float32)
        o_sigs = 0.5 * (sigs + p_sigs).astype(np.float32)
        return off, o_sigs

    # -- Batch mutation ------------------------------------------------
    def _mutate(self, off, o_sigs):
        mut = self.cfg["mutation"]
        if mut == "gaussian":
            tau   = 1.0 / np.sqrt(off.shape[1])
            ns    = self.rng.standard_normal(off.shape).astype(np.float32)
            nx    = self.rng.standard_normal(off.shape).astype(np.float32)
            new_s = np.maximum(o_sigs * np.exp(tau * ns), float(self.cfg.get("sigma_min", 1e-5)))
            return (off + new_s * nx).astype(np.float32), new_s
        else:
            p     = float(self.cfg.get("p_reset", 0.1))
            lo_r  = float(self.cfg.get("reset_low",  -0.5))
            hi_r  = float(self.cfg.get("reset_high",  0.5))
            mask  = self.rng.random(off.shape) < p
            vals  = self.rng.uniform(lo_r, hi_r, off.shape).astype(np.float32)
            return np.where(mask, vals, off).astype(np.float32), o_sigs

    # -- One generation step -------------------------------------------
    def step(self) -> StepRecord:
        n_u = len(self.U);  k = self.cfg["k"]
        sel  = self.cfg["selection"]

        # Selection
        if sel == "tournament":
            pidx = tournament_selection(self.fit_U, n_u, int(self.cfg["tau"]), self.rng)
        else:
            pidx = rank_roulette_selection(self.fit_U, n_u, self.rng)

        U_before   = self.U.copy()
        fit_before = self.fit_U.copy()
        sig_before = self.sig_U.copy()

        # Crossover + mutation
        off_raw, off_sigs = self._crossover(self.U, self.sig_U, pidx)
        off_mut, off_sigs = self._mutate(off_raw, off_sigs)

        # Evaluate offspring
        bv = int(np.argmax(self.fit_V))
        cv = select_collaborators(len(self.V), bv, int(self.cfg["k_random"]), self.rng)
        off_fits = evaluate_population_U(off_mut, self.V, self.R_train, cv)

        # Survivor selection
        if self.cfg["survivor_selection"] == "mu_plus_lambda":
            improve = off_fits > self.fit_U
            self.U[improve]     = off_mut[improve]
            self.sig_U[improve] = off_sigs[improve]
            self.fit_U[improve] = off_fits[improve]
        else:
            self.U[:]     = off_mut
            self.sig_U[:] = off_sigs
            self.fit_U[:] = off_fits

        # Re-evaluate V
        bu = int(np.argmax(self.fit_U))
        cu = select_collaborators(n_u, bu, int(self.cfg["k_random"]), self.rng)
        self.fit_V = evaluate_population_V(self.U, self.V, self.R_train, cu)

        n_imp = int((self.fit_U > fit_before).sum())
        trm   = None
        if self.test_pairs is not None and self.test_rates is not None:
            trm = float(compute_test_rmse(self.U, self.V, self.test_pairs, self.test_rates))

        rec = StepRecord(
            gen=self.gen,
            U_before=U_before,        fit_before=fit_before,
            partner_idx=pidx,
            offspring_raw=off_raw,    offspring_mut=off_mut,
            offspring_fits=off_fits,
            U_after=self.U.copy(),    fit_after=self.fit_U.copy(),
            sigma_mean_before=float(sig_before.mean()),
            sigma_mean_after=float(self.sig_U.mean()),
            n_improved=n_imp,
            best_rmse_before=float(-fit_before.max()),
            best_rmse_after=float(-self.fit_U.max()),
            test_rmse=trm,
        )
        self.gen += 1
        return rec


# ---------------------------------------------------------------------------
# Plotly chart factories
# ---------------------------------------------------------------------------

_DARK_LAYOUT = dict(
    paper_bgcolor="#111520",
    plot_bgcolor="#0d1117",
    font=dict(family="Inter", color="#c0c7d4"),
    margin=dict(l=40, r=20, t=40, b=40),
)

def fitness_curve_fig(log: List[Dict]) -> go.Figure:
    """Animated dual-line fitness curve (U and V populations)."""
    gens   = [r["gen"] for r in log]
    rmse_U = [-r["best_fit_U"] for r in log]
    rmse_V = [-r["best_fit_V"] for r in log]
    avg_U  = [-r["avg_fit_U"]  for r in log]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gens, y=rmse_U, name="Best RMSE (Users)",
        line=dict(color="#6c63ff", width=2.5),
        fill="tozeroy", fillcolor="rgba(108,99,255,0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=gens, y=rmse_V, name="Best RMSE (Items)",
        line=dict(color="#ff6584", width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=gens, y=avg_U, name="Avg RMSE (Users)",
        line=dict(color="#00d4aa", width=1.5, dash="dash"),
    ))
    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text="Fitness Convergence Curve", font=dict(size=14)),
        xaxis=dict(title="Generation", gridcolor="#1e2540", showgrid=True),
        yaxis=dict(title="RMSE  (lower is better)", gridcolor="#1e2540"),
        legend=dict(bgcolor="#111520", bordercolor="#2d3452", borderwidth=1),
        hovermode="x unified",
    )
    return fig


def population_pca_fig(
    populations: Dict[str, np.ndarray],
    fitnesses:   Dict[str, np.ndarray],
    title:       str = "Population in PCA Space",
    marker_size: int = 8,
) -> go.Figure:
    """
    Scatter of individuals projected to 2D via PCA.
    ``populations`` keys: label strings.  Values: (N, k) arrays.
    ``fitnesses``   keys: same labels.    Values: (N,) fitness arrays.
    """
    fig = go.Figure()
    colours = {"before": "#6c63ff", "after": "#00d4aa",
               "offspring": "#ff6584", "partners": "#ffb347"}

    for label, mat in populations.items():
        xy   = pca_2d(mat.astype(np.float64))
        fits = fitnesses.get(label, np.zeros(len(mat)))
        rmse = -fits
        col  = colours.get(label, "#8892a4")
        hover = [f"Individual {i}<br>RMSE: {rmse[i]:.4f}" for i in range(len(mat))]
        fig.add_trace(go.Scatter(
            x=xy[:, 0], y=xy[:, 1],
            mode="markers",
            name=label.capitalize(),
            marker=dict(
                color=rmse, colorscale="Viridis", showscale=(label == "before"),
                size=marker_size, opacity=0.85,
                line=dict(color=col, width=1),
                colorbar=dict(title="RMSE", thickness=12, len=0.7),
            ),
            text=hover, hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text=title, font=dict(size=13)),
        xaxis=dict(title="PC 1", showgrid=False, zeroline=False),
        yaxis=dict(title="PC 2", showgrid=False, zeroline=False),
        showlegend=True,
        legend=dict(bgcolor="#111520", bordercolor="#2d3452", borderwidth=1),
    )
    return fig


def rmse_bar_fig(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart comparing mean RMSE across runs."""
    df_s = df.sort_values("mean_rmse")
    fig  = go.Figure(go.Bar(
        x=df_s["mean_rmse"],
        y=df_s["config_label"],
        orientation="h",
        error_x=dict(type="data", array=df_s.get("std_rmse", [0]*len(df_s)).tolist(),
                     color="#8892a4", thickness=1.5, width=4),
        marker=dict(
            color=df_s["mean_rmse"],
            colorscale="Viridis_r",
            showscale=True,
            colorbar=dict(title="RMSE", thickness=12, len=0.7),
        ),
        hovertemplate="%{y}<br>Mean RMSE: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text="Test RMSE by Configuration (lower = better)", font=dict(size=14)),
        xaxis=dict(title="Mean Test RMSE", gridcolor="#1e2540"),
        yaxis=dict(title="", tickfont=dict(size=10)),
        height=max(300, 36 * len(df_s)),
    )
    return fig


def diversity_curve_fig(log: List[Dict]) -> go.Figure:
    """Population diversity over generations."""
    gens  = [r["gen"] for r in log if "div_U" in r]
    div_u = [r["div_U"] for r in log if "div_U" in r]
    div_v = [r.get("div_V", 0) for r in log if "div_U" in r]
    fig   = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=div_u, name="User pop diversity",
                              line=dict(color="#6c63ff", width=2)))
    fig.add_trace(go.Scatter(x=gens, y=div_v, name="Item pop  diversity",
                              line=dict(color="#ff6584", width=2, dash="dot")))
    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text="Population Diversity (mean pairwise distance)", font=dict(size=13)),
        xaxis=dict(title="Generation", gridcolor="#1e2540"),
        yaxis=dict(title="Diversity", gridcolor="#1e2540"),
        legend=dict(bgcolor="#111520", bordercolor="#2d3452"),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar — all EA parameter controls
# ---------------------------------------------------------------------------

def render_sidebar(R_train: np.ndarray) -> Dict:
    """Render the parameter panel and return a fully merged config dict."""
    st.sidebar.markdown(
        "<div style='text-align:center;padding:0.5rem 0 1rem'>"
        "<span style='font-size:1.6rem'>🧬</span>"
        "<br><span style='color:#6c63ff;font-weight:700;font-size:1.0rem'>"
        "CoEvo Studio</span></div>",
        unsafe_allow_html=True,
    )

    # ---- Problem -------------------------------------------------------
    st.sidebar.markdown("<div class='section-label'>Problem</div>",
                        unsafe_allow_html=True)
    k = st.sidebar.slider("Latent factors k", 4, 40, 10, 2,
                           help="Dimensionality of user/item latent vectors.")

    # ---- Coevolution ---------------------------------------------------
    st.sidebar.markdown("<div class='section-label'>Coevolution</div>",
                        unsafe_allow_html=True)
    n_gen   = st.sidebar.slider("Generations", 10, 300, 40, 10)
    k_rand  = st.sidebar.slider("Collaborators k_random", 1, 10, 3,
                                 help="Random collaborators added to the best individual.")
    seed_ui = st.sidebar.number_input("Random seed", 0, 9999, 42, step=1)

    # ---- Selection -----------------------------------------------------
    st.sidebar.markdown("<div class='section-label'>Parent Selection</div>",
                        unsafe_allow_html=True)
    sel_method = st.sidebar.selectbox("Method", ["tournament", "rank_roulette"],
                                       format_func=lambda x: {
                                           "tournament":    "Tournament",
                                           "rank_roulette": "Rank-based Roulette",
                                       }[x])
    tau = st.sidebar.slider("Tournament size τ", 1, 10, 3,
                             disabled=(sel_method != "tournament"),
                             help="Only used for Tournament selection.")

    # ---- Crossover -----------------------------------------------------
    st.sidebar.markdown("<div class='section-label'>Crossover</div>",
                        unsafe_allow_html=True)
    xover = st.sidebar.selectbox("Operator", ["uniform", "blx_alpha"],
                                  format_func=lambda x: {
                                      "uniform":   "Uniform crossover",
                                      "blx_alpha": "BLX-α  (α=0.5)",
                                  }[x])
    p_swap   = st.sidebar.slider("Swap prob p_swap", 0.1, 1.0, 0.5, 0.05,
                                  disabled=(xover != "uniform"))
    alpha_blx = st.sidebar.slider("BLX α", 0.0, 1.0, 0.5, 0.05,
                                   disabled=(xover != "blx_alpha"))

    # ---- Mutation ------------------------------------------------------
    st.sidebar.markdown("<div class='section-label'>Mutation</div>",
                        unsafe_allow_html=True)
    mut_type = st.sidebar.selectbox("Operator", ["gaussian", "uniform_reset"],
                                     format_func=lambda x: {
                                         "gaussian":      "Gaussian (self-adaptive σ)",
                                         "uniform_reset": "Uniform Reset",
                                     }[x])
    sigma_init = st.sidebar.slider("Initial σ", 0.01, 1.0, 0.3, 0.01,
                                    disabled=(mut_type != "gaussian"))
    p_reset    = st.sidebar.slider("Reset prob p_reset", 0.01, 0.5, 0.10, 0.01,
                                    disabled=(mut_type != "uniform_reset"))

    # ---- Survivor selection --------------------------------------------
    st.sidebar.markdown("<div class='section-label'>Survivor Selection</div>",
                        unsafe_allow_html=True)
    surv = st.sidebar.selectbox("Model",
                                 ["mu_plus_lambda", "mu_comma_lambda"],
                                 format_func=lambda x: {
                                     "mu_plus_lambda":  "(μ + λ)  — elitist",
                                     "mu_comma_lambda": "(μ , λ)  — generational",
                                 }[x])

    # ---- Diversity -----------------------------------------------------
    st.sidebar.markdown("<div class='section-label'>Diversity</div>",
                        unsafe_allow_html=True)
    use_sharing = st.sidebar.toggle("Fitness sharing", value=False)
    sigma_share = st.sidebar.slider("Niche radius σ_share", 0.5, 5.0, 1.5, 0.5,
                                     disabled=not use_sharing)
    use_islands = st.sidebar.toggle("Island model", value=False)
    n_islands   = st.sidebar.slider("Islands", 2, 5, 4,
                                     disabled=not use_islands)
    mig_int     = st.sidebar.slider("Migration interval (gens)", 5, 50, 10, 5,
                                     disabled=not use_islands)

    # ---- Named preset --------------------------------------------------
    st.sidebar.markdown("<div class='section-label'>Load Preset</div>",
                        unsafe_allow_html=True)
    preset_names = ["(manual)"] + list_configs()[:10]
    preset = st.sidebar.selectbox("Preset config", preset_names)
    if preset != "(manual)":
        preset_cfg = get_config(preset)
        st.sidebar.caption(f"ℹ️ {describe_config(preset)}")

    config = dict(
        k=k, n_generations=n_gen, k_random=k_rand, seed=int(seed_ui),
        selection=sel_method, tau=tau,
        crossover=xover, p_swap=p_swap, alpha_blx=alpha_blx,
        mutation=mut_type, sigma_init=sigma_init, p_reset=p_reset,
        reset_low=-0.5, reset_high=0.5,
        survivor_selection=surv,
        fitness_sharing=use_sharing, sigma_share=sigma_share, alpha_share=1.0,
        island_model=use_islands, n_islands=n_islands,
        migration_interval=mig_int, n_migrants=2,
        init_type="uniform", log_every=max(1, n_gen // 50),
    )
    if preset != "(manual)":
        config = {**config, **preset_cfg, "n_generations": n_gen, "seed": int(seed_ui)}

    return config


# ---------------------------------------------------------------------------
# Tab 1 — Live Run
# ---------------------------------------------------------------------------

def tab_live_run(
    config:         Dict,
    R_train:        np.ndarray,
    test_pairs:     np.ndarray,
    test_ratings:   np.ndarray,
    dataset_label:  str,
) -> None:
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown(f"**Dataset:** `{dataset_label}`")
    with col_b:
        run_btn = st.button("▶  Run Evolution", type="primary",
                            use_container_width=True, key="run_btn")

    # Config summary chips
    chips = [
        f"k={config['k']}",
        f"{config['n_generations']} gens",
        f"sel: {config['selection']}",
        f"xover: {config['crossover']}",
        f"mut: {config['mutation']}",
        f"surv: {config['survivor_selection'].replace('mu_','μ').replace('_lambda','λ')}",
    ]
    if config["fitness_sharing"]:  chips.append("sharing ON")
    if config["island_model"]:      chips.append(f"{config['n_islands']} islands")
    st.markdown(
        " ".join(f"<code style='background:#1a1f2e;border:1px solid #2d3452;"
                 f"border-radius:6px;padding:2px 8px;color:#c0c7d4;font-size:0.8rem'>"
                 f"{c}</code>" for c in chips),
        unsafe_allow_html=True,
    )

    if run_btn:
        # ---- Run -------------------------------------------------------
        progress_bar = st.progress(0, text="Initialising …")
        status_text  = st.empty()
        chart_placeholder = st.empty()

        run_log  = []
        t_start  = time.perf_counter()

        # Patch log_every to capture at least 40 data points
        cfg_run = {**config, "log_every": max(1, config["n_generations"] // 40)}

        eng = CoevolutionaryEngine(cfg_run, R_train, test_pairs, test_ratings)

        # Run in one shot (eng.run()), then display animated result
        with st.spinner("Evolving…"):
            results = eng.run()

        log = results["log"]
        st.session_state.setdefault("run_history", []).append({
            "config_label": (
                f"{config['selection'][:5]}-{config['crossover'][:3]}-"
                f"{config['mutation'][:5]}-{config['n_generations']}gen"
            ),
            "config": deepcopy(config),
            "log":    log,
            "final_rmse": results.get("final_test_rmse") or float("nan"),
            "wall_sec":   results["total_wall_sec"],
        })

        progress_bar.progress(1.0, text="Done!")
        wall = results["total_wall_sec"]

        # ---- Metrics row -----------------------------------------------
        m1, m2, m3, m4 = st.columns(4)
        last = log[-1]
        final_rmse = results.get("final_test_rmse") or float("nan")
        with m1: st.metric("Test RMSE",    f"{final_rmse:.4f}")
        with m2: st.metric("Best RMSE (U)", f"{last['best_rmse_U']:.4f}")
        with m3: st.metric("Diversity U",   f"{last['div_U']:.3f}")
        with m4: st.metric("Wall time",     f"{wall:.1f}s")

        # ---- Fitness curve chart ---------------------------------------
        st.plotly_chart(fitness_curve_fig(log), use_container_width=True)

        # ---- Diversity chart -------------------------------------------
        if any("div_U" in r for r in log):
            with st.expander("Population Diversity Curve", expanded=False):
                st.plotly_chart(diversity_curve_fig(log), use_container_width=True)

        # ---- Log table -------------------------------------------------
        with st.expander("Generation Log (raw data)", expanded=False):
            df_log = pd.DataFrame(log).round(5)
            st.dataframe(df_log, use_container_width=True, height=260)

    elif st.session_state.get("run_history"):
        # Show last result while waiting for next run
        last_run = st.session_state["run_history"][-1]
        st.info("Showing last completed run.  Adjust parameters and click ▶ to re-run.")
        st.plotly_chart(fitness_curve_fig(last_run["log"]),
                        use_container_width=True)
    else:
        st.markdown(
            "<div class='info-card'>"
            "<h4>👈 Configure parameters in the sidebar, then click ▶ Run Evolution</h4>"
            "<p>The fitness convergence curve will appear here.</p>"
            "</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Tab 2 — Compare Runs
# ---------------------------------------------------------------------------

def tab_compare() -> None:
    history = st.session_state.get("run_history", [])
    if not history:
        st.markdown(
            "<div class='info-card'>"
            "<h4>No runs yet</h4>"
            "<p>Run at least two experiments in the <b>Run Evolution</b> tab "
            "to see a comparison here.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    df_hist = pd.DataFrame([{
        "Run #":       i + 1,
        "config_label":h["config_label"],
        "Final RMSE":  round(h["final_rmse"], 4),
        "Wall (s)":    h["wall_sec"],
        "k":           h["config"]["k"],
        "Generations": h["config"]["n_generations"],
        "Selection":   h["config"]["selection"],
        "Crossover":   h["config"]["crossover"],
        "Mutation":    h["config"]["mutation"],
        "Survivor":    h["config"]["survivor_selection"],
    } for i, h in enumerate(history)])

    # ---- Summary metrics -----------------------------------------------
    valid_rmse = [r for r in df_hist["Final RMSE"] if r == r]
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Total Runs",    len(history))
    with c2: st.metric("Best RMSE",
                         f"{min(valid_rmse):.4f}" if valid_rmse else "—")
    with c3: st.metric("Mean RMSE",
                         f"{sum(valid_rmse)/len(valid_rmse):.4f}" if valid_rmse else "—")

    # ---- Bar chart -----------------------------------------------------
    df_plot = df_hist.rename(columns={"config_label": "config_label",
                                       "Final RMSE": "mean_rmse"})
    df_plot["std_rmse"] = 0.0
    st.plotly_chart(rmse_bar_fig(df_plot), use_container_width=True)

    # ---- Overlay convergence curves ------------------------------------
    with st.expander("Convergence Curve Overlay", expanded=True):
        fig_conv = go.Figure()
        cmap = px.colors.qualitative.Vivid
        for i, h in enumerate(history):
            if not h["log"]:
                continue
            log  = h["log"]
            gens = [r["gen"] for r in log]
            rmse = [-r["best_fit_U"] for r in log]
            fig_conv.add_trace(go.Scatter(
                x=gens, y=rmse,
                name=f"Run {i+1}: {h['config_label']}",
                line=dict(color=cmap[i % len(cmap)], width=2),
            ))
        fig_conv.update_layout(
            **_DARK_LAYOUT,
            title="All Runs — Best RMSE (Users)",
            xaxis=dict(title="Generation", gridcolor="#1e2540"),
            yaxis=dict(title="RMSE", gridcolor="#1e2540"),
            legend=dict(bgcolor="#111520", bordercolor="#2d3452", orientation="v"),
        )
        st.plotly_chart(fig_conv, use_container_width=True)

    # ---- Full table + CSV download -------------------------------------
    st.markdown("<div class='section-label'>All Runs</div>", unsafe_allow_html=True)
    st.dataframe(
        df_hist.style.background_gradient(subset=["Final RMSE"],
                                           cmap="RdYlGn_r"),
        use_container_width=True,
    )
    csv_bytes = df_hist.to_csv(index=False).encode()
    st.download_button("⬇ Download CSV", csv_bytes, "runs.csv",
                        "text/csv", use_container_width=True)

    if st.button("🗑 Clear run history", type="secondary"):
        st.session_state["run_history"] = []
        st.rerun()


# ---------------------------------------------------------------------------
# Tab 3 — Educational Step-Through
# ---------------------------------------------------------------------------

def tab_step_through(
    R_mini:       np.ndarray,
    test_pairs:   np.ndarray,
    test_ratings: np.ndarray,
    config:       Dict,
) -> None:
    # Use a very small population for speed
    N_STEP_USERS, N_STEP_ITEMS = 30, 40
    n_u, n_i = R_mini.shape
    su = min(N_STEP_USERS, n_u)
    si = min(N_STEP_ITEMS, n_i)
    R_s  = R_mini[:su, :si]

    # Filter test set to only include mini indices
    tp_mask = (test_pairs[:, 0] < su) & (test_pairs[:, 1] < si)
    if tp_mask.sum() < 3:
        tp_s  = np.array([[0,0],[0,1],[1,0]])
        tr_s  = np.array([3.0, 4.0, 2.0], dtype=np.float32)
    else:
        tp_s  = test_pairs[tp_mask]
        tr_s  = test_ratings[tp_mask]

    step_config = {
        **config, "k": min(config["k"], 8),
        "k_random": min(config["k_random"], 2),
        "log_every": 1,
    }

    # ---- Init / Reset controls ------------------------------------------
    ctrl_c1, ctrl_c2 = st.columns([2, 1])
    with ctrl_c1:
        st.markdown(
            "<div class='info-card'>"
            "<h4>🎓 Educational Step-Through</h4>"
            "<p>Advances the coevolutionary algorithm <b>one generation at a time</b>. "
            "After each step you can inspect how the user population moved in "
            "PCA-space and which individuals improved.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with ctrl_c2:
        if st.button("🔄  Reset Engine", use_container_width=True):
            st.session_state["step_engine"]  = None
            st.session_state["step_records"] = []
            st.rerun()

    # Initialise engine in session state if not present
    if st.session_state.get("step_engine") is None:
        st.session_state["step_engine"]  = StepThroughEngine(
            R_s, tp_s, tr_s, step_config
        )
        st.session_state["step_records"] = []
        st.info("Engine initialised.  Click ▶ Next Step to begin.")

    eng     = st.session_state["step_engine"]
    records = st.session_state["step_records"]

    # ---- Step button ---------------------------------------------------
    b1, b2 = st.columns([1, 3])
    with b1:
        step_btn = st.button("▶  Next Generation", type="primary",
                              use_container_width=True, key="step_btn")
    with b2:
        st.caption(
            f"Current generation: **Gen {eng.gen}**  |  "
            f"Population U: **{len(eng.U)} users, k={eng.U.shape[1]}**"
        )

    if step_btn:
        with st.spinner(f"Running generation {eng.gen} …"):
            rec = eng.step()
            records.append(rec)

    if not records:
        # Show initial population
        init_pca = pca_2d(eng.U.astype(np.float64))
        fig_init = population_pca_fig(
            {"before": eng.U},
            {"before": eng.fit_U},
            title="Initial Population (Gen 0) — User latent space",
        )
        st.plotly_chart(fig_init, use_container_width=True)
        return

    # ---- Select which generation to inspect ----------------------------
    gen_options = [f"Gen {r.gen}" for r in records]
    gen_sel     = st.select_slider("Inspect generation", gen_options,
                                    value=gen_options[-1])
    rec = records[int(gen_sel.split()[1])]

    # ---- Metrics row ---------------------------------------------------
    d1, d2, d3, d4 = st.columns(4)
    with d1: st.metric("Best RMSE (before)",   f"{rec.best_rmse_before:.4f}")
    with d2: st.metric("Best RMSE (after)",    f"{rec.best_rmse_after:.4f}",
                        delta=f"{rec.best_rmse_after - rec.best_rmse_before:.4f}",
                        delta_color="inverse")
    with d3: st.metric("Individuals improved", f"{rec.n_improved}/{len(rec.U_before)}")
    with d4: st.metric("Mean σ",
                        f"{rec.sigma_mean_after:.4f}",
                        delta=f"{rec.sigma_mean_after - rec.sigma_mean_before:+.4f}")

    if rec.test_rmse is not None:
        st.caption(f"🧪 Test RMSE after this generation: **{rec.test_rmse:.4f}**")

    # ---- Side-by-side PCA charts ---------------------------------------
    pca_col1, pca_col2 = st.columns(2)
    with pca_col1:
        fig_before = population_pca_fig(
            {"before": rec.U_before},
            {"before": rec.fit_before},
            title=f"Gen {rec.gen} — Before  (pop in PCA space)",
            marker_size=9,
        )
        st.plotly_chart(fig_before, use_container_width=True)

    with pca_col2:
        fig_after = population_pca_fig(
            {"after":    rec.U_after,
             "offspring": rec.offspring_mut},
            {"after":    rec.fit_after,
             "offspring": rec.offspring_fits},
            title=f"Gen {rec.gen} — After  (green = pop, pink = offspring tried)",
            marker_size=9,
        )
        st.plotly_chart(fig_after, use_container_width=True)

    # ---- Fitness distribution change -----------------------------------
    with st.expander("Fitness Distribution Before vs After", expanded=False):
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=-rec.fit_before, name="Before", opacity=0.65,
            marker_color="#6c63ff", nbinsx=20,
        ))
        fig_hist.add_trace(go.Histogram(
            x=-rec.fit_after, name="After", opacity=0.65,
            marker_color="#00d4aa", nbinsx=20,
        ))
        fig_hist.update_layout(
            **_DARK_LAYOUT, barmode="overlay",
            title=f"RMSE distribution at Gen {rec.gen}",
            xaxis=dict(title="RMSE", gridcolor="#1e2540"),
            yaxis=dict(title="Count", gridcolor="#1e2540"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ---- Algorithm pseudocode with step highlighted --------------------
    st.markdown("<div class='section-label'>What happened this generation</div>",
                unsafe_allow_html=True)
    sel_name  = step_config["selection"].replace("_", " ").title()
    xo_name   = step_config["crossover"].replace("_", " ").title()
    mut_name  = step_config["mutation"].replace("_", " ").title()
    surv_name = step_config["survivor_selection"].replace("_", "+").replace("mu","μ").replace("lambda","λ")
    st.markdown(
        f"""<div class='pseudo'>
<span class='kw'>FOR</span> each user i <span class='cmt'># {len(rec.U_before)} users total</span><br>
&nbsp;&nbsp;<span class='hl cur'>1. SELECT</span> partner j via <b>{sel_name}</b> (based on fitness)<br>
&nbsp;&nbsp;<span class='hl cur'>2. CROSSOVER</span>: child_i = <b>{xo_name}</b>(U[i], U[j])<br>
&nbsp;&nbsp;<span class='hl cur'>3. MUTATE</span>: child_i += <b>{mut_name}</b> noise<br>
&nbsp;&nbsp;<span class='hl cur'>4. EVALUATE</span>: fitness(child_i) = <b>-RMSE</b>(R[i,:], child_i · V^T)<br>
&nbsp;&nbsp;<span class='hl cur'>5. SURVIVE</span> (<b>{surv_name}</b>):<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ {rec.n_improved} / {len(rec.U_before)} individuals <b>improved</b> this generation<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ Best RMSE: <b>{rec.best_rmse_before:.4f}</b> → <b>{rec.best_rmse_after:.4f}</b>
        </div>""",
        unsafe_allow_html=True,
    )

    # ---- Convergence miniplot (all steps so far) -----------------------
    if len(records) > 1:
        with st.expander("Convergence so far", expanded=False):
            step_gens  = [r.gen for r in records]
            step_rmses = [r.best_rmse_after for r in records]
            fig_step   = go.Figure()
            fig_step.add_trace(go.Scatter(
                x=step_gens, y=step_rmses,
                mode="lines+markers",
                line=dict(color="#6c63ff", width=2),
                marker=dict(color="#6c63ff", size=6),
                name="Best RMSE (U)",
            ))
            fig_step.update_layout(
                **_DARK_LAYOUT,
                title="Step-Through Convergence",
                xaxis=dict(title="Generation"),
                yaxis=dict(title="Best RMSE"),
            )
            st.plotly_chart(fig_step, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4 — Algorithm Guide
# ---------------------------------------------------------------------------

def tab_guide() -> None:

    def _card(title, icon, body):
        st.markdown(
            f"<div class='info-card'><h4>{icon} {title}</h4>{body}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("## 📖 Coevolutionary Recommender — Algorithm Guide")
    st.caption("A self-contained reference for every component of the engine.")

    with st.expander("🧩 What is Coevolution?", expanded=True):
        _card("Cooperative Coevolution", "🤝",
              "<p>Two populations evolve <b>simultaneously</b> and their fitnesses "
              "are defined <em>by each other</em>:</p>"
              "<ul>"
              "<li><b>Population U</b> (users): each individual <code>U[i]</code> "
              "is a k-dimensional latent vector for user i.</li>"
              "<li><b>Population V</b> (items): each individual <code>V[j]</code> "
              "is a k-dimensional latent vector for item j.</li>"
              "<li>Fitness of <code>U[i]</code> = <code>−RMSE(R[i,:], U[i]·V^T)</code>."
              "</li></ul>"
              "<p>The rating prediction model is matrix factorisation: "
              "<code>r̂ᵢⱼ = U[i] · V[j]</code>.</p>")

    with st.expander("🎯 Parent Selection"):
        _card("Tournament Selection", "⚔️",
              "<p>Sample <code>τ</code> individuals uniformly at random. "
              "The one with the <b>highest fitness wins</b> and becomes a parent. "
              "Larger <code>τ</code> → stronger selection pressure → faster "
              "convergence but less diversity.</p>")
        _card("Rank-Based Roulette", "🎡",
              "<p>Rank individuals by fitness (rank 1 = worst, rank N = best). "
              "Selection probability is <b>proportional to rank</b>: "
              "<code>P(r) = 2r / N(N+1)</code>. "
              "Avoids super-individual dominance that plagues raw fitness-proportionate "
              "selection, especially with our negative-value RMSE fitness.</p>")

    with st.expander("🔀 Crossover (Recombination)"):
        _card("Uniform Crossover", "🧱",
              "<p>For each gene dimension <code>d</code>, flip a "
              "<code>Bernoulli(p_swap)</code> coin. If heads: child gets the "
              "gene from parent 2; otherwise from parent 1. At <code>p_swap=0.5</code> "
              "every gene is equally likely from either parent — maximum gene "
              "shuffling without positional bias.</p>")
        _card("BLX-α Crossover", "🌈",
              "<p>For each gene <code>d</code>, sample a child value from the "
              "<em>extended</em> interval "
              "<code>[min(p1,p2)−α·I, max(p1,p2)+α·I]</code> "
              "where <code>I = |p1−p2|</code>. At <code>α=0.5</code> children "
              "may explore ±50% beyond the parents' values — enabling escape from "
              "local optima while respecting the parents' general neighbourhood.</p>")

    with st.expander("🔧 Mutation"):
        _card("Self-Adaptive Gaussian (ES-style)", "📈",
              "<p>Each individual carries its own <em>strategy parameter</em> σ "
              "that co-evolves alongside the object variables:</p>"
              "<ol>"
              "<li>Update step size: <code>σ' = σ · exp(τ · N(0,1))</code></li>"
              "<li>Perturb gene: &nbsp; <code>x' = x + σ' · N(0,1)</code></li>"
              "</ol>"
              "<p>The EA automatically learns the right step size for each gene "
              "dimension — no manual mutation-rate tuning needed. "
              "τ = 1/√k is the empirically validated default (Beyer & Schwefel 2002).</p>")
        _card("Uniform Reset Mutation", "🎲",
              "<p>Each gene is independently replaced with probability "
              "<code>p_reset</code> by a fresh draw from <code>U(low, high)</code>. "
              "Provides large jumps — effective for escaping local optima and "
              "complementary to Gaussian's small perturbations.</p>")

    with st.expander("⚖️ Survivor Selection"):
        _card("(μ + λ) — Elitist", "🏆",
              "<p>Pool parents AND offspring, keep the best μ individuals. "
              "The global best solution is <b>never lost</b>. "
              "Risk: premature convergence if one super-individual dominates. "
              "Mitigate with fitness sharing or island model.</p>")
        _card("(μ , λ) — Generational", "🔄",
              "<p>Discard all parents; select μ best from offspring only. "
              "More exploratory — bad regions from the parent generation cannot "
              "'block' the search. Requires λ ≥ μ to ensure enough offspring diversity. "
              "Recommended: λ = 7μ (ES convention).</p>")

    with st.expander("🌍 Diversity Mechanisms"):
        _card("Fitness Sharing", "🔢",
              "<p>Penalises individuals that are <em>too close</em> to their "
              "neighbours in genotype space:</p>"
              "<code>f'(i) = sign(f(i)) · |f(i)| / Σⱼ sh(d(i,j))</code>"
              "<p>where <code>sh(d) = 1 − (d/σ_share)^α</code> if <code>d < σ_share</code>, else 0. "
              "Encourages the population to spread across multiple <em>niches</em> "
              "rather than converging to a single peak.</p>")
        _card("Island Model (Ring Topology)", "🏝️",
              "<p>Population is partitioned into <code>n_islands</code> sub-populations. "
              "Each island evolves independently (different genetic drift). "
              "Every <code>migration_interval</code> generations the "
              "<code>n_migrants</code> best individuals from each island are "
              "<em>copied</em> (not moved) to the adjacent island (clockwise ring). "
              "This balances diversity preservation with inter-island gene flow.</p>")

    with st.expander("📐 Collaborator Selection (Coevo Evaluation)"):
        _card("How fitness is computed", "🧮",
              "<p>To evaluate user individual U[i], we need a set of item vectors. "
              "We use a <b>collaborator set</b> drawn from population V:</p>"
              "<ul>"
              "<li>Always include the <b>current best</b> item individual (best ≡ lowest RMSE)</li>"
              "<li>Plus <code>k_random</code> randomly sampled item individuals</li>"
              "</ul>"
              "<p>This ensures the fitness signal isn't purely noisy (anchored by "
              "the best V) while still probing diverse V representatives "
              "(the random sample). The fitness function then computes "
              "<code>RMSE</code> of <code>U[i] · V[observed_items]^T</code> "
              "versus <code>R[i, observed_items]</code>.</p>")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    # ---- Hero header ---------------------------------------------------
    st.markdown(
        "<div class='hero-header'>"
        "<h1>🧬 CoEvo Recommender Studio</h1>"
        "<p>Adaptive Recommendation Engine powered by Coevolutionary Algorithms  "
        "· MovieLens 100K  · Pure NumPy EA</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ---- Sidebar -------------------------------------------------------
    R_train, test_pairs, test_ratings, dataset_label = load_data()
    config = render_sidebar(R_train)

    # ---- Session-state defaults ----------------------------------------
    if "run_history" not in st.session_state:
        st.session_state["run_history"] = []
    if "step_engine" not in st.session_state:
        st.session_state["step_engine"]  = None
        st.session_state["step_records"] = []

    # ---- Tabs ----------------------------------------------------------
    t1, t2, t3, t4 = st.tabs([
        "🚀  Run Evolution",
        "📊  Compare Runs",
        "🎓  Step-Through",
        "📖  Algorithm Guide",
    ])

    with t1:
        tab_live_run(config, R_train, test_pairs, test_ratings, dataset_label)

    with t2:
        tab_compare()

    with t3:
        tab_step_through(R_train, test_pairs, test_ratings, config)

    with t4:
        tab_guide()


if __name__ == "__main__":
    main()
