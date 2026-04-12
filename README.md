# Adaptive Recommendation Engine using Coevolutionary Algorithms

> A university project implementing a cooperative coevolutionary algorithm (CCEA) for collaborative filtering on the MovieLens 100K dataset, with a full Streamlit UI for interactive experimentation and educational visualization.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Algorithm Design](#algorithm-design)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
- [UI Features](#ui-features)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## Team

| Name | LinkedIn |
|---|---|
| Ahmad Al-Kordy | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ahmadalkordy) |
| Yassin Yasser | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yassin-yasser1) |
| Nour Hatem | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nour-hatem-) |
| Mahmoud Hossam | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mahmoud-hossam-090958351) |

---

## Overview

This project frames the **movie recommendation problem as a coevolutionary optimization task**. Instead of solving matrix factorization analytically, two populations — one representing users, one representing items — evolve simultaneously. Each individual is a k-dimensional latent factor vector. Fitness is measured by how accurately a candidate vector predicts real ratings when paired with collaborators from the opposing population.

The system implements the full Potter & De Jong Cooperative Coevolution (CCEA) framework and is designed to meet strict academic EA requirements including multiple operators, diversity preservation, statistical rigor over 30 independent runs, and an educational Streamlit interface.

---

## Project Structure

```
coevo_recommender/
│
├── data/
│   └── ml-100k/
│       └── u.data                  ← MovieLens 100K (download separately)
│
├── core/
│   ├── __init__.py
│   ├── data_loader.py              ← Load & split rating matrix
│   ├── population.py               ← Population class (2 representations, 2 inits)
│   ├── fitness.py                  ← Collaborative fitness evaluation
│   ├── operators.py                ← All selection, crossover, mutation operators
│   ├── diversity.py                ← Fitness sharing + Island model
│   └── coevo_engine.py             ← Main coevolutionary loop
│
├── experiments/
│   ├── __init__.py
│   ├── config.py                   ← All operator combination configs
│   └── batch_runner.py             ← 30-run parallel executor + statistics
│
├── ui/
│   └── app.py                      ← Streamlit dashboard
│
├── results/                        ← Auto-created: seeds.json, CSVs, plots
├── requirements.txt
└── README.md
```

---

## Algorithm Design

### Problem Formulation

- **Type:** Continuous Optimization (matrix factorization via evolutionary search)
- **Representation A:** Real-valued k-dimensional vectors (float32), initialized uniformly from U(−0.5, 0.5) or SVD-seeded
- **Representation B:** Gray-coded binary strings (k × n_bits bits), decoded to reals for evaluation
- **Constraint Handling:** Soft L2-norm penalty added to fitness for out-of-bounds vectors

### Two Co-evolving Populations

| Population | Maps to | Size | Individual |
|---|---|---|---|
| Population U | Users (943) | n_users | k-dim latent vector per user |
| Population V | Items (1682) | n_items | k-dim latent vector per item |

### Fitness Function

For user individual `u_i`, fitness is computed against collaborators drawn from Population V:

```
fitness(u_i) = −RMSE({ r_ij − dot(u_i, v_j) : v_j ∈ collaborators, r_ij is known })
```

Collaborator set = best current item individual + k randomly sampled item individuals.

### EA Components

**Parent Selection (2 methods)**
- Tournament Selection (configurable size τ)
- Rank-based Roulette Wheel Selection

**Recombination (2 operators)**
- Uniform Crossover (p=0.5 per gene)
- BLX-α Crossover (α=0.5, blend of parent genes)

**Mutation (2 operators)**
- Gaussian Mutation with self-adaptive step size σ
- Uniform Reset Mutation (random gene replacement with probability p_reset)

**Survivor Selection (2 models)**
- (μ + λ): parents and offspring compete, top μ survive
- (μ, λ): offspring only, parents discarded

**Diversity Preservation (2 mechanisms)**
- Fitness Sharing: penalizes crowded regions of the search space
- Island Model: ring topology with 3–5 islands, migration every K generations

**Initialization (2 strategies)**
- Uniform Random: U(−0.5, 0.5) per gene
- SVD-Seeded: truncated SVD of R as population mean + Gaussian noise

**Termination**
- Max generations G (default 200)
- Early stopping if best fitness improvement < ε = 1×10⁻⁴ over 20 generations

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/coevo-recommender.git
cd coevo-recommender
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Go to [https://grouplens.org/datasets/movielens/100k/](https://grouplens.org/datasets/movielens/100k/), download `ml-100k.zip`, extract it, and place `u.data` at:

```
data/ml-100k/u.data
```

### 4. Verify setup

Run from the project root:

```bash
python core/data_loader.py
```

Expected output:
```
Total interactions : 100000
Users              : 943
Items              : 1682
Train matrix shape : (943, 1682)
Sparsity           : 94.03%
```

---

## Usage

### Run a single experiment

```bash
python core/coevo_engine.py
```

### Launch the Streamlit UI

```bash
streamlit run ui/app.py
```

### Run the full 30-run batch experiment

```bash
python -m experiments.batch_runner
```

Results are saved to `results/` including:
- `seeds.json` — all random seeds used
- `runs.csv` — per-run RMSE results
- `summary.csv` — mean ± std per configuration
- `wilcoxon.csv` — pairwise statistical significance tests
- `boxplots.png` — distribution plots per configuration

> **Important:** Always run commands from the project root directory, never from inside a subfolder.

---

## Experiments

The batch runner tests all combinations of:

| Factor | Options |
|---|---|
| Representation | Real-valued, Binary (Gray-coded) |
| Initialization | Uniform random, SVD-seeded |
| Parent Selection | Tournament, Rank-RWS |
| Recombination | Uniform crossover, BLX-α |
| Mutation | Gaussian self-adaptive, Uniform reset |
| Survivor Selection | (μ+λ), (μ,λ) |

Each configuration is run **30 times** with independently stored random seeds for full reproducibility. Statistical comparison uses the **Wilcoxon Rank-Sum Test** (p < 0.05).

---

## UI Features

The Streamlit dashboard provides four modes:

- **Parameter Dashboard** — sliders and dropdowns for all EA hyperparameters, live run button
- **Live Evolution Viewer** — real-time Plotly charts of fitness curves and diversity metric per generation for both populations
- **Results Comparison** — side-by-side RMSE table and box plots across all tested configurations
- **Educational Step-Through** — step through a single generation visually, showing which individuals were selected, how crossover combined them, and which offspring survived

---

## Results

After running experiments, summary statistics are printed and saved. Example output format:

```
Configuration               Mean RMSE    Std RMSE    Sig. vs baseline
baseline (tourn+uniform+gauss+(μ+λ))  1.042        0.031       —
blx_alpha_crossover                   1.018        0.028       Yes (p=0.021)
svd_init                              0.987        0.019       Yes (p=0.003)
```

---

## Tech Stack

- **Python 3.11**
- **NumPy** — all EA operations (no ML frameworks in core)
- **Pandas** — data loading
- **Scikit-learn** — train/test split
- **SciPy** — Wilcoxon statistical tests
- **Streamlit** — UI dashboard
- **Plotly** — interactive charts
- **Joblib** — parallel 30-run execution

---

## Academic Context

This project was developed as a university assignment for an AI & Machine Learning course. It targets the following graded requirements:

- Problem formalization as optimization
- Constraint handling via penalty functions
- All 8 EA components explicitly defined and implemented
- Minimum 2 operators per component independently tested
- Diversity preservation (fitness sharing + island model)
- 30 runs per setting with stored seeds
- Statistical reporting with Wilcoxon tests
- Functional UI demonstrating the algorithm

**Bonus features implemented:**
- Two representations (real + binary Gray-coded)
- Two initialization strategies (uniform + SVD-seeded)
- Educational visual step-through interface
- SOTA hybrid suggestion: CCEA + Differential Evolution (DE/rand/1/bin)

---

## License

This project is submitted for academic purposes. Dataset credit: F. Maxwell Harper and Joseph A. Konstan, [GroupLens Research](https://grouplens.org).
