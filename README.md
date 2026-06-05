# ⚽ FIFA WC 2026 Match Predictor

![Version](https://img.shields.io/badge/version-1.0.0-blue?style=flat-square)
![Status](https://img.shields.io/badge/status-active-brightgreen?style=flat-square)
![Python](https://img.shields.io/badge/python-3.11%2B-yellow?style=flat-square)
![Streamlit](https://img.shields.io/badge/streamlit-1.58-red?style=flat-square)
![Accuracy](https://img.shields.io/badge/model%20accuracy-69%25-orange?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square)
![CI](https://github.com/mmlcj5-dev/fifa2026-prediction/actions/workflows/ci.yml/badge.svg)

---

> **Machine learning–powered match predictor and Monte Carlo tournament simulator for FIFA World Cup 2026.**
> Trained on 47,399 international matches across 19 features — predicts match outcomes with a 4-model ensemble and simulates the full WC2026 bracket using Dixon-Coles τ-corrected Poisson scoring.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture Diagram](#-architecture-diagram)
3. [Technology Stack](#-technology-stack)
4. [Repository Structure](#-repository-structure)
5. [Prerequisites](#-prerequisites)
6. [Setup & Installation](#-setup--installation)
7. [Environment Configuration (.env)](#-environment-configuration-env)
8. [Pages & Features](#-pages--features)
9. [Module Explanations](#-module-explanations)
10. [Model Details](#-model-details)
11. [Running the App](#-running-the-app)
12. [Running the Simulation CLI](#-running-the-simulation-cli)
13. [Live Score Integration](#-live-score-integration)
14. [DevOps & Deployment](#-devops--deployment)
15. [Security Notes](#-security-notes)
16. [Roadmap](#-roadmap)
17. [License](#-license)

---

## 🎯 Project Overview

The **FIFA WC 2026 Match Predictor** is a full-stack machine learning application built in Python and Streamlit. It combines a trained 4-model ensemble classifier with a Monte Carlo tournament simulator to predict match outcomes and championship probabilities across all 48 teams and 104 matches of the 2026 FIFA World Cup.

A **Dixon-Coles τ (tau) correction** is applied to the Poisson scoring model in the group stage — fixing the known under-representation of low-scoring results (0–0, 1–0, 0–1, 1–1) that independent Poisson distributions produce, consistent with published football analytics literature (Dixon & Coles, 1997).

A **live score integration layer** (`src/live_scores.py`) is wired to the football-data.org API and activates automatically on June 11, 2026 — the tournament start date.

### Core Capabilities

| Capability | Description |
|---|---|
| **Match Predictor** | Predicts win/draw/loss probabilities for any matchup using 4 independent models |
| **Tournament Schedule** | Full 72-match group stage schedule with CT kick-off times, venues, and live score overlay |
| **Group Stage Predictor** | Monte Carlo simulation of any single group — qualification %, match probs, and top scorelines |
| **Winner Probabilities** | Full tournament simulation (10,000 runs) producing WC2026 championship probabilities |
| **Dixon-Coles τ Correction** | Bivariate Poisson correction for accurate low-scoring scoreline distribution |
| **ELO-Based Rating System** | World Football Elo ratings for all 48 qualified nations |
| **Live Score Integration** | football-data.org API stub — activates on June 11, 2026 |
| **Containerized Deployment** | Docker + GitHub Actions CI + Azure Container Apps deployment pipeline |

---

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                  FIFA WC 2026 MATCH PREDICTOR                   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
            ┌───────────────────▼──────────────────┐
            │          STREAMLIT UI (4 pages)       │
            │  Match Predictor · Tournament Schedule│
            │  Group Predictor · Winner Probs       │
            └──────┬────────────────────┬───────────┘
                   │                    │
   ┌───────────────▼──────┐   ┌────────▼──────────────────┐
   │   MATCH PREDICTOR    │   │   MONTE CARLO SIMULATOR   │
   │  src/simulate.py     │   │   src/simulate.py         │
   │                      │   │                           │
   │ • 4-model ensemble   │   │ • Group stage (Poisson)   │
   │ • ELO-based draw     │   │ • Dixon-Coles τ correction│
   │   prior              │   │ • R32→R16→QF→SF→Final    │
   │ • Bidirectional      │   │ • 10,000 simulations      │
   │   probability        │   │ • Prob cache (fast)       │
   └──────────────────────┘   └────────────────────────────┘
                   │                    │
   ┌───────────────▼──────────────────▼─┐
   │         MODEL ENSEMBLE             │
   │         models/*.pkl               │
   │                                    │
   │  Logistic Regression               │
   │  XGBoost Classifier                │
   │  Random Forest Classifier          │
   │  VotingClassifier (Ensemble)       │
   │                                    │
   │  Trained on 47,399 matches         │
   │  19 features · 69% accuracy        │
   └──────────────┬─────────────────────┘
                  │
   ┌──────────────▼─────────────────────┐
   │         FEATURE LAYER              │
   │  src/team_data.py                  │
   │                                    │
   │ • ELO ratings (48 nations)         │
   │ • Rolling form stats (ELO-derived) │
   │ • H2H defaults (neutral WC venue)  │
   │ • Host nation flags                │
   └──────────────┬─────────────────────┘
                  │
   ┌──────────────▼─────────────────────┐
   │     LIVE DATA LAYER                │
   │  src/live_scores.py                │
   │                                    │
   │ • football-data.org API            │
   │ • 60s cache (free tier safe)       │
   │ • Activates Jun 11 2026            │
   │ • Returns {} before tournament     │
   └────────────────────────────────────┘
```

---

## 🧰 Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| UI Framework | Streamlit 1.58 |
| ML Models | scikit-learn (Logistic Regression, Random Forest, VotingClassifier), XGBoost |
| Data Processing | pandas, numpy |
| Visualization | Plotly |
| Simulation | Monte Carlo · Dixon-Coles τ-corrected bivariate Poisson |
| Rating System | World Football Elo Ratings |
| Live Data | football-data.org REST API v4 |
| Model Persistence | joblib |
| Config | python-dotenv |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud Deployment | Azure Container Apps |
| Model Storage | Git LFS (large `.pkl` files) |

---

## 📁 Repository Structure

```
fifa2026-prediction/
│
├── .env.example                    # Safe config template — commit this, not .env
├── .gitattributes                  # Git LFS rules for large model files
├── .gitignore
├── .github/
│   └── workflows/
│       ├── ci.yml                  # GitHub Actions: lint + smoke tests + Docker build
│       └── azure-deploy.yml        # Azure Container Apps deployment on main merge
├── Dockerfile                      # Containerized Streamlit app
├── README.md
├── requirements.txt
│
├── app/
│   └── streamlit_app.py            # 4-page Streamlit UI (main entrypoint)
│
├── src/                            # Core library modules
│   ├── team_data.py                # 48 WC2026 teams, ELO ratings, group assignments
│   ├── simulate.py                 # Monte Carlo engine + Dixon-Coles τ correction
│   ├── group_predictor.py          # Single-group simulation with rich stats
│   ├── schedule.py                 # Full 72-match group stage schedule (CT times + venues)
│   ├── live_scores.py              # football-data.org live score integration
│   ├── features.py                 # Feature engineering utilities
│   └── live_elo_loader.py          # ELO loader with hardcoded fallback
│
├── models/                         # Trained model artifacts (Git LFS for large files)
│   ├── ensemble.pkl                # VotingClassifier — primary model  [LFS ~20 MB]
│   ├── random_forest.pkl           # Random Forest                     [LFS ~9 MB]
│   ├── xgboost.pkl                 # XGBoost Classifier                [LFS ~750 KB]
│   ├── logistic_regression.pkl     # Logistic Regression               [~1 KB]
│   ├── scaler.pkl                  # StandardScaler                    [~2 KB]
│   └── feature_cols.pkl            # Ordered feature name list         [~1 KB]
│
├── scripts/
│   ├── run_simulation.py           # CLI: run full Monte Carlo, save results
│   ├── train_models.py             # Model training pipeline
│   ├── prepare_data.py             # Data preparation
│   └── predict_game.py             # Single-match CLI predictor
│
├── data/
│   └── simulation_results.json     # Cached Monte Carlo output (10,000 simulations)
│
├── notebooks/                      # Exploratory analysis notebooks
├── azure/                          # [Planned] ARM / Bicep templates
├── docs/                           # [Planned] Extended documentation
└── runbooks/                       # [Planned] Operational runbooks
```

---

## ✅ Prerequisites

- **Python 3.11 or higher**
- **Git LFS** installed (`git lfs install`) — required to pull the large model `.pkl` files
- **pip** for package installation
- *(Optional)* A free **football-data.org API key** for live score integration from June 11, 2026

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```powershell
git clone https://github.com/mmlcj5-dev/fifa2026-prediction.git
cd fifa2026-prediction
```

> **Git LFS required** — the large model files (`ensemble.pkl`, `random_forest.pkl`) are stored in Git LFS.
> Run `git lfs install` before cloning if you have not already.

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```powershell
copy .env.example .env
# Add your football-data.org API key if you want live scores (optional)
```

### 5. Verify Models Load Correctly

```powershell
python -c "
import sys; sys.path.insert(0,'.')
from src.simulate import predict_win_prob, _load_models
_load_models()
p = predict_win_prob('Argentina', 'France')
print(f'Argentina vs France — Win: {p[0]:.1%}  Draw: {p[1]:.1%}  Loss: {p[2]:.1%}')
"
```

Expected output:
```
Argentina vs France — Win: 38.8%  Draw: 23.7%  Loss: 37.4%
```

---

## 🔐 Environment Configuration (.env)

Copy `.env.example` → `.env`. **Never commit `.env` to source control — it is gitignored.**

```dotenv
# ─────────────────────────────────────────────────────────
#  FIFA WC 2026 MATCH PREDICTOR
#  Environment Configuration
# ─────────────────────────────────────────────────────────

# Live score integration (free tier — sign up at football-data.org)
# Leave blank before June 11 2026 — the app works without it
FOOTBALL_DATA_API_KEY=your_key_here
```

> ℹ️ The app is fully functional without the API key. Live scores activate automatically once the key is set **and** the tournament start date (June 11, 2026) is reached.

---

## 📄 Pages & Features

### Page 1 — Match Predictor
Select any two of the 48 WC2026 nations, set neutral/friendly flags, and click **Predict**.

- 3-outcome probabilities (Win / Draw / Loss) using the ensemble model
- Per-model breakdown bar chart (Logistic Regression, XGBoost, Random Forest, Ensemble)
- ELO rating comparison with delta indicator

### Page 2 — Tournament Schedule
Full 72-match group stage schedule organized by date.

- Sub-views: **Group Cards** · **Match Schedule** · **Knockout Rounds**
- All kick-off times in **Central Time (CT / CDT)**
- Venue name and city for all 15 WC2026 stadiums across the USA, Canada, and Mexico
- Teams **color-coded by championship probability** once simulation results are loaded
- **Live score overlay** — activates June 11, 2026 (scores update every 60 seconds)
- Knockout Rounds view shows predicted contenders at each stage

### Page 3 — Group Predictor
Deep-dive Monte Carlo simulation of any single group.

| Output | Detail |
|---|---|
| **Qualification probabilities** | % chance of finishing 1st / 2nd / 3rd / 4th |
| **Stacked bar chart** | Visual finish-position breakdown per team |
| **Match-by-match probs** | Win / Draw / Loss bar for all 6 group games |
| **Top 5 scorelines** | Most common predicted scoreline per match (from N simulations) |
| **Live standings** | Real group table from API when tournament is in progress |

### Page 4 — Winner Probabilities
Full-tournament Monte Carlo simulation producing WC2026 championship probabilities.

- Configurable simulation count (500 – 20,000 runs)
- Gold-gradient bar chart (Opta-style) for top N teams
- Podium metrics (1st / 2nd / 3rd favorites)
- Full results table with group, ELO, and win probability
- Results cached to `data/simulation_results.json` — Tournament Schedule and Group Predictor pages update automatically

---

## 🧩 Module Explanations

### `src/simulate.py`
The core Monte Carlo engine. Contains:
- `predict_win_prob()` — runs the ensemble model in both directions (home/away) and reconciles with an ELO-based draw prior (~28% max, decaying with ELO gap). Results are cached so repeated matchups cost zero inference time.
- `simulate_group()` — round-robin group stage using Dixon-Coles τ-corrected Poisson goal sampling.
- `simulate_tournament()` — full bracket from group stage through the Final.
- `run_monte_carlo()` — runs N full simulations and returns championship probabilities.

Host advantage (`is_home_host`) is applied only in the Match Predictor UI — the tournament simulation uses neutral-venue features throughout for unbiased results.

### `src/group_predictor.py`
Single-group simulation engine returning rich statistics: qualification probabilities, finish-position breakdown, match-by-match probabilities (from the model, not empirical), and top-5 scorelines per fixture (from N simulation runs).

### `src/team_data.py`
Defines all 48 WC2026 nations with World Football Elo ratings, 12-group assignments, and ELO-derived rolling form statistics (win rate, goals scored/conceded, points per game). Edit this file to update ELO ratings or correct group assignments as news develops.

### `src/schedule.py`
Full 72-match group stage schedule: dates, CT kick-off times, home/away teams, and venue keys mapped to the `VENUES` dictionary (name, city, capacity). Knockout round dates and venues are also defined here.

### `src/live_scores.py`
football-data.org API integration. Returns an empty dict before June 11, 2026 or if no API key is configured — the rest of the app is unaffected. On tournament days, fetches all WC2026 match data with a 60-second LRU cache (respects the free-tier 10 req/min rate limit). Exposes `status_badge()` for rendering LIVE / HT / FT chips in the UI.

---

## 📊 Model Details

### Training Data
- **47,399 international matches** — covers all FIFA-recognized nations from 1993 to 2025
- **Binary target:** home team wins (1) vs. does not win (0)
- **Overall accuracy:** 69%

### Feature Set (19 features)

| Feature | Description |
|---|---|
| `home_elo`, `away_elo` | World Football Elo ratings |
| `elo_diff` | Raw ELO difference (home − away) |
| `is_home_host`, `is_away_host` | Host nation flag |
| `is_neutral`, `is_friendly` | Venue and match type flags |
| `home_win_rate`, `away_win_rate` | 10-match rolling win rate |
| `home/away_goals_scored_avg` | 10-match rolling goals scored |
| `home/away_goals_conceded_avg` | 10-match rolling goals conceded |
| `home/away_points_per_game` | 10-match rolling PPG |
| `h2h_home_wins`, `h2h_draws`, `h2h_away_wins` | Historical head-to-head record |
| `h2h_home_goal_diff` | Historical H2H goal difference |

### Models

| Model | Type | Notes |
|---|---|---|
| Logistic Regression | Linear baseline | Fast, interpretable |
| XGBoost | Gradient boosting | Handles non-linear ELO interactions |
| Random Forest | Bagged trees | Robust to outliers |
| **Ensemble** | VotingClassifier (soft) | Primary model used for simulation |

---

## 🚀 Running the App

```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Launch Streamlit app (opens at http://localhost:8501)
streamlit run app/streamlit_app.py
```

**Expected console output:**
```
  You can now view your Streamlit app in your browser.
  Local URL:  http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

---

## 💻 Running the Simulation CLI

Run the full Monte Carlo simulation independently from the command line:

```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Default: 10,000 simulations, seed 42
python scripts/run_simulation.py

# Custom run
python scripts/run_simulation.py --n 20000 --seed 123
```

**Expected output:**
```
Running 10,000 WC2026 simulations (seed=42)...

Completed in 132.4s. Results saved to data/simulation_results.json

Team                   Win %
-----------------------------
Argentina              3.47%
Spain                  3.19%
France                 3.16%
Germany                3.05%
Portugal               3.00%
...
```

Results are saved to `data/simulation_results.json` and loaded automatically by the Streamlit UI on the next page load.

---

## 📡 Live Score Integration

The live score layer uses the **football-data.org** free API tier.

### Setup

1. Register at [football-data.org/client/register](https://www.football-data.org/client/register) — free, no credit card required
2. Add your key to `.env`:
   ```dotenv
   FOOTBALL_DATA_API_KEY=your_key_here
   ```
3. Restart the Streamlit app

### Behavior

| Condition | Result |
|---|---|
| Before June 11, 2026 | Returns empty dict — no API calls made |
| Key not configured + tournament live | Warning banner with signup link shown in UI |
| Key configured + tournament live | Scores appear between team chips; badge shows 🔴 LIVE / ⏸ HT / ✅ FT |
| API error / timeout | Graceful fallback to schedule view — no crash |

Scores refresh every **60 seconds** via an LRU cache — safe for the free tier's 10 req/min limit.

---

## 🔧 DevOps & Deployment

### GitHub Actions CI (`ci.yml`)
Runs on every push to `main`, `master`, or `develop`:

1. **Lint** — flake8 across `src/`, `app/`, `scripts/`
2. **Smoke test: simulation engine** — verifies probabilities sum to 1.0
3. **Smoke test: group predictor** — verifies qualification probabilities are valid
4. **Smoke test: live scores** — verifies empty dict returned without API key
5. **Docker build** — builds the container image and verifies the health endpoint responds

### Azure Deployment (`azure-deploy.yml`)
Triggered on merge to `main`:

1. Builds and pushes Docker image to **GitHub Container Registry** (ghcr.io)
2. Deploys to **Azure Container Apps** using the `azure/container-apps-deploy-action`
3. API key injected as an Azure Container Apps secret reference — never stored in plaintext

### Running with Docker

```powershell
# Build
docker build -t fifa2026-predictor .

# Run locally
docker run -p 8501:8501 -e FOOTBALL_DATA_API_KEY=your_key fifa2026-predictor

# Open http://localhost:8501
```

### Azure Setup (one-time)

To connect the Azure deployment pipeline, add one GitHub repository secret:

| Secret | Value |
|---|---|
| `AZURE_CREDENTIALS` | Output of `az ad sp create-for-rbac --sdk-auth` |

Update `AZURE_APP_NAME` and `AZURE_RG` in `azure-deploy.yml` to match your Azure environment.

---

## 🔒 Security Notes

- `.env` is gitignored — **never commit live API keys**
- Large model files use **Git LFS** — avoids bloating repository history
- The football-data.org free key has read-only scope — no risk of write operations
- `FOOTBALL_DATA_API_KEY` is passed to Azure Container Apps as a **secret reference**, not an environment variable in plaintext

---

## 🗺️ Roadmap

- [x] 4-model ensemble trained on 47,399 matches
- [x] Monte Carlo tournament simulator (10,000 runs)
- [x] Dixon-Coles τ correction for Poisson scoreline accuracy
- [x] ELO-based draw prior for realistic 3-outcome probabilities
- [x] 4-page Streamlit UI
- [x] Full 72-match WC2026 schedule with CT times and venues
- [x] Group stage predictor (qualification %, scorelines, finish-position breakdown)
- [x] Live score integration stub (activates June 11, 2026)
- [x] Dockerfile + GitHub Actions CI pipeline
- [x] Azure Container Apps deployment workflow
- [x] Git LFS for large model files
- [ ] Real rolling form stats from live API (replace ELO-derived estimates)
- [ ] Head-to-head history page
- [ ] Bracket auto-update as live results come in
- [ ] pytest unit tests for simulation engine and feature layer
- [ ] Azure Bicep template for one-click infrastructure provisioning
- [ ] Streamlit Cloud deployment option

---

## 📄 License

MIT License — see [`LICENSE`](LICENSE) for details.

---

<div align="center">

**FIFA WC 2026 Match Predictor**
Machine Learning · Monte Carlo Simulation · Streamlit

*Predicting the beautiful game.*

</div>
