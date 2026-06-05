# =============================================================================
# streamlit_app/match_predictor.py
# -----------------------------------------------------------------------------
# FIFA 2026 Match Predictor — Streamlit UI
#
# Models used:
#   - Logistic Regression (scaled features)
#   - XGBoost
#   - Random Forest
#   - Ensemble (soft voting across all three)
#
# Feature pipeline:
#   Uses get_match_features_for_prediction() from src/features.py which
#   computes live rolling form + H2H stats from historical match data.
#
# Run from project root:
#   streamlit run streamlit_app/match_predictor.py
# =============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import joblib

from src.data_loader import (
    load_fixtures_2026,
    load_raw_elo,
    load_raw_matches,
    load_teams_metadata,
)
from src.features import get_match_features_for_prediction

# =============================================================================
# SECTION 1 — Setup & Data Loading
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
models_dir = PROJECT_ROOT / "models"


@st.cache_resource
def load_models():
    """Load all models once and cache them for the session."""
    log_reg   = joblib.load(models_dir / "logistic_regression.pkl")
    scaler    = joblib.load(models_dir / "scaler.pkl")
    xgb       = joblib.load(models_dir / "xgboost.pkl")
    rf        = joblib.load(models_dir / "random_forest.pkl")
    ensemble  = joblib.load(models_dir / "ensemble.pkl")
    feat_cols = joblib.load(models_dir / "feature_cols.pkl")
    return log_reg, scaler, xgb, rf, ensemble, feat_cols


@st.cache_data(ttl=3600)
def load_data():
    """Load all data sources once per hour."""
    elo_df      = load_raw_elo()
    matches_df  = load_raw_matches()
    fixtures_df = load_fixtures_2026()
    teams_meta  = load_teams_metadata()
    return elo_df, matches_df, fixtures_df, teams_meta


log_reg, scaler, xgb, rf, ensemble, feat_cols = load_models()
elo_df, matches_df, fixtures_df, teams_meta = load_data()

# Normalise fixture column names
fixtures_col_map = {c.strip().lower(): c for c in fixtures_df.columns}
if "home_team" not in fixtures_col_map and "home" in fixtures_col_map:
    fixtures_df = fixtures_df.rename(columns={fixtures_col_map["home"]: "home_team"})
if "away_team" not in fixtures_col_map and "away" in fixtures_col_map:
    fixtures_df = fixtures_df.rename(columns={fixtures_col_map["away"]: "away_team"})
if "location" not in fixtures_col_map and "venue" in fixtures_col_map:
    fixtures_df = fixtures_df.rename(columns={fixtures_col_map["venue"]: "location"})

fixtures_available = (
    not fixtures_df.empty
    and "home_team" in fixtures_df.columns
    and "away_team" in fixtures_df.columns
)

# WC2026 host nations
HOST_NATIONS = {"United States", "Canada", "Mexico"}


# =============================================================================
# SECTION 2 — Prediction Logic
# =============================================================================

def predict_match(home_team, away_team, location="",
                  is_home_host=0, is_away_host=0,
                  is_neutral=1, is_friendly=0):
    """
    Build features and run all four models for a given matchup.
    Returns predictions and probabilities for each model.
    """
    # Build enriched feature vector (ELO + rolling form + H2H)
    features = get_match_features_for_prediction(
        home_team=home_team,
        away_team=away_team,
        matches_df=matches_df,
        elo_df=elo_df,
        is_home_host=is_home_host,
        is_away_host=is_away_host,
        is_neutral=is_neutral,
        is_friendly=is_friendly,
    )

    # Ensure columns match training order exactly
    features = features.reindex(columns=feat_cols, fill_value=0)

    # Scale for Logistic Regression
    features_scaled = scaler.transform(features)

    # Predictions
    lr_proba  = log_reg.predict_proba(features_scaled)[0][1]
    xgb_proba = xgb.predict_proba(features)[0][1]
    rf_proba  = rf.predict_proba(features)[0][1]
    ens_proba = ensemble.predict_proba(features)[0][1]

    home_elo = float(elo_df[elo_df["team"] == home_team]["elo_rating"].iloc[0]) \
               if home_team in elo_df["team"].values else 1500
    away_elo = float(elo_df[elo_df["team"] == away_team]["elo_rating"].iloc[0]) \
               if away_team in elo_df["team"].values else 1500

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_elo": home_elo,
        "away_elo": away_elo,
        "features": features,
        "lr_proba":  lr_proba,
        "xgb_proba": xgb_proba,
        "rf_proba":  rf_proba,
        "ens_proba": ens_proba,
    }


def proba_to_label(proba: float) -> str:
    return "Home Win" if proba >= 0.5 else "Not Home Win"


def form_summary(team_name: str) -> dict:
    """Return last-5 win rate and avg goals for display."""
    tm = matches_df[
        (matches_df["home_team"] == team_name) |
        (matches_df["away_team"] == team_name)
    ].tail(5)
    if tm.empty:
        return {"win_rate": 0.0, "avg_goals": 0.0, "form": "N/A"}

    wins = (
        ((tm["home_team"] == team_name) & (tm["home_score"] > tm["away_score"])) |
        ((tm["away_team"] == team_name) & (tm["away_score"] > tm["home_score"]))
    ).sum()
    draws = (tm["home_score"] == tm["away_score"]).sum()
    losses = len(tm) - wins - draws

    form_str = ""
    for _, row in tm.iterrows():
        if row["home_team"] == team_name:
            gf, ga = row["home_score"], row["away_score"]
        else:
            gf, ga = row["away_score"], row["home_score"]
        form_str += "W" if gf > ga else ("D" if gf == ga else "L")

    return {
        "win_rate": wins / len(tm),
        "avg_goals": (tm["home_score"] + tm["away_score"]).mean(),
        "form": form_str,
        "wins": int(wins), "draws": int(draws), "losses": int(losses),
    }


# =============================================================================
# SECTION 3 — Streamlit UI
# =============================================================================

st.set_page_config(page_title="FIFA 2026 Predictor", page_icon="⚽", layout="wide")
st.title("⚽ FIFA 2026 Match Predictor")
st.caption("Predictions powered by Logistic Regression, XGBoost, Random Forest & Ensemble — "
           "trained on 47,000+ international matches with ELO, form, and H2H features.")

# --- Team Selection ---
teams = sorted(elo_df["team"].unique())

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home Team", teams,
                             index=teams.index("United States") if "United States" in teams else 0)
with col2:
    away_team = st.selectbox("Away Team", teams,
                             index=teams.index("Brazil") if "Brazil" in teams else 1)

# --- Location ---
if fixtures_available:
    matched = fixtures_df[
        (fixtures_df["home_team"] == home_team) &
        (fixtures_df["away_team"] == away_team)
    ]
    default_loc = (
        matched["location"].iloc[0]
        if not matched.empty and "location" in fixtures_df.columns
        else "Neutral Stadium"
    )
else:
    default_loc = "Neutral Stadium"

location = st.text_input("Match Location / Stadium", value=default_loc)

# --- Match Context ---
st.write("### Match Context")
c1, c2, c3, c4 = st.columns(4)
with c1:
    is_neutral = st.checkbox("Neutral Venue", value=True)
with c2:
    is_friendly = st.checkbox("Friendly Match", value=False)
with c3:
    is_home_host = st.checkbox(
        "Home Team is Host Country", value=home_team in HOST_NATIONS)
with c4:
    is_away_host = st.checkbox(
        "Away Team is Host Country", value=away_team in HOST_NATIONS)

# --- Predict Button ---
if st.button("Predict Match", type="primary"):

    with st.spinner("Computing form, H2H, and running models..."):
        result = predict_match(
            home_team=home_team,
            away_team=away_team,
            location=location,
            is_home_host=1 if is_home_host else 0,
            is_away_host=1 if is_away_host else 0,
            is_neutral=1 if is_neutral else 0,
            is_friendly=1 if is_friendly else 0,
        )

    home_form = form_summary(home_team)
    away_form = form_summary(away_team)

    # -------------------------------------------------------------------------
    # Team Stats Side-by-Side
    # -------------------------------------------------------------------------
    st.subheader("📋 Team Overview")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown(f"### 🏠 {home_team}")
        st.metric("ELO Rating", f"{result['home_elo']:,.0f}")
        st.metric("Recent Win Rate (L5)", f"{home_form['win_rate']:.0%}")
        st.metric("Form (last 5)", home_form.get("form", "N/A"))
        if is_home_host:
            st.info("🏆 Host Nation")
    with t2:
        st.markdown(f"### ✈️ {away_team}")
        st.metric("ELO Rating", f"{result['away_elo']:,.0f}")
        st.metric("Recent Win Rate (L5)", f"{away_form['win_rate']:.0%}")
        st.metric("Form (last 5)", away_form.get("form", "N/A"))
        if is_away_host:
            st.info("🏆 Host Nation")

    elo_diff = result["home_elo"] - result["away_elo"]
    st.caption(f"ELO difference: {elo_diff:+.0f} ({'favour of ' + home_team if elo_diff > 0 else 'favour of ' + away_team if elo_diff < 0 else 'even'})")

    # -------------------------------------------------------------------------
    # Model Predictions
    # -------------------------------------------------------------------------
    st.write("---")
    st.subheader("📊 Model Predictions")

    models_output = [
        ("Logistic Regression", result["lr_proba"]),
        ("XGBoost",             result["xgb_proba"]),
        ("Random Forest",       result["rf_proba"]),
        ("⭐ Ensemble",          result["ens_proba"]),
    ]

    cols = st.columns(4)
    for col, (name, proba) in zip(cols, models_output):
        label = proba_to_label(proba)
        with col:
            st.markdown(f"**{name}**")
            if label == "Home Win":
                st.success(label)
            else:
                st.warning(label)
            st.progress(float(proba))
            st.caption(f"{home_team} win prob: {proba:.1%}")

    # -------------------------------------------------------------------------
    # Ensemble Deep Dive
    # -------------------------------------------------------------------------
    st.write("---")
    st.subheader("🎯 Ensemble Confidence Breakdown")

    ens_proba = result["ens_proba"]
    home_prob  = ens_proba
    away_prob  = 1 - ens_proba

    b1, b2, b3 = st.columns(3)
    with b1:
        st.metric(f"{home_team} Win", f"{home_prob:.1%}")
    with b2:
        st.metric(f"{away_team} Win", f"{away_prob:.1%}")
    with b3:
        margin = abs(home_prob - away_prob)
        confidence = "High" if margin > 0.2 else "Medium" if margin > 0.1 else "Low"
        st.metric("Model Confidence", confidence)

    # Visual probability bar
    st.markdown(f"**Win probability: {home_team}** ◀──────────────▶ **{away_team}**")
    st.progress(float(home_prob))
    st.caption(f"{home_team}: {home_prob:.1%}  |  {away_team}: {away_prob:.1%}")

    # -------------------------------------------------------------------------
    # Feature Inputs Used (expandable)
    # -------------------------------------------------------------------------
    with st.expander("🔍 View feature inputs used for this prediction"):
        st.dataframe(result["features"].T.rename(columns={0: "Value"}).round(4))
