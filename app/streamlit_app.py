"""
WC2026 Prediction App — Streamlit UI
Pages:
  1. Match Predictor       — predict any single match with 4 models
  2. Tournament Schedule   — group bracket with dates, times (CT) & venues + live scores
  3. Group Predictor       — deep-dive simulation of any single group
  4. Winner Probabilities  — WC2026 winner bar chart (Monte Carlo)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.team_data       import TEAM_ELO, GROUPS, HOST_NATIONS, get_form_stats
from src.simulate        import predict_win_prob, run_monte_carlo
from src.schedule        import GROUP_MATCHES, VENUES, KNOCKOUT_ROUNDS
from src.live_scores     import (get_live_scores, get_group_standings,
                                  is_api_configured, is_tournament_live,
                                  status_badge)
from src.group_predictor import run_group_simulations

# ── App config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FIFA WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
)

MODEL_DIR        = ROOT / "models"
DATA_DIR         = ROOT / "data"
SIM_RESULTS_PATH = DATA_DIR / "simulation_results.json"

# ── Sidebar nav ───────────────────────────────────────────────────────────────

page = st.sidebar.radio(
    "Navigate",
    ["Match Predictor", "Tournament Schedule", "Group Predictor", "Winner Probabilities"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Models trained on 47,399 matches · 19 features · 69% accuracy")

# ── Load models (cached) ──────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    return {
        "logistic_regression": joblib.load(MODEL_DIR / "logistic_regression.pkl"),
        "xgboost":             joblib.load(MODEL_DIR / "xgboost.pkl"),
        "random_forest":       joblib.load(MODEL_DIR / "random_forest.pkl"),
        "ensemble":            joblib.load(MODEL_DIR / "ensemble.pkl"),
        "scaler":              joblib.load(MODEL_DIR / "scaler.pkl"),
        "feature_cols":        joblib.load(MODEL_DIR / "feature_cols.pkl"),
    }

models       = load_models()
FEATURE_COLS = models["feature_cols"]
ALL_TEAMS    = sorted(TEAM_ELO.keys())

# ── Shared helpers ────────────────────────────────────────────────────────────

def build_features(team_a, team_b, neutral, friendly):
    elo_a = TEAM_ELO.get(team_a, 1500)
    elo_b = TEAM_ELO.get(team_b, 1500)
    fa    = get_form_stats(team_a)
    fb    = get_form_stats(team_b)
    row   = {
        "home_elo":               elo_a,
        "away_elo":               elo_b,
        "elo_diff":               elo_a - elo_b,
        "is_home_host":           int(team_a in HOST_NATIONS),
        "is_away_host":           int(team_b in HOST_NATIONS),
        "is_neutral":             int(neutral),
        "is_friendly":            int(friendly),
        "home_win_rate":          fa["win_rate"],
        "home_goals_scored_avg":  fa["goals_scored_avg"],
        "home_goals_conceded_avg": fa["goals_conceded_avg"],
        "home_points_per_game":   fa["points_per_game"],
        "away_win_rate":          fb["win_rate"],
        "away_goals_scored_avg":  fb["goals_scored_avg"],
        "away_goals_conceded_avg": fb["goals_conceded_avg"],
        "away_points_per_game":   fb["points_per_game"],
        "h2h_home_wins":          1,
        "h2h_draws":              1,
        "h2h_away_wins":          1,
        "h2h_home_goal_diff":     0,
    }
    return pd.DataFrame([row])


def predict_all_models(team_a, team_b, neutral, friendly):
    feats    = build_features(team_a, team_b, neutral, friendly)
    X_scaled = pd.DataFrame(
        models["scaler"].transform(feats[FEATURE_COLS]), columns=FEATURE_COLS
    )
    return {
        name: models[name].predict_proba(X_scaled)[0][1]
        for name in ["logistic_regression", "xgboost", "random_forest", "ensemble"]
    }


def _load_sim_probs() -> dict[str, float]:
    if SIM_RESULTS_PATH.exists():
        with open(SIM_RESULTS_PATH) as f:
            return json.load(f)["win_probs"]
    return {}


# ── Colour helpers ────────────────────────────────────────────────────────────

def _prob_bg(prob: float) -> str:
    """ELO-prob → dark background color (dark gray → amber → gold)."""
    if prob <= 0:
        return "#1a1f2e"
    t  = min(prob / 0.06, 1.0)
    r  = int(50  + t * 205)
    g  = int(55  + t * 125)
    b  = int(80  - t * 60)
    return f"#{r:02x}{g:02x}{b:02x}"


def _team_chip(team: str, prob: float | None, is_host: bool,
               show_pct: bool = True) -> str:
    star  = " ★" if is_host else ""
    bg    = _prob_bg(prob) if (prob is not None and prob > 0) else "#1a1f2e"
    bold  = "font-weight:700;" if (prob or 0) >= 0.028 else ""
    pct   = (f"<span style='font-size:0.68em;opacity:0.85;margin-left:5px'>"
             f"{prob*100:.1f}%</span>") if (show_pct and prob and prob > 0) else ""
    return (
        f"<span style='display:inline-block;background:{bg};{bold}"
        f"padding:3px 9px;margin:2px 1px;border-radius:5px;"
        f"font-size:0.85em;white-space:nowrap'>{team}{star}{pct}</span>"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Match Predictor
# ══════════════════════════════════════════════════════════════════════════════

if page == "Match Predictor":
    st.title("⚽ Match Predictor")

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        home = st.selectbox(
            "Home / Team A", ALL_TEAMS,
            index=ALL_TEAMS.index("Argentina") if "Argentina" in ALL_TEAMS else 0
        )
    with col2:
        st.markdown("<br><h3 style='text-align:center'>vs</h3>",
                    unsafe_allow_html=True)
    with col3:
        away = st.selectbox(
            "Away / Team B", ALL_TEAMS,
            index=ALL_TEAMS.index("France") if "France" in ALL_TEAMS else 1
        )

    c1, c2   = st.columns(2)
    neutral  = c1.checkbox("Neutral venue",  value=True)
    friendly = c2.checkbox("Friendly match", value=False)

    if st.button("Predict", type="primary", use_container_width=True):
        if home == away:
            st.error("Please select two different teams.")
        else:
            preds               = predict_all_models(home, away, neutral, friendly)
            p_a, p_draw, p_b    = predict_win_prob(home, away, neutral, apply_host_flag=True)

            st.markdown("---")
            st.subheader(f"{home}  vs  {away}")

            ca, cd, cb = st.columns(3)
            ca.metric(f"🏆 {home} wins", f"{p_a*100:.1f}%")
            cd.metric("🤝 Draw",          f"{p_draw*100:.1f}%")
            cb.metric(f"🏆 {away} wins",  f"{p_b*100:.1f}%")

            st.markdown("#### Model breakdown — P(home/team A wins)")
            model_labels = {
                "logistic_regression": "Logistic Regression",
                "xgboost":             "XGBoost",
                "random_forest":       "Random Forest",
                "ensemble":            "Ensemble",
            }
            fig = go.Figure(go.Bar(
                x=[model_labels[k] for k in preds],
                y=[v * 100 for v in preds.values()],
                marker_color=["#636EFA", "#EF553B", "#00CC96", "#AB63FA"],
                text=[f"{v*100:.1f}%" for v in preds.values()],
                textposition="outside",
            ))
            fig.update_layout(yaxis_title="Win probability (%)",
                              yaxis_range=[0, 100], height=350,
                              margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### ELO ratings")
            ec1, ec2 = st.columns(2)
            ec1.metric(home, TEAM_ELO.get(home, "N/A"))
            ec2.metric(away, TEAM_ELO.get(away, "N/A"),
                       delta=f"{TEAM_ELO.get(away,1500)-TEAM_ELO.get(home,1500):+d}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Tournament Schedule
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Tournament Schedule":
    st.title("📅 WC 2026 — Tournament Schedule")
    st.caption(
        "Full group-stage schedule with kick-off times (CT) and venues. "
        "Teams color-coded by predicted win probability when simulation results are available."
    )

    probs = _load_sim_probs()
    if probs:
        st.success("✅ Simulation results loaded — teams are color-coded by championship probability.")
    else:
        st.info("Run a simulation on the **Winner Probabilities** page to color-code teams.")

    # ── View selector ─────────────────────────────────────────────────────────
    view = st.radio("View", ["Group Cards", "Match Schedule", "Knockout Rounds"],
                    horizontal=True)

    # ── View A: Group Cards ───────────────────────────────────────────────────
    if view == "Group Cards":
        st.markdown("### Groups")
        group_items = list(GROUPS.items())
        for row_start in range(0, 12, 4):
            cols = st.columns(4)
            for ci, col in enumerate(cols):
                gi = row_start + ci
                if gi >= len(group_items):
                    break
                gname, teams = group_items[gi]
                sorted_teams = (sorted(teams, key=lambda t: probs.get(t,0), reverse=True)
                                if probs else teams)
                chips = "".join(
                    _team_chip(t, probs.get(t) if probs else None, t in HOST_NATIONS)
                    for t in sorted_teams
                )
                col.markdown(
                    f"<div style='background:#111827;border:1px solid #2d3748;"
                    f"border-radius:8px;padding:10px 12px;margin-bottom:10px'>"
                    f"<div style='font-size:0.7em;color:#6b7280;letter-spacing:.08em;"
                    f"margin-bottom:6px'>GROUP {gname}</div>{chips}</div>",
                    unsafe_allow_html=True,
                )

    # ── View B: Match Schedule ────────────────────────────────────────────────
    elif view == "Match Schedule":
        # Filter controls
        fc1, fc2 = st.columns([1, 3])
        sel_group = fc1.selectbox("Filter by group", ["All"] + list(GROUPS.keys()))
        sel_md    = fc2.radio("Matchday", ["All", "1", "2", "3"], horizontal=True)

        # Build display dataframe
        rows = []
        for date, time_et, home_t, away_t, venue_key, grp, md in GROUP_MATCHES:
            if sel_group != "All" and grp != sel_group:
                continue
            if sel_md != "All" and str(md) != sel_md:
                continue
            venue_info = VENUES.get(venue_key, {})
            p_h = probs.get(home_t, 0) if probs else None
            p_a = probs.get(away_t, 0) if probs else None

            # Format date nicely
            from datetime import datetime
            dt = datetime.strptime(date, "%Y-%m-%d")
            date_fmt = dt.strftime("%a %d %b")

            rows.append({
                "Group":   grp,
                "MD":      md,
                "Date":    date_fmt,
                "Time (CT)": time_et,
                "Home":    home_t,
                "Away":    away_t,
                "Venue":   venue_info.get("name", venue_key),
                "City":    venue_info.get("city", ""),
                "p_home":  p_h,
                "p_away":  p_a,
            })

        # Fetch live scores (empty dict before Jun 11 or if no API key)
        live = get_live_scores()
        if live:
            st.markdown(
                "<div style='background:#14532d;border:1px solid #16a34a;"
                "border-radius:6px;padding:6px 14px;margin-bottom:10px;"
                "font-size:0.82em'>🔴 Live score data active — scores update every 60 s</div>",
                unsafe_allow_html=True,
            )
        elif is_tournament_live() and not is_api_configured():
            st.warning(
                "Tournament is live! Add your **FOOTBALL_DATA_API_KEY** environment variable "
                "to enable live scores. Get a free key at football-data.org"
            )

        if not rows:
            st.warning("No matches for this filter.")
        else:
            # Group by date for display
            by_date = defaultdict(list)
            for r in rows:
                by_date[r["Date"]].append(r)

            for date_label, matches in by_date.items():
                st.markdown(
                    f"<div style='background:#0f172a;padding:4px 12px;"
                    f"border-left:3px solid #f59e0b;margin:12px 0 6px 0;"
                    f"font-size:0.85em;color:#f59e0b;font-weight:600'>"
                    f"📆 {date_label}</div>",
                    unsafe_allow_html=True,
                )
                for m in matches:
                    chip_h = _team_chip(m["Home"], m["p_home"], m["Home"] in HOST_NATIONS, show_pct=bool(probs))
                    chip_a = _team_chip(m["Away"], m["p_away"], m["Away"] in HOST_NATIONS, show_pct=bool(probs))

                    # Live score overlay
                    live_key  = (m["Home"], m["Away"])
                    live_data = live.get(live_key)
                    if live_data and live_data["home_score"] is not None:
                        score_str = (
                            f"<span style='color:#f8fafc;font-weight:700;"
                            f"font-size:1em;padding:0 8px'>"
                            f"{live_data['home_score']} – {live_data['away_score']}</span>"
                            f"{status_badge(live_data['status'], live_data.get('minute'))}"
                        )
                    else:
                        score_str = (
                            f"<span style='color:#64748b;font-size:0.8em;padding:0 4px'>vs</span>"
                        )

                    st.markdown(
                        f"<div style='background:#1e293b;border:1px solid #334155;"
                        f"border-radius:7px;padding:8px 14px;margin-bottom:6px;"
                        f"display:flex;align-items:center;gap:8px;flex-wrap:wrap'>"
                        f"<span style='color:#94a3b8;font-size:0.75em;min-width:44px'>"
                        f"Grp {m['Group']}</span>"
                        f"<span style='color:#64748b;font-size:0.75em;min-width:44px'>"
                        f"{m['Time (CT)']} CT</span>"
                        f"{chip_h}{score_str}{chip_a}"
                        f"<span style='color:#475569;font-size:0.72em;margin-left:auto'>"
                        f"📍 {m['Venue']}, {m['City']}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    # ── View C: Knockout Rounds ───────────────────────────────────────────────
    elif view == "Knockout Rounds":
        st.markdown("### Knockout Schedule")

        for rnd in KNOCKOUT_ROUNDS:
            venue_str = f" &nbsp;·&nbsp; 📍 {rnd['venue']}" if "venue" in rnd else ""
            st.markdown(
                f"<div style='background:#111827;border:1px solid #1f2937;"
                f"border-radius:8px;padding:12px 16px;margin-bottom:10px'>"
                f"<div style='font-size:1em;font-weight:700;color:#f59e0b'>"
                f"{rnd['round']}</div>"
                f"<div style='font-size:0.82em;color:#9ca3af;margin-top:3px'>"
                f"📅 {rnd['dates']} &nbsp;·&nbsp; {rnd['matches']} match"
                f"{'es' if rnd['matches']>1 else ''}{venue_str}</div>",
                unsafe_allow_html=True,
            )

            if probs and rnd["round"] != "Final":
                # Show predicted contenders for this round
                thresholds = {
                    "Round of 32":    0.015,
                    "Round of 16":    0.020,
                    "Quarter-Finals": 0.025,
                    "Semi-Finals":    0.030,
                }
                thresh = thresholds.get(rnd["round"], 0)
                contenders = sorted(
                    [(t, p) for t, p in probs.items() if p >= thresh],
                    key=lambda x: x[1], reverse=True
                )
                chips = " ".join(
                    _team_chip(t, p, t in HOST_NATIONS, show_pct=True)
                    for t, p in contenders
                )
                st.markdown(
                    f"<div style='margin-top:7px;line-height:2'>{chips}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                if probs:
                    winner = max(probs, key=probs.get)
                    chip   = _team_chip(winner, probs[winner], winner in HOST_NATIONS)
                    st.markdown(
                        f"<div style='margin-top:7px'>🏆 Predicted winner: {chip}</div></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Group Predictor
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Group Predictor":
    st.title("🔬 Group Stage Predictor")
    st.caption("Monte Carlo simulation of a single group — standings, qualification odds, scorelines")

    # ── Controls ──────────────────────────────────────────────────────────────
    gc1, gc2, gc3 = st.columns([1, 1, 2])
    sel_group = gc1.selectbox("Select group", list(GROUPS.keys()), index=0)
    n_sims    = gc2.selectbox("Simulations", [1_000, 5_000, 10_000], index=1)
    run_btn   = gc3.button("▶ Run group simulation", type="primary", use_container_width=True)

    # Cache key so we don't re-run on every widget interaction
    cache_key = f"group_result_{sel_group}_{n_sims}"
    if run_btn or cache_key not in st.session_state:
        with st.spinner(f"Simulating Group {sel_group} × {n_sims:,}…"):
            st.session_state[cache_key] = run_group_simulations(sel_group, n=n_sims)

    res = st.session_state.get(cache_key)
    if not res:
        st.info("Select a group and click **▶ Run group simulation**.")
        st.stop()

    teams           = res["teams"]
    finish_probs    = res["finish_probs"]
    qualify_probs   = res["qualify_probs"]
    match_probs     = res["match_probs"]
    top_scorelines  = res["top_scorelines"]
    avg_pts         = res["avg_pts"]
    predicted_order = res["predicted_order"]

    st.markdown(f"### Group {sel_group}  ·  {n_sims:,} simulations")

    # ── Live standings override ───────────────────────────────────────────────
    live_standings = get_group_standings(sel_group) if is_tournament_live() else None
    if live_standings:
        st.markdown("#### 📡 Live Standings")
        st.dataframe(pd.DataFrame(live_standings), use_container_width=True, hide_index=True)
        st.markdown("---")

    # ── 1. Qualification probability table ───────────────────────────────────
    st.markdown("#### 🎯 Qualification Probabilities")
    qual_rows = []
    for team in predicted_order:
        fp = finish_probs[team]
        qual_rows.append({
            "Team":      team,
            "1st 🥇":   f"{fp[0]*100:.1f}%",
            "2nd 🥈":   f"{fp[1]*100:.1f}%",
            "3rd":       f"{fp[2]*100:.1f}%",
            "4th":       f"{fp[3]*100:.1f}%",
            "Qualify %": f"{qualify_probs[team]*100:.1f}%",
            "Avg Pts":   f"{avg_pts[team]:.2f}",
        })
    st.dataframe(pd.DataFrame(qual_rows), use_container_width=True, hide_index=True)

    # ── 2. Predicted final standings bar ─────────────────────────────────────
    st.markdown("#### 📊 Predicted Standings — Finish Position Breakdown")
    pos_labels = ["1st", "2nd", "3rd", "4th"]
    colors     = ["#f59e0b", "#9ca3af", "#cd7f32", "#374151"]
    fig = go.Figure()
    for i, (pos, color) in enumerate(zip(pos_labels, colors)):
        fig.add_trace(go.Bar(
            name=pos,
            x=predicted_order,
            y=[finish_probs[t][i] * 100 for t in predicted_order],
            marker_color=color,
        ))
    fig.update_layout(
        barmode="stack",
        yaxis_title="Probability (%)",
        height=340,
        margin=dict(t=10, b=10),
        legend=dict(orientation="h", y=1.1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 3. Match-by-match breakdown ───────────────────────────────────────────
    st.markdown("#### ⚔️ Match Probabilities & Top Scorelines")
    match_pairs = list(combinations(teams, 2))
    for team_a, team_b in match_pairs:
        mp  = match_probs[(team_a, team_b)]
        scl = top_scorelines[(team_a, team_b)]

        # Scoreline pills
        score_pills = " ".join(
            f"<span style='background:#1e293b;border:1px solid #334155;"
            f"border-radius:4px;padding:2px 7px;font-size:0.75em;margin:1px'>"
            f"{s} <span style='color:#94a3b8'>({c}×)</span></span>"
            for s, c in scl
        )

        # Win/draw/loss bar (inline HTML)
        pw = int(mp["win"]  * 200)
        pd_ = int(mp["draw"] * 200)
        pl = 200 - pw - pd_

        st.markdown(
            f"<div style='background:#111827;border:1px solid #1f2937;"
            f"border-radius:8px;padding:10px 14px;margin-bottom:8px'>"
            f"<div style='display:flex;justify-content:space-between;margin-bottom:6px'>"
            f"<span style='font-weight:600'>{team_a}</span>"
            f"<span style='color:#64748b;font-size:0.8em'>vs</span>"
            f"<span style='font-weight:600'>{team_b}</span></div>"
            # probability bar
            f"<div style='display:flex;height:18px;border-radius:4px;overflow:hidden;"
            f"margin-bottom:6px'>"
            f"<div style='width:{mp['win']*100:.0f}%;background:#16a34a;"
            f"display:flex;align-items:center;justify-content:center;"
            f"font-size:0.7em;color:#fff'>{mp['win']*100:.0f}%</div>"
            f"<div style='width:{mp['draw']*100:.0f}%;background:#4b5563;"
            f"display:flex;align-items:center;justify-content:center;"
            f"font-size:0.7em;color:#fff'>{mp['draw']*100:.0f}%</div>"
            f"<div style='width:{mp['loss']*100:.0f}%;background:#dc2626;"
            f"display:flex;align-items:center;justify-content:center;"
            f"font-size:0.7em;color:#fff'>{mp['loss']*100:.0f}%</div>"
            f"</div>"
            # scoreline pills
            f"<div style='margin-top:4px'>Most likely scorelines: {score_pills}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Winner Probabilities
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Winner Probabilities":
    st.title("🏆 WC 2026 — Winner Probabilities")
    st.caption("Monte Carlo simulation · Dixon-Coles τ correction · no host bias")

    # ── Simulation controls ───────────────────────────────────────────────────
    with st.expander("⚙️ Simulation settings", expanded=False):
        n_sims  = st.slider("Number of simulations", 500, 20_000, 10_000, step=500)
        seed    = st.number_input("Random seed", value=42, min_value=0, step=1)
        run_new = st.button("▶ Run simulation", type="primary")

    # ── Load or run ───────────────────────────────────────────────────────────
    probs: dict[str, float] = {}

    if run_new:
        with st.spinner(f"Running {n_sims:,} simulations… (~2 min)"):
            from src.simulate import _prob_cache
            _prob_cache.clear()
            probs = run_monte_carlo(n_simulations=n_sims, seed=seed)
        DATA_DIR.mkdir(exist_ok=True)
        with open(SIM_RESULTS_PATH, "w") as f:
            json.dump({"n_simulations": n_sims, "seed": seed, "win_probs": probs}, f, indent=2)
        st.success("Done! Results saved — switch to Tournament Schedule to see the bracket update.")

    elif SIM_RESULTS_PATH.exists():
        with open(SIM_RESULTS_PATH) as f:
            data = json.load(f)
        probs = data["win_probs"]
        st.info(
            f"Showing saved results ({data['n_simulations']:,} simulations, "
            f"seed {data['seed']}). Expand settings above to re-run."
        )
    else:
        st.warning("No simulation results yet. Expand settings above and click **▶ Run simulation**.")

    if probs:
        probs_sorted = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
        teams_list   = list(probs_sorted.keys())
        pct_list     = [v * 100 for v in probs_sorted.values()]

        # Podium
        top3 = teams_list[:3]
        m1, m2, m3 = st.columns(3)
        m1.metric("🥇 Favourite", top3[0], f"{probs_sorted[top3[0]]*100:.1f}%")
        m2.metric("🥈 2nd",       top3[1], f"{probs_sorted[top3[1]]*100:.1f}%")
        m3.metric("🥉 3rd",       top3[2], f"{probs_sorted[top3[2]]*100:.1f}%")

        st.markdown("---")

        show_top   = st.slider("Show top N teams", 5, len(teams_list), min(20, len(teams_list)))
        teams_show = teams_list[:show_top]
        pct_show   = pct_list[:show_top]

        colors = [
            f"rgba({max(0,255-int(i*200/show_top))},"
            f"{max(0,180-int(i*150/show_top))},"
            f"{max(0,50-int(i*30/show_top))},0.85)"
            for i in range(show_top)
        ]

        fig = go.Figure(go.Bar(
            x=teams_show, y=pct_show,
            marker_color=colors,
            text=[f"{p:.1f}%" for p in pct_show],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Win probability: %{y:.2f}%<extra></extra>",
        ))
        fig.update_layout(
            title=f"WC 2026 Winner Probabilities — Top {show_top} teams",
            xaxis_title="", yaxis_title="Win probability (%)",
            height=500, margin=dict(t=50, b=100),
            xaxis_tickangle=-35,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Full results table"):
            df = pd.DataFrame({
                "Team":            teams_list,
                "Win probability": [f"{p:.2f}%" for p in pct_list],
                "Group": [next((g for g, ts in GROUPS.items() if t in ts), "—")
                          for t in teams_list],
                "ELO":   [TEAM_ELO.get(t, "—") for t in teams_list],
            })
            st.dataframe(df, use_container_width=True, hide_index=True)
