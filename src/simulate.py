"""
Monte Carlo simulator for WC2026.
Uses trained ensemble model to predict match outcomes.

Key modelling choices:
  - Host advantage (is_home_host) is applied in the Match Predictor UI
    but DISABLED in the tournament simulation — WC games are effectively
    neutral for all teams once the tournament starts.
  - Dixon-Coles τ (tau) correction is applied to the Poisson scoreline
    sampler in the group stage.  The independent-Poisson assumption
    under-counts 0-0, 1-0, 0-1, and 1-1 scorelines; τ fixes that by
    introducing a small negative correlation between the two teams' goals
    at low scores (Dixon & Coles 1997).

WC2026 format: 12 groups of 4 → top 2 + 8 best 3rd → R32 → R16 → QF → SF → Final.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from itertools import combinations

from src.team_data import TEAM_ELO, GROUPS, HOST_NATIONS, get_form_stats

MODEL_DIR = Path(__file__).parent.parent / "models"

_ensemble = None
_scaler = None
_feature_cols = None
_prob_cache: dict[tuple, tuple[float, float, float]] = {}

# Dixon-Coles rho parameter — negative correlation between low goal counts.
# Typical empirical range for international football: -0.10 to -0.18.
DC_RHO = -0.13


def _load_models():
    global _ensemble, _scaler, _feature_cols
    if _ensemble is None:
        _ensemble = joblib.load(MODEL_DIR / "ensemble.pkl")
        _scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        _feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")


# ── Dixon-Coles tau correction ────────────────────────────────────────────────

def _tau(x: int, y: int, mu: float, nu: float, rho: float) -> float:
    """
    Dixon-Coles τ adjustment factor for scoreline (x, y).
    Only deviates from 1.0 for low-scoring outcomes {0,1} x {0,1}.
    mu  = expected goals for team A (lambda_a)
    nu  = expected goals for team B (lambda_b)
    rho = correlation parameter (typically small and negative)
    """
    if x == 0 and y == 0:
        return 1 - mu * nu * rho
    elif x == 1 and y == 0:
        return 1 + nu * rho
    elif x == 0 and y == 1:
        return 1 + mu * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


def _dc_sample_score(lambda_a: float, lambda_b: float,
                     rho: float = DC_RHO) -> tuple[int, int]:
    """
    Sample a scoreline from the Dixon-Coles bivariate Poisson with τ correction.
    Uses rejection sampling: draw from independent Poissons, accept with
    probability proportional to τ(x, y).  Since τ is close to 1 for most
    scores the acceptance rate is very high (>95%), keeping it fast.
    """
    max_iter = 50
    for _ in range(max_iter):
        x = np.random.poisson(lambda_a)
        y = np.random.poisson(lambda_b)
        tau = _tau(x, y, lambda_a, lambda_b, rho)
        # tau can be slightly >1 or <1; normalise acceptance threshold
        # We bound tau to [0, 2] for safety then accept with p = tau / max_tau
        # For typical rho ≈ -0.13 the max tau deviation is small.
        accept_prob = np.clip(tau, 0.0, 2.0) / 2.0
        if np.random.random() < accept_prob:
            return x, y
    # Fallback if rejection loop exhausted (shouldn't happen in practice)
    return np.random.poisson(lambda_a), np.random.poisson(lambda_b)


# ── Feature builder ───────────────────────────────────────────────────────────

def _build_features(team_a: str, team_b: str,
                    neutral: bool = True,
                    apply_host_flag: bool = False,
                    elo_map: dict | None = None) -> pd.DataFrame:
    """
    Build a one-row feature DataFrame for team_a (home/left) vs team_b (away/right).

    apply_host_flag — set True only in the Match Predictor UI where a real
    home advantage matters.  False (default) for tournament simulation where
    all games are played at neutral WC venues.
    elo_map — alternative Elo ratings (e.g. updated with live tournament
    results); defaults to the pre-tournament TEAM_ELO.
    """
    elo = elo_map if elo_map is not None else TEAM_ELO
    elo_a = elo.get(team_a, 1500)
    elo_b = elo.get(team_b, 1500)
    form_a = get_form_stats(team_a)
    form_b = get_form_stats(team_b)

    row = {
        "home_elo":               elo_a,
        "away_elo":               elo_b,
        "elo_diff":               elo_a - elo_b,
        "is_home_host":           int(apply_host_flag and team_a in HOST_NATIONS),
        "is_away_host":           int(apply_host_flag and team_b in HOST_NATIONS),
        "is_neutral":             int(neutral),
        "is_friendly":            0,
        "home_win_rate":          form_a["win_rate"],
        "home_goals_scored_avg":  form_a["goals_scored_avg"],
        "home_goals_conceded_avg": form_a["goals_conceded_avg"],
        "home_points_per_game":   form_a["points_per_game"],
        "away_win_rate":          form_b["win_rate"],
        "away_goals_scored_avg":  form_b["goals_scored_avg"],
        "away_goals_conceded_avg": form_b["goals_conceded_avg"],
        "away_points_per_game":   form_b["points_per_game"],
        # Neutral H2H defaults — no meaningful WC2026 H2H data yet
        "h2h_home_wins":          1,
        "h2h_draws":              1,
        "h2h_away_wins":          1,
        "h2h_home_goal_diff":     0,
    }
    return pd.DataFrame([row])


# ── Win-probability model ─────────────────────────────────────────────────────

def predict_win_prob(team_a: str, team_b: str,
                     neutral: bool = True,
                     apply_host_flag: bool = False,
                     elo_map: dict | None = None) -> tuple[float, float, float]:
    """
    Return (p_a_wins, p_draw, p_b_wins) for a match.

    The ensemble model is binary (home wins vs doesn't).  We run it in both
    directions and reconcile, then inject an ELO-based draw prior so draw
    probabilities are realistic rather than near-zero.
    Results are cached per (teams, flags, Elo values) — the Elo values are
    part of the key so live-updated ratings don't collide with the
    pre-tournament ones.
    """
    elo = elo_map if elo_map is not None else TEAM_ELO
    cache_key = (team_a, team_b, neutral, apply_host_flag,
                 round(elo.get(team_a, 1500)), round(elo.get(team_b, 1500)))
    if cache_key in _prob_cache:
        return _prob_cache[cache_key]
    _load_models()

    feats_ab = _build_features(team_a, team_b, neutral, apply_host_flag, elo_map)
    feats_ba = _build_features(team_b, team_a, neutral, apply_host_flag, elo_map)

    X_ab = pd.DataFrame(
        _scaler.transform(feats_ab[_feature_cols]), columns=_feature_cols
    )
    X_ba = pd.DataFrame(
        _scaler.transform(feats_ba[_feature_cols]), columns=_feature_cols
    )

    p_a_wins_ab = _ensemble.predict_proba(X_ab)[0][1]
    p_b_wins_ba = _ensemble.predict_proba(X_ba)[0][1]

    # Symmetrised win probabilities
    p_a_raw = (p_a_wins_ab + (1 - p_b_wins_ba)) / 2
    p_b_raw = (p_b_wins_ba + (1 - p_a_wins_ab)) / 2

    # ELO-based draw prior: ~28% max for equal teams, decays as gap widens
    elo_diff = abs(elo.get(team_a, 1500) - elo.get(team_b, 1500))
    draw_prior = 0.28 * np.exp(-elo_diff / 500)

    scale = 1.0 - draw_prior
    p_a    = p_a_raw * scale
    p_b    = p_b_raw * scale
    p_draw = draw_prior

    total  = p_a + p_draw + p_b
    result = p_a / total, p_draw / total, p_b / total

    _prob_cache[cache_key] = result
    _prob_cache[(team_b, team_a, neutral, apply_host_flag,
                 round(elo.get(team_b, 1500)), round(elo.get(team_a, 1500)))] = \
        (result[2], result[1], result[0])
    return result


# ── Single-match simulator ────────────────────────────────────────────────────

def simulate_match(team_a: str, team_b: str,
                   neutral: bool = True,
                   allow_draw: bool = True,
                   elo_map: dict | None = None) -> str:
    """
    Simulate a single match outcome.
    In knockout rounds (allow_draw=False) the draw probability is redistributed
    proportionally to the two win probabilities (i.e. extra time / penalties).
    """
    p_a, p_draw, p_b = predict_win_prob(team_a, team_b, neutral,
                                        apply_host_flag=False,
                                        elo_map=elo_map)
    if not allow_draw:
        p_a_ko = p_a + p_draw * (p_a / (p_a + p_b))
        p_b_ko = 1.0 - p_a_ko
        return team_a if np.random.random() < p_a_ko else team_b

    r = np.random.random()
    if r < p_a:
        return team_a
    elif r < p_a + p_draw:
        return "draw"
    return team_b


# ── Group stage ───────────────────────────────────────────────────────────────

def simulate_group(teams: list[str]) -> dict:
    """
    Simulate one group (round-robin, 6 matches).
    Scorelines are drawn from a Dixon-Coles corrected Poisson distribution.
    Returns standings sorted by (pts, gd, gf).
    """
    pts = {t: 0 for t in teams}
    gd  = {t: 0 for t in teams}
    gf  = {t: 0 for t in teams}

    for team_a, team_b in combinations(teams, 2):
        elo_a = TEAM_ELO.get(team_a, 1500)
        elo_b = TEAM_ELO.get(team_b, 1500)

        # Expected goals anchored at ~1.1 per team for even match, scaled by ELO
        lambda_a = max(0.3, 1.1 * np.exp((elo_a - elo_b) / 600))
        lambda_b = max(0.3, 1.1 * np.exp((elo_b - elo_a) / 600))

        # Dixon-Coles τ-corrected scoreline sample
        goals_a, goals_b = _dc_sample_score(lambda_a, lambda_b)

        gf[team_a] += goals_a
        gf[team_b] += goals_b
        gd[team_a] += goals_a - goals_b
        gd[team_b] += goals_b - goals_a

        if goals_a > goals_b:
            pts[team_a] += 3
        elif goals_a == goals_b:
            pts[team_a] += 1
            pts[team_b] += 1
        else:
            pts[team_b] += 3

    standings = sorted(
        teams,
        key=lambda t: (pts[t], gd[t], gf[t]),
        reverse=True,
    )
    return {"standings": standings, "pts": pts, "gd": gd, "gf": gf}


# ── Full tournament ───────────────────────────────────────────────────────────

def simulate_tournament() -> str:
    """Run one full WC2026 simulation; return the winner's name."""
    group_results   = {}
    third_placers   = []

    for group_name, teams in GROUPS.items():
        result = simulate_group(teams)
        group_results[group_name] = result
        t3 = result["standings"][2]
        third_placers.append((
            t3,
            (result["pts"][t3], result["gd"][t3], result["gf"][t3]),
        ))

    # Best 8 third-place teams advance to Round of 32
    third_placers.sort(key=lambda x: x[1], reverse=True)
    qualified_third = {t[0] for t in third_placers[:8]}

    r32_teams = []
    for result in group_results.values():
        r32_teams.append(result["standings"][0])   # 1st
        r32_teams.append(result["standings"][1])   # 2nd
    r32_teams.extend(qualified_third)

    # Knockout: R32 → R16 → QF → SF → Final
    survivors = list(r32_teams)
    np.random.shuffle(survivors)

    while len(survivors) > 1:
        next_round = []
        for i in range(0, len(survivors), 2):
            if i + 1 < len(survivors):
                winner = simulate_match(
                    survivors[i], survivors[i + 1],
                    neutral=True, allow_draw=False
                )
                next_round.append(winner)
            else:
                next_round.append(survivors[i])
        survivors = next_round

    return survivors[0]


# ── Monte Carlo runner ────────────────────────────────────────────────────────

def run_monte_carlo(n_simulations: int = 10_000,
                    seed: int = 42) -> dict[str, float]:
    """
    Run n_simulations full WC2026 tournaments.
    Returns dict mapping team_name → win_probability (0–1).
    """
    np.random.seed(seed)
    _load_models()
    _prob_cache.clear()   # ensure cache is fresh for new run

    win_counts: dict[str, int] = {}
    for _ in range(n_simulations):
        winner = simulate_tournament()
        win_counts[winner] = win_counts.get(winner, 0) + 1

    all_teams = [t for grp in GROUPS.values() for t in grp]
    return {team: win_counts.get(team, 0) / n_simulations for team in all_teams}
