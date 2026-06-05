"""
Group Stage Predictor — runs N Monte Carlo simulations of a single WC2026 group
and returns rich statistics:
  - Qualification probabilities (1st / 2nd / 3rd / 4th)
  - Match-by-match win/draw/loss probabilities
  - Most common predicted scorelines per match
  - Predicted final standings (most likely finish order)
"""

import numpy as np
from itertools import combinations
from collections import Counter

from src.simulate import predict_win_prob, _load_models
from src.team_data import TEAM_ELO, GROUPS


def _poisson_score(elo_a: int, elo_b: int) -> tuple[int, int]:
    """Dixon-Coles corrected Poisson scoreline (imported logic inline to avoid circular import)."""
    from src.simulate import _dc_sample_score, DC_RHO
    lam_a = max(0.3, 1.1 * np.exp((elo_a - elo_b) / 600))
    lam_b = max(0.3, 1.1 * np.exp((elo_b - elo_a) / 600))
    return _dc_sample_score(lam_a, lam_b, DC_RHO)


def simulate_group_once(teams: list[str]) -> dict:
    """
    Simulate one full round-robin group.
    Returns final standings list (sorted) and match scorelines.
    """
    pts    = {t: 0 for t in teams}
    gd     = {t: 0 for t in teams}
    gf     = {t: 0 for t in teams}
    scores = {}   # (team_a, team_b) → (ga, gb)

    for team_a, team_b in combinations(teams, 2):
        elo_a = TEAM_ELO.get(team_a, 1500)
        elo_b = TEAM_ELO.get(team_b, 1500)
        ga, gb = _poisson_score(elo_a, elo_b)
        scores[(team_a, team_b)] = (ga, gb)

        gf[team_a] += ga;  gf[team_b] += gb
        gd[team_a] += ga - gb;  gd[team_b] += gb - ga

        if ga > gb:   pts[team_a] += 3
        elif ga == gb: pts[team_a] += 1; pts[team_b] += 1
        else:         pts[team_b] += 3

    standings = sorted(teams, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
    return {"standings": standings, "pts": pts, "gd": gd, "gf": gf, "scores": scores}


def run_group_simulations(group_name: str, n: int = 5_000, seed: int = 42) -> dict:
    """
    Run n simulations of a single group and return aggregated stats.

    Returns
    -------
    {
        "teams":          list[str],
        "finish_probs":   {team: [p_1st, p_2nd, p_3rd, p_4th]},
        "qualify_probs":  {team: float},   # P(finish 1st or 2nd)
        "match_probs":    {(ta,tb): {"win":f, "draw":f, "loss":f}},
        "top_scorelines": {(ta,tb): [(score_str, count), ...]},  # top 5
        "avg_pts":        {team: float},
        "avg_gd":         {team: float},
    }
    """
    np.random.seed(seed)
    _load_models()

    teams = GROUPS[group_name]
    n_teams = len(teams)

    finish_counts   = {t: [0] * n_teams for t in teams}
    scoreline_counts = {pair: Counter() for pair in combinations(teams, 2)}
    total_pts       = {t: 0 for t in teams}
    total_gd        = {t: 0 for t in teams}

    for _ in range(n):
        result = simulate_group_once(teams)
        for pos, team in enumerate(result["standings"]):
            finish_counts[team][pos] += 1
        for team in teams:
            total_pts[team] += result["pts"][team]
            total_gd[team]  += result["gd"][team]
        for pair, (ga, gb) in result["scores"].items():
            scoreline_counts[pair][f"{ga}–{gb}"] += 1

    # Finish probabilities
    finish_probs  = {t: [c / n for c in counts] for t, counts in finish_counts.items()}
    qualify_probs = {t: finish_probs[t][0] + finish_probs[t][1] for t in teams}

    # Average pts / gd
    avg_pts = {t: total_pts[t] / n for t in teams}
    avg_gd  = {t: total_gd[t]  / n for t in teams}

    # Most-likely standings (sort by expected pts then gd)
    predicted_order = sorted(teams, key=lambda t: (avg_pts[t], avg_gd[t]), reverse=True)

    # Match probabilities (from the model, not empirical — faster and exact)
    match_probs = {}
    for team_a, team_b in combinations(teams, 2):
        p_a, p_d, p_b = predict_win_prob(team_a, team_b, neutral=True, apply_host_flag=False)
        match_probs[(team_a, team_b)] = {"win": p_a, "draw": p_d, "loss": p_b}

    # Top 5 scorelines per match
    top_scorelines = {
        pair: counts.most_common(5)
        for pair, counts in scoreline_counts.items()
    }

    return {
        "group":           group_name,
        "teams":           teams,
        "n_simulations":   n,
        "finish_probs":    finish_probs,
        "qualify_probs":   qualify_probs,
        "match_probs":     match_probs,
        "top_scorelines":  top_scorelines,
        "avg_pts":         avg_pts,
        "avg_gd":          avg_gd,
        "predicted_order": predicted_order,
    }
