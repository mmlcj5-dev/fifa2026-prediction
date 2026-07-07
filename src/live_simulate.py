"""
Live tournament-conditioned Monte Carlo for WC2026.

Once the tournament is under way the pre-tournament simulation goes stale:
group outcomes are decided, most knockout results are known, and most of
the 48 teams are already eliminated.  This module rebuilds championship
probabilities from the CURRENT tournament state:

  1. Fetch all matches from football-data.org (via src.live_scores).
  2. Update the pre-tournament Elo ratings with every finished match
     (eloratings.net formula: K=60 for World Cup finals, goal-difference
     multiplier), so in-tournament form feeds the win-probability model.
  3. Reconstruct the knockout bracket from the API match list.  Finished
     matches are fixed; matches whose teams the API hasn't filled in yet
     inherit the winner of their feeding match from the previous round.
  4. Monte Carlo only the remaining matches.

Every function degrades gracefully: if the API is unreachable, the key is
missing, or the knockout draw isn't published yet, run_live_monte_carlo()
returns None and callers fall back to the pre-tournament simulation.
"""

import numpy as np
from collections import Counter

from src.live_scores import fetch_matches, _normalize_team
from src.simulate import simulate_match
from src.team_data import TEAM_ELO, GROUPS

# eloratings.net weight for World Cup finals matches
K_FACTOR = 60

# Knockout stages in bracket order (WC2026: R32 → R16 → QF → SF → Final).
# THIRD_PLACE is deliberately excluded — it never affects the champion.
STAGE_ORDER = ["LAST_32", "LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "FINAL"]


# ── Elo updated with real results ─────────────────────────────────────────────

def _goal_multiplier(goal_diff: int) -> float:
    """eloratings.net goal-difference multiplier."""
    if goal_diff >= 3:
        return (11 + goal_diff) / 8
    if goal_diff == 2:
        return 1.5
    return 1.0


def elo_with_results(matches: list[dict] | None = None) -> dict[str, float]:
    """
    Return a copy of TEAM_ELO updated with every finished tournament match,
    applied in chronological order.  Penalty shootouts count as draws for
    rating purposes (fullTime score excludes the shootout).
    """
    if matches is None:
        matches = fetch_matches()

    elo = {t: float(r) for t, r in TEAM_ELO.items()}
    played = sorted(
        (m for m in matches if m.get("status") == "FINISHED"),
        key=lambda m: m.get("utcDate", ""),
    )
    for m in played:
        home = _normalize_team((m.get("homeTeam") or {}).get("name") or "")
        away = _normalize_team((m.get("awayTeam") or {}).get("name") or "")
        ft = (m.get("score") or {}).get("fullTime", {})
        hs, as_ = ft.get("home"), ft.get("away")
        if not home or not away or hs is None or as_ is None:
            continue

        r_home, r_away = elo.get(home, 1500.0), elo.get(away, 1500.0)
        expected_home = 1.0 / (1.0 + 10 ** ((r_away - r_home) / 400))
        result_home = 1.0 if hs > as_ else 0.0 if hs < as_ else 0.5
        delta = K_FACTOR * _goal_multiplier(abs(hs - as_)) * (result_home - expected_home)
        elo[home] = r_home + delta
        elo[away] = r_away - delta
    return elo


# ── Bracket reconstruction ────────────────────────────────────────────────────

def build_bracket(matches: list[dict]) -> list[list[dict]] | None:
    """
    Rebuild the knockout bracket from the API match list.

    Returns a list of stages (STAGE_ORDER order).  Each stage is a list of
    match dicts sorted by API match id:
        {"home": str|None, "away": str|None,      # normalized names
         "finished": bool, "winner": str|None,
         "home_src": int|None, "away_src": int|None}  # feeder index in prev stage

    Feeder resolution: a known team "consumes" the previous-stage match it
    won; empty slots are then filled from the unconsumed previous-stage
    matches in id order.  (The API doesn't expose bracket links directly,
    and its id ordering is not adjacent-pairs — e.g. QF2 2026 drew from R16
    matches 5 and 6 — so consumption is the only reliable mapping for slots
    whose teams are already published.)

    Returns None if the knockout draw isn't available yet (first stage has
    unfilled, unresolvable teams).
    """
    stages: list[list[dict]] = []
    for stage_name in STAGE_ORDER:
        api_ms = sorted(
            (m for m in matches if m.get("stage") == stage_name),
            key=lambda m: m.get("id", 0),
        )
        if not api_ms:
            return None
        stage = []
        for m in api_ms:
            home = _normalize_team((m.get("homeTeam") or {}).get("name") or "") or None
            away = _normalize_team((m.get("awayTeam") or {}).get("name") or "") or None
            ft = (m.get("score") or {}).get("fullTime", {})
            hs, as_ = ft.get("home"), ft.get("away")
            finished = (m.get("status") == "FINISHED"
                        and hs is not None and as_ is not None)
            winner = None
            if finished:
                if hs != as_:
                    winner = home if hs > as_ else away
                else:
                    # Decided on penalties
                    pens = (m.get("score") or {}).get("penalties") or {}
                    ph, pa = pens.get("home"), pens.get("away")
                    winner = home if (ph or 0) > (pa or 0) else away
            stage.append({"home": home, "away": away,
                          "finished": finished, "winner": winner,
                          "home_src": None, "away_src": None})
        stages.append(stage)

    # First stage must be fully drawn — otherwise groups aren't decided yet.
    if any(not m["home"] or not m["away"] for m in stages[0]):
        return None

    # Wire feeders for later stages.
    for prev, stage in zip(stages, stages[1:]):
        consumed = set()
        for m in stage:
            for side in ("home", "away"):
                team = m[side]
                if not team:
                    continue
                for j, pm in enumerate(prev):
                    if j not in consumed and pm["finished"] and pm["winner"] == team:
                        consumed.add(j)
                        break
        unconsumed = iter(j for j in range(len(prev)) if j not in consumed)
        for m in stage:
            for side in ("home", "away"):
                if not m[side]:
                    m[side + "_src"] = next(unconsumed, None)
                    if m[side + "_src"] is None:
                        return None   # bracket inconsistent — bail out
    return stages


# ── Monte Carlo over the remaining matches ────────────────────────────────────

def run_live_monte_carlo(n_simulations: int = 10_000,
                         seed: int = 42) -> dict | None:
    """
    Championship probabilities conditioned on all results so far.

    Returns None when live data is unavailable, otherwise:
        {
            "win_probs":  {team: float},   # all 48 teams; eliminated = 0.0
            "elo":        {team: float},   # Elo updated with real results
            "n_finished": int,             # real results conditioned on
            "n_remaining": int,            # matches still to be simulated
        }
    """
    matches = fetch_matches()
    if not matches:
        return None
    bracket = build_bracket(matches)
    if bracket is None:
        return None

    elo = elo_with_results(matches)
    n_finished = sum(1 for m in matches if m.get("status") == "FINISHED")
    n_remaining = sum(1 for stage in bracket for m in stage if not m["finished"])

    np.random.seed(seed)
    champion_counts: Counter = Counter()
    for _ in range(n_simulations):
        prev_winners: list[str] = []
        for stage in bracket:
            winners = []
            for m in stage:
                if m["finished"]:
                    winners.append(m["winner"])
                    continue
                home = m["home"] or prev_winners[m["home_src"]]
                away = m["away"] or prev_winners[m["away_src"]]
                winners.append(simulate_match(home, away, neutral=True,
                                              allow_draw=False, elo_map=elo))
            prev_winners = winners
        champion_counts[prev_winners[0]] += 1

    all_teams = [t for grp in GROUPS.values() for t in grp]
    win_probs = {t: champion_counts.get(t, 0) / n_simulations for t in all_teams}
    return {
        "win_probs": win_probs,
        "elo": elo,
        "n_finished": n_finished,
        "n_remaining": n_remaining,
    }
