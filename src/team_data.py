"""
WC2026 team data: ELO ratings, form stats, and group assignments.
ELO ratings sourced from World Football Elo Ratings (eloratings.net) circa 2025.
Form stats (win_rate, goals, etc.) are estimated from ELO tier.
"""

import numpy as np

# 48 qualified teams with ELO ratings
TEAM_ELO = {
    # Hosts
    "USA":          1728,
    "Canada":       1642,
    "Mexico":       1720,
    # CONMEBOL
    "Argentina":    2088,
    "Brazil":       1839,
    "Colombia":     1780,
    "Uruguay":      1795,
    "Ecuador":      1720,
    "Venezuela":    1659,
    "Paraguay":     1637,
    # UEFA
    "France":       2005,
    "England":      1939,
    "Spain":        1985,
    "Portugal":     1946,
    "Germany":      1939,
    "Netherlands":  1906,
    "Belgium":      1838,
    "Croatia":      1826,
    "Switzerland":  1820,
    "Denmark":      1790,
    "Italy":        1800,
    "Austria":      1763,
    "Serbia":       1762,
    "Poland":       1741,
    "Hungary":      1718,
    "Turkey":       1731,
    "Scotland":     1720,
    "Ukraine":      1712,
    "Greece":       1680,
    # CAF
    "Morocco":      1786,
    "Senegal":      1729,
    "Egypt":        1695,
    "Nigeria":      1674,
    "Ivory Coast":  1668,
    "Cameroon":     1632,
    "Mali":         1628,
    "Algeria":      1654,
    "South Africa": 1578,
    # AFC
    "Japan":        1804,
    "South Korea":  1726,
    "Iran":         1729,
    "Australia":    1698,
    "Saudi Arabia": 1639,
    "Iraq":         1617,
    "Jordan":       1598,
    "Uzbekistan":   1603,
    # CONCACAF
    "Panama":       1621,
    "Honduras":     1543,
    "Jamaica":      1566,
    # OFC
    "New Zealand":  1545,
    # Repechage/play-off
    "Indonesia":    1471,
    "Bolivia":      1537,
    "Cuba":         1399,
}

# WC2026 Groups (12 groups of 4)
GROUPS = {
    "A": ["Argentina", "Chile_placeholder", "Poland", "Saudi Arabia"],
    "B": ["France", "Morocco", "Croatia", "Uruguay"],
    "C": ["Spain", "Serbia", "Ivory Coast", "New Zealand"],
    "D": ["England", "Senegal", "Netherlands_alt", "Bolivia"],
    "E": ["Brazil", "Colombia", "Germany_alt", "Cuba"],
    "F": ["Portugal", "Turkey", "Egypt", "Venezuela"],
    "G": ["Netherlands", "Denmark", "Japan", "Honduras"],
    "H": ["Germany", "Belgium", "South Korea", "Indonesia"],
    "I": ["Italy", "Switzerland", "Iran", "Jamaica"],
    "J": ["USA", "Panama", "Uzbekistan", "Algeria"],
    "K": ["Mexico", "Canada", "Australia", "Ecuador"],
    "L": ["Croatia_alt", "Austria", "Nigeria", "Iraq"],
}

# WC2026 groups — source: FIFA Official Match Schedule v17, April 10 2026
GROUPS = {
    "A": ["Mexico",      "South Africa",        "Korea Republic",  "Czechia"],
    "B": ["Canada",      "Bosnia & Herzegovina","Qatar",           "Switzerland"],
    "C": ["Brazil",      "Morocco",             "Haiti",           "Scotland"],
    "D": ["USA",         "Paraguay",            "Australia",       "Türkiye"],
    "E": ["Germany",     "Curaçao",             "Côte d'Ivoire",   "Ecuador"],
    "F": ["Netherlands", "Japan",               "Sweden",          "Tunisia"],
    "G": ["Belgium",     "Egypt",               "IR Iran",         "New Zealand"],
    "H": ["Spain",       "Cabo Verde",          "Saudi Arabia",    "Uruguay"],
    "I": ["France",      "Senegal",             "Iraq",            "Norway"],
    "J": ["Argentina",   "Algeria",             "Austria",         "Jordan"],
    "K": ["Portugal",    "Congo DR",            "Uzbekistan",      "Colombia"],
    "L": ["England",     "Croatia",             "Ghana",           "Panama"],
}

# No placeholder fixes needed — all teams are confirmed
_team_fixes = {}

def _apply_fixes():
    for group, teams in GROUPS.items():
        GROUPS[group] = [_team_fixes.get(t, t) for t in teams]
    for old, new in _team_fixes.items():
        if old in TEAM_ELO:
            TEAM_ELO[new] = TEAM_ELO.pop(old)

_apply_fixes()

# Host nations (get slight home advantage boost in ELO lookup)
HOST_NATIONS = {"USA", "Canada", "Mexico"}


def get_form_stats(team: str) -> dict:
    """Estimate rolling form stats from ELO rating."""
    elo = TEAM_ELO.get(team, 1500)
    # Normalise ELO to [0,1] range across realistic range 1300–2100
    strength = np.clip((elo - 1300) / 800, 0, 1)

    win_rate = 0.25 + strength * 0.55          # 0.25 – 0.80
    goals_scored = 0.8 + strength * 1.4        # 0.8 – 2.2
    goals_conceded = 2.0 - strength * 1.2      # 0.8 – 2.0
    points_per_game = win_rate * 3 * 0.9       # rough PPG

    return {
        "win_rate":          round(win_rate, 3),
        "goals_scored_avg":  round(goals_scored, 2),
        "goals_conceded_avg": round(goals_conceded, 2),
        "points_per_game":   round(points_per_game, 2),
    }


def get_all_teams() -> list[str]:
    """Return all 48 WC2026 teams."""
    teams = set()
    for group_teams in GROUPS.values():
        teams.update(group_teams)
    return sorted(teams)
