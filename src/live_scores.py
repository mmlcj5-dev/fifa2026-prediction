"""
Live score integration for WC 2026.
Uses the football-data.org API (free tier — 10 req/min, no credit card needed).
Sign up at https://www.football-data.org/client/register to get a free API key.

Set the key as an environment variable:
    Windows:  $env:FOOTBALL_DATA_API_KEY = "your_key_here"
    .env file: FOOTBALL_DATA_API_KEY=your_key_here

Before the tournament starts (Jun 11 2026) this module returns an empty result
so the rest of the app works unchanged.  On and after Jun 11 it fetches live /
recent match data automatically.
"""

import os
import datetime
import requests
from functools import lru_cache

# ── Constants ─────────────────────────────────────────────────────────────────

TOURNAMENT_START = datetime.date(2026, 6, 11)
API_BASE         = "https://api.football-data.org/v4"
WC2026_ID        = 2000          # football-data.org competition ID for FIFA World Cup
REQUEST_TIMEOUT  = 8             # seconds

# Status codes that mean the match has a real score
FINISHED_STATUSES = {"FINISHED", "IN_PLAY", "PAUSED", "EXTRA_TIME", "PENALTY_SHOOTOUT"}


def _api_key() -> str | None:
    return os.environ.get("FOOTBALL_DATA_API_KEY")


def is_api_configured() -> bool:
    return bool(_api_key())


def is_tournament_live() -> bool:
    """True on or after the tournament start date."""
    return datetime.date.today() >= TOURNAMENT_START


# ── API fetch (cached for 60 s to avoid hammering the free-tier rate limit) ───

@lru_cache(maxsize=1)
def _fetch_matches_cached(cache_bust: int) -> list[dict]:
    """
    Fetch all WC2026 matches from football-data.org.
    cache_bust is the current minute (time.time() // 60) so the cache
    refreshes every 60 seconds automatically.
    """
    key = _api_key()
    if not key:
        return []
    headers = {"X-Auth-Token": key}
    url     = f"{API_BASE}/competitions/{WC2026_ID}/matches"
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("matches", [])
    except Exception:
        return []


def fetch_matches() -> list[dict]:
    import time
    cache_bust = int(time.time() // 60)   # refresh cache every minute
    return _fetch_matches_cached(cache_bust)


# ── Public interface ──────────────────────────────────────────────────────────

def get_live_scores() -> dict[tuple[str, str], dict]:
    """
    Return a dict keyed by (home_team, away_team) with match result info.

    Each value:
        {
            "home_score":  int | None,
            "away_score":  int | None,
            "status":      str,          # "FINISHED", "IN_PLAY", "SCHEDULED", etc.
            "minute":      int | None,   # match minute if in play
            "matchday":    int,
            "date":        str,          # ISO date
        }

    Returns an empty dict if:
        - The API key is not configured
        - The tournament hasn't started yet
        - The API request fails
    """
    if not is_tournament_live() or not is_api_configured():
        return {}

    results = {}
    for m in fetch_matches():
        home = m.get("homeTeam", {}).get("name", "")
        away = m.get("awayTeam", {}).get("name", "")
        if not home or not away:
            continue

        score  = m.get("score", {})
        full   = score.get("fullTime", {})
        status = m.get("status", "SCHEDULED")

        results[(home, away)] = {
            "home_score": full.get("home"),
            "away_score": full.get("away"),
            "status":     status,
            "minute":     m.get("minute"),
            "matchday":   m.get("matchday", 0),
            "date":       m.get("utcDate", "")[:10],
        }
    return results


def get_group_standings(group: str) -> list[dict] | None:
    """
    Return current group standings from the API, or None if unavailable.
    Each entry: {"team": str, "pts": int, "played": int, "gd": int, "gf": int}
    """
    if not is_tournament_live() or not is_api_configured():
        return None

    key = _api_key()
    headers = {"X-Auth-Token": key}
    url     = f"{API_BASE}/competitions/{WC2026_ID}/standings"
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        for standing in data.get("standings", []):
            if standing.get("group") == f"Group {group}":
                return [
                    {
                        "team":   row["team"]["name"],
                        "pts":    row["points"],
                        "played": row["playedGames"],
                        "gd":     row["goalDifference"],
                        "gf":     row["goalsFor"],
                    }
                    for row in standing["table"]
                ]
    except Exception:
        pass
    return None


# ── Status badge helper (used by Streamlit UI) ────────────────────────────────

def status_badge(status: str, minute: int | None = None) -> str:
    """Return an HTML badge string for a match status."""
    if status == "IN_PLAY":
        m = f" {minute}'" if minute else ""
        return f"<span style='background:#dc2626;color:#fff;padding:2px 7px;border-radius:4px;font-size:0.75em'>🔴 LIVE{m}</span>"
    if status == "PAUSED":
        return "<span style='background:#d97706;color:#fff;padding:2px 7px;border-radius:4px;font-size:0.75em'>⏸ HT</span>"
    if status == "FINISHED":
        return "<span style='background:#166534;color:#fff;padding:2px 7px;border-radius:4px;font-size:0.75em'>✅ FT</span>"
    if status == "EXTRA_TIME":
        return "<span style='background:#7c3aed;color:#fff;padding:2px 7px;border-radius:4px;font-size:0.75em'>⚡ ET</span>"
    if status == "PENALTY_SHOOTOUT":
        return "<span style='background:#7c3aed;color:#fff;padding:2px 7px;border-radius:4px;font-size:0.75em'>🎯 PSO</span>"
    return ""
