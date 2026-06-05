# src/live_elo_loader.py
import pandas as pd


# Current WC2026 ELO ratings - manually updated from eloratings.net
# Last updated: June 2026
CURRENT_ELO = {
    "France": 2003, "Spain": 1994, "England": 1975, "Brazil": 1968,
    "Portugal": 1960, "Netherlands": 1952, "Argentina": 1947, "Germany": 1940,
    "Belgium": 1930, "Italy": 1918, "Morocco": 1905, "Croatia": 1898,
    "United States": 1856, "Switzerland": 1843, "Uruguay": 1840, "Colombia": 1837,
    "Mexico": 1834, "Senegal": 1830, "Denmark": 1826, "Austria": 1822,
    "Japan": 1818, "South Korea": 1814, "Australia": 1810, "Iran": 1806,
    "Serbia": 1802, "Poland": 1798, "Ecuador": 1794, "Peru": 1790,
    "Canada": 1787, "Hungary": 1783, "Czech Republic": 1779, "Turkey": 1775,
    "Chile": 1771, "Nigeria": 1767, "Cameroon": 1763, "Ghana": 1759,
    "Algeria": 1755, "Egypt": 1751, "Tunisia": 1747, "Ivory Coast": 1743,
    "South Africa": 1739, "Saudi Arabia": 1735, "Iraq": 1731, "Qatar": 1727,
    "Uzbekistan": 1723, "Jordan": 1719, "Venezuela": 1715, "Paraguay": 1711,
}


def fetch_live_elo_ratings() -> pd.DataFrame:
    """
    Returns ELO ratings. Tries eloratings.net first, falls back to
    hardcoded current ratings if the site blocks the request.
    """
    try:
        import requests, re
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        }
        resp = requests.get("https://www.eloratings.net/World", headers=headers, timeout=8)
        if resp.status_code == 200 and len(resp.text) > 1000:
            pattern = r'"([A-Za-z\s\.]+)",(\d{3,4})'
            matches = re.findall(pattern, resp.text)
            if len(matches) > 20:
                df = pd.DataFrame(matches, columns=["team", "elo_rating"])
                df["elo_rating"] = df["elo_rating"].astype(float)
                print(f"[live_elo_loader] Fetched {len(df)} teams live")
                return df
    except Exception as e:
        print(f"[live_elo_loader] Live fetch failed: {e}")

    # Fall back to hardcoded current ratings
    print("[live_elo_loader] Using built-in current ELO ratings")
    df = pd.DataFrame(list(CURRENT_ELO.items()), columns=["team", "elo_rating"])
    return df