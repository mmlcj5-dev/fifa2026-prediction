"""
WC 2026 match schedule — group stage dates, kick-off times (CT / CDT, UTC-5), and venues.
Source: FIFA Official Match Schedule v17 — 10 April 2026.
All times converted to US Central Daylight Time (CDT = ET − 1 hr).
"""

# ── Venues ────────────────────────────────────────────────────────────────────
VENUES = {
    # USA
    "MetLife":    {"name": "MetLife Stadium",          "city": "New York / NJ",      "cap": 82_500},
    "ATT":        {"name": "AT&T Stadium",             "city": "Dallas, TX",         "cap": 80_000},
    "SoFi":       {"name": "SoFi Stadium",             "city": "Los Angeles, CA",    "cap": 77_500},
    "Levis":      {"name": "Levi's Stadium",           "city": "San Francisco, CA",  "cap": 68_500},
    "LincolnFin": {"name": "Lincoln Financial Field",  "city": "Philadelphia, PA",   "cap": 69_800},
    "HardRock":   {"name": "Hard Rock Stadium",        "city": "Miami, FL",          "cap": 65_000},
    "Arrowhead":  {"name": "Arrowhead Stadium",        "city": "Kansas City, MO",    "cap": 76_416},
    "Gillette":   {"name": "Gillette Stadium",         "city": "Boston, MA",         "cap": 65_878},
    "Lumen":      {"name": "Lumen Field",              "city": "Seattle, WA",        "cap": 69_000},
    "NRG":        {"name": "NRG Stadium",              "city": "Houston, TX",        "cap": 72_220},
    "MercedesBenz":{"name": "Mercedes-Benz Stadium",  "city": "Atlanta, GA",        "cap": 71_000},
    # Canada
    "BMO":        {"name": "BMO Field",                "city": "Toronto, ON",        "cap": 30_000},
    "BCPlace":    {"name": "BC Place",                 "city": "Vancouver, BC",      "cap": 54_500},
    # Mexico
    "Azteca":     {"name": "Estadio Azteca",           "city": "Mexico City, MX",    "cap": 87_523},
    "BBVA":       {"name": "Estadio BBVA",             "city": "Monterrey, MX",      "cap": 53_500},
    "Akron":      {"name": "Estadio Akron",            "city": "Guadalajara, MX",    "cap": 49_850},
}

# ── Group-stage schedule ───────────────────────────────────────────────────────
# Columns: (date, time_CT, home, away, venue_key, group, matchday)
# All times in Central Daylight Time (CDT = ET − 1 hr).
# Source: FIFA Official Match Schedule v17, April 10 2026.
GROUP_MATCHES = [

    # ── GROUP A — Mexico · South Africa · Korea Republic · Czechia ────────────
    ("2026-06-11", "14:00", "Mexico",          "South Africa",   "Azteca",     "A", 1),
    ("2026-06-11", "21:00", "Korea Republic",  "Czechia",        "Akron",      "A", 1),
    ("2026-06-20", "11:00", "Czechia",         "South Africa",   "BBVA",       "A", 2),
    ("2026-06-21", "20:00", "Mexico",          "Korea Republic", "Akron",      "A", 2),
    ("2026-06-29", "14:00", "South Africa",    "Korea Republic", "BBVA",       "A", 3),
    ("2026-06-29", "14:00", "Czechia",         "Mexico",         "Azteca",     "A", 3),

    # ── GROUP B — Canada · Bosnia & Herzegovina · Qatar · Switzerland ─────────
    ("2026-06-12", "14:00", "Canada",          "Bosnia & Herzegovina", "BMO",  "B", 1),
    ("2026-06-13", "14:00", "Qatar",           "Switzerland",    "BCPlace",    "B", 1),
    ("2026-06-21", "14:00", "Switzerland",     "Bosnia & Herzegovina","Lumen", "B", 2),
    ("2026-06-21", "17:00", "Canada",          "Qatar",          "BMO",        "B", 2),
    ("2026-06-29", "14:00", "Bosnia & Herzegovina","Qatar",       "BCPlace",   "B", 3),
    ("2026-06-29", "14:00", "Switzerland",     "Canada",         "Lumen",      "B", 3),

    # ── GROUP C — Brazil · Morocco · Haiti · Scotland ─────────────────────────
    ("2026-06-13", "20:00", "Haiti",           "Scotland",       "Gillette",   "C", 1),
    ("2026-06-13", "17:00", "Brazil",          "Morocco",        "MetLife",    "C", 1),
    ("2026-06-22", "19:30", "Brazil",          "Haiti",          "MetLife",    "C", 2),
    ("2026-06-22", "17:00", "Scotland",        "Morocco",        "Gillette",   "C", 2),
    ("2026-06-28", "12:00", "Morocco",         "Haiti",          "HardRock",   "C", 3),
    ("2026-06-28", "12:00", "Scotland",        "Brazil",         "HardRock",   "C", 3),

    # ── GROUP D — USA · Paraguay · Australia · Türkiye ────────────────────────
    ("2026-06-12", "20:00", "USA",             "Paraguay",       "SoFi",       "D", 1),
    ("2026-06-13", "23:00", "Australia",       "Türkiye",        "BCPlace",    "D", 1),
    ("2026-06-22", "14:00", "USA",             "Australia",      "Levis",      "D", 2),
    ("2026-06-22", "21:00", "Türkiye",         "Paraguay",       "Levis",      "D", 2),
    ("2026-06-29", "18:00", "Paraguay",        "Australia",      "SoFi",       "D", 3),
    ("2026-06-29", "18:00", "Türkiye",         "USA",            "SoFi",       "D", 3),

    # ── GROUP E — Germany · Curaçao · Côte d'Ivoire · Ecuador ────────────────
    ("2026-06-14", "18:00", "Côte d'Ivoire",   "Ecuador",        "ATT",        "E", 1),
    ("2026-06-14", "12:00", "Germany",         "Curaçao",        "LincolnFin", "E", 1),
    ("2026-06-23", "15:00", "Germany",         "Côte d'Ivoire",  "ATT",        "E", 2),
    ("2026-06-23", "19:00", "Ecuador",         "Curaçao",        "LincolnFin", "E", 2),
    ("2026-06-28", "17:00", "Ecuador",         "Germany",        "ATT",        "E", 3),
    ("2026-06-28", "17:00", "Curaçao",         "Côte d'Ivoire",  "LincolnFin", "E", 3),

    # ── GROUP F — Netherlands · Japan · Sweden · Tunisia ─────────────────────
    ("2026-06-15", "15:00", "Netherlands",     "Japan",          "BBVA",       "F", 1),
    ("2026-06-15", "21:00", "Sweden",          "Tunisia",        "Lumen",      "F", 1),
    ("2026-06-23", "12:00", "Netherlands",     "Sweden",         "Arrowhead",  "F", 2),
    ("2026-06-23", "23:00", "Tunisia",         "Japan",          "BBVA",       "F", 2),
    ("2026-06-30", "15:00", "Tunisia",         "Netherlands",    "Arrowhead",  "F", 3),
    ("2026-06-30", "15:00", "Japan",           "Sweden",         "Lumen",      "F", 3),

    # ── GROUP G — Belgium · Egypt · IR Iran · New Zealand ────────────────────
    ("2026-06-17", "20:00", "IR Iran",         "New Zealand",    "BCPlace",    "G", 1),
    ("2026-06-17", "14:00", "Belgium",         "Egypt",          "BCPlace",    "G", 1),
    ("2026-06-24", "14:00", "New Zealand",     "Egypt",          "Levis",      "G", 2),
    ("2026-06-25", "12:00", "Belgium",         "IR Iran",        "Levis",      "G", 2),
    ("2026-06-30", "21:00", "New Zealand",     "Belgium",        "SoFi",       "G", 3),
    ("2026-06-30", "21:00", "Egypt",           "IR Iran",        "SoFi",       "G", 3),

    # ── GROUP H — Spain · Cabo Verde · Saudi Arabia · Uruguay ────────────────
    ("2026-06-16", "17:00", "Saudi Arabia",    "Uruguay",        "HardRock",   "H", 1),
    ("2026-06-16", "11:00", "Spain",           "Cabo Verde",     "MercedesBenz","H", 1),
    ("2026-06-24", "17:00", "Uruguay",         "Cabo Verde",     "HardRock",   "H", 2),
    ("2026-06-24", "11:00", "Spain",           "Saudi Arabia",   "MercedesBenz","H", 2),
    ("2026-07-01", "22:00", "Cabo Verde",      "Saudi Arabia",   "HardRock",   "H", 3),
    ("2026-07-01", "22:00", "Uruguay",         "Spain",          "MercedesBenz","H", 3),

    # ── GROUP I — France · Senegal · Iraq · Norway ────────────────────────────
    ("2026-06-17", "14:00", "France",          "Senegal",        "MetLife",    "I", 1),
    ("2026-06-17", "17:00", "Iraq",            "Norway",         "BMO",        "I", 1),
    ("2026-06-25", "16:00", "France",          "Iraq",           "MetLife",    "I", 2),
    ("2026-06-25", "19:00", "Norway",          "Senegal",        "BMO",        "I", 2),
    ("2026-07-01", "14:00", "Senegal",         "Iraq",           "MetLife",    "I", 3),
    ("2026-07-01", "14:00", "Norway",          "France",         "BMO",        "I", 3),

    # ── GROUP J — Argentina · Algeria · Austria · Jordan ─────────────────────
    ("2026-06-18", "20:00", "Argentina",       "Algeria",        "Arrowhead",  "J", 1),
    ("2026-06-18", "23:00", "Austria",         "Jordan",         "SoFi",       "J", 1),
    ("2026-06-25", "12:00", "Argentina",       "Austria",        "Arrowhead",  "J", 2),
    ("2026-06-25", "20:00", "Jordan",          "Algeria",        "SoFi",       "J", 2),
    ("2026-07-01", "21:00", "Algeria",         "Austria",        "Arrowhead",  "J", 3),
    ("2026-07-01", "21:00", "Jordan",          "Argentina",      "SoFi",       "J", 3),

    # ── GROUP K — Portugal · Congo DR · Uzbekistan · Colombia ────────────────
    ("2026-06-20", "12:00", "Portugal",        "Congo DR",       "NRG",        "K", 1),
    ("2026-06-20", "21:00", "Uzbekistan",      "Colombia",       "Azteca",     "K", 1),
    ("2026-06-26", "12:00", "Colombia",        "Congo DR",       "NRG",        "K", 2),
    ("2026-06-26", "21:00", "Portugal",        "Uzbekistan",     "MercedesBenz","K", 2),
    ("2026-07-02", "18:30", "Congo DR",        "Uzbekistan",     "Akron",      "K", 3),
    ("2026-07-02", "18:30", "Colombia",        "Portugal",       "NRG",        "K", 3),

    # ── GROUP L — England · Croatia · Ghana · Panama ──────────────────────────
    ("2026-06-19", "18:00", "Ghana",           "Panama",         "BMO",        "L", 1),
    ("2026-06-19", "15:00", "England",         "Croatia",        "MercedesBenz","L", 1),
    ("2026-06-26", "15:00", "England",         "Ghana",          "Gillette",   "L", 2),
    ("2026-06-26", "18:00", "Panama",          "Croatia",        "Gillette",   "L", 2),
    ("2026-07-02", "14:00", "Panama",          "England",        "MercedesBenz","L", 3),
    ("2026-07-02", "14:00", "Croatia",         "Ghana",          "BMO",        "L", 3),
]

# ── Knockout schedule ─────────────────────────────────────────────────────────
KNOCKOUT_ROUNDS = [
    {"round": "Round of 32",    "dates": "Jul 4–8, 2026",   "matches": 16},
    {"round": "Round of 16",    "dates": "Jul 10–13, 2026", "matches": 8},
    {"round": "Quarter-Finals", "dates": "Jul 15–16, 2026", "matches": 4},
    {"round": "Semi-Finals",    "dates": "Jul 18–19, 2026", "matches": 2},
    {"round": "Bronze Final",   "dates": "Jul 18, 2026",    "matches": 1,
     "venue": "Hard Rock Stadium, Miami FL"},
    {"round": "Final",          "dates": "Jul 19, 2026",    "matches": 1,
     "venue": "MetLife Stadium, New York / NJ"},
]
