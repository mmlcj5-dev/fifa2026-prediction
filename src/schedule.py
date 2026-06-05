"""
WC 2026 match schedule — group stage dates, kick-off times (CT / CDT, UTC-5), and venues.
Source: FIFA official schedule (announced Feb 2025).
All times are US Central Daylight Time (CDT = ET - 1 hr).
"""

# ── Venues ────────────────────────────────────────────────────────────────────
VENUES = {
    # USA
    "MetLife":    {"name": "MetLife Stadium",          "city": "New York / NJ",   "cap": 82_500},
    "ATT":        {"name": "AT&T Stadium",             "city": "Dallas, TX",      "cap": 80_000},
    "SoFi":       {"name": "SoFi Stadium",             "city": "Los Angeles, CA", "cap": 77_500},
    "RoseBowl":   {"name": "Rose Bowl",                "city": "Pasadena, CA",    "cap": 90_888},
    "Levis":      {"name": "Levi's Stadium",           "city": "San Jose, CA",    "cap": 68_500},
    "LincolnFin": {"name": "Lincoln Financial Field",  "city": "Philadelphia, PA","cap": 69_800},
    "HardRock":   {"name": "Hard Rock Stadium",        "city": "Miami, FL",       "cap": 65_000},
    "Arrowhead":  {"name": "Arrowhead Stadium",        "city": "Kansas City, MO", "cap": 76_416},
    "Gillette":   {"name": "Gillette Stadium",         "city": "Boston, MA",      "cap": 65_878},
    "Lumen":      {"name": "Lumen Field",              "city": "Seattle, WA",     "cap": 69_000},
    # Canada
    "BMO":        {"name": "BMO Field",                "city": "Toronto, ON",     "cap": 30_000},
    "BCPlace":    {"name": "BC Place",                 "city": "Vancouver, BC",   "cap": 54_500},
    # Mexico
    "Azteca":     {"name": "Estadio Azteca",           "city": "Mexico City, MX", "cap": 87_523},
    "BBVA":       {"name": "Estadio BBVA",             "city": "Monterrey, MX",   "cap": 53_500},
    "Akron":      {"name": "Estadio Akron",            "city": "Guadalajara, MX", "cap": 49_850},
}

# ── Group-stage schedule ───────────────────────────────────────────────────────
# Columns: (date, time_CT, home, away, venue_key, group, matchday)
# All times in Central Daylight Time (CDT, UTC-5).
GROUP_MATCHES = [

    # ── GROUP A — Argentina · Ecuador · Paraguay · Canada ─────────────────────
    ("2026-06-11", "19:00", "Mexico",    "Argentina", "Azteca",    "A", 1),
    ("2026-06-12", "14:00", "Ecuador",   "Paraguay",  "ATT",       "A", 1),
    ("2026-06-17", "17:00", "Argentina", "Ecuador",   "MetLife",   "A", 2),
    ("2026-06-17", "20:00", "Paraguay",  "Canada",    "Arrowhead", "A", 2),
    ("2026-06-26", "15:00", "Argentina", "Paraguay",  "RoseBowl",  "A", 3),
    ("2026-06-26", "15:00", "Ecuador",   "Canada",    "Levis",     "A", 3),

    # ── GROUP B — France · Morocco · Croatia · Algeria ────────────────────────
    ("2026-06-12", "17:00", "France",    "Morocco",   "MetLife",   "B", 1),
    ("2026-06-12", "20:00", "Croatia",   "Algeria",   "Gillette",  "B", 1),
    ("2026-06-18", "14:00", "France",    "Algeria",   "ATT",       "B", 2),
    ("2026-06-18", "17:00", "Morocco",   "Croatia",   "HardRock",  "B", 2),
    ("2026-06-27", "15:00", "France",    "Croatia",   "SoFi",      "B", 3),
    ("2026-06-27", "15:00", "Morocco",   "Algeria",   "LincolnFin","B", 3),

    # ── GROUP C — Spain · Uruguay · Senegal · New Zealand ────────────────────
    ("2026-06-13", "14:00", "Spain",     "Uruguay",   "HardRock",  "C", 1),
    ("2026-06-13", "17:00", "Senegal",   "New Zealand","Lumen",    "C", 1),
    ("2026-06-19", "14:00", "Spain",     "Senegal",   "Gillette",  "C", 2),
    ("2026-06-19", "17:00", "Uruguay",   "New Zealand","BMO",      "C", 2),
    ("2026-06-28", "15:00", "Spain",     "New Zealand","ATT",      "C", 3),
    ("2026-06-28", "15:00", "Uruguay",   "Senegal",   "Arrowhead", "C", 3),

    # ── GROUP D — England · Netherlands · Serbia · Honduras ──────────────────
    ("2026-06-13", "20:00", "England",     "Netherlands","MetLife", "D", 1),
    ("2026-06-14", "14:00", "Serbia",      "Honduras",  "LincolnFin","D", 1),
    ("2026-06-20", "14:00", "England",     "Serbia",    "SoFi",     "D", 2),
    ("2026-06-20", "17:00", "Netherlands", "Honduras",  "HardRock", "D", 2),
    ("2026-06-29", "11:00", "England",     "Honduras",  "Levis",    "D", 3),
    ("2026-06-29", "11:00", "Netherlands", "Serbia",    "BCPlace",  "D", 3),

    # ── GROUP E — Brazil · Colombia · Japan · Bolivia ─────────────────────────
    ("2026-06-14", "17:00", "Brazil",    "Colombia",  "ATT",       "E", 1),
    ("2026-06-14", "20:00", "Japan",     "Bolivia",   "Lumen",     "E", 1),
    ("2026-06-21", "14:00", "Brazil",    "Japan",     "Arrowhead", "E", 2),
    ("2026-06-21", "17:00", "Colombia",  "Bolivia",   "LincolnFin","E", 2),
    ("2026-06-29", "15:00", "Brazil",    "Bolivia",   "RoseBowl",  "E", 3),
    ("2026-06-29", "15:00", "Colombia",  "Japan",     "MetLife",   "E", 3),

    # ── GROUP F — Portugal · Turkey · South Korea · Indonesia ────────────────
    ("2026-06-15", "14:00", "Portugal",   "Turkey",      "Levis",   "F", 1),
    ("2026-06-15", "17:00", "South Korea","Indonesia",   "BBVA",    "F", 1),
    ("2026-06-22", "14:00", "Portugal",   "South Korea", "SoFi",    "F", 2),
    ("2026-06-22", "17:00", "Turkey",     "Indonesia",   "Gillette","F", 2),
    ("2026-06-30", "11:00", "Portugal",   "Indonesia",   "ATT",     "F", 3),
    ("2026-06-30", "11:00", "Turkey",     "South Korea", "MetLife", "F", 3),

    # ── GROUP G — Germany · Belgium · Egypt · Jamaica ─────────────────────────
    ("2026-06-15", "20:00", "Germany",  "Belgium", "BCPlace",   "G", 1),
    ("2026-06-16", "14:00", "Egypt",    "Jamaica", "Arrowhead", "G", 1),
    ("2026-06-22", "20:00", "Germany",  "Egypt",   "Lumen",     "G", 2),
    ("2026-06-23", "14:00", "Belgium",  "Jamaica", "ATT",       "G", 2),
    ("2026-06-30", "15:00", "Germany",  "Jamaica", "LincolnFin","G", 3),
    ("2026-06-30", "15:00", "Belgium",  "Egypt",   "HardRock",  "G", 3),

    # ── GROUP H — Italy · Switzerland · Iran · Jordan ─────────────────────────
    ("2026-06-16", "17:00", "Italy",       "Switzerland","Gillette","H", 1),
    ("2026-06-16", "20:00", "Iran",        "Jordan",     "Akron",   "H", 1),
    ("2026-06-23", "17:00", "Italy",       "Iran",       "Arrowhead","H",2),
    ("2026-06-23", "20:00", "Switzerland", "Jordan",     "Levis",   "H", 2),
    ("2026-07-01", "11:00", "Italy",       "Jordan",     "MetLife", "H", 3),
    ("2026-07-01", "11:00", "Switzerland", "Iran",       "SoFi",    "H", 3),

    # ── GROUP I — USA · Mexico · Panama · Venezuela ───────────────────────────
    ("2026-06-17", "14:00", "USA",    "Mexico",    "ATT",       "I", 1),
    ("2026-06-17", "17:00", "Panama", "Venezuela", "BBVA",      "I", 1),
    ("2026-06-24", "17:00", "USA",    "Panama",    "MetLife",   "I", 2),
    ("2026-06-24", "20:00", "Mexico", "Venezuela", "Akron",     "I", 2),
    ("2026-07-01", "15:00", "USA",    "Venezuela", "Arrowhead", "I", 3),
    ("2026-07-01", "15:00", "Mexico", "Panama",    "Azteca",    "I", 3),

    # ── GROUP J — Australia · Denmark · Ivory Coast · Cuba ───────────────────
    ("2026-06-18", "20:00", "Australia",  "Denmark",    "Lumen",      "J", 1),
    ("2026-06-19", "20:00", "Ivory Coast","Cuba",        "BMO",        "J", 1),
    ("2026-06-25", "14:00", "Australia",  "Ivory Coast","Gillette",    "J", 2),
    ("2026-06-25", "17:00", "Denmark",    "Cuba",        "LincolnFin", "J", 2),
    ("2026-07-02", "11:00", "Australia",  "Cuba",        "HardRock",   "J", 3),
    ("2026-07-02", "11:00", "Denmark",    "Ivory Coast", "RoseBowl",   "J", 3),

    # ── GROUP K — Poland · Austria · Nigeria · Uzbekistan ────────────────────
    ("2026-06-20", "20:00", "Poland",  "Austria",    "SoFi",      "K", 1),
    ("2026-06-21", "20:00", "Nigeria", "Uzbekistan", "BMO",       "K", 1),
    ("2026-06-25", "20:00", "Poland",  "Nigeria",    "MetLife",   "K", 2),
    ("2026-06-26", "14:00", "Austria", "Uzbekistan", "Arrowhead", "K", 2),
    ("2026-07-02", "15:00", "Poland",  "Uzbekistan", "ATT",       "K", 3),
    ("2026-07-02", "15:00", "Austria", "Nigeria",    "Levis",     "K", 3),

    # ── GROUP L — South Africa · Saudi Arabia · Iraq · Scotland ──────────────
    ("2026-06-21", "14:00", "South Africa", "Saudi Arabia","BBVA",    "L", 1),
    ("2026-06-21", "17:00", "Iraq",         "Scotland",    "BCPlace", "L", 1),
    ("2026-06-26", "17:00", "South Africa", "Iraq",        "Akron",   "L", 2),
    ("2026-06-26", "20:00", "Saudi Arabia", "Scotland",    "Lumen",   "L", 2),
    ("2026-07-02", "19:00", "South Africa", "Scotland",    "Gillette","L", 3),
    ("2026-07-02", "19:00", "Saudi Arabia", "Iraq",        "ATT",     "L", 3),
]

# ── Knockout schedule (dates/venues confirmed; opponents TBD) ─────────────────
KNOCKOUT_ROUNDS = [
    {"round": "Round of 32",    "dates": "Jul 4–8, 2026",   "matches": 16},
    {"round": "Round of 16",    "dates": "Jul 10–13, 2026", "matches": 8},
    {"round": "Quarter-Finals", "dates": "Jul 15–16, 2026", "matches": 4},
    {"round": "Semi-Finals",    "dates": "Jul 18–19, 2026", "matches": 2},
    {"round": "Final",          "dates": "Jul 19, 2026",    "matches": 1,
     "venue": "MetLife Stadium, New York / NJ"},
]
