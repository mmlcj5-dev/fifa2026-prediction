# =============================================================================
# src/features.py
# -----------------------------------------------------------------------------
# Feature engineering for the FIFA 2026 match prediction model.
#
# Functions:
#   build_features()                  - Base ELO + goal features (stateless)
#   build_rolling_features()          - Adds form + H2H via rolling window
#   get_match_features_for_prediction() - Single-row feature vector at inference
#
# Typical usage:
#   base_df   = build_features(matches_df, elo_df)
#   full_df   = build_rolling_features(base_df)
#   full_df.to_csv("data/processed/features.csv", index=False)
# =============================================================================

import pandas as pd


# =============================================================================
# SECTION 1 — Column Normalisation Helpers
# =============================================================================

def _normalize_match_columns(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names across different source datasets.
    Handles variations like 'HomeTeam' / 'home' / 'Home' → 'home_team', etc.
    Also ensures 'date' is parsed as datetime and 'result' (1=home win) exists.
    """
    rename_map = {
        'Date': 'date', 'date': 'date',
        'Home': 'home_team', 'home': 'home_team', 'HomeTeam': 'home_team', 'home_team': 'home_team',
        'Away': 'away_team', 'away': 'away_team', 'AwayTeam': 'away_team', 'away_team': 'away_team',
        'HG': 'home_goals', 'HomeGoals': 'home_goals', 'home_goals': 'home_goals',
        'home_score': 'home_goals', 'HomeScore': 'home_goals',
        'AG': 'away_goals', 'AwayGoals': 'away_goals', 'away_goals': 'away_goals',
        'away_score': 'away_goals', 'AwayScore': 'away_goals',
        'Result': 'result', 'result': 'result',
    }
    df = matches.copy()
    rename_dict = {old: new for old, new in rename_map.items() if old in df.columns}
    if rename_dict:
        df = df.rename(columns=rename_dict)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if 'home_goals' in df.columns and 'away_goals' in df.columns:
        if 'result' not in df.columns:
            df['result'] = (df['home_goals'] > df['away_goals']).astype(int)
    return df


def _normalize_elo_columns(elo: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise ELO DataFrame column names.
    Handles variations like 'Team' / 'country' / 'country_name' → 'team', etc.
    """
    rename_map = {
        'team': 'team', 'Team': 'team', 'country': 'team', 'country_name': 'team',
        'elo_rating': 'elo_rating', 'ELO': 'elo_rating',
        'rating': 'elo_rating', 'rating_elo': 'elo_rating',
    }
    df = elo.copy()
    rename_dict = {old: new for old, new in rename_map.items() if old in df.columns}
    if rename_dict:
        df = df.rename(columns=rename_dict)
    return df


# =============================================================================
# SECTION 2 — Base Feature Builder (stateless, no rolling window)
# =============================================================================

def build_features(matches: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    """
    Build the base feature set by joining ELO ratings onto match data.

    Adds: home_elo, away_elo, elo_diff, goal_diff, total_goals, home_advantage.
    Missing ELO values are filled with the dataset median (fallback: 1500).

    Parameters
    ----------
    matches : DataFrame with match results (home/away teams + scores)
    elo     : DataFrame with team ELO ratings

    Returns
    -------
    DataFrame with original match columns plus engineered base features.
    """
    matches = _normalize_match_columns(matches)
    elo = _normalize_elo_columns(elo)

    if 'home_team' not in matches.columns or 'away_team' not in matches.columns:
        raise ValueError('Matches dataframe must contain home_team and away_team columns.')
    if 'home_goals' not in matches.columns or 'away_goals' not in matches.columns:
        raise ValueError('Matches dataframe must contain home_goals and away_goals columns.')
    if 'team' not in elo.columns or 'elo_rating' not in elo.columns:
        elo = pd.DataFrame(columns=['team', 'elo_rating'])

    home_elo = elo[['team', 'elo_rating']].rename(columns={'team': 'home_team', 'elo_rating': 'home_elo'})
    away_elo = elo[['team', 'elo_rating']].rename(columns={'team': 'away_team', 'elo_rating': 'away_elo'})

    df = matches.merge(home_elo, on='home_team', how='left')
    df = df.merge(away_elo, on='away_team', how='left')

    for col in ['home_elo', 'away_elo']:
        median_value = df[col].median()
        if pd.isna(median_value):
            median_value = 1500
        df[col] = df[col].fillna(median_value)

    df['elo_diff'] = df['home_elo'] - df['away_elo']
    df['goal_diff'] = df['home_goals'] - df['away_goals']
    df['total_goals'] = df['home_goals'] + df['away_goals']
    df['home_advantage'] = 1

    return df


# =============================================================================
# SECTION 3 — Rolling Stats Helpers (used internally)
# =============================================================================

def _team_rolling_stats(df: pd.DataFrame, team: str, before_date, n: int = 5) -> dict:
    """
    Compute rolling performance stats for a team using their last N matches
    strictly before `before_date` (prevents data leakage during training).

    Returns a dict with: win_rate, draw_rate, goals_scored_avg,
    goals_conceded_avg, points_per_game, matches_played.
    Falls back to neutral defaults if no history exists.
    """
    past = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (df['date'] < before_date)
    ].sort_values('date').tail(n)

    if past.empty:
        return {
            'win_rate': 0.5,
            'draw_rate': 0.2,
            'goals_scored_avg': 1.2,
            'goals_conceded_avg': 1.2,
            'points_per_game': 1.0,
            'matches_played': 0,
        }

    wins = draws = goals_scored = goals_conceded = 0
    for _, row in past.iterrows():
        gf = row['home_goals'] if row['home_team'] == team else row['away_goals']
        ga = row['away_goals'] if row['home_team'] == team else row['home_goals']
        goals_scored += gf
        goals_conceded += ga
        if gf > ga:
            wins += 1
        elif gf == ga:
            draws += 1

    n_actual = len(past)
    return {
        'win_rate': wins / n_actual,
        'draw_rate': draws / n_actual,
        'goals_scored_avg': goals_scored / n_actual,
        'goals_conceded_avg': goals_conceded / n_actual,
        'points_per_game': (wins * 3 + draws) / n_actual,
        'matches_played': n_actual,
    }


def _h2h_stats(df: pd.DataFrame, home_team: str, away_team: str, before_date, n: int = 5) -> dict:
    """
    Compute head-to-head record between two teams using their last N meetings
    strictly before `before_date`.

    Returns a dict with: h2h_home_wins, h2h_draws, h2h_away_wins,
    h2h_home_goal_diff (avg), h2h_games.
    All rates are per-game (0.0–1.0). Falls back to zeros if no history.
    """
    h2h = df[
        (
            ((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
            ((df['home_team'] == away_team) & (df['away_team'] == home_team))
        ) &
        (df['date'] < before_date)
    ].sort_values('date').tail(n)

    if h2h.empty:
        return {
            'h2h_home_wins': 0,
            'h2h_draws': 0,
            'h2h_away_wins': 0,
            'h2h_home_goal_diff': 0.0,
            'h2h_games': 0,
        }

    home_wins = draws = away_wins = goal_diff_sum = 0
    for _, row in h2h.iterrows():
        gf = row['home_goals'] if row['home_team'] == home_team else row['away_goals']
        ga = row['away_goals'] if row['home_team'] == home_team else row['home_goals']
        goal_diff_sum += gf - ga
        if gf > ga:
            home_wins += 1
        elif gf == ga:
            draws += 1
        else:
            away_wins += 1

    n_actual = len(h2h)
    return {
        'h2h_home_wins': home_wins / n_actual,
        'h2h_draws': draws / n_actual,
        'h2h_away_wins': away_wins / n_actual,
        'h2h_home_goal_diff': goal_diff_sum / n_actual,
        'h2h_games': n_actual,
    }


# =============================================================================
# SECTION 4 — Rolling Feature Builder (for training data generation)
# =============================================================================

def build_rolling_features(df: pd.DataFrame, n_form: int = 5, n_h2h: int = 5) -> pd.DataFrame:
    """
    Enrich a features DataFrame (output of build_features) with rolling
    form statistics and head-to-head records.

    Each row's stats are computed using only matches that occurred strictly
    BEFORE that row's date — this prevents any data leakage into training.

    Adds columns: home_win_rate, home_goals_scored_avg, home_goals_conceded_avg,
    home_points_per_game, home_matches_played (and away_ equivalents),
    plus h2h_home_wins, h2h_draws, h2h_away_wins, h2h_home_goal_diff, h2h_games.

    Parameters
    ----------
    df      : Output of build_features()
    n_form  : Number of recent matches to use for form calculation (default 5)
    n_h2h   : Number of recent H2H meetings to use (default 5)

    Returns
    -------
    Enriched DataFrame ready for model training.

    Note: This iterates over all rows and can take several minutes on 47k matches.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)

    form_home, form_away, h2h_records = [], [], []
    total = len(df)

    print(f"Computing rolling features for {total:,} matches...")
    for i, row in df.iterrows():
        if i % 5000 == 0:
            print(f"  Progress: {i:,} / {total:,} ({100*i//total}%)")

        home_stats = _team_rolling_stats(df, row['home_team'], row['date'], n_form)
        away_stats = _team_rolling_stats(df, row['away_team'], row['date'], n_form)
        h2h = _h2h_stats(df, row['home_team'], row['away_team'], row['date'], n_h2h)

        form_home.append({f'home_{k}': v for k, v in home_stats.items()})
        form_away.append({f'away_{k}': v for k, v in away_stats.items()})
        h2h_records.append(h2h)

    df = pd.concat([
        df,
        pd.DataFrame(form_home),
        pd.DataFrame(form_away),
        pd.DataFrame(h2h_records),
    ], axis=1)

    print(f"Done. Final shape: {df.shape}")
    return df


# =============================================================================
# SECTION 5 — Single Match Feature Vector (for Streamlit inference)
# =============================================================================

def get_match_features_for_prediction(
    home_team: str,
    away_team: str,
    matches_df: pd.DataFrame,
    elo_df: pd.DataFrame,
    is_home_host: int = 0,
    is_away_host: int = 0,
    is_neutral: int = 1,
    is_friendly: int = 0,
) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame for use at prediction time in the
    Streamlit app. Uses all available historical matches as context (no date
    cutoff needed since we are predicting a future match).

    Parameters
    ----------
    home_team    : Name of the home team (must match elo_df 'team' column)
    away_team    : Name of the away team
    matches_df   : Full historical matches DataFrame
    elo_df       : Current ELO ratings DataFrame
    is_home_host : 1 if home team is a WC2026 host nation (USA/CAN/MEX)
    is_away_host : 1 if away team is a WC2026 host nation
    is_neutral   : 1 if match is at a neutral venue (default for WC)
    is_friendly  : 1 if match is a friendly (0 for WC matches)

    Returns
    -------
    Single-row DataFrame with all features expected by the trained models.
    """
    matches_df = _normalize_match_columns(matches_df)
    matches_df['date'] = pd.to_datetime(matches_df['date'], errors='coerce')

    # Use current timestamp so all history is available
    today = pd.Timestamp.now()

    # ELO lookup with fallback
    home_elo = (
        float(elo_df[elo_df['team'] == home_team]['elo_rating'].iloc[0])
        if home_team in elo_df['team'].values else 1500
    )
    away_elo = (
        float(elo_df[elo_df['team'] == away_team]['elo_rating'].iloc[0])
        if away_team in elo_df['team'].values else 1500
    )

    home_form = _team_rolling_stats(matches_df, home_team, today, n=5)
    away_form = _team_rolling_stats(matches_df, away_team, today, n=5)
    h2h = _h2h_stats(matches_df, home_team, away_team, today, n=5)

    features = {
        # ELO features (match original model input)
        'home_elo': home_elo,
        'away_elo': away_elo,
        'elo_diff': home_elo - away_elo,
        'is_home_host': is_home_host,
        'is_away_host': is_away_host,
        'is_neutral': is_neutral,
        'is_friendly': is_friendly,
        # Form features
        'home_win_rate': home_form['win_rate'],
        'home_goals_scored_avg': home_form['goals_scored_avg'],
        'home_goals_conceded_avg': home_form['goals_conceded_avg'],
        'home_points_per_game': home_form['points_per_game'],
        'away_win_rate': away_form['win_rate'],
        'away_goals_scored_avg': away_form['goals_scored_avg'],
        'away_goals_conceded_avg': away_form['goals_conceded_avg'],
        'away_points_per_game': away_form['points_per_game'],
        # Head-to-head features
        'h2h_home_wins': h2h['h2h_home_wins'],
        'h2h_draws': h2h['h2h_draws'],
        'h2h_away_wins': h2h['h2h_away_wins'],
        'h2h_home_goal_diff': h2h['h2h_home_goal_diff'],
    }

    return pd.DataFrame([features])
