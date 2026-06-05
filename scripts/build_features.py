# =============================================================================
# scripts/build_features.py
# -----------------------------------------------------------------------------
# One-time script to regenerate data/processed/features.csv with the full
# enriched feature set (base ELO + rolling form + head-to-head).
#
# Run from project root:
#   python scripts/build_features.py
#
# Expected runtime: 5-15 minutes for ~47k matches.
# Output: data/processed/features.csv (overwrites existing file)
# =============================================================================

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.data_loader import load_raw_matches, load_raw_elo
from src.features import build_features, build_rolling_features

def main():
    print("=" * 60)
    print("FIFA 2026 — Feature Engineering Pipeline")
    print("=" * 60)

    # Load raw data
    print("\n[1/4] Loading raw matches...")
    matches_df = load_raw_matches()
    print(f"      Loaded {len(matches_df):,} matches")

    print("\n[2/4] Loading ELO ratings...")
    elo_df = load_raw_elo()
    print(f"      Loaded {len(elo_df):,} teams")

    # Build base features
    print("\n[3/4] Building base features (ELO join)...")
    base_df = build_features(matches_df, elo_df)
    print(f"      Base features shape: {base_df.shape}")
    print(f"      Columns: {base_df.columns.tolist()}")

    # Build rolling features
    print("\n[4/4] Building rolling features (form + H2H)...")
    print("      This will take 5-15 minutes — progress shown every 5,000 rows.")
    full_df = build_rolling_features(base_df, n_form=5, n_h2h=5)

    # Save
    out_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "features.csv"
    full_df.to_csv(out_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"Done! Saved {len(full_df):,} rows x {len(full_df.columns)} columns")
    print(f"Output: {out_path}")
    print(f"New columns added:")
    original_cols = set(base_df.columns)
    new_cols = [c for c in full_df.columns if c not in original_cols]
    for c in new_cols:
        print(f"  + {c}")
    print("=" * 60)

if __name__ == "__main__":
    main()