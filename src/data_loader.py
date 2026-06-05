import os
import pandas as pd
from pathlib import Path
import zipfile
try:
    import yaml  # type: ignore
except Exception:
    # yaml may not be installed in some environments (linting/CI). Defer a clear
    # ImportError until runtime when config is actually needed.
    yaml = None

# Resolve config path relative to project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATHS = PROJECT_ROOT / "config" / "paths.yaml"

with open(CONFIG_PATHS, "r") as f:
    if yaml is None:
        raise ImportError(
            "PyYAML is required to load config/paths.yaml. Install it with: pip install pyyaml"
        )
    PATHS = yaml.safe_load(f)

RAW_DIR = str(PROJECT_ROOT / PATHS["data"]["raw_dir"])
PROC_DIR = str(PROJECT_ROOT / PATHS["data"]["processed_dir"])
FILES = PATHS["files"]


def _read_csv_with_encoding(path):
    """Try reading CSV or Excel files with encoding/format detection."""
    import io
    
    # Check if file is actually an Excel file (ZIP with xl/ structure)
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path, 'r') as zip_ref:
                all_files = zip_ref.namelist()
                # Check if it's an Excel file
                if any(f.startswith('xl/') for f in all_files):
                    # It's an XLSX file disguised with a .csv extension.
                    import tempfile
                    from pathlib import Path as _Path

                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
                            tmp_path = tmp.name
                            tmp.write(_Path(path).read_bytes())
                        return pd.read_excel(tmp_path, engine='openpyxl')
                    finally:
                        if tmp_path is not None:
                            try:
                                _Path(tmp_path).unlink(missing_ok=True)
                            except Exception:
                                pass
                # Otherwise try to find and read CSV
                csv_files = [f for f in all_files if f.endswith('.csv')]
                if csv_files:
                    with zip_ref.open(csv_files[0]) as f:
                        content = f.read()
                        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                        for encoding in encodings:
                            try:
                                text = content.decode(encoding)
                                return pd.read_csv(io.StringIO(text))
                            except (UnicodeDecodeError, pd.errors.ParserError):
                                continue
                        return pd.read_csv(io.BytesIO(content), encoding='utf-8', errors='replace', on_bad_lines='skip')
    except (zipfile.BadZipFile, NotImplementedError, ImportError):
        pass  # Not a ZIP file or openpyxl not available, continue
    
    # Regular CSV reading with encoding/delimiter fallback
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    delimiters = [',', ';', '\t', '|']
    
    for encoding in encodings:
        for delimiter in delimiters:
            try:
                return pd.read_csv(path, encoding=encoding, delimiter=delimiter)
            except (UnicodeDecodeError, LookupError, pd.errors.ParserError):
                continue
    # If all fail, read with errors='replace' and flexible delimiter as last resort
    return pd.read_csv(path, encoding='utf-8', delimiter=',', errors='replace', on_bad_lines='skip')


def load_raw_matches():
    path = os.path.join(RAW_DIR, FILES["matches"])
    return _read_csv_with_encoding(path)


def load_raw_elo() -> pd.DataFrame:
    from src.live_elo_loader import fetch_live_elo_ratings
    return fetch_live_elo_ratings()

    # Fallback to your local static file
    print("[data_loader] Falling back to local ELO data")
    return pd.read_csv(PROJECT_ROOT / "data" / "raw" / "elo_ratings.csv")


def load_fixtures_2026():
    path = os.path.join(RAW_DIR, FILES["fixtures_2026"])
    return _read_csv_with_encoding(path)


def load_teams_metadata():
    path = os.path.join(RAW_DIR, FILES["teams_metadata"])
    return _read_csv_with_encoding(path)


def save_processed(df: pd.DataFrame, key: str):
    filename = FILES[key]
    path = os.path.join(PROC_DIR, filename)
    df.to_csv(path, index=False)
