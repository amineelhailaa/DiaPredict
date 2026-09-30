"""Shared project constants. Importing this module does not read or write data."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RANDOM_STATE = 42
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "dataset-diabete-raw.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
