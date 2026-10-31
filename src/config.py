"""Pre-registered parameters. See "Pre-registered design" in README.md for when each was fixed.

Do not change a value without telling the group first and adding a line to
"Deviations from plan" in README.md.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"

# Data
SYMBOL = "BTCUSDT"
INTERVAL = "1d"
PULL_START = "2018-12-01"   # warm-up for the moving averages only
SAMPLE_START = "2019-01-01"
SAMPLE_END = "2025-12-31"
N_PULL = 2588               # daily candles from PULL_START to SAMPLE_END
N_SAMPLE = 2557             # daily returns in the analysis window

# Rule
SHORT = 5
LONG = 20
COST = 0.001                # 0.1% per position switch

# Inference
ALPHA = 0.05                # two-sided
BLOCK_LENGTH = round(N_SAMPLE ** (1 / 3))   # = 14
BLOCK_GRID = [1, 5, 10, 14, 20, 40]
B = 9_999
SEED = 607

# Robustness sub-periods (fixed in the proposal)
SUB_PERIODS = {
    "bull_2020_2021": ("2020-01-01", "2021-12-31"),
    "bear_2022": ("2022-01-01", "2022-12-31"),
}
