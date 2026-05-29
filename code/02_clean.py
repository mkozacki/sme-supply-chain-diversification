"""
02_clean.py
-----------
Clean raw Compustat data, apply SME filter, construct variables.

Input:  data/raw/<most recent timestamp folder>/fyear_*.parquet
Output: data/processed/panel_clean.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Find most recent pull folder ──────────────────────────────────────────────
RAW_DIR = Path("data/raw")
folders = sorted([f for f in RAW_DIR.iterdir() if f.is_dir()], reverse=True)

if not folders:
    raise FileNotFoundError("No pull folders found in data/raw/. Run 01_pull_data.py first.")

LATEST = folders[0]
print(f"Using pull folder: {LATEST}")

# ── Output path ───────────────────────────────────────────────────────────────
OUT_PATH = Path("data/processed/panel_clean.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Load all year files ───────────────────────────────────────────────────────
print("Loading raw data...")
parquet_files = sorted(LATEST.glob("fyear_*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
df.columns = [c.strip().lower() for c in df.columns]
n_raw = len(df)
print(f"  Raw observations: {n_raw:,} | firms: {df['gvkey'].nunique():,}")

# ── Basic cleaning ────────────────────────────────────────────────────────────
# Drop duplicates and rows missing gvkey or fyear
df = df.drop_duplicates(subset=["gvkey", "fyear"])
df = df.dropna(subset=["gvkey", "fyear"])

# Convert key columns to numeric (some may be stored as strings)
numeric_cols = ["at", "ib", "ibc", "capx", "dp", "dltt", "dlc", "seq",
                "sale", "xrd", "emp", "che", "ppent", "ebit", "ebitda"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Require positive total assets and sales
df = df[(df["at"] > 0) & (df["sale"] > 0)].copy()
print(f"  After basic cleaning: {len(df):,}")

# ── SME Filter ────────────────────────────────────────────────────────────────
# EU definition: < 250 employees OR total assets <= 43m
# emp in Compustat is in thousands → 0.25 = 250 employees
sme_mask = (df["emp"] < 0.25) | (df["at"] <= 43)
n_before = len(df)
df = df[sme_mask].copy()
print(f"  After SME filter: {len(df):,} (removed {n_before - len(df):,})")

# ── Construct Variables ───────────────────────────────────────────────────────
# Dependent variable (Y): Return on Assets
df["roa"] = df["ib"] / df["at"]

# Independent variable (X): Capital Intensity
df["capital_intensity"] = df["capx"].fillna(0) / df["at"]

# Controls
df["ln_at"] = np.log(df["at"])
df["leverage"] = (df["dltt"].fillna(0) + df["dlc"].fillna(0)) / df["seq"]
df["cash_flow"] = (df["ibc"].fillna(0) + df["dp"].fillna(0)) / df["at"]
df["rd_intensity"] = df["xrd"].fillna(0) / df["at"]

# Firm age: years since first appearance in dataset
first_year = df.groupby("gvkey")["fyear"].transform("min")
df["age"] = df["fyear"] - first_year

# ── Winsorize at 1%–99% ──────────────────────────────────────────────────────
def winsorize(series, lower=0.01, upper=0.99):
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)

for col in ["roa", "capital_intensity", "leverage", "cash_flow", "rd_intensity"]:
    if col in df.columns:
        df[col] = winsorize(df[col])

# ── Drop observations with missing core variables ────────────────────────────
core_vars = ["roa", "capital_intensity", "ln_at", "leverage", "age"]
n_before = len(df)
df = df.dropna(subset=core_vars).copy()
print(f"  After dropping missing core vars: {len(df):,} (removed {n_before - len(df):,})")

# ── Require >= 3 observations per firm ────────────────────────────────────────
obs_per_firm = df.groupby("gvkey")["fyear"].count()
valid_firms = obs_per_firm[obs_per_firm >= 3].index
n_before = len(df)
df = df[df["gvkey"].isin(valid_firms)].copy()
print(f"  After min-obs filter (>=3 per firm): {len(df):,} (removed {n_before - len(df):,})")
print(f"  Final: {len(df):,} obs | {df['gvkey'].nunique():,} firms | {df['loc'].nunique()} countries")
print(f"  Years: {int(df['fyear'].min())}–{int(df['fyear'].max())}")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_parquet(OUT_PATH, index=False)
print(f"\nSaved cleaned panel to {OUT_PATH}")

# ── Log file ──────────────────────────────────────────────────────────────────
log_path = OUT_PATH.parent / "clean_log.txt"
log_path.write_text(
    f"Clean log\n"
    f"=========\n"
    f"Input:       {LATEST}\n"
    f"Raw rows:    {n_raw:,}\n"
    f"Clean rows:  {len(df):,}\n"
    f"Firms:       {df['gvkey'].nunique():,}\n"
    f"Countries:   {df['loc'].nunique()}\n"
    f"Years:       {int(df['fyear'].min())}–{int(df['fyear'].max())}\n"
    f"Columns:     {len(df.columns)}\n"
)
print(f"Log saved to {log_path}")