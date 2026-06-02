"""
02_clean.py
-----------
Clean raw Compustat data, apply SME filter, construct variables.

Input:  data/raw/<most recent timestamp folder>/fyear_*.parquet
Output: data/processed/panel_clean.parquet

Variables constructed
--------------------
roa              = ib / at                    (Y: Return on Assets)
capital_intensity = capx / at                 (X1: Capital Intensity)
cash_holdings    = che / at                   (X2: Cash Holdings)
leverage         = dltt / at                  (X3: Leverage)
disruption       = 1 if 2020<=fyear<=2022     (Moderator: disruption period)
capint_x_disrupt = capital_intensity * disruption  (H1 interaction)
cash_x_disrupt   = cash_holdings * disruption      (H2 interaction)
lev_x_disrupt    = leverage * disruption           (H3 interaction)
ln_at            = log(at)                    (Control: firm size)
age              = fyear - first year in data (Control: firm age)
cash_flow        = (ibc + dp) / at            (Control: operating cash flow)
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
df = df.drop_duplicates(subset=["gvkey", "fyear"])
df = df.dropna(subset=["gvkey", "fyear"])

numeric_cols = ["at", "ib", "ibc", "capx", "dp", "dltt", "dlc", "seq",
                "sale", "xrd", "emp", "che", "ppent", "ebit", "ebitda"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[(df["at"] > 0) & (df["sale"] > 0)].copy()
print(f"  After basic cleaning: {len(df):,}")

# ── SME Filter ────────────────────────────────────────────────────────────────
# EU definition: < 250 employees OR total assets <= 43m
sme_mask = (df["emp"] < 0.25) | (df["at"] <= 43)
n_before = len(df)
df = df[sme_mask].copy()
print(f"  After SME filter: {len(df):,} (removed {n_before - len(df):,})")

# ── Construct Variables ───────────────────────────────────────────────────────
# Dependent variable (Y)
df["roa"] = df["ib"] / df["at"]

# Independent variables (X)
df["capital_intensity"] = df["capx"].fillna(0) / df["at"]
df["cash_holdings"] = df["che"].fillna(0) / df["at"]
df["leverage"] = df["dltt"].fillna(0) / df["at"]

# Disruption dummy (2020-2022: COVID + supply chain crisis)
df["disruption"] = ((df["fyear"] >= 2020) & (df["fyear"] <= 2022)).astype(int)

# Interaction terms for hypothesis testing
df["capint_x_disrupt"] = df["capital_intensity"] * df["disruption"]
df["cash_x_disrupt"] = df["cash_holdings"] * df["disruption"]
df["lev_x_disrupt"] = df["leverage"] * df["disruption"]

# Controls
df["ln_at"] = np.log(df["at"])
df["cash_flow"] = (df["ibc"].fillna(0) + df["dp"].fillna(0)) / df["at"]

# Firm age: years since first appearance in dataset
first_year = df.groupby("gvkey")["fyear"].transform("min")
df["age"] = df["fyear"] - first_year

# ── Winsorize at 1%–99% ──────────────────────────────────────────────────────
def winsorize(series, lower=0.01, upper=0.99):
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)

for col in ["roa", "capital_intensity", "cash_holdings", "leverage", "cash_flow"]:
    df[col] = winsorize(df[col])

# Recompute interaction terms after winsorizing
df["capint_x_disrupt"] = df["capital_intensity"] * df["disruption"]
df["cash_x_disrupt"] = df["cash_holdings"] * df["disruption"]
df["lev_x_disrupt"] = df["leverage"] * df["disruption"]

# ── Drop observations with missing core variables ────────────────────────────
core_vars = ["roa", "capital_intensity", "cash_holdings", "leverage",
             "ln_at", "cash_flow", "age", "disruption"]
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
print(f"  Disruption obs: {df['disruption'].sum():,} | Non-disruption: {(1-df['disruption']).sum():.0f}")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_parquet(OUT_PATH, index=False)
print(f"\nSaved cleaned panel to {OUT_PATH}")

# ── Log file ──────────────────────────────────────────────────────────────────
log_path = OUT_PATH.parent / "clean_log.txt"
log_path.write_text(
    f"Clean log\n"
    f"=========\n"
    f"Input:           {LATEST}\n"
    f"Raw rows:        {n_raw:,}\n"
    f"Clean rows:      {len(df):,}\n"
    f"Firms:           {df['gvkey'].nunique():,}\n"
    f"Countries:       {df['loc'].nunique()}\n"
    f"Years:           {int(df['fyear'].min())}–{int(df['fyear'].max())}\n"
    f"Disruption obs:  {df['disruption'].sum():,}\n"
    f"Columns:         {len(df.columns)}\n"
)
print(f"Log saved to {log_path}")