"""
03_descriptives.py
------------------
Descriptive statistics, correlation matrix, and plots.

Input:  data/processed/panel_clean.parquet
Output: output/tables/summary_statistics.csv
        output/figures/correlation_matrix.png
        output/figures/dv_distribution.png
        output/figures/main_relationship.png
        data/processed/panel_with_vars.parquet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
IN_PATH = Path("data/processed/panel_clean.parquet")
TABLE_DIR = Path("output/tables")
FIG_DIR = Path("output/figures")
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading cleaned panel...")
df = pd.read_parquet(IN_PATH)
print(f"  Loaded: {len(df):,} obs | {df['gvkey'].nunique():,} firms")

# ── Data quality filters ──────────────────────────────────────────────────────
df = df[(df["at"] > 0.1) & (df["sale"] > 0)].copy()
df = df[df["at"] >= 1].copy()  # remove micro-firms
sme_mask = (df["emp"] < 0.25) | (df["at"] <= 43)
df = df[sme_mask].copy()
print(f"  After quality filters: {len(df):,} obs")

# ── Variable construction ─────────────────────────────────────────────────────
df["roa"] = df["ib"] / df["at"]
df["capital_intensity"] = df["capx"].fillna(0) / df["at"]
df["cash_holdings"] = df["che"].fillna(0) / df["at"]
df["leverage"] = df["dltt"].fillna(0) / df["at"]
df["ln_at"] = np.log(df["at"])
df["cash_flow"] = (df["ibc"].fillna(0) + df["dp"].fillna(0)) / df["at"]
first_year = df.groupby("gvkey")["fyear"].transform("min")
df["age"] = df["fyear"] - first_year

# Interaction term
df["capint_x_cash"] = df["capital_intensity"] * df["cash_holdings"]

# ── Core variables ────────────────────────────────────────────────────────────
CORE_VARS = ["roa", "capital_intensity", "cash_holdings", "leverage",
             "ln_at", "cash_flow", "age"]

WINSORIZE_VARS = ["roa", "capital_intensity", "cash_holdings",
                  "leverage", "cash_flow"]

VAR_LABELS = {
    "roa":                "RoA (ib/at)",
    "capital_intensity":  "Capital Intensity (capx/at)",
    "cash_holdings":      "Cash Holdings (che/at)",
    "leverage":           "Leverage (dltt/at)",
    "ln_at":              "Firm Size (log at)",
    "cash_flow":          "Cash Flow ((ibc+dp)/at)",
    "age":                "Firm Age (years in panel)",
}

# ── Drop missing core vars ────────────────────────────────────────────────────
df = df.dropna(subset=CORE_VARS).copy()
print(f"  After dropping missing core vars: {len(df):,} obs")

# ── Winsorize ─────────────────────────────────────────────────────────────────
for col in WINSORIZE_VARS:
    lo = df[col].quantile(0.01)
    hi = df[col].quantile(0.99)
    df[col] = df[col].clip(lo, hi)
    print(f"  Winsorized {col}: [{lo:.4f}, {hi:.4f}]")

# Recompute interaction after winsorizing
df["capint_x_cash"] = df["capital_intensity"] * df["cash_holdings"]

# ── 1. Summary statistics ─────────────────────────────────────────────────────
print("\n--- Summary Statistics ---")
summary = (
    df[list(VAR_LABELS.keys())]
    .rename(columns=VAR_LABELS)
    .describe(percentiles=[0.25, 0.5, 0.75])
    .T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    .round(4)
)
print(summary.to_string())
summary.to_csv(TABLE_DIR / "summary_statistics.csv")
print(f"\nSaved to {TABLE_DIR / 'summary_statistics.csv'}")

# ── 2. Correlation matrix ─────────────────────────────────────────────────────
print("\n--- Correlation Matrix ---")
corr_cols = ["roa", "capital_intensity", "cash_holdings",
             "leverage", "ln_at", "cash_flow", "age"]
corr = df[corr_cols].corr().round(3)
print(corr.to_string())

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f",
            square=True, ax=ax)
ax.set_title("Correlation Matrix")
plt.tight_layout()
fig.savefig(FIG_DIR / "correlation_matrix.png", dpi=150)
plt.close()
print(f"Saved to {FIG_DIR / 'correlation_matrix.png'}")

# ── 3. DV distribution ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(df["roa"], bins=50, edgecolor="black", alpha=0.7)
axes[0].set_xlabel("RoA (ib/at)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Distribution of RoA")
axes[0].axvline(df["roa"].median(), color="red", linestyle="--", label="Median")
axes[0].legend()

axes[1].hist(df["ln_at"], bins=50, edgecolor="black", alpha=0.7, color="orange")
axes[1].set_xlabel("Firm Size (log AT)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Distribution of Firm Size")
axes[1].axvline(df["ln_at"].median(), color="red", linestyle="--", label="Median")
axes[1].legend()

plt.tight_layout()
fig.savefig(FIG_DIR / "dv_distribution.png", dpi=150)
plt.close()
print(f"Saved to {FIG_DIR / 'dv_distribution.png'}")

# ── 4. Main relationship plots ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# H1: Capital Intensity vs RoA
axes[0].scatter(df["capital_intensity"], df["roa"], alpha=0.05, s=5)
z = np.polyfit(df["capital_intensity"], df["roa"], 1)
p = np.poly1d(z)
x_line = np.linspace(df["capital_intensity"].min(), df["capital_intensity"].max(), 100)
axes[0].plot(x_line, p(x_line), "r-", linewidth=2, label=f"slope = {z[0]:.4f}")
axes[0].set_xlabel("Capital Intensity (capx/at)")
axes[0].set_ylabel("RoA (ib/at)")
axes[0].set_title("H1: Capital Intensity → RoA")
axes[0].legend()

# H2: Cash Holdings vs RoA
axes[1].scatter(df["cash_holdings"], df["roa"], alpha=0.05, s=5)
z2 = np.polyfit(df["cash_holdings"], df["roa"], 1)
p2 = np.poly1d(z2)
x_line2 = np.linspace(df["cash_holdings"].min(), df["cash_holdings"].max(), 100)
axes[1].plot(x_line2, p2(x_line2), "r-", linewidth=2, label=f"slope = {z2[0]:.4f}")
axes[1].set_xlabel("Cash Holdings (che/at)")
axes[1].set_ylabel("RoA (ib/at)")
axes[1].set_title("H2: Cash Holdings → RoA")
axes[1].legend()

plt.tight_layout()
fig.savefig(FIG_DIR / "main_relationship.png", dpi=150)
plt.close()
print(f"Saved to {FIG_DIR / 'main_relationship.png'}")

# ── Save panel with constructed variables ─────────────────────────────────────
df.to_parquet(Path("data/processed/panel_with_vars.parquet"), index=False)
print(f"\nSaved panel_with_vars.parquet ({len(df):,} obs)")
print("Done.")