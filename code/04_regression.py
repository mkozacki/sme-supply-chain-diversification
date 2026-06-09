"""
04_regression.py
----------------
Panel regression: Capital Intensity and Cash Holdings on RoA.

Models:
  (1) Pooled OLS
  (2) Two-Way Fixed Effects (Entity + Time)
  (3) TWFE + Interaction (capital_intensity × cash_holdings)

Input:  data/processed/panel_with_vars.parquet
Output: output/tables/regression_results.csv
"""

import warnings
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DV = "roa"
X_MAIN1 = "capital_intensity"
X_MAIN2 = "cash_holdings"
INTERACT = "capint_x_cash"
CONTROLS = ["ln_at", "leverage"]

# ── Paths ─────────────────────────────────────────────────────────────────────
IN_PATH = Path("data/processed/panel_with_vars.parquet")
OUT_DIR = Path("output/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helper to extract standard errors ─────────────────────────────────────────
def get_se(result, param):
    """Extract SE from statsmodels or linearmodels result."""
    try:
        return result.std_errors[param]
    except (AttributeError, KeyError):
        try:
            return result.bse[param]
        except (AttributeError, KeyError):
            return float("nan")

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading panel_with_vars.parquet...")
df = pd.read_parquet(IN_PATH)

all_vars = [DV, X_MAIN1, X_MAIN2, INTERACT] + CONTROLS
df = df.dropna(subset=all_vars).copy()
print(f"  Observations: {len(df):,} | Firms: {df['gvkey'].nunique():,}")

# ── Model 1: Pooled OLS ──────────────────────────────────────────────────────
print("\n=== Model (1): Pooled OLS ===")
controls_str = " + ".join(CONTROLS)
formula_ols = f"{DV} ~ {X_MAIN1} + {X_MAIN2} + {controls_str}"

m1 = smf.ols(formula_ols, data=df).fit(cov_type="cluster",
     cov_kwds={"groups": df["gvkey"]})
print(m1.summary().tables[1])

# ── Model 2: Two-Way Fixed Effects ────────────────────────────────────────────
print("\n=== Model (2): Two-Way Fixed Effects ===")
df_panel = df.set_index(["gvkey", "fyear"])
formula_fe = f"{DV} ~ {X_MAIN1} + {X_MAIN2} + {controls_str} + EntityEffects + TimeEffects"

m2 = PanelOLS.from_formula(formula_fe, data=df_panel).fit(
    cov_type="clustered", cluster_entity=True)
print(m2.summary.tables[1])

# ── Model 3: TWFE + Interaction ───────────────────────────────────────────────
print("\n=== Model (3): TWFE + Interaction (CapInt × Cash) ===")
formula_int = f"{DV} ~ {X_MAIN1} + {X_MAIN2} + {INTERACT} + {controls_str} + EntityEffects + TimeEffects"

m3 = PanelOLS.from_formula(formula_int, data=df_panel).fit(
    cov_type="clustered", cluster_entity=True)
print(m3.summary.tables[1])

# ── Results table ─────────────────────────────────────────────────────────────
results = pd.DataFrame({
    "OLS": m1.params,
    "OLS_se": m1.bse,
    "OLS_p": m1.pvalues,
    "FE": m2.params,
    "FE_se": m2.std_errors,
    "FE_p": m2.pvalues,
    "FE_interact": m3.params,
    "FE_int_se": m3.std_errors,
    "FE_int_p": m3.pvalues,
}).round(6)

results.to_csv(OUT_DIR / "regression_results.csv")
print(f"\nSaved to {OUT_DIR / 'regression_results.csv'}")

# ── H1 Diagnostic ────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("H1: Capital Intensity → RoA (expected: NEGATIVE)")
print("=" * 55)
b1 = m2.params.get(X_MAIN1, float("nan"))
p1 = m2.pvalues.get(X_MAIN1, float("nan"))
print(f"  Beta(FE):  {b1:.4f}")
print(f"  p-value:   {p1:.4f}")
if p1 < 0.05 and b1 < 0:
    print("  → H1 SUPPORTED: significant negative effect")
elif p1 < 0.05 and b1 > 0:
    print("  → H1 NOT SUPPORTED: significant but positive")
else:
    print("  → H1 NOT SUPPORTED: not significant at 5%")

# ── H2 Diagnostic ────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("H2: Cash Holdings → RoA (expected: POSITIVE)")
print("=" * 55)
b2 = m2.params.get(X_MAIN2, float("nan"))
p2 = m2.pvalues.get(X_MAIN2, float("nan"))
print(f"  Beta(FE):  {b2:.4f}")
print(f"  p-value:   {p2:.4f}")
if p2 < 0.05 and b2 > 0:
    print("  → H2 SUPPORTED: significant positive effect")
elif p2 < 0.05 and b2 < 0:
    print("  → H2 NOT SUPPORTED: significant but negative")
else:
    print("  → H2 NOT SUPPORTED: not significant at 5%")

# ── Interaction Diagnostic ────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Interaction: Capital Intensity × Cash Holdings")
print("=" * 55)
b3 = m3.params.get(INTERACT, float("nan"))
p3 = m3.pvalues.get(INTERACT, float("nan"))
print(f"  Beta(FE):  {b3:.4f}")
print(f"  p-value:   {p3:.4f}")
if p3 < 0.05:
    print("  → SIGNIFICANT: Cash holdings moderate the capital intensity effect")
else:
    print("  → NOT SIGNIFICANT at 5%")

# ── OLS vs FE comparison ─────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("OLS vs FE comparison")
print("=" * 55)
print(f"  OLS R²:    {m1.rsquared:.4f}")
print(f"  FE  R²:    {m2.rsquared:.4f}")
print(f"  Difference: {m2.rsquared - m1.rsquared:.4f}")
if abs(m2.rsquared - m1.rsquared) > 0.05:
    print("  → Large difference: firm FE explain important variance (OVB in OLS)")
else:
    print("  → Small difference: limited omitted variable bias")

print("\nDone.")