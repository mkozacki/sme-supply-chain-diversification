# SME Supply Chain Diversification

**Author:** Marcel Kozacki (h12531214)
**Course:** ExInt II — Research Designs in SME Research, WU Vienna, SS 2026

## Research Question

Welche Firmencharakteristiken beeinflussen die Performance-Resilienz europäischer KMU während Perioden globaler Lieferkettenstörungen (2020–2022)?

## Theoretical Framework

Die Studie kombiniert zwei komplementäre theoretische Perspektiven:

**Dynamic Capabilities (Teece et al., 1997; Teece, 2007):** KMU benötigen Sensing-, Seizing- und Reconfiguring-Fähigkeiten, um auf Lieferkettenstörungen zu reagieren. Finanzielle Ressourcen (Cash Holdings) ermöglichen Seizing — die schnelle Mobilisierung von Ressourcen bei Störungen. Hohe Kapitalintensität erhöht die Asset Specificity und erschwert Reconfiguring.

**Transaction Cost Economics (Williamson, 1989):** Lieferkettenstörungen erhöhen die Transaktionskosten. Kapitalintensive Firmen tragen höhere Switching Costs, da spezialisierte Assets schwerer umzuwidmen sind. Hoher Leverage reduziert die finanzielle Flexibilität und erhöht Monitoring-Kosten in Krisenzeiten.

## Hypotheses and Operationalization

### H1: Capital Intensity und Performance während Lieferkettenstörungen

Kapitalintensive KMU zeigen einen stärkeren Performance-Rückgang während der Störungsperiode (2020–2022) als weniger kapitalintensive KMU.

- **Logik (TCE):** Hohe Asset Specificity erhöht die Transaktionskosten bei Störungen, da spezialisierte physische Assets schwerer anpassbar sind.
- **Y:** RoA (IB / AT)
- **X:** Capital Intensity (CAPX / AT) × Disruption-Dummy
- **Erwartetes Vorzeichen:** negativ (kapitalintensive Firmen performen schlechter während Störungen)

### H2: Cash Holdings als Resilienzfaktor

KMU mit höheren Cash Holdings zeigen eine stabilere Performance während der Störungsperiode.

- **Logik (Dynamic Capabilities — Seizing):** Liquiditätsreserven ermöglichen schnelle Reaktion auf Störungen — z.B. Wechsel zu alternativen Lieferanten, Aufbau von Lagerbeständen.
- **Y:** RoA (IB / AT)
- **X:** Cash Holdings (CHE / AT) × Disruption-Dummy
- **Erwartetes Vorzeichen:** positiv (höhere Cash Holdings stabilisieren Performance)

### H3: Leverage verstärkt negative Disruption-Effekte

Der negative Performance-Effekt von hohem Leverage ist während der Störungsperiode stärker als in normalen Jahren.

- **Logik (TCE + Dynamic Capabilities — Reconfiguring):** Hohe Verschuldung reduziert die finanzielle Flexibilität für Reconfiguring und erhöht Monitoring-Kosten in Krisenzeiten.
- **Y:** RoA (IB / AT)
- **X:** Leverage (DLTT / AT) × Disruption-Dummy
- **Erwartetes Vorzeichen:** negativ (Leverage schadet stärker während Störungen)

## Variables

### Dependent variable (Y)

| Construct | Data Item(s) | Formula |
|-----------|-------------|---------|
| RoA | IB, AT | IB / AT |

### Independent variables (X)

| Construct | Data Item(s) | Formula |
|-----------|-------------|---------|
| Capital Intensity | CAPX, AT | CAPX / AT |
| Cash Holdings | CHE, AT | CHE / AT |
| Leverage | DLTT, AT | DLTT / AT |
| Disruption | FYEAR | 1 if 2020 <= FYEAR <= 2022, else 0 |

### Interaction terms

| Term | Formula | Tests |
|------|---------|-------|
| CapInt × Disruption | capital_intensity × disruption | H1 |
| Cash × Disruption | cash_holdings × disruption | H2 |
| Leverage × Disruption | leverage × disruption | H3 |

### Controls

| Construct | Data Item(s) | Formula |
|-----------|-------------|---------|
| Firm Size | AT | log(AT) |
| Firm Age | FYEAR | FYEAR - min(FYEAR) per firm |
| Cash Flow | IBC, DP, AT | (IBC + DP) / AT |
| Industry | SICH | categorical fixed effect |

### Additional variables pulled

REVT, EBIT, EBITDA, OIADP, CAPX, XSGA, COGS, DLTR, DLTIS, CEQ, LT, LCT, CHE, CH, PPENT, INVT, ACT, EMP, SALE, CONM, FIC, CURCD, XRD

## Data

| Item | Detail |
|--------------|--------------------------------------|
| Source | WRDS / Compustat Global |
| Table | comp_global_daily.g_funda |
| Downloaded | 2026-05-29 |
| License | WRDS subscriber agreement |
| Fiscal years | 2015–2024 |
| Raw rows | 338,464 |
| Clean rows | 70,718 |
| Firms | 10,703 |
| Countries | 103 |