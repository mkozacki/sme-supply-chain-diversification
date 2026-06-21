# SME Supply Chain Diversification

**Author:** Marcel Kozacki (h12531214)
**Course:** ExInt II — Research Designs in SME Research, WU Vienna, SS 2026

## Research Question

Wie beeinflussen Capital Intensity und Cash Holdings die Performance von KMU?

## Theoretical Framework

**Resource-Based View (Barney, 1991) / Transaction Cost Economics (Williamson, 1989):** Hohe Investitionen in physische Assets (Capital Intensity) binden Ressourcen und reduzieren die Flexibilität von KMU. Die hohen Fixkosten erhöhen das operative Risiko und die Transaktionskosten, was sich negativ auf die Performance auswirkt.

**Dynamic Capabilities (Teece et al., 1997):** Liquiditätsreserven (Cash Holdings) ermöglichen es KMU, schnell auf Chancen zu reagieren (Seizing). Firmen mit höherer Liquidität können flexibler agieren und Investitionsmöglichkeiten wahrnehmen, was sich positiv auf die Performance auswirkt.

## Hypotheses

### H1: Capital Intensity und Performance

Höhere Capital Intensity führt zu niedrigerer Performance bei KMU.

- **Y:** RoA (IB / AT)
- **X:** Capital Intensity (CAPX / AT)
- **Erwartetes Vorzeichen:** negativ

### H2: Cash Holdings und Performance

Höhere Cash Holdings führen zu besserer Performance bei KMU.

- **Y:** RoA (IB / AT)
- **X:** Cash Holdings (CHE / AT)
- **Erwartetes Vorzeichen:** positiv

## Variables

| Variable | Field(s) | Formula | Role |
|-------------------|----------|----------------------|----------------|
| RoA | ib, at | ib / at | Dependent (Y) |
| Capital Intensity | capx, at | capx.fillna(0) / at | Independent H1 |
| Cash Holdings | che, at | che.fillna(0) / at | Independent H2 |
| CapInt × Cash | — | capital_intensity × cash_holdings | Interaction |
| Firm Size | at | log(at) | Control |
| Leverage | dltt, at | dltt / at | Control |


## Data

| Item | Detail |
|--------------|--------------------------------------|
| Source | WRDS / Compustat Global |
| Table | comp_global_daily.g_funda |
| Downloaded | 2026-05-29 |
| License | WRDS subscriber agreement |
| Fiscal years | 2015–2024 |
| Raw rows | 338,464 |
| Clean rows | 69,528 |
| Firms | 10,664 |
| Countries | 103 |