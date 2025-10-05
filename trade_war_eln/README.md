# Bullish Equity Linked Note – US-China Trade War View

This workspace captures the thought process, assumptions, and artefacts used to
structure and price a one-year bullish ELN referencing a two-stock U.S. basket.
It is organised so you can regenerate the numbers, update the views, and
assemble deliverables (Excel + PPT) requested in the original brief.

## Repository map

```
trade_war_eln/
├── README.md                 # This overview
├── src/
│   ├── eln_core.py           # Dependency-light pricing + CSV export
│   ├── eln_analysis.py       # Full workflow with charts/PPT (optional libs)
│   ├── data_snapshot.py      # Pulls market metrics & correlations via yfinance
│   └── shortlist.py          # Automated scoring to shortlist beneficiaries
└── outputs/
    ├── payoff_profile.csv    # Payoff curve data (normalised to $1 notional)
    ├── terminal_delta.csv    # Terminal delta (slope of payoff) vs performance
    ├── delta_t0.csv          # Time-0 PV & delta vs basket level scenarios
    ├── market_metrics.csv    # 1Y returns, realised vol, liquidity stats (yfinance)
    ├── correlation_matrix.csv# Pairwise equity correlations over last 12 months
    ├── shortlist_results.csv # Scored universe supporting automated selection
    └── summary.txt           # Key pricing numbers and component PVs
```

## Quick start – Dashboard only

1. **Install dependencies** (Python 3.10+, virtual environment recommended):
   ```bash
   pip install -r requirements.txt
   ```
   (If you don’t have a requirements file, install `flask numpy pandas matplotlib python-pptx openpyxl yfinance`.)

2. **Launch the dashboard** from the project root:
   ```bash
   export FLASK_APP=trade_war_eln.dashboard.app
   python3 -m flask run
   ```

3. **Open the browser** at `http://127.0.0.1:5000/`.
   - The home page shows the latest shortlist and “interview talking points”.
   - Adjust ETF sleeves, thresholds, or weights in the form and click *Run Shortlist*.
   - The backend automatically regenerates payoff charts, pricing tables, and delta profiles (no manual script execution required).

4. **Download artefacts** directly from the `trade_war_eln/outputs/` folder: payoff and delta PNGs, Excel workbook, PPT template, and CSV exports are refreshed each time the dashboard recomputes.

### Optional command-line utilities

All underlying scripts remain available if you prefer the CLI:

- `python3 src/shortlist.py` – regenerate the scored shortlist (reads `src/shortlist_config.json` or overrides via environment variable `SHORTLIST_CONFIG`).
- `python3 src/eln_core.py` – dependency-light pricing breakdown.
- `python3 src/eln_analysis.py` – full analytics pipeline (called automatically by the dashboard when charts are missing).
- `python3 src/data_snapshot.py` – refresh market metrics and correlations from yfinance.

## Next steps for deliverables

- **Excel** – import the CSV files (or run `eln_analysis.py` once dependencies
  are available) to populate the calculation workbook requested in the brief.
- **PPT** – populate the template (or auto-generate via `eln_analysis.py`) using
  the charts based on `payoff_profile.csv`, `terminal_delta.csv`, and
  `delta_t0.csv`.

All numbers are based on the deterministic assumptions documented in the code –
update them with live data as needed before client use.
