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

## Quick start – Interactive Dashboard (Plotly Dash)

Run the app with fully interactive Plotly charts (no static images).

1) Install dependencies (Python 3.10+):
```bash
pip install dash plotly pandas numpy yfinance
```

2) Launch from the repo root (auto‑opens your browser):
```bash
python run_dash.py
```
Optional environment overrides:
```bash
DASH_HOST=0.0.0.0 DASH_PORT=8050 python run_dash.py
```

Alternatively:
```bash
python -m trade_war_eln.dashboard.dash_app
```

What you’ll see:
- Horizontal filter/weights bar at the top. Click “Run Shortlist” to refresh the table and bar charts.
- View toggle: “Static ELN” (payoff, terminal delta, time‑0 delta) or “Autocall” (Monte Carlo expected PV vs basket ratio, call‑timing probabilities, knock‑in donut). Click “Run Autocall” to generate the MC charts.
- The shortlist table is an interactive Dash DataTable with visible headers.

- **Scoring factors** – The composite is data-driven: 1-year momentum, 63-day
  max drawdown, beta versus FXI (highest weight), revenue growth, and net
  margin. All metrics come from yfinance and are normalised across the
  shortlisted universe before weights are applied (`src/shortlist_config.json`).

### Optional command-line utilities

- `python3 src/shortlist.py` – regenerate the scored shortlist (reads `src/shortlist_config.json` or overrides via environment variable `SHORTLIST_CONFIG`).
- `python3 src/eln_core.py` – dependency‑light ELN pricing breakdown (static structure).
- `python3 src/data_snapshot.py` – refresh market metrics and correlations from yfinance.
- `python3 src/autocall_pricer.py --S0 4900 --sigma 0.22 --q 0.025 --r 0.03 --T 5 --obs "0.5,1,1.5,2,2.5,3,3.5,4,4.5,5" --acb "1,1,1,1,1,1,1,1,1,1" --coupon "0.025837,0.025837,0.025837,0.025837,0.025837,0.025837,0.025837,0.025837,0.025837,0.141" --kib 0.6 --kib_dir down` – Monte Carlo pricer for the autocall/knock‑in note.

## Next steps for deliverables

- **Excel** – import the CSV files (or run `eln_analysis.py` once dependencies
  are available) to populate the calculation workbook requested in the brief.
- **PPT** – populate the template (or auto-generate via `eln_analysis.py`) using
  the charts based on `payoff_profile.csv`, `terminal_delta.csv`, and
  `delta_t0.csv`.

All numbers are based on clear, editable assumptions in the code – update them with live data before client use.
