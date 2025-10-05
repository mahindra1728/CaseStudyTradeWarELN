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

## How to reproduce the numbers

1. Activate a Python 3.10+ environment.
2. From `trade_war_eln/`, run the lightweight script:
   ```bash
   python3 src/eln_core.py
   ```
   This prints pricing outputs and refreshes the CSV files and summary text in
   `outputs/`.
3. (Optional) Install `numpy pandas matplotlib python-pptx openpyxl yfinance`
   and run `python3 src/eln_analysis.py` to build ready-to-plot charts, an Excel
   workbook, and a 4-slide PPT deck automatically.
4. To refresh the quantitative evidence behind the stock screen, run
   `python3 src/data_snapshot.py` (requires `yfinance`). This regenerates
   `outputs/market_metrics.csv` and `outputs/correlation_matrix.csv`.
5. To reproduce the automated shortlist, run `python3 src/shortlist.py`. The
   script filters candidate ETFs (ITA, SOXX, PAVE, IGM, IYW, XLI, VOX) to those
   with positive 1Y returns and sub-30% realised volatility, harvests their top
   holdings via yfinance, adds essential names, and scores each equity across
   policy, China exposure (capped at 10%), liquidity, returns, and volatility.
   Results are saved to `outputs/shortlist_results.csv` with source ETF tags and
   an Excel rank formula.

### Interactive dashboard

Launch the Flask dashboard to tune parameters visually:

```bash
export FLASK_APP=trade_war_eln.dashboard.app
export FLASK_ENV=development  # optional auto-reload
python3 -m flask run
```

The dashboard lets you adjust thresholds/weights, choose ETF sleeves, and
preview the resulting shortlist directly in your browser. Overrides entered in
the form are applied in-memory (they do not overwrite
`src/shortlist_config.json`).

### Tuning inputs

- Edit `src/shortlist_config.json` to change ETF sleeves, weights, and
  thresholds (lookback, min return, max volatility, max China share). You can
  point to an alternate config via `SHORTLIST_CONFIG=/path/to/config.json
  python3 src/shortlist.py`.

## Next steps for deliverables

- **Excel** – import the CSV files (or run `eln_analysis.py` once dependencies
  are available) to populate the calculation workbook requested in the brief.
- **PPT** – populate the template (or auto-generate via `eln_analysis.py`) using
  the charts based on `payoff_profile.csv`, `terminal_delta.csv`, and
  `delta_t0.csv`.

All numbers are based on the deterministic assumptions documented in the code –
update them with live data as needed before client use.
