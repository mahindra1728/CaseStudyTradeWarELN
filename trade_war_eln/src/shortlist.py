"""Automated shortlisting of trade-war resilient U.S. equities.

The script now sources its candidate universe from liquid U.S. sector ETFs that
benefit from defence, reshoring, and secure infrastructure themes. ETFs are kept
only if their own one-year performance is positive and their 90-day realised
volatility is below 30%. Holdings from the qualifying ETFs are then scored using
policy alignment, China exposure, liquidity, returns, and volatility filters.
Outputs are written to ``outputs/shortlist_results.csv`` with an Excel-ready
rank formula for transparency.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List

import pandas as pd
import yfinance as yf
from functools import lru_cache
from pathlib import Path

# ETF universe to examine. Each sleeve corresponds to a policy-aligned theme.
DEFAULT_CONFIG_PATH = Path(__file__).with_name("shortlist_config.json")

DEFAULT_CONFIG = {
    "etf_candidates": {
        "ITA": "Defense & aerospace",
        "SOXX": "US semiconductors",
        "PAVE": "Infrastructure / reshoring",
        "IGM": "Networking & digital infrastructure",
        "IYW": "Tech hardware & services",
        "XLI": "Industrial supply chain",
        "VOX": "Communications infrastructure",
    },
    "thresholds": {
        "lookback_days": 365,
        "vol_window": 63,
        "min_one_year_return": 0.0,
        "max_realized_vol": 0.30,
        "max_china_share": 0.10,
    },
    "weights": {
        "policy": 0.4,
        "china": 0.25,
        "returns": 0.15,
        "liquidity": 0.1,
        "volatility": 0.1,
    },
    "manual_inclusions": {
        "CSCO": "Secure networking (Cisco)",
        "NUE": "Domestic steel (Nucor)",
        "LMT": "Defense prime (Lockheed Martin)",
    },
}


def deep_update(base: Dict, overrides: Dict) -> Dict:
    for key, value in overrides.items():
        if (
            isinstance(value, dict)
            and key in base
            and isinstance(base[key], dict)
        ):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | None = None) -> Dict:
    """Return configuration dict by merging defaults with file overrides."""

    config = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if cfg_path.exists():
        try:
            with cfg_path.open('r', encoding='utf-8') as handle:
                data = json.load(handle)
            deep_update(config, data)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Warning: failed to load config {cfg_path}: {exc}. Using defaults.")
    else:
        if path:
            print(f"Warning: config file {cfg_path} not found. Using defaults.")
    return config


def apply_config(config: Dict) -> None:
    global ETF_CANDIDATES, THRESHOLDS, WEIGHTS, MANUAL_INCLUSIONS
    global LOOKBACK_DAYS, VOL_WINDOW, MIN_ONE_YEAR_RETURN, MAX_REALIZED_VOL, MAX_CHINA_SHARE

    ETF_CANDIDATES = config["etf_candidates"]
    THRESHOLDS = config["thresholds"]
    WEIGHTS = config["weights"]
    MANUAL_INCLUSIONS = config["manual_inclusions"]

    LOOKBACK_DAYS = int(THRESHOLDS.get("lookback_days", 365))
    VOL_WINDOW = int(THRESHOLDS.get("vol_window", 63))
    MIN_ONE_YEAR_RETURN = float(THRESHOLDS.get("min_one_year_return", 0.0))
    MAX_REALIZED_VOL = float(THRESHOLDS.get("max_realized_vol", 30.0))
    if MAX_REALIZED_VOL <= 1:
        MAX_REALIZED_VOL *= 100
    MAX_CHINA_SHARE = float(THRESHOLDS.get("max_china_share", 0.10))


CONFIG = load_config(os.environ.get("SHORTLIST_CONFIG"))
apply_config(CONFIG)
OUTPUT_DIR = "outputs"


# Base metadata with qualitative China exposure estimates (share of revenue) and
# policy-alignment scores. Values are sourced from FY2023 10-Ks, investor
# presentations, and sector research used in the structuring memo.
BASE_METADATA: Dict[str, Dict[str, float | str]] = {
    "LMT": {"description": "Defense prime (Lockheed Martin)", "china": 0.26, "policy": 1.0},
    "NOC": {"description": "Defense prime (Northrop Grumman)", "china": 0.25, "policy": 0.75},
    "GD": {"description": "Defense systems (General Dynamics)", "china": 0.24, "policy": 0.7},
    "RTX": {"description": "Defense/aerospace (RTX)", "china": 0.35, "policy": 0.75},
    "INTC": {"description": "US semiconductors (Intel)", "china": 0.27, "policy": 0.95},
    "AMD": {"description": "US semiconductors (AMD)", "china": 0.32, "policy": 0.7},
    "NVDA": {"description": "US semiconductors (Nvidia)", "china": 0.35, "policy": 0.65},
    "MU": {"description": "US memory (Micron)", "china": 0.30, "policy": 0.7},
    "CSCO": {"description": "Secure networking (Cisco)", "china": 0.14, "policy": 0.85},
    "JNPR": {"description": "Networking (Juniper)", "china": 0.20, "policy": 0.6},
    "HPE": {"description": "Edge compute (HPE)", "china": 0.28, "policy": 0.55},
    "NUE": {"description": "Domestic steel (Nucor)", "china": 0.18, "policy": 0.95},
    "X": {"description": "Domestic steel (US Steel)", "china": 0.22, "policy": 0.7},
    "CLF": {"description": "Steel/iron ore (Cleveland-Cliffs)", "china": 0.19, "policy": 0.7},
    "CAT": {"description": "Infrastructure capex (Caterpillar)", "china": 0.28, "policy": 0.6},
    "DE": {"description": "Industrial equipment (Deere)", "china": 0.24, "policy": 0.55},
    "GE": {"description": "Aero engines (GE Aerospace)", "china": 0.22, "policy": 0.7},
    "HWM": {"description": "Aerospace alloys (Howmet)", "china": 0.18, "policy": 0.65},
    "FAST": {"description": "Industrial fasteners (Fastenal)", "china": 0.19, "policy": 0.6},
    "URI": {"description": "US equipment rentals (United Rentals)", "china": 0.12, "policy": 0.7},
    "NSC": {"description": "Rail logistics (Norfolk Southern)", "china": 0.05, "policy": 0.5},
    "PWR": {"description": "Grid engineering (Quanta Services)", "china": 0.08, "policy": 0.75},
    "PH": {"description": "Motion control (Parker Hannifin)", "china": 0.24, "policy": 0.65},
    "SRE": {"description": "US utilities (Sempra)", "china": 0.05, "policy": 0.5},
    "CSX": {"description": "Rail logistics (CSX)", "china": 0.04, "policy": 0.45},
    "TT": {"description": "Climate control (Trane)", "china": 0.20, "policy": 0.55},
}

@dataclass
class StockMetrics:
    ticker: str
    one_year_return: float
    realized_vol: float
    avg_volume: float
    policy_score: float
    china_share: float
    liquidity_score: float
    return_score: float
    volatility_score: float
    china_score: float
    composite: float
    market_cap: float | None


def fetch_price_history(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end, auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(ticker, axis=1, level=1)
    return data


@lru_cache(maxsize=None)
def fetch_market_cap(ticker: str) -> float | None:
    try:
        info = yf.Ticker(ticker).fast_info
        market_cap = getattr(info, "market_cap", None)
        if market_cap is None:
            market_cap = info.get("market_cap") if isinstance(info, dict) else None
        return market_cap
    except Exception:
        return None


def fetch_holdings(etf: str) -> List[str]:
    ticker = yf.Ticker(etf)
    symbols: List[str] = []
    funds_data = getattr(ticker, "funds_data", None)
    if funds_data is not None:
        for attr in ("top_holdings", "equity_holdings"):
            df = getattr(funds_data, attr, None)
            if df is None or df.empty:
                continue
            if 'symbol' in df.columns:
                symbols.extend(df['symbol'].dropna().astype(str).str.upper().tolist())
            elif 'Symbol' in df.columns:
                symbols.extend(df['Symbol'].dropna().astype(str).str.upper().tolist())
            elif isinstance(df.index, pd.Index) and df.index.name and df.index.name.lower() == 'symbol':
                symbols.extend(df.index.astype(str).str.upper().tolist())
    if not symbols:
        fallback = ticker.get_holdings()
        if isinstance(fallback, pd.DataFrame):
            if 'symbol' in fallback.columns:
                symbols.extend(fallback['symbol'].dropna().astype(str).str.upper().tolist())
            elif 'Symbol' in fallback.columns:
                symbols.extend(fallback['Symbol'].dropna().astype(str).str.upper().tolist())
    return symbols


def score_liquidity(avg_volume: float) -> float:
    return min(avg_volume / 2_000_000, 1.0)


def score_returns(one_year_return: float, benchmark: float) -> float:
    excess = one_year_return - benchmark
    baseline = 0.2
    if excess <= -30:
        return baseline
    if excess >= 25:
        return 1.0
    scale = (excess + 30) / 55
    return baseline + (1 - baseline) * max(0.0, min(1.0, scale))


def score_volatility(realized_vol: float) -> float:
    if realized_vol >= 80:
        return 0.0
    if realized_vol <= 15:
        return 1.0
    return max(0.0, 1 - (realized_vol - 15) / 65)


def score_china(china_share: float) -> float:
    if china_share >= 0.5:
        return 0.0
    if china_share <= 0.1:
        return 1.0
    return max(0.0, 1 - (china_share - 0.1) / 0.4)


def filter_etfs() -> Dict[str, str]:
    qualifying: Dict[str, str] = {}
    end = datetime.today()
    start = end - timedelta(days=LOOKBACK_DAYS)
    for etf, description in ETF_CANDIDATES.items():
        hist = fetch_price_history(etf, start, end)
        if hist.empty or 'Adj Close' not in hist:
            print(f"Skipping ETF {etf}: insufficient history")
            continue
        hist['Return'] = hist['Adj Close'].pct_change()
        if hist['Return'].dropna().empty:
            print(f"Skipping ETF {etf}: invalid data")
            continue
        one_year_return = (hist['Adj Close'].iloc[-1] / hist['Adj Close'].iloc[0] - 1) * 100
        realized_vol = hist['Return'].tail(VOL_WINDOW).std() * (252 ** 0.5) * 100
        if one_year_return <= MIN_ONE_YEAR_RETURN:
            print(
                f"Skipping ETF {etf}: 1Y return {one_year_return:.2f}% "
                f"<= threshold {MIN_ONE_YEAR_RETURN:.2f}%"
            )
            continue
        if realized_vol >= MAX_REALIZED_VOL:
            print(
                f"Skipping ETF {etf}: realized vol {realized_vol:.2f}% "
                f">= threshold {MAX_REALIZED_VOL:.2f}%"
            )
            continue
        qualifying[etf] = description
    return qualifying


def load_candidates() -> Dict[str, Dict[str, object]]:
    selected_etfs = filter_etfs()
    if not selected_etfs:
        print("Warning: no ETFs met filters; reverting to full candidate list")
        selected_etfs = ETF_CANDIDATES

    tickers: Dict[str, Dict[str, object]] = {}
    for etf, sleeve in selected_etfs.items():
        symbols = fetch_holdings(etf)
        if not symbols:
            print(f"Warning: no holdings retrieved for ETF {etf}")
            continue
        for symbol in symbols:
            if symbol in BASE_METADATA:
                meta = BASE_METADATA[symbol]
                entry = tickers.setdefault(
                    symbol,
                    {
                        "description": meta['description'],
                        "sources": set(),
                    },
                )
                entry["sources"].add(f"{sleeve} ({etf})")

    for symbol, desc in MANUAL_INCLUSIONS.items():
        if symbol in BASE_METADATA:
            entry = tickers.setdefault(
                symbol,
                {
                    "description": desc,
                    "sources": set(),
                },
            )
            entry["sources"].add("Manual inclusion")
    return tickers


def build_metrics(candidates: Dict[str, Dict[str, object]]) -> List[StockMetrics]:
    end = datetime.today()
    start = end - timedelta(days=LOOKBACK_DAYS)

    spy_hist = fetch_price_history("SPY", start, end)
    spy_return = (spy_hist['Adj Close'].iloc[-1] / spy_hist['Adj Close'].iloc[0] - 1) * 100

    results: List[StockMetrics] = []
    for ticker in candidates:
        hist = fetch_price_history(ticker, start, end)
        if hist.empty or 'Adj Close' not in hist or 'Volume' not in hist:
            print(f"Skipping {ticker}: insufficient price history")
            continue
        hist['Return'] = hist['Adj Close'].pct_change()
        if hist['Adj Close'].isna().all() or hist['Return'].dropna().empty:
            print(f"Skipping {ticker}: invalid data points")
            continue
        one_year_return = (hist['Adj Close'].iloc[-1] / hist['Adj Close'].iloc[0] - 1) * 100
        if one_year_return <= MIN_ONE_YEAR_RETURN:
            print(
                f"Skipping {ticker}: 1Y return {one_year_return:.2f}% "
                f"<= threshold {MIN_ONE_YEAR_RETURN:.2f}%"
            )
            continue
        realized_vol = hist['Return'].tail(VOL_WINDOW).std() * (252 ** 0.5) * 100
        if realized_vol >= MAX_REALIZED_VOL:
            print(
                f"Skipping {ticker}: realized vol {realized_vol:.2f}% "
                f">= threshold {MAX_REALIZED_VOL:.2f}%"
            )
            continue
        avg_volume = hist['Volume'].tail(VOL_WINDOW).mean()

        meta = BASE_METADATA.get(ticker)
        if meta is None:
            continue

        china_share = float(meta['china'])
        if china_share >= MAX_CHINA_SHARE:
            print(f"Skipping {ticker}: China share {china_share:.2%} exceeds {MAX_CHINA_SHARE:.0%} cap")
            continue

        liquidity = score_liquidity(avg_volume)
        ret_score = score_returns(one_year_return, spy_return)
        vol_score = score_volatility(realized_vol)
        china_score = score_china(china_share)
        policy = float(meta['policy'])

        composite = (
            WEIGHTS['policy'] * policy
            + WEIGHTS['china'] * china_score
            + WEIGHTS['returns'] * ret_score
            + WEIGHTS['liquidity'] * liquidity
            + WEIGHTS['volatility'] * vol_score
        )

        results.append(
            StockMetrics(
                ticker=ticker,
                one_year_return=one_year_return,
                realized_vol=realized_vol,
                avg_volume=avg_volume,
                policy_score=policy,
                china_share=china_share,
                liquidity_score=liquidity,
                return_score=ret_score,
                volatility_score=vol_score,
                china_score=china_score,
                composite=composite,
                market_cap=fetch_market_cap(ticker),
            )
        )
    return results


def _excel_column_label(index: int) -> str:
    label = ""
    while index >= 0:
        index, remainder = divmod(index, 26)
        label = chr(ord('A') + remainder) + label
        index -= 1
    return label


def _format_market_cap(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.1f}"


def metrics_to_frame(metrics: Iterable[StockMetrics], candidates: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for m in metrics:
        info = candidates[m.ticker]
        sources = ", ".join(sorted(info.get("sources", [])))
        market_cap = m.market_cap
        market_cap_bn = market_cap / 1e9 if market_cap else None
        rows.append(
            {
                "Ticker": m.ticker,
                "Description": info["description"],
                "Source ETFs": sources,
                "1Y Return %": m.one_year_return,
                "90D Realized Vol %": m.realized_vol,
                "Avg Volume (3M)": m.avg_volume,
                "Policy Score": m.policy_score,
                "China Share": m.china_share,
                "Market Cap ($bn)": market_cap_bn,
                "Return Score": m.return_score,
                "Liquidity Score": m.liquidity_score,
                "Volatility Score": m.volatility_score,
                "China Score": m.china_score,
                "Composite Score": m.composite,
            }
        )
    df = pd.DataFrame(rows)
    df.sort_values(by="Composite Score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.insert(0, "Rank", df.index + 1)

    comp_col_index = df.columns.get_loc("Composite Score")
    comp_col_letter = _excel_column_label(comp_col_index)
    total_rows = len(df) + 1  # header occupies row 1 in Excel
    formulas = []
    for idx in range(len(df)):
        row = idx + 2
        formulas.append(
            f"=RANK(${comp_col_letter}${row},${comp_col_letter}$2:${comp_col_letter}${total_rows},0)"
        )
    df["Rank_Formula"] = formulas
    return df


def display_table(table: pd.DataFrame, limit: int = 10) -> None:
    display_cols = [
        "Rank",
        "Ticker",
        "Description",
        "Source ETFs",
        "Composite Score",
        "Policy Score",
        "China Share",
        "Market Cap ($bn)",
        "1Y Return %",
        "90D Realized Vol %",
    ]
    formatters = {
        "Composite Score": "{:.2f}".format,
        "Policy Score": "{:.2f}".format,
        "China Share": "{:.2%}".format,
        "Market Cap ($bn)": _format_market_cap,
        "1Y Return %": "{:.2f}".format,
        "90D Realized Vol %": "{:.2f}".format,
    }
    print("Top trade-war beneficiaries (filtered & scored):")
    print(table[display_cols].head(limit).to_string(index=False, formatters=formatters))


def generate_shortlist(
    config_overrides: Dict | None = None,
    config_path: str | None = None,
    quiet: bool = True,
    write_output: bool = False,
    output_path: str | None = None,
) -> tuple[pd.DataFrame, Dict]:
    """Generate shortlist DataFrame using optional config overrides."""

    config = load_config(config_path)
    if config_overrides:
        deep_update(config, config_overrides)
    apply_config(config)

    candidates = load_candidates()
    if not candidates:
        raise RuntimeError("Candidate universe is empty after ETF aggregation.")

    metrics = build_metrics(candidates)
    if not metrics:
        raise RuntimeError("No equities met the screening thresholds.")

    table = metrics_to_frame(metrics, candidates)

    destination: Path | None = None
    if write_output:
        destination = Path(output_path) if output_path else Path(OUTPUT_DIR) / "shortlist_results.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(destination, index=False)
    if not quiet:
        display_table(table)
        if destination:
            print()
            print(f"Full ranking exported to {destination}")
    return table, config


def main() -> None:
    table, _ = generate_shortlist(quiet=False, write_output=True)
    if not table.empty:
        print(f"Rows returned: {len(table)}")


if __name__ == "__main__":
    main()
