"""Automated shortlisting of trade-war resilient U.S. equities.

This module aggregates ETF sleeves and optional index baskets, applies
observable risk filters, and ranks candidates using data-driven factors sourced
from yfinance (momentum, drawdown, China beta, revenue growth, net margin).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Set

import numpy as np
import pandas as pd
import yfinance as yf

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
    "index_lists": {},  # Optional manual index universes
    "thresholds": {
        "lookback_days": 365,
        "vol_window": 90,
        "min_one_year_return": 0.0,
        "max_realized_vol": 30.0,
        "max_china_share": 0.10,
        "long_lookback_days": 1825
    },
    "weights": {
        "momentum_252": 0.30,
        "max_drawdown_63": 0.10,
        "beta_cn": 0.40,
        "revenue_growth": 0.12,
        "net_margin": 0.08,
    },
    "manual_inclusions": {
        "CSCO": "Secure networking (Cisco)",
        "NUE": "Domestic steel (Nucor)",
        "LMT": "Defense prime (Lockheed Martin)",
    },
}

BASE_METADATA: Dict[str, Dict[str, float | str]] = {
    "LMT": {"description": "Defense prime (Lockheed Martin)", "china": 0.26},
    "NOC": {"description": "Defense prime (Northrop Grumman)", "china": 0.25},
    "GD": {"description": "Defense systems (General Dynamics)", "china": 0.24},
    "RTX": {"description": "Defense/aerospace (RTX)", "china": 0.35},
    "INTC": {"description": "US semiconductors (Intel)", "china": 0.27},
    "AMD": {"description": "US semiconductors (AMD)", "china": 0.32},
    "NVDA": {"description": "US semiconductors (Nvidia)", "china": 0.35},
    "MU": {"description": "US memory (Micron)", "china": 0.30},
    "CSCO": {"description": "Secure networking (Cisco)", "china": 0.14},
    "JNPR": {"description": "Networking (Juniper)", "china": 0.20},
    "HPE": {"description": "Edge compute (HPE)", "china": 0.28},
    "NUE": {"description": "Domestic steel (Nucor)", "china": 0.18},
    "X": {"description": "Domestic steel (US Steel)", "china": 0.22},
    "CLF": {"description": "Steel/iron ore (Cleveland-Cliffs)", "china": 0.19},
    "CAT": {"description": "Infrastructure capex (Caterpillar)", "china": 0.28},
    "DE": {"description": "Industrial equipment (Deere)", "china": 0.24},
    "GE": {"description": "Aero engines (GE Aerospace)", "china": 0.22},
    "HWM": {"description": "Aerospace alloys (Howmet)", "china": 0.18},
    "FAST": {"description": "Industrial fasteners (Fastenal)", "china": 0.19},
    "URI": {"description": "US equipment rentals (United Rentals)", "china": 0.12},
    "NSC": {"description": "Rail logistics (Norfolk Southern)", "china": 0.05},
    "PWR": {"description": "Grid engineering (Quanta Services)", "china": 0.08},
    "PH": {"description": "Motion control (Parker Hannifin)", "china": 0.24},
    "SRE": {"description": "US utilities (Sempra)", "china": 0.05},
    "CSX": {"description": "Rail logistics (CSX)", "china": 0.04},
    "TT": {"description": "Climate control (Trane)", "china": 0.20},
}

NEGATIVE_FACTORS = {"max_drawdown_63", "beta_cn"}
REFERENCE_RETURNS: Dict[str, pd.Series] = {}


@lru_cache(maxsize=None)
def _download_history(ticker: str, start_str: str, end_str: str) -> pd.DataFrame:
    data = yf.download(ticker, start=start_str, end=end_str, auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(ticker, axis=1, level=1)
    return data


@dataclass
class StockMetrics:
    ticker: str
    description: str
    sources: Set[str]
    one_year_return: float
    realized_vol: float
    avg_volume: float
    china_share: float
    market_cap: float | None
    factors: Dict[str, float]
    sharpe_5y: float
    max_drawdown_5y: float
    composite: float = 0.0


ETF_CANDIDATES: Dict[str, str] = {}
INDEX_LISTS: Dict[str, object] = {}
THRESHOLDS: Dict[str, float] = {}
FACTOR_WEIGHTS: Dict[str, float] = {}
MANUAL_INCLUSIONS: Dict[str, str] = {}
LOOKBACK_DAYS = 365
VOL_WINDOW = 90
LONG_LOOKBACK_DAYS = 1825
MIN_ONE_YEAR_RETURN = 0.0
MAX_REALIZED_VOL = 30.0
MAX_CHINA_SHARE = 0.10
OUTPUT_DIR = "outputs"
FACTOR_ORDER: List[str] = []


def compute_long_horizon_metrics(ticker: str) -> Tuple[float, float]:
    end = datetime.today()
    start = end - timedelta(days=LONG_LOOKBACK_DAYS)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    data = _download_history(ticker.upper(), start_str, end_str)
    if data.empty or "Adj Close" not in data:
        return 0.0, 0.0
    price_series = data["Adj Close"].copy()
    returns = price_series.pct_change().dropna()
    sharpe = 0.0
    if not returns.empty:
        std = returns.std(ddof=0)
        if std and not np.isnan(std) and std > 0:
            sharpe = float((returns.mean() / std) * np.sqrt(252))
    rolling_max = price_series.cummax()
    drawdown = price_series / rolling_max - 1
    max_drawdown = float(abs(drawdown.min()) * 100) if not drawdown.empty else 0.0
    return sharpe, max_drawdown


def deep_update(base: Dict, overrides: Dict) -> Dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | None = None) -> Dict:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if cfg_path.exists():
        try:
            with cfg_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            deep_update(config, data)
        except Exception as exc:  # pragma: no cover
            print(f"Warning: failed to load config {cfg_path}: {exc}. Using defaults.")
    else:
        if path:
            print(f"Warning: config file {cfg_path} not found. Using defaults.")
    return config


def apply_config(config: Dict) -> None:
    global ETF_CANDIDATES, INDEX_LISTS, THRESHOLDS, FACTOR_WEIGHTS, MANUAL_INCLUSIONS
    global LOOKBACK_DAYS, VOL_WINDOW, LONG_LOOKBACK_DAYS, MIN_ONE_YEAR_RETURN, MAX_REALIZED_VOL, MAX_CHINA_SHARE, FACTOR_ORDER

    ETF_CANDIDATES = config.get("etf_candidates", {})
    INDEX_LISTS = config.get("index_lists", {})
    THRESHOLDS = config.get("thresholds", {})
    FACTOR_WEIGHTS = config.get("weights", {})
    MANUAL_INCLUSIONS = config.get("manual_inclusions", {})

    LOOKBACK_DAYS = int(THRESHOLDS.get("lookback_days", 365))
    VOL_WINDOW = int(THRESHOLDS.get("vol_window", 90))
    MIN_ONE_YEAR_RETURN = float(THRESHOLDS.get("min_one_year_return", 0.0))
    MAX_REALIZED_VOL = float(THRESHOLDS.get("max_realized_vol", 30.0))
    if MAX_REALIZED_VOL <= 1:
        MAX_REALIZED_VOL *= 100
    MAX_CHINA_SHARE = float(THRESHOLDS.get("max_china_share", 0.10))
    LONG_LOOKBACK_DAYS = int(THRESHOLDS.get("long_lookback_days", 1825))
    FACTOR_ORDER = list(FACTOR_WEIGHTS.keys())


def fetch_price_history(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    data = _download_history(ticker.upper(), start_str, end_str)
    return data.copy()


@lru_cache(maxsize=None)
def fetch_market_cap(ticker: str) -> float | None:
    try:
        info = yf.Ticker(ticker).fast_info
        market_cap = getattr(info, "market_cap", None)
        if market_cap is None and isinstance(info, dict):
            market_cap = info.get("market_cap")
        return market_cap
    except Exception:
        return None


@lru_cache(maxsize=None)
def fetch_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}


def get_reference_returns(symbol: str) -> pd.Series:
    symbol = symbol.upper()
    cached = REFERENCE_RETURNS.get(symbol)
    if cached is not None:
        return cached
    data = yf.download(symbol, period="1y", interval="1d", auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(symbol, axis=1, level=1)
    series = data["Adj Close"].pct_change().dropna()
    REFERENCE_RETURNS[symbol] = series
    return series


def compute_factor_values(ticker: str, prices: pd.DataFrame, returns: pd.Series) -> Dict[str, float]:
    price_series = prices["Adj Close"]
    momentum = (price_series.iloc[-1] / price_series.iloc[0] - 1) * 100

    rolling_max = price_series.rolling(63).max()
    drawdown_series = price_series / rolling_max - 1
    max_drawdown = abs(drawdown_series.min() * 100) if not drawdown_series.empty else 0.0

    fx_returns = get_reference_returns("FXI")
    aligned_fx = fx_returns.reindex(returns.index).dropna()
    aligned_returns = returns.reindex(aligned_fx.index).dropna()
    if len(aligned_returns) >= 2 and aligned_fx.var(ddof=0) > 0:
        beta_cn = float(np.cov(aligned_returns, aligned_fx)[0, 1] / aligned_fx.var(ddof=0))
    else:
        beta_cn = 0.0

    info = fetch_info(ticker)
    revenue_growth = float((info.get("revenueGrowth") or 0.0) * 100)
    net_margin = float((info.get("profitMargins") or 0.0) * 100)

    return {
        "momentum_252": float(momentum),
        "max_drawdown_63": float(max_drawdown),
        "beta_cn": beta_cn,
        "revenue_growth": revenue_growth,
        "net_margin": net_margin,
    }


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


def load_candidates() -> Dict[str, Dict[str, object]]:
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
                        "description": meta.get("description", symbol),
                        "sources": set(),
                    },
                )
                entry["sources"].add(f"{sleeve} ({etf})")

    for index_name, sources in INDEX_LISTS.items():
        symbols: List[str] = []
        if isinstance(sources, list):
            symbols = [str(t).upper() for t in sources]
        elif isinstance(sources, str):
            path = Path(sources)
            if path.exists():
                symbols = [line.strip().upper() for line in path.read_text().splitlines() if line.strip()]
            else:
                symbols = [token.strip().upper() for token in sources.split(',') if token.strip()]
        for symbol in symbols:
            if symbol not in BASE_METADATA:
                continue
            meta = BASE_METADATA[symbol]
            entry = tickers.setdefault(
                symbol,
                {
                    "description": meta.get("description", symbol),
                    "sources": set(),
                },
            )
            entry["sources"].add(f"{index_name} (index)")

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

    metrics: List[StockMetrics] = []
    for ticker, meta in candidates.items():
        base_meta = BASE_METADATA.get(ticker)
        if base_meta is None:
            continue
        china_share = float(base_meta.get("china", 0.5))
        if china_share >= MAX_CHINA_SHARE:
            continue

        hist = fetch_price_history(ticker, start, end)
        if hist.empty or "Adj Close" not in hist or "Volume" not in hist:
            continue
        returns = hist["Adj Close"].pct_change().dropna()
        if returns.empty:
            continue

        one_year_return = (hist["Adj Close"].iloc[-1] / hist["Adj Close"].iloc[0] - 1) * 100
        if one_year_return <= MIN_ONE_YEAR_RETURN:
            continue

        realized_vol = returns.tail(VOL_WINDOW).std() * np.sqrt(252) * 100
        if realized_vol >= MAX_REALIZED_VOL:
            continue

        avg_volume = hist["Volume"].tail(VOL_WINDOW).mean()

        factors = compute_factor_values(ticker, hist, returns)
        market_cap = fetch_market_cap(ticker)
        sharpe_5y, max_drawdown_5y = compute_long_horizon_metrics(ticker)

        metrics.append(
            StockMetrics(
                ticker=ticker,
                description=meta.get("description", ticker),
                sources=set(meta.get("sources", set())),
                one_year_return=float(one_year_return),
                realized_vol=float(realized_vol),
                avg_volume=float(avg_volume or 0.0),
                china_share=china_share,
                market_cap=market_cap,
                factors=factors,
                sharpe_5y=sharpe_5y,
                max_drawdown_5y=max_drawdown_5y,
            )
        )

    if not metrics:
        return []

    factor_df = pd.DataFrame([m.factors for m in metrics], index=[m.ticker for m in metrics])
    factor_df = factor_df.reindex(columns=FACTOR_ORDER, fill_value=np.nan).fillna(0.0)

    normalized = factor_df.copy()
    for col in factor_df.columns:
        series = factor_df[col]
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            normalized[col] = 0.0
        else:
            normalized[col] = (series - series.mean()) / std

    for m in metrics:
        score = 0.0
        for factor, weight in FACTOR_WEIGHTS.items():
            val = normalized.at[m.ticker, factor]
            if factor in NEGATIVE_FACTORS:
                val *= -1
            score += val * weight
        m.composite = float(score)

    return metrics


def metrics_to_frame(metrics: Iterable[StockMetrics]) -> pd.DataFrame:
    rows = []
    for m in metrics:
        rows.append(
            {
                "Ticker": m.ticker,
                "Description": m.description,
                "Source Tags": "; ".join(sorted(m.sources)) if m.sources else "N/A",
                "1Y Return %": m.one_year_return,
                "90D Realized Vol %": m.realized_vol,
                "Avg Volume (3M)": m.avg_volume,
                "China Share %": m.china_share * 100,
                "Market Cap ($bn)": (m.market_cap / 1e9) if m.market_cap else None,
                "Momentum 1Y %": m.factors["momentum_252"],
                "Max Drawdown 63d %": m.factors["max_drawdown_63"],
                "Beta vs FXI": m.factors["beta_cn"],
                "Revenue Growth %": m.factors["revenue_growth"],
                "Net Margin %": m.factors["net_margin"],
                "Composite Score": m.composite,
            }
        )
    df = pd.DataFrame(rows)
    df.sort_values(by="Composite Score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.insert(0, "Rank", df.index + 1)
    return df


def display_table(table: pd.DataFrame, limit: int = 10) -> None:
    columns = [
        "Rank",
        "Ticker",
        "Description",
        "Composite Score",
        "Beta vs FXI",
        "China Share %",
        "1Y Return %",
        "90D Realized Vol %",
    ]
    available = [c for c in columns if c in table.columns]
    print(table[available].head(limit).to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def generate_shortlist(
    config_overrides: Dict | None = None,
    config_path: str | None = None,
    quiet: bool = True,
    write_output: bool = False,
    output_path: str | None = None,
) -> tuple[pd.DataFrame, Dict]:
    """Generate the shortlist DataFrame and effective config."""

    config = load_config(config_path)
    if config_overrides:
        deep_update(config, config_overrides)
    apply_config(config)

    candidates = load_candidates()
    if not candidates:
        raise RuntimeError("Candidate universe is empty after aggregation.")

    metrics = build_metrics(candidates)
    if not metrics:
        raise RuntimeError("No equities met the screening thresholds.")

    table = metrics_to_frame(metrics)

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
