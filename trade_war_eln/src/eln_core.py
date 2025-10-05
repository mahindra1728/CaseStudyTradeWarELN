"""Minimal dependency pricing for the bullish ELN structure.

This module mirrors the logic in ``eln_analysis`` but only uses the Python
standard library. It prints key outputs and writes CSV snapshots so results are
available even when optional packages are absent.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


NOTIONAL = 10_000_000


@dataclass
class Stock:
    ticker: str
    price: float
    volatility: float
    dividend_yield: float
    china_revenue_share: float
    commentary: str


@dataclass
class Basket:
    first: str
    second: str
    weights: Tuple[float, float] = (0.5, 0.5)


@dataclass
class Market:
    risk_free_rate: float
    time_to_maturity: float
    correlation: float


@dataclass
class Results:
    note_price: float
    basket_spot: float
    basket_volatility: float
    basket_dividend: float
    component_values: Dict[str, float]
    payoff_table: List[Tuple[float, float]]
    terminal_delta: List[Tuple[float, float]]
    delta_t0: List[Tuple[float, float, float]]


# --- Analytics helpers ----------------------------------------------------


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    time_to_maturity: float,
) -> Tuple[float, float]:
    sqrt_t = math.sqrt(time_to_maturity)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility ** 2) * time_to_maturity
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    discount_factor = math.exp(-rate * time_to_maturity)
    dividend_discount = math.exp(-dividend_yield * time_to_maturity)
    price = spot * dividend_discount * norm_cdf(d1) - strike * discount_factor * norm_cdf(d2)
    delta = dividend_discount * norm_cdf(d1)
    return price, delta


def payoff_function(performance_ratio: float) -> float:
    if performance_ratio >= 1.0:
        return 2.0 * performance_ratio - 1.0
    if performance_ratio >= 0.8:
        return 1.0
    return performance_ratio + 0.2


def terminal_delta(performance_ratio: float) -> float:
    if performance_ratio < 0.8:
        return 1.0
    if performance_ratio < 1.0:
        return 0.0
    return 2.0


# --- Business logic ------------------------------------------------------


def build_stocks() -> Dict[str, Stock]:
    return {
        "LMT": Stock(
            ticker="LMT",
            price=470.0,
            volatility=0.22,
            dividend_yield=0.027,
            china_revenue_share=0.03,
            commentary=(
                "Defense prime with minimal China revenue exposure; poised to benefit from "
                "elevated Indo-Pacific security spending under aggressive trade stances."
            ),
        ),
        "INTC": Stock(
            ticker="INTC",
            price=34.0,
            volatility=0.35,
            dividend_yield=0.015,
            china_revenue_share=0.22,
            commentary=(
                "U.S.-based semiconductor manufacturer supported by CHIPS Act incentives and "
                "strategic realignment of advanced node supply chains."
            ),
        ),
        "CSCO": Stock(
            ticker="CSCO",
            price=48.0,
            volatility=0.24,
            dividend_yield=0.029,
            china_revenue_share=0.05,
            commentary=(
                "Critical networking supplier benefiting from federal security procurement and "
                "restrictions on Chinese telecom equipment."
            ),
        ),
        "NUE": Stock(
            ticker="NUE",
            price=170.0,
            volatility=0.30,
            dividend_yield=0.015,
            china_revenue_share=0.02,
            commentary=(
                "Domestic steel leader with electric arc capacity that benefits from U.S. infrastructure "
                "and reshoring themes amid tariff barriers."
            ),
        ),
    }


def pick_basket() -> Basket:
    return Basket(first="LMT", second="INTC")


def effective_basket(stocks: Dict[str, Stock], basket: Basket, market: Market) -> Tuple[float, float, float]:
    w1, w2 = basket.weights
    s1 = stocks[basket.first]
    s2 = stocks[basket.second]
    spot = w1 * s1.price + w2 * s2.price
    dividend = (w1 * s1.price * s1.dividend_yield + w2 * s2.price * s2.dividend_yield) / spot
    variance = (
        (w1 * s1.volatility) ** 2
        + (w2 * s2.volatility) ** 2
        + 2 * w1 * w2 * s1.volatility * s2.volatility * market.correlation
    )
    return spot, dividend, math.sqrt(variance)


def price_structure(stocks: Dict[str, Stock], basket: Basket, market: Market) -> Results:
    spot, dividend, volatility = effective_basket(stocks, basket, market)
    weight_spot = NOTIONAL / spot
    strike_80 = 0.8 * spot
    strike_100 = spot
    call_80, delta_80 = black_scholes_call(spot, strike_80, market.risk_free_rate, dividend, volatility, market.time_to_maturity)
    call_100, delta_100 = black_scholes_call(spot, strike_100, market.risk_free_rate, dividend, volatility, market.time_to_maturity)

    discount_factor = math.exp(-market.risk_free_rate * market.time_to_maturity)

    components = {
        "Long basket": weight_spot * spot,
        "Long 20% ZCB": 0.2 * NOTIONAL * discount_factor,
        "Short call 80%": -weight_spot * call_80,
        "Long 2 calls 100%": 2.0 * weight_spot * call_100,
    }

    note_price = sum(components.values())

    payoff_table = [(x, payoff_function(x)) for x in frange(0.4, 1.61, 0.02)]
    terminal_delta_profile = [(x, terminal_delta(x)) for x in frange(0.4, 1.61, 0.02)]

    delta_t0 = []
    for ratio in frange(0.6, 1.41, 0.02):
        current_spot = spot * ratio
        call_80_s, delta_80_s = black_scholes_call(
            current_spot,
            strike_80,
            market.risk_free_rate,
            dividend,
            volatility,
            market.time_to_maturity,
        )
        call_100_s, delta_100_s = black_scholes_call(
            current_spot,
            strike_100,
            market.risk_free_rate,
            dividend,
            volatility,
            market.time_to_maturity,
        )
        price = (
            weight_spot * current_spot
            + 0.2 * NOTIONAL * discount_factor
            - weight_spot * call_80_s
            + 2.0 * weight_spot * call_100_s
        )
        delta = weight_spot - weight_spot * delta_80_s + 2.0 * weight_spot * delta_100_s
        delta_t0.append((ratio, price, delta))

    return Results(
        note_price=note_price,
        basket_spot=spot,
        basket_volatility=volatility,
        basket_dividend=dividend,
        component_values=components,
        payoff_table=payoff_table,
        terminal_delta=terminal_delta_profile,
        delta_t0=delta_t0,
    )


def frange(start: float, stop: float, step: float) -> Iterable[float]:
    value = start
    while value < stop - 1e-9:
        yield round(value, 6)
        value += step


def save_csv(path: Path, headers: List[str], rows: Iterable[Iterable[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


BASE_DIR = Path(__file__).resolve().parent.parent


def main() -> None:
    stocks = build_stocks()
    basket = pick_basket()
    market = Market(risk_free_rate=0.045, time_to_maturity=1.0, correlation=0.35)
    results = price_structure(stocks, basket, market)

    print("Note present value: ${:,.2f}".format(results.note_price))
    print("Basket spot: {:.2f}".format(results.basket_spot))
    print("Basket volatility: {:.2%}".format(results.basket_volatility))
    print("Effective dividend yield: {:.2%}".format(results.basket_dividend))
    print("Component contribution:")
    for name, pv in results.component_values.items():
        print(f"  {name:<20s} ${pv:,.2f}")

    out_dir = BASE_DIR / "outputs"
    save_csv(out_dir / "payoff_profile.csv", ["Performance", "Payoff_per_$1"], results.payoff_table)
    save_csv(out_dir / "terminal_delta.csv", ["Performance", "Terminal_Delta"], results.terminal_delta)
    save_csv(out_dir / "delta_t0.csv", ["Basket_Ratio", "Note_Value", "Delta_vs_Basket"], results.delta_t0)

    # Summary text for quick reference
    summary_path = out_dir / "summary.txt"
    with summary_path.open("w") as handle:
        handle.write("Bullish ELN pricing summary\n")
        handle.write(f"Note PV: ${results.note_price:,.2f}\n")
        handle.write(f"Basket spot: {results.basket_spot:.2f}\n")
        handle.write(f"Basket volatility: {results.basket_volatility:.2%}\n")
        handle.write(f"Basket dividend yield: {results.basket_dividend:.2%}\n")
        handle.write("Component breakdown:\n")
        for name, pv in results.component_values.items():
            handle.write(f"  {name}: ${pv:,.2f}\n")


if __name__ == "__main__":
    main()
