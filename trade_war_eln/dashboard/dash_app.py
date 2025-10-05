"""Dash dashboard with interactive Plotly charts.

This app provides interactive payoff and delta plots for the ELN structure and
renders the shortlist table with visible column headers using Dash DataTable.

Run:
    python -m trade_war_eln.dashboard.dash_app
or:
    python trade_war_eln/dashboard/dash_app.py

Dependencies:
    dash plotly pandas numpy yfinance (for shortlist)  
    (eln_core uses only stdlib)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from dash.dash_table import DataTable

# Wire package path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR.parent))

from trade_war_eln.src import eln_core
from trade_war_eln.src.autocall_pricer import AutocallSpec, MCSpec, price_autocall
from trade_war_eln.src.shortlist import (
    DEFAULT_CONFIG,
    DEFAULT_CONFIG_PATH,
    deep_update,
    generate_shortlist,
    load_config,
)


def compute_eln_frames() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stocks = eln_core.build_stocks()
    basket = eln_core.pick_basket()
    market = eln_core.Market(risk_free_rate=0.045, time_to_maturity=1.0, correlation=0.35)
    res = eln_core.price_structure(stocks, basket, market)
    payoff = pd.DataFrame(res.payoff_table, columns=["Basket_Performance", "Payoff_per_$1"]).copy()
    terminal = pd.DataFrame(res.terminal_delta, columns=["Basket_Performance", "Terminal_Delta"]).copy()
    t0 = pd.DataFrame(res.delta_t0, columns=["Basket_Ratio", "Note_Value", "Delta_vs_Basket"]).copy()
    return payoff, terminal, t0


def _clone_stocks_with_vol(stocks: dict, dvol: float = 0.0, scale: float | None = None) -> dict:
    new = {}
    for k, v in stocks.items():
        # dataclass without import; mimic structure
        new[k] = type(v)(
            ticker=v.ticker,
            price=v.price,
            volatility=(v.volatility * scale if scale is not None else v.volatility + dvol),
            dividend_yield=v.dividend_yield,
            china_revenue_share=getattr(v, "china_revenue_share", 0.0),
            commentary=getattr(v, "commentary", ""),
        )
    return new


def compute_static_risk_figs() -> Dict[str, go.Figure]:
    # Base inputs
    stocks0 = eln_core.build_stocks()
    basket = eln_core.pick_basket()
    market0 = eln_core.Market(risk_free_rate=0.045, time_to_maturity=1.0, correlation=0.35)
    res0 = eln_core.price_structure(stocks0, basket, market0)
    pv0 = res0.note_price
    # Delta near x=1
    t0_df = pd.DataFrame(res0.delta_t0, columns=["Basket_Ratio", "Note_Value", "Delta_vs_Basket"])
    try:
        delta_x1 = float(t0_df.loc[abs(t0_df["Basket_Ratio"] - 1.0) < 1e-9, "Delta_vs_Basket"].iloc[0])
    except Exception:
        delta_x1 = float(t0_df.iloc[(t0_df["Basket_Ratio"] - 1.0).abs().argmin()]["Delta_vs_Basket"])  # type: ignore

    # Vega: bump all stock vols by +1 vol point
    stocks_up = _clone_stocks_with_vol(stocks0, dvol=0.01)
    stocks_dn = _clone_stocks_with_vol(stocks0, dvol=-0.01)
    pv_up = eln_core.price_structure(stocks_up, basket, market0).note_price
    pv_dn = eln_core.price_structure(stocks_dn, basket, market0).note_price
    vega_1pt = (pv_up - pv_dn) / 2.0  # $ per 1 vol point

    # Rho: bump rate by 1%
    m_up = eln_core.Market(risk_free_rate=market0.risk_free_rate + 0.01, time_to_maturity=1.0, correlation=market0.correlation)
    m_dn = eln_core.Market(risk_free_rate=market0.risk_free_rate - 0.01, time_to_maturity=1.0, correlation=market0.correlation)
    rho_1pct = (eln_core.price_structure(stocks0, basket, m_up).note_price - eln_core.price_structure(stocks0, basket, m_dn).note_price) / 2.0

    # Corr sensitivity: bump rho by 0.1
    m_rho_up = eln_core.Market(risk_free_rate=market0.risk_free_rate, time_to_maturity=1.0, correlation=min(0.9, market0.correlation + 0.1))
    m_rho_dn = eln_core.Market(risk_free_rate=market0.risk_free_rate, time_to_maturity=1.0, correlation=max(-0.9, market0.correlation - 0.1))
    corr_sens_0p1 = (eln_core.price_structure(stocks0, basket, m_rho_up).note_price - eln_core.price_structure(stocks0, basket, m_rho_dn).note_price) / 2.0

    # Tornado chart
    labels = ["Delta (1% basket)", "Vega (1 vol pt)", "Rho (1% rate)", "Corr (Δρ=0.1)"]
    # Convert delta to $ per 1%: delta * 1% * B0
    delta_1pct = delta_x1 * 0.01 * res0.basket_spot
    values = [delta_1pct, vega_1pt, rho_1pct, corr_sens_0p1]
    fig_tornado = go.Figure(data=[go.Bar(x=values, y=labels, orientation="h", marker_color=["#60a5fa", "#fbbf24", "#34d399", "#f87171"])])
    fig_tornado.update_layout(template="plotly_dark", title="Static ELN: Local Sensitivities (ΔPV)", xaxis_title="ΔPV (USD)", height=320, margin=dict(l=60, r=20, t=50, b=40))

    # PV vs sigma (scale vols) and vs correlation & rates
    sig_scales = [0.8, 0.9, 1.0, 1.1, 1.2]
    pv_sigma = []
    for sc in sig_scales:
        s_scaled = _clone_stocks_with_vol(stocks0, scale=sc)
        pv_sigma.append(eln_core.price_structure(s_scaled, basket, market0).note_price)
    fig_sigma = go.Figure(data=[go.Scatter(x=sig_scales, y=pv_sigma, mode="lines+markers")])
    fig_sigma.update_layout(template="plotly_dark", title="PV vs Volatility Scale", xaxis_title="Vol scale (×)", yaxis_title="PV (USD)", height=320, margin=dict(l=40, r=20, t=50, b=40))

    r_values = [0.02, 0.03, 0.04, 0.05, 0.06]
    pv_rate = []
    for rv in r_values:
        m = eln_core.Market(risk_free_rate=rv, time_to_maturity=1.0, correlation=market0.correlation)
        pv_rate.append(eln_core.price_structure(stocks0, basket, m).note_price)
    fig_rate = go.Figure(data=[go.Scatter(x=r_values, y=pv_rate, mode="lines+markers")])
    fig_rate.update_layout(template="plotly_dark", title="PV vs Risk‑free Rate", xaxis_title="r (continuous)", yaxis_title="PV (USD)", height=320, margin=dict(l=40, r=20, t=50, b=40))

    rho_values = [0.0, 0.2, 0.35, 0.5, 0.7]
    pv_rho = []
    for rh in rho_values:
        m = eln_core.Market(risk_free_rate=market0.risk_free_rate, time_to_maturity=1.0, correlation=rh)
        pv_rho.append(eln_core.price_structure(stocks0, basket, m).note_price)
    fig_rho = go.Figure(data=[go.Scatter(x=rho_values, y=pv_rho, mode="lines+markers")])
    fig_rho.update_layout(template="plotly_dark", title="PV vs Correlation", xaxis_title="ρ", yaxis_title="PV (USD)", height=320, margin=dict(l=40, r=20, t=50, b=40))

    return {
        "tornado": fig_tornado,
        "sigma": fig_sigma,
        "rate": fig_rate,
        "rho": fig_rho,
    }


def payoff_figure(payoff_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=payoff_df["Basket_Performance"],
            y=payoff_df["Payoff_per_$1"],
            mode="lines",
            name="ELN payoff",
        )
    )
    for x in (0.8, 1.0):
        fig.add_vline(x=x, line=dict(color="gray", width=1, dash="dash"))
    fig.update_layout(
        title="Bullish ELN Payoff",
        xaxis_title="Basket performance (Final / Initial)",
        yaxis_title="Payoff per $1 notional",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def terminal_delta_figure(delta_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=delta_df["Basket_Performance"],
            y=delta_df["Terminal_Delta"],
            mode="lines",
            line_shape="hv",
            name="Terminal delta",
        )
    )
    fig.update_layout(
        title="Terminal Delta Profile",
        xaxis_title="Basket performance (Final / Initial)",
        yaxis_title="dPayoff/dBasket",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def t0_delta_figure(t0_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t0_df["Basket_Ratio"],
            y=t0_df["Delta_vs_Basket"],
            mode="lines",
            name="Time-0 delta",
        )
    )
    fig.update_layout(
        title="Time-0 Delta Sensitivity",
        xaxis_title="Current basket level / Initial",
        yaxis_title="Delta vs basket",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def _prepare_overrides(values: Dict) -> Dict:
    overrides: Dict = {"thresholds": {}, "weights": {}, "manual_inclusions": {}, "etf_candidates": {}}
    min_ret = values.get("min_return")
    max_vol = values.get("max_vol")
    max_china = values.get("max_china")
    if min_ret not in (None, ""):
        v = float(min_ret)
        if 0 < v <= 1:
            v *= 100.0
        overrides["thresholds"]["min_one_year_return"] = v
    if max_vol not in (None, ""):
        overrides["thresholds"]["max_realized_vol"] = float(max_vol)
    if max_china not in (None, ""):
        overrides["thresholds"]["max_china_share"] = float(max_china)

    for key in ["momentum_252", "max_drawdown_63", "beta_cn", "revenue_growth", "net_margin"]:
        v = values.get(f"weight_{key}")
        if v not in (None, ""):
            overrides["weights"][key] = float(v)

    manual = values.get("manual_inclusions", "").strip()
    if manual:
        manual_map: Dict[str, str] = {}
        for line in manual.splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                t, d = line.split(":", 1)
                manual_map[t.strip().upper()] = d.strip()
            else:
                manual_map[line.upper()] = line.upper()
        overrides["manual_inclusions"] = manual_map

    selected_etfs = values.get("etfs", [])
    if selected_etfs:
        etf_map = load_config().get("etf_candidates", {})
        overrides["etf_candidates"] = {etf: etf_map.get(etf, etf) for etf in selected_etfs}

    # Clean empties
    overrides = {k: v for k, v in overrides.items() if v}
    return overrides


def shortlist_table_and_charts(config_overrides: Dict | None = None) -> Tuple[pd.DataFrame, Dict, Dict[str, go.Figure]]:
    table = pd.DataFrame()
    effective_config: Dict = load_config()
    try:
        table, effective_config = generate_shortlist(config_overrides=config_overrides, quiet=True)
    except Exception:
        # Keep empty table; charts fall back to empty figs
        pass

    figs: Dict[str, go.Figure] = {}
    # Weight allocation
    weights = effective_config.get("weights", {})
    if weights:
        figs["weights"] = go.Figure(
            data=[
                go.Pie(
                    labels=[
                        "Momentum (1Y)",
                        "Max Drawdown",
                        "Beta vs FXI",
                        "Revenue Growth",
                        "Net Margin",
                    ],
                    values=[
                        weights.get("momentum_252", 0) * 100,
                        weights.get("max_drawdown_63", 0) * 100,
                        weights.get("beta_cn", 0) * 100,
                        weights.get("revenue_growth", 0) * 100,
                        weights.get("net_margin", 0) * 100,
                    ],
                    hole=0.6,
                )
            ]
        )
        figs["weights"].update_layout(template="plotly_dark", showlegend=True, margin=dict(l=10, r=10, t=40, b=10), title="Weight Allocation (%)")

    # Bar charts from top rows
    if not table.empty:
        top_rows = table.head(min(6, len(table)))
        figs["returns"] = go.Figure(data=[
            go.Bar(x=top_rows["Ticker"], y=top_rows["1Y Return %"], name="1Y Return %", marker_color="#2563eb")
        ])
        figs["returns"].update_layout(template="plotly_dark", margin=dict(l=40, r=20, t=40, b=40), title="Top Returns")

        figs["vols"] = go.Figure(data=[
            go.Bar(x=top_rows["Ticker"], y=top_rows["90D Realized Vol %"], name="90d Vol %", marker_color="#f97316")
        ])
        figs["vols"].update_layout(template="plotly_dark", margin=dict(l=40, r=20, t=40, b=40), title="Realised Volatility")

        if "Beta vs FXI" in top_rows.columns:
            figs["betas"] = go.Figure(data=[
                go.Bar(x=top_rows["Ticker"], y=top_rows["Beta vs FXI"], name="Beta vs FXI", marker_color="#10b981")
            ])
            figs["betas"].update_layout(template="plotly_dark", margin=dict(l=40, r=20, t=40, b=40), title="China Beta (vs FXI)")

    return table, effective_config, figs


def build_app() -> Dash:
    payoff_df, term_df, t0_df = compute_eln_frames()
    table, effective_config, figs = shortlist_table_and_charts()

    app = Dash(__name__, title="Trade-War ELN Dash")
    app.layout = html.Div(
        [
            html.H2("Trade-War ELN – Interactive Dashboard"),
            # Horizontal filter bar spanning full width
            html.Div([
                html.Div([
                    html.Label("View"),
                    dcc.RadioItems(
                        id="view_mode",
                        options=[
                            {"label": "Static ELN", "value": "static"},
                            {"label": "Autocall", "value": "autocall"},
                        ],
                        value="static",
                        labelStyle={"display": "inline-block", "marginRight": "0.75rem"},
                        style={"color": "#e5e7eb"},
                    ),
                ]),
                html.Div([
                    html.Label("Min 1Y Return (%)"),
                    dcc.Input(id="min_return", type="number", step=0.1, placeholder=str(effective_config["thresholds"].get("min_one_year_return", 0.0)), debounce=True, style={"width": "100%"}),
                ]),
                html.Div([
                    html.Label("Max 90d Realised Vol (%)"),
                    dcc.Input(id="max_vol", type="number", step=0.1, placeholder=str(effective_config["thresholds"].get("max_realized_vol", 30.0)), debounce=True, style={"width": "100%"}),
                ]),
                html.Div([
                    html.Label("Max China Share (0-1)"),
                    dcc.Input(id="max_china", type="number", step=0.01, placeholder=str(effective_config["thresholds"].get("max_china_share", 0.1)), debounce=True, style={"width": "100%"}),
                ]),
                html.Div([
                    html.Label("Wgt: Momentum"),
                    dcc.Input(id="w_momentum_252", type="number", step=0.01, placeholder=str(effective_config["weights"].get("momentum_252", 0.3)), debounce=True, style={"width": "100%"}),
                ]),
                html.Div([
                    html.Label("Wgt: Max Drawdown"),
                    dcc.Input(id="w_max_drawdown_63", type="number", step=0.01, placeholder=str(effective_config["weights"].get("max_drawdown_63", 0.1)), debounce=True, style={"width": "100%"}),
                ]),
                html.Div([
                    html.Label("Wgt: Beta vs FXI"),
                    dcc.Input(id="w_beta_cn", type="number", step=0.01, placeholder=str(effective_config["weights"].get("beta_cn", 0.4)), debounce=True, style={"width": "100%"}),
                ]),
                html.Div([
                    html.Label("Wgt: Revenue"),
                    dcc.Input(id="w_revenue_growth", type="number", step=0.01, placeholder=str(effective_config["weights"].get("revenue_growth", 0.12)), debounce=True, style={"width": "100%"}),
                ]),
                html.Div([
                    html.Label("Wgt: Net Margin"),
                    dcc.Input(id="w_net_margin", type="number", step=0.01, placeholder=str(effective_config["weights"].get("net_margin", 0.08)), debounce=True, style={"width": "100%"}),
                ]),
                html.Div([
                    html.Button("Run Shortlist", id="run_btn", style={"marginTop": "1.55rem", "width": "100%"}),
                ]),
                html.Div(id="error_box", style={"color": "#fca5a5", "alignSelf": "end"}),
            ], style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(170px, 1fr))",
                "gap": "0.75rem",
                "alignItems": "end",
                "marginBottom": "1rem",
            }),
            html.Div(
                [
                    # Left panel removed to free horizontal space
                    html.Div([], style={"display": "none"}),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div([dcc.Graph(id="payoff_fig", figure=payoff_figure(payoff_df))], style={"minWidth": "0"}),
                                    html.Div([dcc.Graph(id="terminal_fig", figure=terminal_delta_figure(term_df))], style={"minWidth": "0"}),
                                    html.Div([dcc.Graph(id="t0_fig", figure=t0_delta_figure(t0_df))], style={"minWidth": "0"}),
                                ],
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(auto-fit, minmax(300px, 1fr))",
                                    "gap": "1rem",
                                }, id="static_section"),
                            html.Br(),
                            html.H4("Static ELN – Risk (Sensitivities)"),
                            html.Div([
                                html.Div([dcc.Graph(figure=compute_static_risk_figs()["tornado"])], style={"minWidth": "0"}),
                                html.Div([dcc.Graph(figure=compute_static_risk_figs()["sigma"])], style={"minWidth": "0"}),
                                html.Div([dcc.Graph(figure=compute_static_risk_figs()["rho"])], style={"minWidth": "0"}),
                                html.Div([dcc.Graph(figure=compute_static_risk_figs()["rate"])], style={"minWidth": "0"}),
                            ], style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(auto-fit, minmax(260px, 1fr))",
                                "gap": "1rem",
                            }),
                            html.Br(),
                            html.Div(
                                [
                                    html.Div([dcc.Graph(id="weights_fig", figure=figs.get("weights", go.Figure()))], style={"minWidth": "0"}),
                                    html.Div([dcc.Graph(id="returns_fig", figure=figs.get("returns", go.Figure()))], style={"minWidth": "0"}),
                                    html.Div([dcc.Graph(id="vols_fig", figure=figs.get("vols", go.Figure()))], style={"minWidth": "0"}),
                                    html.Div([dcc.Graph(id="betas_fig", figure=figs.get("betas", go.Figure()))], style={"minWidth": "0"}),
                                ],
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(auto-fit, minmax(260px, 1fr))",
                                    "gap": "1rem",
                                },
                            ),
                            html.Br(),
                            html.H4("Shortlist Results"),
                            DataTable(
                                id="shortlist_table",
                                columns=[{"name": c, "id": c} for c in (table.columns.tolist() if not table.empty else [])],
                                data=(table.to_dict("records") if not table.empty else []),
                                page_size=10,
                                style_table={"overflowX": "auto", "width": "100%", "minWidth": "100%"},
                                style_cell={"backgroundColor": "#0b1120", "color": "#e5e7eb", "minWidth": 60, "maxWidth": 280, "whiteSpace": "nowrap", "textOverflow": "ellipsis"},
                                style_header={"backgroundColor": "#1f2937", "color": "#f8fafc", "fontWeight": "600"},
                            ),
                            html.Br(),
                            html.H4("Autocall (Monte Carlo)"),
                            html.Div([
                                html.Div([
                                    html.Label("Risk-free r"),
                                    dcc.Input(id="ac_r", type="number", step=0.001, value=0.03, debounce=True),
                                ], style={"marginRight": "0.75rem"}),
                                html.Div([
                                    html.Label("Dividend q (basket)"),
                                    dcc.Input(id="ac_q", type="number", step=0.001, placeholder="from basket", debounce=True),
                                ], style={"marginRight": "0.75rem"}),
                                html.Div([
                                    html.Label("Vol σ (basket)"),
                                    dcc.Input(id="ac_sigma", type="number", step=0.001, placeholder="from basket", debounce=True),
                                ], style={"marginRight": "0.75rem"}),
                                html.Div([
                                    html.Label("Maturity T (y)"),
                                    dcc.Input(id="ac_T", type="number", step=0.5, value=5.0, debounce=True),
                                ], style={"marginRight": "0.75rem"}),
                                html.Div([
                                    html.Label("KI level (×S0)"),
                                    dcc.Input(id="ac_kib", type="number", step=0.01, value=0.6, debounce=True),
                                ], style={"marginRight": "0.75rem"}),
                                html.Div([
                                    html.Label("Bump ε"),
                                    dcc.Input(id="ac_eps", type="number", step=0.005, value=0.01, debounce=True),
                                ]),
                                html.Button("Run Autocall", id="run_autocall", style={"marginLeft": "0.75rem", "height": "38px", "alignSelf": "end"}),
                                html.Div(id="ac_error", style={"color": "#fca5a5", "marginLeft": "1rem", "alignSelf": "end"}),
                            ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "end"}),
                            html.Div([
                                dcc.Graph(id="ac_payoff_fig", style={"height": "360px"}),
                                dcc.Graph(id="ac_delta_fig", style={"height": "320px"}),
                                dcc.Graph(id="ac_timing_fig", style={"height": "320px"}),
                                dcc.Graph(id="ac_ki_fig", style={"height": "320px"}),
                            ], style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "1rem"}, id="autocall_section"),
                        ],
                        style={"flex": "1 1 auto", "minWidth": "0"},
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap", "gap": "1rem"},
            ),
        ],
        style={"padding": "1rem", "backgroundColor": "#0b1120", "color": "#e5e7eb", "maxWidth": "1400px", "margin": "0 auto", "overflowX": "hidden", "width": "100%"},
    )

    # Callback
    @app.callback(
        Output("shortlist_table", "columns"),
        Output("shortlist_table", "data"),
        Output("weights_fig", "figure"),
        Output("returns_fig", "figure"),
        Output("vols_fig", "figure"),
        Output("betas_fig", "figure"),
        Output("error_box", "children"),
        Input("run_btn", "n_clicks"),
        State("min_return", "value"),
        State("max_vol", "value"),
        State("max_china", "value"),
        State("w_momentum_252", "value"),
        State("w_max_drawdown_63", "value"),
        State("w_beta_cn", "value"),
        State("w_revenue_growth", "value"),
        State("w_net_margin", "value"),
        prevent_initial_call=True,
    )
    def on_run(_n, min_return, max_vol, max_china, w_mom, w_mdd, w_beta, w_rev, w_margin):
        values = {
            "min_return": min_return,
            "max_vol": max_vol,
            "max_china": max_china,
            "weight_momentum_252": w_mom,
            "weight_max_drawdown_63": w_mdd,
            "weight_beta_cn": w_beta,
            "weight_revenue_growth": w_rev,
            "weight_net_margin": w_margin,
            "manual_inclusions": "",
            "etfs": [],  # keep defaults
        }
        # Convert weight keys to expected mapping
        normalized = {
            "min_return": values["min_return"],
            "max_vol": values["max_vol"],
            "max_china": values["max_china"],
            "weight_momentum_252": values["weight_momentum_252"],
            "weight_max_drawdown_63": values["weight_max_drawdown_63"],
            "weight_beta_cn": values["weight_beta_cn"],
            "weight_revenue_growth": values["weight_revenue_growth"],
            "weight_net_margin": values["weight_net_margin"],
            "manual_inclusions": values["manual_inclusions"],
        }

        overrides = _prepare_overrides(normalized)
        try:
            table, eff_cfg, figs = shortlist_table_and_charts(overrides)
            columns = [{"name": c, "id": c} for c in table.columns] if not table.empty else []
            data = table.to_dict("records") if not table.empty else []
            return (
                columns,
                data,
                figs.get("weights", go.Figure()),
                figs.get("returns", go.Figure()),
                figs.get("vols", go.Figure()),
                figs.get("betas", go.Figure()),
                "",
            )
        except Exception as exc:  # pragma: no cover
            return [], [], go.Figure(), go.Figure(), go.Figure(), go.Figure(), f"Error: {exc}"

    # Toggle static/autocall visibility
    @app.callback(
        Output("static_section", "style"),
        Output("autocall_section", "style"),
        Input("view_mode", "value"),
    )
    def on_mode(mode):
        if mode == "autocall":
            return {"display": "none"}, {"display": "block"}
        return {"display": "block"}, {"display": "none"}

    # Autocall callback
    @app.callback(
        Output("ac_payoff_fig", "figure"),
        Output("ac_delta_fig", "figure"),
        Output("ac_timing_fig", "figure"),
        Output("ac_ki_fig", "figure"),
        Output("ac_error", "children"),
        Input("run_autocall", "n_clicks"),
        State("ac_r", "value"),
        State("ac_q", "value"),
        State("ac_sigma", "value"),
        State("ac_T", "value"),
        State("ac_kib", "value"),
        State("ac_eps", "value"),
        prevent_initial_call=True,
    )
    def on_run_autocall(_n, r, q, sigma, T, kib, eps):
        try:
            # Default basket inputs
            stocks = eln_core.build_stocks()
            basket = eln_core.pick_basket()
            market = eln_core.Market(risk_free_rate=0.045, time_to_maturity=1.0, correlation=0.35)
            S0, qB, sigmaB = eln_core.effective_basket(stocks, basket, market)
            r = float(r if r is not None else 0.03)
            q = float(q if q is not None else qB)
            sigma = float(sigma if sigma is not None else sigmaB)
            T = float(T if T is not None else 5.0)
            kib = float(kib if kib is not None else 0.6)
            eps = float(eps if eps is not None else 0.01)

            # Observation schedule: semiannual
            obs = [0.5 * i for i in range(1, int(round(T / 0.5)) + 1)]
            acb_rel = [1.0] * len(obs)
            coupons = [0.025837] * (len(obs) - 1) + [0.141]
            # Freeze absolute barriers at baseline S0 so expected PV varies with starting level
            acb_abs = [x * S0 for x in acb_rel]
            kib_abs = kib * S0

            # Expected payoff vs basket ratio
            ratios = np.linspace(0.6, 1.4, 19)
            pvs = []
            deltas = []
            mc = MCSpec(n_paths=8000, steps_per_year=252, seed=42)
            for rratio in ratios:
                S_init = S0 * float(rratio)
                spec = AutocallSpec(
                    S0=S_init, r=r, q=q, sigma=sigma, T=T,
                    obs_times=obs, ac_barriers=acb_abs, coupons=coupons,
                    kib_level=kib_abs, kib_direction="down", notional=1.0,
                    barriers_relative=False,
                )
                pv, _ = price_autocall(spec, mc)
                pvs.append(pv)
                # local delta via central bump at this ratio
                spec_up = AutocallSpec(S0=S_init * (1 + eps), r=r, q=q, sigma=sigma, T=T, obs_times=obs, ac_barriers=acb_abs, coupons=coupons, kib_level=kib_abs, kib_direction="down", notional=1.0, barriers_relative=False)
                spec_dn = AutocallSpec(S0=S_init * (1 - eps), r=r, q=q, sigma=sigma, T=T, obs_times=obs, ac_barriers=acb_abs, coupons=coupons, kib_level=kib_abs, kib_direction="down", notional=1.0, barriers_relative=False)
                pv_up, _ = price_autocall(spec_up, mc)
                pv_dn, _ = price_autocall(spec_dn, mc)
                deltas.append((pv_up - pv_dn) / (2 * S_init * eps))

            # Call timing and KI at S0
            spec0 = AutocallSpec(S0=S0, r=r, q=q, sigma=sigma, T=T, obs_times=obs, ac_barriers=acb_abs, coupons=coupons, kib_level=kib_abs, kib_direction="down", notional=1.0, barriers_relative=False)
            _, det = price_autocall(spec0, mc)

            # Figures
            fig_pay = go.Figure()
            fig_pay.add_trace(go.Scatter(x=ratios, y=pvs, mode="lines", name="Expected PV"))
            fig_pay.add_vline(x=1.0, line=dict(dash="dash", color="#64748b"))
            fig_pay.update_layout(template="plotly_dark", title="Autocall: Expected PV vs Basket Ratio", xaxis_title="Basket ratio (S/S0)", yaxis_title="PV per $1", height=360, margin=dict(l=40,r=20,t=60,b=40))

            fig_delta = go.Figure(data=[go.Scatter(x=ratios, y=deltas, mode="lines", name="Delta")])
            fig_delta.add_vline(x=1.0, line=dict(dash="dash", color="#64748b"))
            fig_delta.update_layout(template="plotly_dark", title="Autocall: Time‑0 Delta vs Basket Ratio", xaxis_title="Basket ratio (S/S0)", yaxis_title="Delta", height=320, margin=dict(l=40,r=20,t=50,b=40))

            labels = [f"t{i+1}" for i in range(len(det.get("call_probs", [])))]
            fig_timing = go.Figure(data=[go.Bar(x=labels, y=det.get("call_probs", []), name="Call probability")])
            fig_timing.update_layout(template="plotly_dark", title="Autocall Timing Probabilities", xaxis_title="Observation", yaxis_title="Probability", height=320, margin=dict(l=40,r=20,t=50,b=40))

            fig_ki = go.Figure(data=[go.Pie(labels=["Knock-in hit", "No knock-in"], values=[det.get("ki_rate", 0.0), 1 - det.get("ki_rate", 0.0)])])
            fig_ki.update_layout(template="plotly_dark", title="Knock-in vs No Knock-in", height=320, margin=dict(l=40,r=20,t=50,b=40))

            return fig_pay, fig_delta, fig_timing, fig_ki, ""
        except Exception as exc:  # pragma: no cover
            return go.Figure(), go.Figure(), go.Figure(), go.Figure(), f"Autocall error: {exc}"

    return app


def main() -> None:
    app = build_app()
    run = getattr(app, "run", None)
    if callable(run):
        run(debug=True)
    else:  # pragma: no cover
        app.run_server(debug=True)


if __name__ == "__main__":
    main()
