"""Flask dashboard for interactive ELN shortlist tuning and Q&A."""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from flask import Flask, flash, render_template, request

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR.parent))

from trade_war_eln.src.shortlist import (
    DEFAULT_CONFIG,
    DEFAULT_CONFIG_PATH,
    deep_update,
    generate_shortlist,
    load_config,
)
from trade_war_eln.src.eln_analysis import generate_outputs as generate_eln_outputs

OUTPUT_DIR = BASE_DIR / "outputs"

app = Flask(__name__)
app.secret_key = "trade-war-eln"  # for flash messages; replace in production


def _prepare_overrides(form: dict) -> dict:
    overrides: dict = {"thresholds": {}, "weights": {}, "manual_inclusions": {}, "etf_candidates": {}}

    # Thresholds
    min_return = form.get("min_return")
    max_vol = form.get("max_vol")
    max_china = form.get("max_china")
    if min_return:
        overrides["thresholds"]["min_one_year_return"] = float(min_return)
    if max_vol:
        overrides["thresholds"]["max_realized_vol"] = float(max_vol)
    if max_china:
        overrides["thresholds"]["max_china_share"] = float(max_china)

    # Weights
    for key in ["policy", "china", "returns", "liquidity", "volatility"]:
        value = form.get(f"weight_{key}")
        if value:
            overrides["weights"][key] = float(value)

    # Manual inclusions (format: TKR:Description per line)
    manual_text = form.get("manual_inclusions", "").strip()
    if manual_text:
        manual_entries = {}
        for line in manual_text.splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                ticker, desc = line.split(":", 1)
                manual_entries[ticker.strip().upper()] = desc.strip()
            else:
                manual_entries[line.upper()] = line.upper()
        overrides["manual_inclusions"] = manual_entries

    # ETF selection
    selected_etfs = form.getlist("etf_candidates")
    if selected_etfs:
        etf_map = load_config().get("etf_candidates", {})
        overrides["etf_candidates"] = {
            etf: etf_map.get(etf, etf)
            for etf in selected_etfs
        }

    # Clean empty dicts
    overrides = {k: v for k, v in overrides.items() if v}
    return overrides


def _load_payoff_snapshot() -> Tuple[str | None, str]:
    payoff_img = None
    message = (
        "Bullish ELN payoff: 2x upside above par, full principal protection between "
        "80-100%, and 20% downside cushion (x + 0.2) below 80%."
    )
    img_path = OUTPUT_DIR / "payoff_profile.png"
    if not img_path.exists():
        try:
            generate_eln_outputs(OUTPUT_DIR)
        except Exception as exc:
            return None, f"Run analytics failed: {exc}"
    if img_path.exists():
        payoff_img = base64.b64encode(img_path.read_bytes()).decode("ascii")
    return payoff_img, message


def _load_pricing_summary() -> Dict[str, str]:
    summary_path = OUTPUT_DIR / "summary.txt"
    if not summary_path.exists():
        return {
            "note_price": "Run pricing scripts (`src/eln_core.py` or `src/eln_analysis.py`) to refresh summary.txt.",
            "components": "",
        }
    lines = summary_path.read_text().strip().splitlines()
    components = []
    note_price = ""
    for line in lines:
        if line.startswith("Note PV"):
            note_price = line.replace("Note PV:", "").strip()
        if line.startswith("  "):
            components.append(line.strip())
    return {
        "note_price": note_price,
        "components": "\n".join(components),
    }


def _load_delta_summary() -> Dict[str, str]:
    terminal_path = OUTPUT_DIR / "terminal_delta.csv"
    t0_path = OUTPUT_DIR / "delta_t0.csv"
    if not terminal_path.exists() or not t0_path.exists():
        return {
            "message": "Run `src/eln_analysis.py` to regenerate delta tables and charts.",
            "terminal": None,
            "t0": None,
        }
    terminal = pd.read_csv(terminal_path)
    t0 = pd.read_csv(t0_path)
    terminal_summary = (
        f"Terminal delta ranges {terminal['Terminal_Delta'].min():.0f} to "
        f"{terminal['Terminal_Delta'].max():.0f} with flat zone between 0.8-1.0."
    )
    t0_summary = (
        f"Time-0 delta spans {t0['Delta_vs_Basket'].min():.2f} to "
        f"{t0['Delta_vs_Basket'].max():.2f}; gamma increases past par due to the long 2x calls."
    )
    return {
        "message": "",
        "terminal": terminal_summary,
        "t0": t0_summary,
    }


def _answer_pack(table: pd.DataFrame, config: Dict) -> Dict[str, str]:
    top4 = table.head(4)
    thresholds = config["thresholds"]
    weights = config["weights"]

    filter_lines = [
        f"Positive 1Y return ≥ {thresholds['min_one_year_return']:.1f}%",
        f"90d realised vol ≤ {thresholds['max_realized_vol']:.1f}%",
        f"China revenue ≤ {thresholds['max_china_share']:.0%}",
        "Liquidity via 3M average volume and policy scores from BASE_METADATA.",
    ]
    stock_rationale = []
    for _, row in top4.iterrows():
        stock_rationale.append(
            f"{row['Ticker']} – {row['Description']} | Source: {row['Source ETFs']} | "
            f"Return {row['1Y Return %']:.2f}% | Vol {row['90D Realized Vol %']:.2f}% | China {row['China Share']:.2%}"
        )

    payoff_img, payoff_message = _load_payoff_snapshot()
    pricing = _load_pricing_summary()
    delta = _load_delta_summary()

    best_two = table.head(2)
    best_two_lines = [
        f"{row['Ticker']} – Composite {row['Composite Score']:.2f}, Policy {row['Policy Score']:.2f}, China {row['China Share']:.2%}"
        for _, row in best_two.iterrows()
    ]

    answers = {
        "filters": {
            "thresholds": filter_lines,
            "weights": [f"{k.title()}: {v:.2f}" for k, v in weights.items()],
            "top4": stock_rationale,
        },
        "payoff": {
            "message": payoff_message,
            "image": payoff_img,
        },
        "pricing": pricing,
        "best_two": best_two_lines,
        "delta": delta,
    }
    return answers


@app.route("/", methods=["GET", "POST"])
def index():
    base_config = load_config()
    overrides = {}
    table_html = None
    effective_config = base_config
    error = None
    answers = {}

    if request.method == "POST":
        overrides = _prepare_overrides(request.form)
        try:
            table, effective_config = generate_shortlist(config_overrides=overrides, quiet=True)
            table_html = table.to_html(classes="table table-striped table-sm", index=False, float_format="{:.2f}".format)
            answers = _answer_pack(table, effective_config)
        except Exception as exc:  # pragma: no cover - web surface
            error = str(exc)
            flash(error, "danger")
    else:
        try:
            table, effective_config = generate_shortlist(quiet=True)
            table_html = table.to_html(classes="table table-striped table-sm", index=False, float_format="{:.2f}".format)
            answers = _answer_pack(table, effective_config)
        except Exception as exc:  # pragma: no cover
            error = str(exc)

    etf_map = base_config["etf_candidates"]
    selected_etfs = effective_config["etf_candidates"].keys()

    return render_template(
        "index.html",
        table_html=table_html,
        config=effective_config,
        base_config=base_config,
        overrides_json=json.dumps(overrides, indent=2) if overrides else None,
        etf_map=etf_map,
        selected_etfs=selected_etfs,
        error=error,
        answers=answers,
    )


if __name__ == "__main__":
    app.run(debug=True)
