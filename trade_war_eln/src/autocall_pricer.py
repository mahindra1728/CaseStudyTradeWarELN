"""Path-dependent autocall/knock-in coupon pricer (Monte Carlo).

This module prices a simple but flexible autocallable note with:
  - Observation dates with per-period coupons and autocall barriers
  - Global knock-in barrier (up or down) monitored continuously over steps
  - Final redemption depending on knock-in status and terminal level

It is designed to map to the questionnaire diagram: variables such as
SAP (underlying), t^ID (observation dates), t^FPD (final payment date),
KIBD/KIBL (knock-in direction/level), and p^(3) (participation/coupon) can be
configured through the parameters below.

Usage (example):
  python -m trade_war_eln.src.autocall_pricer --underlying STOXX50E \
      --S0 4900 --sigma 0.22 --q 0.025 --r 0.03 --T 5 \
      --obs "0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0" \
      --acb "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0" \
      --coupon "0.025837,0.025837,0.025837,0.025837,0.025837,0.025837,0.025837,0.025837,0.025837,0.141" \
      --kib_dir down --kib 0.6 --n 20000

Notes
 - We simulate GBM under the risk-neutral measure with dt steps; monitor
   knock-in by tracking path extrema; autocall only at observation dates.
 - The final payoff can be adjusted via Participation parameters.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple

import math
import numpy as np


@dataclass
class AutocallSpec:
    S0: float                 # initial level
    r: float                  # risk-free (cont.)
    q: float                  # dividend yield (cont.)
    sigma: float              # volatility (annualized)
    T: float                  # maturity in years
    obs_times: List[float]    # observation times in years (ascending, <= T)
    ac_barriers: List[float]  # autocall barriers as fraction of S0 per obs time
    coupons: List[float]      # coupon per observation period (paid on call or at maturity if no call and no KI)
    kib_level: float          # knock-in barrier (fraction of S0 if barriers_relative=True; absolute level otherwise)
    kib_direction: str        # 'down' or 'up' (StrictlyDown/StrictlyUp)
    notional: float = 1.0     # scale
    participation_final_up: float = 1.0  # p^(3) for up outcome at maturity (if KI not triggered)
    participation_final_down: float = -1.0  # p^(3) for down outcome if KI triggered (loss participation)
    barriers_relative: bool = True  # if True, ac_barriers & kib_level are ×S0; else absolute levels


@dataclass
class MCSpec:
    n_paths: int = 20000
    steps_per_year: int = 252
    seed: int | None = 42


def _simulate_paths(S0: float, r: float, q: float, sigma: float, T: float, n_paths: int, n_steps: int, seed: int | None = 42) -> np.ndarray:
    dt = T / n_steps
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    # Euler GBM exact discretization
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)
    Z = rng.standard_normal((n_paths, n_steps))
    increments = drift + vol * Z
    log_paths = np.cumsum(increments, axis=1)
    S = S0 * np.exp(log_paths)
    # prepend S0 column for convenience (t=0)
    S = np.concatenate([np.full((n_paths, 1), S0), S], axis=1)
    return S


def price_autocall(spec: AutocallSpec, mc: MCSpec = MCSpec()) -> Tuple[float, dict]:
    # Setup timeline
    obs_times = list(spec.obs_times)
    assert len(obs_times) == len(spec.ac_barriers) == len(spec.coupons), "Observation arrays must align"
    total_steps = max(1, int(mc.steps_per_year * spec.T))
    S = _simulate_paths(spec.S0, spec.r, spec.q, spec.sigma, spec.T, mc.n_paths, total_steps, mc.seed)

    # Map obs times to step indices
    times = np.linspace(0.0, spec.T, total_steps + 1)
    obs_idx = [int(round(t / spec.T * total_steps)) for t in obs_times]

    # Knock-in monitoring: track min or max over entire path (continuous approx.)
    # Determine absolute KI threshold
    ki_threshold = spec.kib_level * spec.S0 if spec.barriers_relative else spec.kib_level
    if spec.kib_direction.lower() in {"down", "strictlydown"}:
        hit_ki = (S.min(axis=1) <= ki_threshold)
    else:
        hit_ki = (S.max(axis=1) >= ki_threshold)

    # Initialize
    called = np.zeros(mc.n_paths, dtype=bool)
    call_index = np.full(mc.n_paths, -1, dtype=int)
    # Autocall check at obs dates
    for i, (idx, barrier) in enumerate(zip(obs_idx, spec.ac_barriers)):
        levels = S[:, idx]
        barrier_abs = barrier * spec.S0 if spec.barriers_relative else barrier
        trigger = levels >= barrier_abs if spec.kib_direction.lower() in {"down", "strictlydown"} else levels <= barrier_abs
        newly_called = (~called) & trigger
        called[newly_called] = True
        call_index[newly_called] = i
        # Once called, skip further checks for those paths

    # Cashflows per path
    cf = np.zeros(mc.n_paths)
    # Discount factors at obs dates and maturity
    disc_obs = [math.exp(-spec.r * t) for t in obs_times]
    disc_T = math.exp(-spec.r * spec.T)

    # Autocall redemption: notional + accrued coupon at call date
    for i in range(len(obs_idx)):
        mask = called & (call_index == i)
        if not np.any(mask):
            continue
        coupon_sum = sum(spec.coupons[: i + 1])
        cf[mask] = spec.notional * (1.0 + coupon_sum) * disc_obs[i]

    # Paths not called -> payoff at maturity
    not_called = ~called
    if np.any(not_called):
        ST = S[not_called, -1]
        # If knock-in never hit, pay notional plus final coupon stack and possibly participation_up
        no_ki = ~hit_ki[not_called]
        if np.any(no_ki):
            coupon_sum = sum(spec.coupons)
            cf_nc = spec.notional * (1.0 + coupon_sum) * np.ones_like(ST[no_ki])
            cf[not_called][no_ki] = cf_nc * disc_T
        # If knock-in hit, pay linear participation on downside at maturity
        ki = hit_ki[not_called]
        if np.any(ki):
            # Common variant: repay notional times (ST/S0) if ST<S0 else notional
            ratio = ST[ki] / spec.S0
            payoff = spec.notional * np.minimum(ratio, 1.0)  # loss participation to 100%
            cf[not_called][ki] = payoff * disc_T

    pv = float(cf.mean())
    # Distribution of call timing
    call_probs = []
    total = max(1, len(call_index))
    for i in range(len(obs_idx)):
        call_probs.append(float(np.mean(call_index == i)))
    not_called_prob = float(np.mean(call_index == -1))
    details = {
        "pv": pv,
        "autocall_rate": float(called.mean()),
        "ki_rate": float(hit_ki.mean()),
        "mean_ST": float(S[:, -1].mean()),
        "obs_times": obs_times,
        "call_probs": call_probs,
        "not_called_prob": not_called_prob,
        "barriers_relative": spec.barriers_relative,
    }
    return pv, details


def main() -> None:
    p = argparse.ArgumentParser(description="Autocall/knock-in Monte Carlo pricer")
    p.add_argument("--S0", type=float, required=True)
    p.add_argument("--r", type=float, required=True)
    p.add_argument("--q", type=float, required=True)
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--T", type=float, required=True)
    p.add_argument("--obs", type=str, required=True, help="Comma-separated observation times in years")
    p.add_argument("--acb", type=str, required=True, help="Comma-separated autocall barriers as fraction of S0")
    p.add_argument("--coupon", type=str, required=True, help="Comma-separated coupons per period (e.g., 0.025837,...,0.141)")
    p.add_argument("--kib", type=float, required=True, help="Knock-in level as fraction of S0 (e.g., 0.6)")
    p.add_argument("--kib_dir", type=str, choices=["down", "up", "StrictlyDown", "StrictlyUp"], default="down")
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--steps_per_year", type=int, default=252)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    obs_times = [float(x) for x in args.obs.split(",") if x.strip()]
    acb = [float(x) for x in args.acb.split(",") if x.strip()]
    coupons = [float(x) for x in args.coupon.split(",") if x.strip()]
    spec = AutocallSpec(
        S0=args.S0, r=args.r, q=args.q, sigma=args.sigma, T=args.T,
        obs_times=obs_times, ac_barriers=acb, coupons=coupons,
        kib_level=args.kib, kib_direction=args.kib_dir,
    )
    pv, details = price_autocall(spec, MCSpec(n_paths=args.n, steps_per_year=args.steps_per_year, seed=args.seed))
    print("Autocall PV: {:.4f}".format(pv))
    print("Autocall rate: {:.2%}".format(details["autocall_rate"]))
    print("Knock-in hit rate: {:.2%}".format(details["ki_rate"]))


if __name__ == "__main__":
    main()
