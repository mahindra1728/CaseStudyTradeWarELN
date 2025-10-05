# Composite Score Construction

The automated shortlist evaluates each candidate equity with a weighted score
that reflects how well it fits the trade-war bullish ELN thesis. The score is
only calculated after a name passes three hard filters:

- 1-year total return strictly greater than 0%
- 90-day realised volatility strictly below 30%
- Normalised China revenue share strictly below 10%

For the names that survive, the composite score is the weighted sum of five
normalised pillars:

| Pillar | Symbol | Weight | Definition |
| --- | --- | --- | --- |
| Policy alignment | `S_policy` | 40% | Analyst-assigned conviction (0–1) based on defence/onshoring tailwinds. |
| China exposure | `S_china` | 25% | Score from 1 (≤10% sales from China) to 0 (≥50%), linearly interpolated. |
| Relative returns | `S_return` | 15% | Scaled excess return vs SPY with a 20% floor and 30% upside taper. |
| Liquidity | `S_liquidity` | 10% | Normalised 3‑month average volume capped at ≥2m shares. |
| Volatility | `S_vol` | 10% | Higher score for calmer names; linearly decreases to 0 at 80% σ. |

The composite is:

```
Composite = 0.40 * S_policy
          + 0.25 * S_china
          + 0.15 * S_return
          + 0.10 * S_liquidity
          + 0.10 * S_vol
```

All inputs are computed inside `trade_war_eln/src/shortlist.py`:

- Policy and China scores pull from `BASE_METADATA` in the same file.
- Returns and volatility use yfinance price history over the past 12 months.
- Liquidity uses the trailing-63-day average share volume.

The CSV exported by `shortlist.py` includes each pillar, the composite score,
and an Excel formula to reproduce the ranking (`Rank_Formula` column).
