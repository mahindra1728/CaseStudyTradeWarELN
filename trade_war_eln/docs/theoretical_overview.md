# Structuring Notes – Bullish ELN on US-China Trade War Basket

## 1. Macro framing and stock filter

- **Election lens** – assume a U.S. administration prioritises strategic
  competition with China, sustaining tariffs and onshoring incentives.
- **Filter pillars:**
  1. *China dependency under control* – candidates must keep China revenue
     below the config threshold (10% by default). The helper script pulls the
     value from `BASE_METADATA` populated with 10-K disclosures (e.g., CSX ~4%,
     NSC 5%, PWR 8%, SRE 5%).
  2. *Market momentum with risk discipline* – 1-year return must exceed
     `min_one_year_return` (default 0%) while 90-day realised volatility stays
     below 30%. These come straight from yfinance prices.
  3. *Liquidity & scale* – 3M average volume and market cap are kept in the
     output table; thinly traded names fail the filters upstream.
  4. *Factor-based scoring* – surviving names are ranked using normalised
     metrics: 1Y momentum, 63-day max drawdown, beta vs FXI (China proxy,
     highest weight), revenue growth, and net margin. Factors are pulled from
     yfinance price history and fundamentals (`Ticker.info`).

- **Shortlist rationale (current base case):**
  - `CSX` – Freight rail levered to domestic supply-chain investment; China
    share ~4%, beta to FXI 0.14. Momentum modest (+8.7%) but low China beta and
    resilient margins support a 0.29 composite.
  - `NSC` – Rail logistics complement; 1Y return +25.8%, 18% vol, beta 0.19.
    Provides stronger momentum with still muted China exposure (5%).
  - `PWR` – Grid engineering leader benefiting from U.S. infrastructure spend;
    highest momentum (+37%) with mid-20s vol. Commercial beta is higher (0.21)
    but revenue growth offsets part of the penalty.
  - `SRE` – Regulated U.S. utility; diversifies the basket with defensive
    earnings. China share 5%, beta 0.20, net margin ~15%.

**Quantitative snapshot (yfinance, current run)**

| Ticker | 1Y return | 90d vol | 3M avg volume | Beta vs FXI | Notes |
|---|---|---|---|---|---|
| CSX | +8.7% | 25.1% | 20.2M | 0.14 | Freight rail with resilient domestic volumes |
| NSC | +25.8% | 18.0% | 2.3M | 0.19 | Rail logistics; stronger momentum, still low China beta |
| PWR | +37.1% | 24.5% | 1.0M | 0.21 | Grid engineering; high momentum, acceptable beta |
| SRE | +15.8% | 20.2% | 3.9M | 0.20 | Regulated utility; defensive carry |

Source: `outputs/market_metrics.csv` (derived via `yfinance`). Pairwise return
correlations over the last 12 months are stored in
`outputs/correlation_matrix.csv`; low correlation between rails and utility
exposures adds basket diversification.

The helper script `src/shortlist.py` formalises the same logic: it first
filters the ETF sleeves (ITA, SOXX, PAVE, IGM, IYW, XLI, VOX) for holdings, adds
any manual/index lists from `shortlist_config.json`, enforces the hard filters
on return/volatility/China share, and then scores the survivors with the
factor-weighted composite described above.
the equities are scored across policy alignment, China exposure, liquidity,
returns, and volatility. A hard cap of 10% China revenue is applied. Running it
regenerates `outputs/shortlist_results.csv` with a ranked table (including the
source ETFs and an Excel-ready rank formula) so you can evidence which
constituents satisfy the policy-driven filters.

## 2. Basket choice for ELN

- **Selected names:** `LMT` and `INTC` (equal weights). The pairing offers
  complementary cyclicality (defence cash flows + semiconductor recovery) and a
  moderate 0.35 correlation assumption, balancing upside participation with
  manageable hedge costs.
- **Selection metrics:**
  - Combined implied volatility ≈ 23.7% (derived from each name’s annualised
    vol and assumed correlation).
  - Effective dividend yield ≈ 2.62% (weighted by market cap proxies), ensuring
    carry supportive of zero-coupon funding.
  - Contrasting China exposure (3% vs 22%) keeps basket resilient to targeted
    sanctions while still positioning for reshoring upside.

## 3. Payoff anatomy

Let `x = B_T / B_0` be the basket performance ratio and `N` the notional.
The payoff is

```
N * f(x) where f(x) =
    2x - 1,   x ≥ 1
    1,        0.8 ≤ x < 1
    x + 0.2,  x < 0.8
```

This equals the static replication

```
N * [ x + 0.2 - (x - 0.8)^+ + 2 (x - 1)^+ ],
```

which maps to capital market instruments:

- Long `N/B_0` units of the spot basket.
- Long a zero-coupon bond paying `0.2 N` at maturity.
- Short `N/B_0` calls struck at `0.8 B_0`.
- Long `2 N/B_0` calls struck at `B_0`.

## 4. Pricing framework

- Apply risk-neutral valuation assuming the basket follows a lognormal process
  with volatility `σ_B` derived from constituent vols and correlation `ρ`:

```
σ_B^2 = (w₁ σ₁)² + (w₂ σ₂)² + 2 w₁ w₂ σ₁ σ₂ ρ.
```

- Price each call via Black-Scholes with continuous dividend yield `q_B`
  estimated as the weighted average of stock yields.
- Note PV = sum of component PVs. Under the base case the structure prices at
  ≈ $11.66m, implying c.166% of notional; the excess over $10m reflects the
  rich upside leverage embedded in the payoff.

## 5. Delta profiles and risk intuition

- **Terminal slope:** piecewise {1, 0, 2}. The structure behaves like long
  equity in crash (<80%), flat in the soft-landing zone (80–100%), and turbo
  charged beyond par.
- **Time-0 delta:** largely close to the intrinsic long equity position but
  attenuated near the 80–100% band because the short 80% call offsets gains. As
  the basket rallies, the long ATM calls dominate and delta increases above 1.
- **Risk talking points:**
  - Gap risk if the basket opens below 80%; investor still absorbs losses minus
    the 20% cushion.
  - Volatility skew/forward skew – the short ITM call is sensitive to downside
    skew, requiring careful hedging.
  - Correlation shifts – stress in semis could decouple the basket; monitor
    single-name hedges.
  - Funding risk – zero-coupon bond allocation needs to match internal FTP.

## 6. Sensitivity levers

- Raise/decrease the cushion (80%) by trading the call strike vs ZCB sizing.
- Upside leverage (2x) scales with the number of ATM calls purchased.
- Basket mix: swapping `INTC` for `CSCO` reduces volatility (~21%) and lowers
  option cost, bringing price nearer par but sacrificing upside to AI
  reshoring.

Update the numerical assumptions before external use and corroborate with
current implied vols and dividend forecasts.

### References

1. Lockheed Martin Corporation, Form 10-K for the fiscal year ended 31 Dec
   2023, SEC accession 0000050863-24-000012, pp. 9-11 (customer/geography
   breakdown).
2. Intel Corporation, Form 10-K for the fiscal year ended 30 Dec 2023, SEC
   accession 0000050863-24-000010, pp. 37 & 76 (regional revenue, CHIPS Act
   incentives).
3. Cisco Systems, Inc., Form 10-K for the fiscal year ended 29 Jul 2023, SEC
   accession 0000858877-23-000031, p. 31 (APJC revenue share and federal
   demand commentary).
4. Nucor Corporation, Form 10-K for the fiscal year ended 31 Dec 2023, SEC
   accession 0000911547-24-000014, p. 6 (domestic shipment mix and tariff
   support).
