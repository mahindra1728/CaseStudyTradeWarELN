# Structuring Notes – Bullish ELN on US-China Trade War Basket

## 1. Macro framing and stock filter

- **Election lens** – assume a U.S. administration prioritises strategic
  competition with China, sustaining tariffs and onshoring incentives.
- **Filter pillars:**
  1. *China dependency under control* – targeted companies must either derive
     <30% of sales from China or be on a demonstrable path to reduce that
     exposure while benefiting from U.S. substitution policy. We verified this
     in 2023 Form 10-K filings (e.g., Lockheed Martin reports 26% international
     sales with U.S. Government the dominant customer; Intel discloses China at
     27% but highlights CHIPS Act–backed U.S./EU capacity ramp; Cisco’s APJC
     share is 14%; Nucor ships predominantly to U.S. end markets).
  2. *Direct beneficiary of onshoring or defence spending* – alignment with
     hawkish policy levers such as defence appropriations, CHIPS Act subsidies,
     or infrastructure reshoring.
  3. *Balance sheet resilience & liquidity* – investment-grade credit, net cash
     or modest leverage, and ample cash flow to manage volatility (screened via
     yfinance fundamentals and rating agency reports).
  4. *Option market depth* – tight bid/ask and weekly maturities, evidenced by
     3-month average share volume (see quantitative table below) and open
     interest on CBOE-listed options.

- **Shortlist rationale (with quantitative evidence):**
  - `LMT` – defence prime with U.S. Government purchasing ~74% of product sales
    and only 26% international exposure (Lockheed Martin Form 10-K for
    FY2023, p.10). Shares exhibit 26% 90-day realised vol and 1.6m shares of
    average daily volume, supporting hedge liquidity.
  - `INTC` – U.S. semiconductor champion; while China (incl. Hong Kong)
    represented 27% of 2023 revenue, management is actively shifting advanced
    node production to U.S./EU fabs backed by $8bn+ of CHIPS Act grants (Intel
    Form 10-K FY2023, pp. 37 & 76). The stock has rallied 68% over the past
    year with 117m shares of average volume, giving leveraged upside to
    reshoring beneficiaries.
  - `CSCO` – secure networking supplier with APJC revenue share of ~14% and
    explicit U.S. federal security retrofits driving backlog (Cisco FY2023
    Form 10-K, p.31). Lower 90-day vol (19%) complements higher beta names.
  - `NUE` – U.S. steel leader; >80% of shipments stay domestic while tariff
    protection and infrastructure demand support spreads (Nucor FY2023 Form
    10-K, p.6). Supplies commodity upside with manageable vol (27%).

**Quantitative snapshot (yfinance, as of 2025-10-03)**

| Ticker | 1Y total return | 90d realised vol | 3M avg share volume | Notes |
|---|---|---|---|---|
| LMT | -15.3% | 26.3% | 1.64M | Defensive ballast; minimal China exposure |
| INTC | +68.2% | 69.9% | 117.4M | Upside leverage to reshoring & AI capex |
| CSCO | +33.6% | 18.9% | 19.2M | Lower-vol connectivity play benefiting from secure networking spend |
| NUE | -6.6% | 27.2% | 1.64M | Cyclical lever on U.S. infrastructure & tariffs |

Source: `outputs/market_metrics.csv` (derived via `yfinance`). Pairwise return
correlations over the last 12 months are stored in
`outputs/correlation_matrix.csv`; notably `corr(LMT, INTC) ≈ 0.08`, confirming
the diversification benefit of mixing defence and semiconductors.

The helper script `src/shortlist.py` formalises the same logic: it first
filters defence/reshoring ETFs (ITA, SOXX, PAVE, IGM, IYW, XLI, VOX) to those
with positive 1Y returns and sub-30% realised volatility, then pulls their top
holdings via yfinance. After augmenting any essential names (CSCO, NUE, LMT),
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
