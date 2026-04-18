"""CPI fundamental schema.

The CPI quantamental forecaster consumes a fixed set of fundamental inputs,
each with a specific integration mechanism. The mapping between an input and
the CPI component it should move is documented inline — this is the audit
trail the "why am I smarter than the market here?" question requires.

All four inputs are available FREE via FRED:

    gasoline_weekly_delta
        Source: FRED series ``GASREGW`` (U.S. regular all-formulations retail
        gasoline, weekly, $/gal, 1990-08 onwards).
        Rationale: retail gasoline feeds directly into the CPI motor-fuel
        subindex, which is the single most volatile component of headline
        CPI. A 4-week change in gasoline is observable BEFORE the CPI
        release and the loading onto monthly CPI residuals is large and
        stable in-sample. This is the highest-information input.

    supply_chain_pressure
        Source: FRED series ``GSCPI`` (NY Fed Global Supply Chain Pressure
        Index, monthly z-score, 1998-01 onwards).
        Rationale: GSCPI captures freight/shipping/delivery pressure that
        leads goods CPI. In normal regimes the loading is small; in supply
        shocks (2021-22) it's the dominant signal.

    hpi_lead_12m
        Source: FRED series ``CSUSHPISA`` (Case-Shiller national home price
        index, monthly, 1987-01 onwards).
        Rationale: home-price growth leads OER and rent CPI by ~12-18
        months with a well-documented lag structure. We use the YoY %
        change 12 months ago as the signal for today's CPI release.

    ppi_passthrough
        Source: FRED series ``PPIACO`` (PPI all commodities, monthly,
        1913-01 onwards — very long history, great for loading
        calibration).
        Rationale: producer prices pass through to consumer prices with a
        short lag. Captures goods-inflation pressure not already in
        GSCPI/gasoline.

Each spec below records the exact FRED series, the integration mechanism,
human-readable units, and the ``applies_to`` scope (so the forecaster can
tell MoM vs YoY apart when needed). The calibration module (separate
file) estimates the loading ``β`` for each input from historical data.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from kalshi_edge.fundamental.schemas.base import (
    FundamentalInput,
    InputSpec,
    IntegrationMechanism,
)


# ---------------------------------------------------------------------------
# Declarative input specs.
# These drive (a) the automated data pulls, (b) the loading calibration, and
# (c) the research brief generator. They are the single source of truth for
# "what fundamental inputs does CPI forecasting accept?".
# ---------------------------------------------------------------------------


def cpi_input_specs() -> list[InputSpec]:
    """Return the canonical CPI input specs."""
    return [
        InputSpec(
            name="gasoline_weekly_delta",
            category="economics",
            mechanism=IntegrationMechanism.PRIOR_SHIFT,
            units="log-change over trailing 4 weeks",
            description=(
                "Trailing 4-week log-change in retail gasoline ($/gal). Feeds "
                "directly into CPI motor-fuel subindex; strong lead signal on "
                "headline CPI MoM."
            ),
            expected_source="FRED:GASREGW",
            # Weekly series — a ~14-day max staleness is generous but safe.
            freshness_hours=24.0 * 14,
            applies_to={"transform": "mom"},
            shrinkage_prior_n=24.0,   # monthly residuals, ~2y to earn full trust
        ),
        InputSpec(
            name="supply_chain_pressure",
            category="economics",
            mechanism=IntegrationMechanism.PRIOR_SHIFT,
            units="z-score (anomaly vs 5-year rolling mean)",
            description=(
                "NY Fed Global Supply Chain Pressure Index (GSCPI). Monthly z-score. "
                "Dominant during 2021-22 supply shocks; small loading in normal regimes."
            ),
            expected_source="FRED:GSCPI",
            freshness_hours=24.0 * 45,
            applies_to={"transform": "mom"},
            shrinkage_prior_n=36.0,
        ),
        InputSpec(
            name="hpi_lead_12m",
            category="economics",
            mechanism=IntegrationMechanism.PRIOR_SHIFT,
            units="YoY % change in Case-Shiller, 12 months ago",
            description=(
                "Leading indicator for OER/rent CPI components. Case-Shiller YoY "
                "12 months in the past predicts today's shelter inflation."
            ),
            expected_source="FRED:CSUSHPISA",
            freshness_hours=24.0 * 60,  # Case-Shiller is 2-month delayed already
            applies_to={"transform": "mom"},
            shrinkage_prior_n=60.0,     # slow-moving lag; need more history
        ),
        InputSpec(
            name="ppi_passthrough",
            category="economics",
            mechanism=IntegrationMechanism.PRIOR_SHIFT,
            units="3-month log-change in PPI all-commodities",
            description=(
                "Producer price change trailing 3 months. Captures goods-inflation "
                "pressure on consumer prices with a well-documented ~1-2 month lag."
            ),
            expected_source="FRED:PPIACO",
            freshness_hours=24.0 * 45,
            applies_to={"transform": "mom"},
            shrinkage_prior_n=36.0,
        ),
    ]


# ---------------------------------------------------------------------------
# Bundle type. The QF forecaster consumes this and nothing else.
# ---------------------------------------------------------------------------


class CPIFundamentals(BaseModel):
    """Bundle of fundamental inputs the CPI forecaster will consume.

    ``inputs`` is a dict keyed by spec name. The forecaster is responsible
    for looking up each expected spec; missing entries are fine (integration
    skips them and notes the miss in the attribution log) but expired entries
    are also dropped.
    """

    model_config = ConfigDict(extra="forbid")

    inputs: dict[str, FundamentalInput] = Field(default_factory=dict)

    def get(self, name: str) -> FundamentalInput | None:
        return self.inputs.get(name)

    def active(self, specs: list[InputSpec] | None = None) -> dict[str, FundamentalInput]:
        """Return the non-expired subset, optionally filtered to the given specs."""
        wanted: set[str] | None = {s.name for s in specs} if specs is not None else None
        out: dict[str, FundamentalInput] = {}
        for name, inp in self.inputs.items():
            if wanted is not None and name not in wanted:
                continue
            if inp.is_expired():
                continue
            out[name] = inp
        return out
