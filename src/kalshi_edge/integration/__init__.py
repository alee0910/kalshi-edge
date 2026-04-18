"""Integration engine.

Takes a quant forecaster's posterior on the underlying (samples) plus a set
of ``FundamentalInput`` + matching ``InputLoading`` pairs, applies each
input's integration mechanism, and returns a new posterior together with a
complete attribution log. Every step is auditable — the dashboard pulls
the attribution log verbatim.

The engine handles ``PRIOR_SHIFT`` today. The other mechanisms
(``LIKELIHOOD_WEIGHT``, ``REGIME_INDICATOR``, ``STRUCTURAL_BREAK``,
``ADDITIONAL_OBSERVATION``) have clean no-op paths and explicit NotImplemented
markers so a forecaster using them fails loud rather than silently ignoring
a signal. ``ADDITIONAL_OBSERVATION`` has a Kalman-style closed-form update
for the Gaussian case implemented below.
"""

from kalshi_edge.integration.engine import (
    AttributionEntry,
    IntegrationOutcome,
    IntegrationRequest,
    apply_prior_shifts,
    integrate,
)

__all__ = [
    "AttributionEntry",
    "IntegrationOutcome",
    "IntegrationRequest",
    "apply_prior_shifts",
    "integrate",
]
