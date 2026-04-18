"""Auto-generated research briefs.

A brief is a Markdown document with one file per contract. It is an INPUT
to the analyst workflow — "here's what the system already thinks about
this market, here are the fundamental inputs currently feeding the
forecast, here's where the market disagrees with the posterior" — not
an output to be shown to end users. Analysts read briefs to decide
whether to enter a manual input under ``analyst_inputs/``.

Hard rules:

    * No LLM. Every line of the brief comes from a structured computation
      over the same objects the integration engine consumes. If an analyst
      reads a claim, they can trace it to a ``FundamentalInput``, an
      ``InputLoading``, or a ``ForecastResult`` field.
    * Determinism. Given the same inputs, the brief bytes are identical.
      This lets us diff briefs across time and flag when the system's
      view of a contract meaningfully shifts.
    * Safe with missing data. A brief can be built from just a contract +
      quant forecast (no fundamentals yet). Sections degrade to "N/A"
      with a note rather than crashing.
"""

from kalshi_edge.fundamental.briefs.generator import (
    BriefContext,
    render_brief,
    write_briefs,
)

__all__ = ["BriefContext", "render_brief", "write_briefs"]
