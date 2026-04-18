"""Fundamental research layer.

This package implements Component 1-4 of the quantamental system:

  * schemas/     — per-category structured input schemas (Pydantic v2).
                   Each fundamental input has a value, uncertainty, provenance,
                   expiration, and an integration mechanism that specifies
                   HOW it modifies the quant forecast (not a black-box
                   adjustment).
  * automated/   — pipelines that fetch the numeric inputs from free public
                   sources (FRED, EIA-via-FRED, etc) on a source-appropriate
                   cadence.
  * manual/      — loader for analyst-entered YAML inputs committed under
                   ``analyst_inputs/`` in the repo. Git is the audit trail:
                   every input is timestamped, attributable (commit author),
                   and versioned.
  * calibration/ — historical loadings for each input, estimated by
                   regressing fundamental anomalies on AR(1,1,0) residuals
                   or the appropriate analogue. No hand-picked multipliers.
  * briefs/      — auto-generated research brief per contract (Markdown).
                   Brief is an INPUT to the analyst workflow, not an output.

The invariant every addition to this package must preserve: fundamental
inputs are numbers with uncertainties, not text blobs, and each one names
the exact mechanism by which it moves a forecast.
"""

from kalshi_edge.fundamental.schemas.base import (
    FundamentalInput,
    IntegrationMechanism,
    InputLoading,
)

__all__ = ["FundamentalInput", "IntegrationMechanism", "InputLoading"]
