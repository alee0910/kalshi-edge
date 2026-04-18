"""Per-category fundamental input schemas.

Each category module defines a Pydantic model enumerating the fundamental
inputs that category accepts, plus the integration mechanism for each. The
quantamental forecaster for that category consumes the schema, pulls values
from the fundamental store (automated + manual), and applies the integration
mechanism.

Design rule: every field is a ``FundamentalInput`` — never a bare float.
This keeps provenance, uncertainty, and expiration attached end-to-end.
"""

from kalshi_edge.fundamental.schemas.base import (
    FundamentalInput,
    IntegrationMechanism,
    InputLoading,
    InputProvenance,
)
from kalshi_edge.fundamental.schemas.cpi import CPIFundamentals, cpi_input_specs

__all__ = [
    "FundamentalInput",
    "IntegrationMechanism",
    "InputLoading",
    "InputProvenance",
    "CPIFundamentals",
    "cpi_input_specs",
]
