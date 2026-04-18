"""Manual analyst-entered fundamental inputs.

Analysts commit YAML files under ``analyst_inputs/<category>/`` in the repo
root. Each file holds one or more fundamental inputs, scoped to a category
(e.g. ``economics``, ``elections``). The loader validates each entry against
the declared ``InputSpec`` for that category and emits ``FundamentalInput``
objects with ``source_kind="manual"`` and a provenance string identifying
the analyst.

The git log is the audit trail: every entry is committed, attributable to
its author via git blame, and versioned. The CLI's ``pull-fundamental``
command runs the loader and persists valid inputs to the DB with the file's
last-commit author recorded in ``provenance.notes``.

Invariant: the loader never silently coerces. If a YAML entry fails schema
validation (missing field, wrong units, nonsensical expiry), it raises —
better to break the pipeline than write a broken input to the DB.
"""

from kalshi_edge.fundamental.manual.loader import (
    ManualInputError,
    load_manual_inputs,
    load_manual_inputs_from_dir,
)

__all__ = [
    "ManualInputError",
    "load_manual_inputs",
    "load_manual_inputs_from_dir",
]
