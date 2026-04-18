"""Automated pulls for fundamental inputs.

Each module in this package fetches a specific data source, computes the
anomaly-form the integration engine expects, wraps it in a
``FundamentalInput``, and returns it. The CLI's ``pull-fundamental`` command
enumerates these modules and writes every input to the DB.

Design rule every pull respects: the output is a computed anomaly (e.g.
4-week log-change in gasoline), not the raw series value. The anomaly is
exactly what the loading regression consumed during calibration, so the
engine can plug it in without any additional transformation.
"""

from kalshi_edge.fundamental.automated.cpi import pull_cpi_fundamentals

__all__ = ["pull_cpi_fundamentals"]
