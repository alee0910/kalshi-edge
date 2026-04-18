"""Fundamental loading calibration.

Every loading in the system comes from a calibration procedure in this
package — never hand-picked. A loading is the linear β (with uncertainty)
that translates a fundamental input anomaly into a shift in the quant
forecaster's underlying variable. The Bayesian shrinkage-prior posterior
over β is what the integration engine samples from.

Current scope: CPI (MoM, ARIMA(1,1,0) residuals). The design extends cleanly
to other categories; the pattern is always the same:

    1. Reconstruct the quant forecaster's residuals on historical data.
    2. Build the same anomaly series the live pipeline will produce
       (reusing the transform functions from ``fundamental/automated``).
    3. Regress residuals on the anomalies, with HAC- / bootstrap-SE β_std.
    4. Store (β, β_std, baseline, n_obs, fit_method, fit_at) in the DB.

Principles:

    * No look-ahead. The residual at month t is computed from an
      ARIMA(1,1,0) fit on data strictly BEFORE t (rolling or expanding
      window — expanding is fine for this slice since the model is simple).
    * Per-input regressions (one β per input). A full multivariate
      regression is available in ``fit_cpi_loadings_multi`` for diagnostic
      purposes (to see collinearity), but the production loadings are
      univariate to keep each input's contribution interpretable.
    * Sample sizes drive shrinkage. The integration engine applies
      shrinkage toward zero based on ``n_obs``; short-history inputs
      don't get to move the posterior much.
"""

from kalshi_edge.fundamental.calibration.cpi_loadings import (
    CalibratedLoading,
    fit_cpi_loadings,
)

__all__ = ["CalibratedLoading", "fit_cpi_loadings"]
