"""Forecaster package.

``build_default_registry`` wires every shipped forecaster into a fresh
registry. Keep that explicit (vs. import-side-effect auto-registration) so
tests can mint their own registries and so the CLI's forecaster set is
obvious from reading one file.
"""

from kalshi_edge.forecasters.base import (
    Forecaster,
    ForecasterRegistry,
    bayesian_model_average,
    default_registry,
)
from kalshi_edge.forecasters.economics import EconomicsCPIForecaster
from kalshi_edge.forecasters.politics import PoliticsAbstainForecaster
from kalshi_edge.forecasters.rates import RatesAbstainForecaster
from kalshi_edge.forecasters.sports import SportsAbstainForecaster
from kalshi_edge.forecasters.weather import WeatherForecaster


def build_default_registry() -> ForecasterRegistry:
    """Instantiate every shipped forecaster and register it."""
    reg = ForecasterRegistry()
    reg.register(WeatherForecaster())
    reg.register(EconomicsCPIForecaster())
    reg.register(RatesAbstainForecaster())
    reg.register(SportsAbstainForecaster())
    reg.register(PoliticsAbstainForecaster())
    return reg


__all__ = [
    "Forecaster",
    "ForecasterRegistry",
    "bayesian_model_average",
    "default_registry",
    "build_default_registry",
    "WeatherForecaster",
    "EconomicsCPIForecaster",
    "RatesAbstainForecaster",
    "SportsAbstainForecaster",
    "PoliticsAbstainForecaster",
]
