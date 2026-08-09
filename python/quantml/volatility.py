"""ARIMA/GARCH volatility modeling.

ARIMA models the conditional mean of returns; GARCH models the conditional
variance (volatility clustering -- large moves tend to be followed by large
moves, calm periods by calm periods). Used together they give both a
return forecast and a volatility forecast, which is what position sizing
and risk models actually need -- a constant-volatility assumption
understates risk right after a shock and overstates it in genuinely calm
markets.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA


@dataclass
class ArimaResult:
    order: tuple
    aic: float
    forecast: float
    fitted_values: pd.Series


def fit_arima(returns: pd.Series, order: tuple = (1, 0, 1)) -> ArimaResult:
    model = ARIMA(returns.dropna(), order=order)
    fit = model.fit()
    forecast = float(fit.forecast(steps=1).iloc[0])
    return ArimaResult(order=order, aic=float(fit.aic), forecast=forecast, fitted_values=fit.fittedvalues)


@dataclass
class GarchResult:
    forecast_vol: float
    conditional_vol: pd.Series
    params: dict


def fit_garch(returns: pd.Series, p: int = 1, q: int = 1) -> GarchResult:
    # arch_model wants returns in percent for numerical stability during
    # optimization; we convert back to return units on the way out.
    scaled = returns.dropna() * 100
    model = arch_model(scaled, vol="Garch", p=p, q=q, dist="normal")
    fit = model.fit(disp="off")
    forecast = fit.forecast(horizon=1)
    next_var = forecast.variance.iloc[-1, 0]
    next_vol = np.sqrt(next_var) / 100
    return GarchResult(
        forecast_vol=float(next_vol),
        conditional_vol=fit.conditional_volatility / 100,
        params=dict(fit.params),
    )


def naive_rolling_vol(returns: pd.Series, window: int = 20) -> float:
    """Baseline to compare GARCH against: a plain trailing rolling std,
    which assumes the recent past window is equally representative of
    tomorrow regardless of whether volatility is currently clustering."""
    return float(returns.rolling(window).std().iloc[-1])
