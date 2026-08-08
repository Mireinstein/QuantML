import numpy as np
import pandas as pd
import pytest

from quantiq.volatility import fit_arima, fit_garch, naive_rolling_vol


def simulate_garch_series(n=800, omega=1e-5, alpha=0.15, beta=0.8, seed=3):
    """Simulates a real GARCH(1,1) process (known ground-truth clustering)
    so we can test whether a fitted GARCH model actually recovers/tracks
    that structure, not just that the library call doesn't crash."""
    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        returns[t] = rng.normal(0, np.sqrt(sigma2[t]))
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.Series(returns, index=idx), pd.Series(sigma2, index=idx)


def test_garch_forecast_is_positive():
    returns, _ = simulate_garch_series()
    result = fit_garch(returns)
    assert result.forecast_vol > 0
    assert len(result.conditional_vol) == len(returns.dropna())


def test_garch_tracks_high_vs_low_volatility_regime():
    """Real correctness check: fit GARCH separately on a synthetic
    high-volatility window and a low-volatility window (built with
    different GARCH parameters) and confirm the model's forecast reflects
    the regime it was fit on, not a constant number regardless of input."""
    calm, _ = simulate_garch_series(n=600, omega=1e-6, alpha=0.03, beta=0.5, seed=10)
    turbulent, _ = simulate_garch_series(n=600, omega=1e-4, alpha=0.15, beta=0.8, seed=11)

    calm_result = fit_garch(calm)
    turbulent_result = fit_garch(turbulent)

    assert turbulent_result.forecast_vol > calm_result.forecast_vol


def test_garch_conditional_vol_correlates_with_true_variance():
    """The fitted in-sample conditional volatility should correlate
    strongly with the TRUE simulated variance path -- confirms GARCH is
    recovering real structure, not fitting noise."""
    returns, true_var = simulate_garch_series(n=1000, seed=5)
    result = fit_garch(returns)

    true_vol = np.sqrt(true_var).reindex(result.conditional_vol.index)
    corr = np.corrcoef(result.conditional_vol.to_numpy(), true_vol.to_numpy())[0, 1]
    assert corr > 0.5


def test_naive_rolling_vol_matches_manual_calc():
    returns = pd.Series(np.random.default_rng(1).normal(0, 0.01, 100))
    expected = returns.iloc[-20:].std()
    assert naive_rolling_vol(returns, window=20) == pytest.approx(expected)


def test_fit_arima_produces_finite_forecast():
    returns, _ = simulate_garch_series(n=300, seed=8)
    result = fit_arima(returns, order=(1, 0, 1))
    assert np.isfinite(result.forecast)
    assert np.isfinite(result.aic)
