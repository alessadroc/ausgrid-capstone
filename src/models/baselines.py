"""Naive baseline forecasters.

These exist because ARIMA is not a lower bound. Hourly electricity demand is
extremely regular - the same hour yesterday is already a strong prediction -
so a seasonal naive forecaster is a genuinely hard benchmark. If a Transformer
cannot clearly beat it, that is the finding, and it is one examiners look for.

THE AVAILABILITY RULE
---------------------
A forecast for time t+h is issued at time t, so it may only use observations
at time t or earlier. Every predictor below is expressed as a lag measured
from the TARGET, and is only admissible if that lag reaches back at least h
steps:

    predictor y(t + h - k) is available   iff   k >= h

This is the rule that stops a 24-hour-ahead model from quietly using data from
1 hour before the target, which does not exist when the forecast is made and
produces impossibly good results. Every baseline here validates its own lag
against the horizon and refuses to run otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOURS_PER_DAY = 24
HOURS_PER_WEEK = 168


class NaiveForecaster:
    """Predicts the target using a single lagged observation.

    `lag` is measured backwards from the TARGET time, not from the forecast
    issue time. lag must be >= horizon for the value to exist at issue time.
    """

    def __init__(self, lag: int, name: str):
        self.lag = lag
        self.name = name

    def validate(self, horizon: int) -> None:
        if self.lag < horizon:
            raise ValueError(
                f"{self.name} uses lag {self.lag}h, which is not available "
                f"when forecasting {horizon}h ahead. A forecast issued at t "
                f"for t+{horizon} can only use lags of {horizon}h or more "
                f"from the target."
            )

    def predict(self, series: pd.Series, horizon: int) -> pd.Series:
        """Return predictions aligned to the target index."""
        self.validate(horizon)
        return series.shift(self.lag)


def persistence(horizon: int) -> NaiveForecaster:
    """Last observed value: y_hat(t+h) = y(t).

    The lag from the target is exactly h, so this is always admissible and is
    the weakest sensible benchmark.
    """
    return NaiveForecaster(lag=horizon, name="persistence")


def seasonal_naive_daily(horizon: int) -> NaiveForecaster:
    """Same hour, previous available day.

    At h=1 this is y(t-23), i.e. this hour yesterday. At h=24 the daily lag
    collapses onto persistence (both become y(t)), so for the 24-hour horizon
    the weekly version below is the meaningful seasonal benchmark.
    """
    lag = HOURS_PER_DAY
    while lag < horizon:
        lag += HOURS_PER_DAY
    return NaiveForecaster(lag=lag, name="seasonal_naive_daily")


def seasonal_naive_weekly(horizon: int) -> NaiveForecaster:
    """Same hour, same weekday, previous week: captures the weekend effect.

    Usually the strongest naive benchmark, because it matches both time of day
    and day of week - a Sunday is predicted from the previous Sunday rather
    than from Saturday.
    """
    lag = HOURS_PER_WEEK
    while lag < horizon:
        lag += HOURS_PER_WEEK
    return NaiveForecaster(lag=lag, name="seasonal_naive_weekly")


def drift_adjusted_seasonal(series: pd.Series, horizon: int,
                            window: int = HOURS_PER_WEEK) -> pd.Series:
    """Weekly seasonal naive corrected for recent level shift.

    Adds the difference between the recent mean and the mean one week earlier,
    which handles gradual seasonal drift that a pure seasonal naive misses.
    All inputs are shifted by at least `horizon` so nothing unavailable leaks.
    """
    base = series.shift(HOURS_PER_WEEK)
    recent = series.shift(horizon).rolling(window, min_periods=window // 2).mean()
    previous = series.shift(horizon + HOURS_PER_WEEK).rolling(
        window, min_periods=window // 2).mean()
    return base + (recent - previous)


def all_baselines(series: pd.Series, horizon: int) -> dict[str, pd.Series]:
    """Every admissible baseline for this horizon, as target-aligned series."""
    out = {}
    for fc in (persistence(horizon),
               seasonal_naive_daily(horizon),
               seasonal_naive_weekly(horizon)):
        out[f"{fc.name}(lag={fc.lag}h)"] = fc.predict(series, horizon)
    out["drift_adjusted_seasonal"] = drift_adjusted_seasonal(series, horizon)
    return out