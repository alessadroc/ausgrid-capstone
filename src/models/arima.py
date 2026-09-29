"""ARIMA baseline with Fourier seasonal terms.

WHY NOT SEASONAL ARIMA
----------------------
Hourly load has two seasonal cycles: daily (s=24) and weekly (s=168). A
SARIMA with s=24 on 20,000 observations is already slow; s=168 requires
estimating 168 seasonal lags and is not practical on a laptop. Fitting both
simultaneously is not supported by standard SARIMA at all.

The established alternative is a regression with ARIMA errors, where the
seasonality enters as Fourier terms - pairs of sine and cosine harmonics at
each cycle length - supplied as exogenous regressors. A handful of harmonics
reproduces a smooth daily and weekly shape with far fewer parameters than
seasonal differencing, and the ARIMA part then models only the remaining
short-range autocorrelation. This is the approach Hyndman & Athanasopoulos
recommend for multiple or long seasonal periods.

DIFFERENCING
------------
The ADF tests in results/metrics/stationarity.csv reject the unit-root null
for every selected substation, so d=0. That is an empirical result from the
data rather than a default, and it should be cited as such in the report.

FORECAST PROTOCOL
-----------------
The model is fitted once on the training split, then rolled forward through
the test period WITHOUT refitting: at each origin t the observed history up
to t is appended, and an h-step forecast is produced. Refitting at every test
hour would take days and is not standard practice. Because no test-period
observation is ever used to estimate a parameter, and each forecast only
conditions on data available at its own origin, this is a valid backtest.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

HOURS_PER_DAY = 24
HOURS_PER_WEEK = 168


# ---------------------------------------------------------------------------
# Fourier seasonality
# ---------------------------------------------------------------------------

def fourier_terms(index: pd.DatetimeIndex,
                  periods=((HOURS_PER_DAY, 3), (HOURS_PER_WEEK, 2))
                  ) -> pd.DataFrame:
    """Sine/cosine harmonics for each seasonal period.

    `periods` is a sequence of (cycle_length_hours, n_harmonics). Three
    harmonics on the daily cycle capture the overnight trough plus the morning
    and evening peaks; two on the weekly cycle capture the weekday/weekend
    step. More harmonics fit a sharper shape at the cost of parameters and
    overfitting risk.

    These are deterministic functions of the timestamp alone, so they are
    known arbitrarily far ahead and are always available at forecast time -
    unlike lagged load, they raise no leakage concern at any horizon.
    """
    t = np.arange(len(index), dtype=float)
    out = {}
    for period, n_harm in periods:
        for k in range(1, n_harm + 1):
            out[f"sin_{period}_{k}"] = np.sin(2 * np.pi * k * t / period)
            out[f"cos_{period}_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(out, index=index)


def calendar_exog(index: pd.DatetimeIndex, holidays=None) -> pd.DataFrame:
    """Fourier terms plus weekend and holiday indicators."""
    X = fourier_terms(index)
    X["is_weekend"] = (index.dayofweek >= 5).astype(float)
    if holidays is not None:
        X["is_holiday"] = pd.Series(holidays, index=index).astype(float).values
    return X


# ---------------------------------------------------------------------------
# Order selection
# ---------------------------------------------------------------------------

def select_order(series: pd.Series, exog: pd.DataFrame,
                 p_range=(1, 2, 3), q_range=(0, 1, 2), d: int = 0,
                 max_obs: int = 8000, verbose: bool = True) -> tuple:
    """Grid-search (p,d,q) by AIC on a subset of the training data.

    Run this ONCE, on one representative substation, and reuse the chosen
    order everywhere. Re-selecting per substation multiplies runtime and
    weakens the controlled comparison, since the model would then differ
    across sites in more than its fitted coefficients.

    A subset is used because AIC ranking stabilises long before the full
    20,000 points, and fitting every candidate on the full series is slow.
    """
    from statsmodels.tsa.arima.model import ARIMA

    y = series.iloc[-max_obs:]
    X = exog.loc[y.index]

    best, best_aic = None, np.inf
    for p in p_range:
        for q in q_range:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = ARIMA(y, exog=X, order=(p, d, q),
                                enforce_stationarity=False,
                                enforce_invertibility=False).fit()
                if verbose:
                    print(f"    ARIMA({p},{d},{q}) AIC={fit.aic:,.1f}")
                if fit.aic < best_aic:
                    best, best_aic = (p, d, q), fit.aic
            except Exception as exc:
                if verbose:
                    print(f"    ARIMA({p},{d},{q}) failed: {type(exc).__name__}")
    if best is None:
        raise RuntimeError("no ARIMA order could be fitted")
    if verbose:
        print(f"  selected ARIMA{best}  AIC={best_aic:,.1f}")
    return best


# ---------------------------------------------------------------------------
# Fit and forecast
# ---------------------------------------------------------------------------

class ArimaForecaster:
    """Regression with ARIMA errors, rolled forward over a test period."""

    def __init__(self, order=(2, 0, 1), periods=((HOURS_PER_DAY, 3),
                                                 (HOURS_PER_WEEK, 2))):
        self.order = order
        self.periods = periods
        self.result = None

    def fit(self, train: pd.Series, exog: pd.DataFrame) -> "ArimaForecaster":
        from statsmodels.tsa.arima.model import ARIMA

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.result = ARIMA(train, exog=exog.loc[train.index],
                                order=self.order,
                                enforce_stationarity=False,
                                enforce_invertibility=False).fit()
        return self

    def forecast_one_step(self, full: pd.Series, exog: pd.DataFrame,
                          test_index: pd.DatetimeIndex) -> pd.Series:
        """All 1-step-ahead forecasts in a single Kalman filter pass.

        `apply(..., refit=False)` re-runs the filter over the full series with
        the training-estimated parameters held fixed, and `get_prediction`
        with dynamic=False then returns, at each time t, the forecast of y(t)
        formed from information up to t-1 only. The Kalman filter is causal,
        so this is exactly the rolling 1-step-ahead backtest - no test
        observation influences its own prediction or any parameter.

        Doing it this way is roughly a thousand times faster than appending
        one observation at a time: on a 3,000-hour test period the loop takes
        about 2.5 minutes, this takes about a tenth of a second.
        """
        if self.result is None:
            raise RuntimeError("call fit() first")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            applied = self.result.apply(full, exog=exog, refit=False)
            pred = applied.get_prediction(start=test_index[0]).predicted_mean
        return pd.Series(np.asarray(pred), index=pred.index, name="arima")

    def forecast_rolling(self, full: pd.Series, exog: pd.DataFrame,
                         test_index: pd.DatetimeIndex, horizon: int,
                         verbose: bool = True,
                         origin_step: int = 1) -> pd.Series:
        """h-step-ahead forecasts at every origin in the test period.

        `append(refit=False)` extends the state space with newly observed
        values while holding the parameters fixed. This is the mechanism that
        makes a rolling backtest tractable: the Kalman filter updates on new
        data, but nothing from the test period informs an estimate.

        The returned series is indexed by TARGET time, so it aligns directly
        with the actual values for evaluation.
        """
        if self.result is None:
            raise RuntimeError("call fit() first")
        if horizon == 1 and origin_step == 1:
            return self.forecast_one_step(full, exog, test_index)

        preds, targets = [], []
        state = self.result
        n_train = len(state.data.endog)
        all_index = full.index

        # Origins: every test timestamp that has a target h steps later
        start_pos = all_index.get_loc(test_index[0])
        end_pos = all_index.get_loc(test_index[-1])

        # origin_step=1 gives a forecast from every hour (most thorough);
        # larger steps trade evaluation points for runtime.
        positions = range(start_pos, end_pos - horizon + 1, max(1, origin_step))

        last_pos = n_train
        for i, pos in enumerate(positions):
            # Append everything observed between the previous origin and this
            new = full.iloc[last_pos:pos + 1]
            if len(new):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    state = state.append(new, exog=exog.iloc[last_pos:pos + 1],
                                         refit=False)
                last_pos = pos + 1

            fc_exog = exog.iloc[pos + 1:pos + 1 + horizon]
            if len(fc_exog) < horizon:
                break
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = state.forecast(steps=horizon, exog=fc_exog)

            preds.append(float(np.asarray(fc)[-1]))
            targets.append(all_index[pos + horizon])

            if verbose and i and i % 200 == 0:
                print(f"      {i}/{len(positions)} origins", flush=True)

        return pd.Series(preds, index=pd.DatetimeIndex(targets), name="arima")