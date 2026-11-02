"""Forecast evaluation metrics.

Three groups:

  1. Accuracy      - MAE, RMSE, MAPE (Hyndman & Athanasopoulos, 2018)
  2. Proxy reliability - peak error and peak bias (Hong & Fan, 2016)
  3. Significance  - Diebold-Mariano test (Diebold & Mariano, 1995)

The significance test matters more than it looks. Comparing raw error numbers
cannot tell you whether an MAE of 0.42 beats an MAE of 0.45 for any reason
other than noise. The DM test answers exactly that question, and without it a
claim that one architecture outperforms another is unsupported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------

def mae(actual, predicted) -> float:
    a, p = _clean(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual, predicted) -> float:
    a, p = _clean(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual, predicted, floor: float = 1e-3) -> float:
    """Mean absolute percentage error.

    Undefined where actual is zero, and unstable where actual is near zero.
    Observations below `floor` are excluded rather than silently producing
    enormous percentages, and the exclusion count should be reported.
    """
    a, p = _clean(actual, predicted)
    keep = np.abs(a) > floor
    if keep.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((a[keep] - p[keep]) / a[keep])) * 100)


def mase(actual, predicted, train_series, season: int = 24) -> float:
    """Mean absolute scaled error (Hyndman & Koehler, 2006).

    MAPE breaks on this dataset. High rooftop-PV penetration drives net load
    through zero and negative at some substations, so dividing by the actual
    value produces meaningless percentages - Avondale reported MAPE above 180%
    purely because its denominator approaches zero around the solar peak.

    MASE divides instead by the in-sample error of a seasonal naive forecast,
    a denominator that is fixed, positive and independent of the test values.
    It is scale-free like MAPE, so it can be averaged across substations of
    very different sizes, but it is defined everywhere.

    Interpretation is direct:
        MASE < 1  beats seasonal naive on the training data
        MASE = 1  matches it
        MASE > 1  worse than simply repeating last week/day
    """
    a, p = _clean(actual, predicted)
    train = np.asarray(train_series, dtype=float)
    train = train[~np.isnan(train)]
    if len(train) <= season:
        return float("nan")

    scale = np.mean(np.abs(train[season:] - train[:-season]))
    if scale <= 0:
        return float("nan")
    return float(np.mean(np.abs(a - p)) / scale)


def _clean(actual, predicted):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if a.shape != p.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {p.shape}")
    keep = ~(np.isnan(a) | np.isnan(p))
    return a[keep], p[keep]


def accuracy_table(actual, predicted, train_series=None,
                   season: int = 24) -> dict:
    out = {"MAE": mae(actual, predicted),
           "RMSE": rmse(actual, predicted),
           "MAPE": mape(actual, predicted),
           "n": int(len(_clean(actual, predicted)[0]))}
    if train_series is not None:
        out["MASE"] = mase(actual, predicted, train_series, season)
    return out


# ---------------------------------------------------------------------------
# Proxy reliability metrics
# ---------------------------------------------------------------------------

def peak_mask(actual, quantile: float = 0.95) -> np.ndarray:
    """Boolean mask for peak-demand hours.

    'Peak' is defined as the top (1 - quantile) of hours by ACTUAL load, so
    the definition is a property of the data rather than of any model - every
    model is then judged on the same set of hours.
    """
    a = np.asarray(actual, dtype=float)
    threshold = np.nanquantile(a, quantile)
    return a >= threshold


def peak_error(actual, predicted, quantile: float = 0.95) -> dict:
    """PRM 1 - forecast error restricted to high-demand periods.

    Errors during peaks are disproportionately associated with system stress
    and capacity shortfalls, so a model with good average accuracy but poor
    peak accuracy is operationally worse than the headline number suggests.
    """
    a, p = _clean(actual, predicted)
    m = peak_mask(a, quantile)
    if m.sum() == 0:
        return {"peak_MAE": float("nan"), "peak_RMSE": float("nan"),
                "n_peak_quantile": 0}
    return {"peak_MAE": mae(a[m], p[m]),
            "peak_RMSE": rmse(a[m], p[m]),
            "peak_threshold_MW": float(np.nanquantile(a, quantile)),
            "n_peak_quantile": int(m.sum())}


# Southern-hemisphere summer evening peak: the window a network operator
# plans capacity around. Fixed in advance, independent of any outcome.
SUMMER_MONTHS = (12, 1, 2)
EVENING_PEAK_HOURS = (16, 17, 18, 19, 20)


def operational_peak_mask(timestamps, months=SUMMER_MONTHS,
                          hours=EVENING_PEAK_HOURS,
                          weekdays_only: bool = True) -> np.ndarray:
    """Peak window defined by the CLOCK, not by the outcome.

    This is the key difference from `peak_mask`. Selecting hours where actual
    load happened to be extreme conditions the sample on the target, and any
    forecast correlated below 1.0 then falls short on average by construction:
    E[forecast | actual extreme] = rho * extreme, with rho < 1. A binomial test
    against 50% therefore rejects for essentially every model, measuring
    correlation strength rather than operational bias.

    Fixing the window by month and hour in advance breaks that conditioning,
    so the resulting bias figure is interpretable - and it matches how peak
    periods are actually defined for capacity planning.
    """
    ts = pd.DatetimeIndex(timestamps)
    m = ts.month.isin(months) & ts.hour.isin(hours)
    if weekdays_only:
        m &= ts.dayofweek < 5
    return np.asarray(m)


def peak_bias(actual, predicted, timestamps=None, quantile: float = 0.95,
              **window) -> dict:
    """PRM 2 - directional bias during peak-demand periods.

    Under- and over-prediction are not operationally symmetric: systematic
    underestimation during peaks is associated with supply inadequacy, while
    overestimation mostly costs money. A signed mean error lets the two
    cancel, so the proportion of underestimated hours is reported separately.

    Pass `timestamps` to use the exogenous operational window, which is the
    version to report. Without them the function falls back to the outcome-
    conditioned definition, whose absolute level is not interpretable - use
    it only to compare models against each other, never against 50%.
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(p))

    if timestamps is not None:
        m = operational_peak_mask(np.asarray(timestamps)[keep], **window)
        definition = "operational window (summer weekday evenings)"
    else:
        m = peak_mask(a[keep], quantile)
        definition = "top %.0f%% of actual load (not interpretable vs 50%%)" % (
            100 * (1 - quantile))

    a, p = a[keep], p[keep]
    if m.sum() == 0:
        return {"underestimation_rate": float("nan"), "n_peak_window": 0,
                "peak_definition": definition}

    under = int(np.sum(p[m] < a[m]))
    n = int(m.sum())
    rate = under / n
    pval = float(stats.binomtest(under, n, 0.5).pvalue)

    return {"underestimation_rate": float(rate),
            "mean_peak_error_MW": float(np.mean(p[m] - a[m])),
            "bias_pvalue": pval,
            "biased_low": bool(rate > 0.5 and pval < 0.05
                               and timestamps is not None),
            "peak_definition": definition,
            "n_peak_window": n}


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------

def diebold_mariano(actual, pred_a, pred_b, horizon: int = 1,
                    loss: str = "squared") -> dict:
    """Diebold-Mariano test of equal predictive accuracy.

    Tests H0: model A and model B have equal expected loss. A significant
    result licenses the claim that one model genuinely outperforms the other;
    without it, a difference in MAE could easily be sampling noise.

    The variance of the loss differential is corrected for autocorrelation up
    to horizon-1 lags, because multi-step forecast errors are serially
    correlated by construction and ignoring that inflates significance.

    Returns a negative statistic when model A has lower loss (A is better).
    """
    a = np.asarray(actual, dtype=float)
    pa = np.asarray(pred_a, dtype=float)
    pb = np.asarray(pred_b, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(pa) | np.isnan(pb))
    a, pa, pb = a[keep], pa[keep], pb[keep]

    ea, eb = a - pa, a - pb
    if loss == "squared":
        d = ea ** 2 - eb ** 2
    elif loss == "absolute":
        d = np.abs(ea) - np.abs(eb)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    n = len(d)
    d_bar = float(np.mean(d))

    # Newey-West style variance with lags up to h-1
    gamma0 = float(np.var(d, ddof=0))
    var = gamma0
    for lag in range(1, horizon):
        if lag >= n:
            break
        cov = float(np.cov(d[lag:], d[:-lag], ddof=0)[0, 1])
        var += 2 * cov
    var = max(var, 1e-12)

    dm_stat = d_bar / np.sqrt(var / n)

    # Harvey-Leybourne-Newbold small-sample correction
    if n > horizon:
        adj = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
        dm_stat *= adj

    pval = float(2 * (1 - stats.t.cdf(abs(dm_stat), df=max(n - 1, 1))))

    if pval >= 0.05:
        verdict = "no significant difference"
    else:
        verdict = "A better" if dm_stat < 0 else "B better"

    return {"dm_statistic": float(dm_stat), "p_value": pval,
            "verdict": verdict, "n": n}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def evaluate(actual, predicted, model: str, substation: str,
             horizon: int, peak_quantile: float = 0.95,
             train_series=None, season: int = 24, timestamps=None) -> dict:
    """Full evaluation row for one model / substation / horizon.

    Pass `train_series` (the substation's training-split load) to get MASE,
    which is the metric to report when load approaches or crosses zero.
    """
    row = {"model": model, "substation": substation, "horizon_h": horizon}
    row.update(accuracy_table(actual, predicted, train_series, season))
    row.update(peak_error(actual, predicted, peak_quantile))
    row.update(peak_bias(actual, predicted, timestamps=timestamps,
                         quantile=peak_quantile))
    return row


def results_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    order = ["model", "substation", "horizon_h", "MAE", "RMSE", "MASE",
             "MAPE", "peak_MAE", "peak_RMSE", "underestimation_rate",
             "bias_pvalue", "biased_low", "peak_definition", "n",
             "n_peak_quantile", "n_peak_window"]
    return df[[c for c in order if c in df.columns]]