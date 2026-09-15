"""Stage 3 - Feature engineering, splitting, scaling, sequence windows.

The governing constraint here is leakage. Every derived feature at time t may
use information available strictly before t and nothing else. Two rules
enforce it:

  * rolling statistics are shifted by one step before the window is applied
  * scaling parameters are fitted on the training block only

A rolling mean that includes the current timestep, or a scaler fitted on the
full series, will both produce validation scores that collapse the moment the
model meets genuinely unseen data.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from .config import Config, log


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------

def add_calendar(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df["timestamp"]

    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["dayofyear"] = ts.dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Cyclical encoding. Without it a model sees a 23-unit discontinuity every
    # midnight and a 6-unit one every Sunday night; sin/cos pairs make hour 23
    # and hour 0 adjacent in feature space.
    for col, period in (("hour", 24), ("dayofweek", 7), ("dayofyear", 365.25)):
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)

    df["is_holiday"] = _nsw_holidays(cfg, ts)
    return df


def _nsw_holidays(cfg: Config, ts: pd.Series) -> pd.Series:
    """NSW public holiday indicator. Holidays shift load toward a weekend
    shape on a weekday, which no calendar feature otherwise captures."""
    try:
        import holidays as hol
    except ImportError:
        warnings.warn("`holidays` not installed; is_holiday = 0. "
                      "pip install holidays")
        return pd.Series(0, index=ts.index, dtype=int)

    naive = ts.dt.tz_localize(None) if ts.dt.tz is not None else ts
    years = range(int(naive.dt.year.min()), int(naive.dt.year.max()) + 1)
    nsw = hol.country_holidays("AU", subdiv="NSW", years=years)
    flags = naive.dt.date.map(lambda d: d in nsw).astype(int)
    log(cfg, "calendar", f"{flags.sum():,} holiday hours flagged")
    return flags


# ---------------------------------------------------------------------------
# Autoregressive features
# ---------------------------------------------------------------------------

def add_lags(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    """Lags and rolling statistics, computed within each substation.

    Lag 168 (one week) and lag 24 carry most of the signal in load data; lag 25
    is included because the 24-hour-ahead model cannot see lag 1..24 and needs
    a same-hour-yesterday anchor that remains available at forecast time.
    """
    df = df.sort_values(["substation", "timestamp"], ignore_index=True).copy()
    grp = df.groupby("substation")["load_mw"]

    for lag in cfg.lags:
        df[f"lag_{lag}h"] = grp.shift(lag)

    shifted = grp.shift(1)
    by_sub = shifted.groupby(df["substation"])
    for w in cfg.rolling_windows:
        df[f"roll_mean_{w}h"] = by_sub.transform(
            lambda s, w=w: s.rolling(w, min_periods=w // 2).mean())
        df[f"roll_std_{w}h"] = by_sub.transform(
            lambda s, w=w: s.rolling(w, min_periods=w // 2).std())
        df[f"roll_max_{w}h"] = by_sub.transform(
            lambda s, w=w: s.rolling(w, min_periods=w // 2).max())

    df["diff_1h"] = grp.diff(1)
    df["diff_24h"] = grp.diff(24)

    log(cfg, "lags", f"{len(cfg.lags)} lags, "
                     f"{3 * len(cfg.rolling_windows)} rolling features")
    return df


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

def add_weather(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    """Join hourly weather and derive degree-hour terms.

    Free hourly history for Sydney, no API key required:

      https://archive-api.open-meteo.com/v1/archive
        ?latitude=-33.87&longitude=151.21
        &start_date=2015-01-01&end_date=2025-06-30
        &hourly=temperature_2m,relative_humidity_2m,wind_speed_10m
        &timezone=Australia/Sydney&format=csv

    Raw temperature alone underperforms because the load/temperature
    relationship is V-shaped - demand rises in both directions away from a
    comfort band. Cooling and heating degree-hours linearise each arm, which
    matters most for ARIMA and for any linear baseline.
    """
    if cfg.weather_csv is None or not cfg.weather_csv.exists():
        log(cfg, "weather", "no weather file supplied - skipped")
        return df

    w = pd.read_csv(cfg.weather_csv)
    time_col = next(c for c in w.columns if c.lower() in ("time", "date", "timestamp"))
    w[time_col] = pd.to_datetime(w[time_col])
    w = w.rename(columns={time_col: "timestamp",
                          "temperature_2m (°C)": "temp_c",
                          "temperature_2m": "temp_c",
                          "relative_humidity_2m (%)": "humidity",
                          "relative_humidity_2m": "humidity",
                          "wind_speed_10m (km/h)": "wind_kmh",
                          "wind_speed_10m": "wind_kmh"})

    tz = df["timestamp"].dt.tz
    if tz is not None and w["timestamp"].dt.tz is None:
        w["timestamp"] = w["timestamp"].dt.tz_localize(
            tz, ambiguous=True, nonexistent="shift_forward")
    elif tz is None and w["timestamp"].dt.tz is not None:
        w["timestamp"] = w["timestamp"].dt.tz_localize(None)

    keep = ["timestamp"] + [c for c in ("temp_c", "humidity", "wind_kmh")
                            if c in w.columns]
    df = df.merge(w[keep], on="timestamp", how="left")

    if "temp_c" in df.columns:
        df["cooling_degree_hours"] = (df["temp_c"] - cfg.cooling_base_c).clip(lower=0)
        df["heating_degree_hours"] = (cfg.heating_base_c - df["temp_c"]).clip(lower=0)
        df["temp_lag_24h"] = df.groupby("substation")["temp_c"].shift(24)
        df["temp_roll_24h"] = (df.groupby("substation")["temp_c"]
                                 .transform(lambda s: s.rolling(24, min_periods=12).mean()))
        unmatched = int(df["temp_c"].isna().sum())
        log(cfg, "weather", f"joined; {unmatched:,} hours unmatched")

    return df


# ---------------------------------------------------------------------------
# Splitting and scaling
# ---------------------------------------------------------------------------

def add_split(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    """Strictly chronological train / validation / test labels.

    Random splitting leaks the future into training and is the most common
    methodological error in published load-forecasting work. Use this fixed
    holdout for final reporting and rolling-origin cross-validation inside the
    training block for hyperparameter selection.
    """
    df = df.copy()
    t = df["timestamp"]
    if t.dt.tz is not None:
        t = t.dt.tz_localize(None)

    # Explicit cut dates win over fractions, so a chosen study period gives a
    # reproducible split that can be quoted directly in the methodology.
    if cfg.split_train_end and cfg.split_val_end:
        t_train = pd.Timestamp(cfg.split_train_end)
        t_val = pd.Timestamp(cfg.split_val_end)
    else:
        t_train = t.quantile(cfg.train_fraction)
        t_val = t.quantile(cfg.train_fraction + cfg.val_fraction)

    df["split"] = np.select([t <= t_train, t <= t_val],
                            ["train", "val"], default="test")

    counts = df["split"].value_counts().to_dict()
    log(cfg, "split", f"train<= {t_train:%Y-%m-%d}, val<= {t_val:%Y-%m-%d}; "
                      + ", ".join(f"{k}={v:,}" for k, v in counts.items()))
    return df


def fit_scalers(cfg: Config, df: pd.DataFrame) -> dict:
    """Per-substation standardisation parameters, fitted on TRAIN only.

    Per-substation rather than global because zone substations differ by an
    order of magnitude in absolute MW; a single global scaler would let the
    largest substations dominate the loss and leave small ones near zero.
    """
    train = df[df["split"] == "train"]
    stats = (train.groupby("substation")["load_mw"]
                  .agg(["mean", "std"])
                  .rename(columns={"mean": "load_mean", "std": "load_std"}))
    stats["load_std"] = stats["load_std"].replace(0, 1.0)

    path = cfg.output_dir / "scaler_params.json"
    path.write_text(json.dumps(stats.to_dict(orient="index"), indent=2))
    log(cfg, "scale", f"fitted on train only, saved to {path.name}")
    return stats.to_dict(orient="index")


def apply_scaling(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()
    means = df["substation"].map(lambda s: stats[s]["load_mean"])
    stds = df["substation"].map(lambda s: stats[s]["load_std"])
    df["load_scaled"] = (df["load_mw"] - means) / stds
    return df


# ---------------------------------------------------------------------------
# Sequence windows
# ---------------------------------------------------------------------------

def make_windows(cfg: Config, df: pd.DataFrame, substation: str,
                 horizon: int, feature_cols: list[str] | None = None
                 ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build (samples, lookback, features) tensors for LSTM / Transformer.

    Returns X, y and the target timestamps. Windows that span an imputed or
    flagged target are excluded so the models are never trained or scored on
    synthetic values.

    ARIMA does not need this - feed it the univariate `load_mw` series
    directly - but the two deep architectures both expect fixed-length
    sequences, and building them once here keeps the comparison honest by
    guaranteeing identical samples across models.
    """
    g = (df[df["substation"] == substation]
         .sort_values("timestamp")
         .reset_index(drop=True))

    if feature_cols is None:
        feature_cols = ["load_scaled", "hour_sin", "hour_cos",
                        "dayofweek_sin", "dayofweek_cos",
                        "dayofyear_sin", "dayofyear_cos",
                        "is_weekend", "is_holiday"]
    feature_cols = [c for c in feature_cols if c in g.columns]

    values = g[feature_cols].to_numpy(dtype=np.float32)
    target = g["load_scaled"].to_numpy(dtype=np.float32)
    invalid = (g["was_imputed"] | g["is_outlier"]).to_numpy()

    L, H = cfg.lookback_hours, horizon
    X, y, stamps = [], [], []

    for i in range(L, len(g) - H + 1):
        t_idx = i + H - 1
        if invalid[t_idx] or np.isnan(values[i - L:i]).any():
            continue
        X.append(values[i - L:i])
        y.append(target[t_idx])
        stamps.append(g["timestamp"].iloc[t_idx])

    log(cfg, "windows", f"{substation} h={H}: {len(X):,} samples "
                        f"({L}x{len(feature_cols)})")
    return (np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.float32),
            pd.DatetimeIndex(stamps))


def run(cfg: Config, hourly: pd.DataFrame) -> pd.DataFrame:
    df = add_calendar(cfg, hourly)
    df = add_lags(cfg, df)
    df = add_weather(cfg, df)
    df = add_split(cfg, df)
    stats = fit_scalers(cfg, df)
    df = apply_scaling(df, stats)
    return df
