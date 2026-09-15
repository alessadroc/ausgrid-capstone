"""Stage 2 - Quality control, temporal standardisation, cleaning.

Design rules applied throughout:

  * Nothing is silently deleted. Suspect values are flagged in a boolean
    column so the effect of every cleaning decision can be quantified, and
    so imputed points can be excluded from evaluation.
  * Missing periods become explicit NaN rows rather than absent rows.
    Sequence models slice by position; an absent row silently splices two
    non-adjacent hours together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, log


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------

def quality_report(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    """Per-substation data quality summary. Table-ready for the report."""
    rows = []

    for sub, g in df.groupby("substation"):
        g = g.sort_values("timestamp")
        # Resolution is a property of the source file, not of the config, and
        # a substation may appear in files of different vintages.
        res = (int(g["resolution_min"].mode().iloc[0])
               if "resolution_min" in g.columns
               else cfg.native_interval_minutes)
        step = pd.Timedelta(minutes=res)
        span = g["timestamp"].max() - g["timestamp"].min()
        expected_n = span / step + 1
        gaps = g["timestamp"].diff().dropna()

        rows.append({
            "substation": sub,
            "resolution_min": res,
            "start": g["timestamp"].min(),
            "end": g["timestamp"].max(),
            "years": round(span.days / 365.25, 2),
            "n_obs": len(g),
            "completeness_pct": round(100 * len(g) / expected_n, 2),
            "duplicate_timestamps": int(g["timestamp"].duplicated().sum()),
            "regular_steps_pct": round(100 * (gaps == step).mean(), 2),
            "largest_gap_hours": round(gaps.max().total_seconds() / 3600, 1),
            "null_load": int(g["load_mw"].isna().sum()),
            "nonpositive_load": int((g["load_mw"] <= 0).sum()),
            "mean_mw": round(g["load_mw"].mean(), 3),
            "max_mw": round(g["load_mw"].max(), 3),
        })

    report = pd.DataFrame(rows).sort_values("completeness_pct")
    log(cfg, "qc", f"{len(report)} substations, median completeness "
                   f"{report['completeness_pct'].median():.1f}%")
    return report


def exclude_poor_substations(cfg: Config, df: pd.DataFrame,
                             report: pd.DataFrame) -> pd.DataFrame:
    """Drop substations too sparse to model, and say which and why."""
    keep = report.loc[
        report["completeness_pct"] >= cfg.min_completeness_pct, "substation"]
    dropped = sorted(set(report["substation"]) - set(keep))

    if dropped:
        log(cfg, "qc", f"excluding {len(dropped)} substations below "
                       f"{cfg.min_completeness_pct}% completeness: "
                       f"{', '.join(dropped[:5])}"
                       f"{' ...' if len(dropped) > 5 else ''}")
    return df[df["substation"].isin(keep)].copy()


def deduplicate(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated (substation, timestamp) pairs.

    These arise legitimately where financial-year files overlap at the
    boundary. Averaging is safer than keep-first: if the two copies disagree,
    an arbitrary pick hides the disagreement.
    """
    before = len(df)
    out = (df.groupby(["substation", "timestamp"], as_index=False)
             .agg(load_mw=("load_mw", "mean"),
                  Unit=("Unit", "first"),
                  resolution_min=("resolution_min", "min")))
    removed = before - len(out)
    if removed:
        log(cfg, "dedupe", f"collapsed {removed:,} duplicate timestamps")
    return out


# ---------------------------------------------------------------------------
# Temporal standardisation
# ---------------------------------------------------------------------------

def resample(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate native intervals to the target frequency.

    MEAN, not sum. The Unit column is MW - instantaneous average power over
    the interval, not energy - so summing four 15-minute readings would
    inflate hourly load fourfold. Multiply the hourly mean by 1 h if you
    later need MWh.

    n_intervals records how many native readings backed each aggregate, which
    is a far more informative completeness flag than a null check alone: an
    hour built from one reading out of four is present but untrustworthy.
    """
    agg = (df.set_index("timestamp")
             .groupby("substation")["load_mw"]
             .resample(cfg.target_freq)
             .agg(["mean", "count"])
             .rename(columns={"mean": "load_mw", "count": "n_intervals"})
             .reset_index())

    agg.loc[agg["n_intervals"] == 0, "load_mw"] = np.nan

    # Expected readings per target period varies with the source vintage, so
    # completeness is judged against each substation's own resolution.
    if "resolution_min" in df.columns:
        res = (df.groupby("substation")["resolution_min"].min()
                 .reindex(agg["substation"]).to_numpy())
    else:
        res = np.full(len(agg), cfg.native_interval_minutes)
    expected = np.maximum(60 // res, 1)
    agg["n_expected"] = expected
    partial = int(((agg["n_intervals"] > 0) &
                   (agg["n_intervals"] < expected)).sum())
    log(cfg, "resample", f"{len(agg):,} hourly rows; {partial:,} incomplete")
    return agg


def complete_index(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    """Force a gap-free time index per substation."""
    out = []
    for sub, g in df.groupby("substation", sort=False):
        tz = g["timestamp"].dt.tz
        idx = pd.date_range(g["timestamp"].min(), g["timestamp"].max(),
                            freq=cfg.target_freq, tz=tz)
        g = (g.set_index("timestamp")
               .reindex(idx)
               .rename_axis("timestamp")
               .reset_index())
        g["substation"] = sub
        g["n_intervals"] = g["n_intervals"].fillna(0).astype(int)
        if "n_expected" in g.columns:
            g["n_expected"] = g["n_expected"].ffill().bfill().fillna(1).astype(int)
        out.append(g)

    full = pd.concat(out, ignore_index=True)
    log(cfg, "reindex", f"inserted {len(full) - len(df):,} explicit missing hours")
    return full


# ---------------------------------------------------------------------------
# Gap filling
# ---------------------------------------------------------------------------

def fill_gaps(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    """Tiered imputation.

    Tier 1 - gaps up to `short_gap_hours`: linear interpolation. Over a few
             hours load is smooth enough that this introduces little bias.
    Tier 2 - longer gaps: the substation's median load for that hour-of-week.
             A flat mean or a moving average would erase the daily and weekly
             cycle precisely where the model most needs shape.
    Tier 3 - anything still missing stays NaN and is dropped at training time.

    Every filled value is flagged. Imputed points must never appear as targets
    in validation or test, or you are partly scoring the model on how well it
    reproduces your own interpolation.
    """
    df = df.sort_values(["substation", "timestamp"], ignore_index=True).copy()
    df["was_imputed"] = df["load_mw"].isna()
    n_missing = int(df["was_imputed"].sum())

    df["load_mw"] = (df.groupby("substation")["load_mw"]
                       .transform(lambda s: s.interpolate(
                           method="linear",
                           limit=cfg.short_gap_hours,
                           limit_area="inside")))
    after_tier1 = int(df["load_mw"].isna().sum())

    hour_of_week = df["timestamp"].dt.dayofweek * 24 + df["timestamp"].dt.hour
    profile = (df.groupby(["substation", hour_of_week])["load_mw"]
                 .transform("median"))
    df["load_mw"] = df["load_mw"].fillna(profile)
    remaining = int(df["load_mw"].isna().sum())

    log(cfg, "impute", f"{n_missing:,} missing -> interpolated "
                       f"{n_missing - after_tier1:,}, seasonal-filled "
                       f"{after_tier1 - remaining:,}, unresolved {remaining:,}")
    return df


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------

def flag_outliers(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    """Rolling median-absolute-deviation test.

    A global IQR or z-score threshold is the wrong tool for load data: summer
    afternoon peaks sit far above the annual median and would be stripped out.
    Since peak prediction error is one of the proxy reliability metrics, that
    would quietly remove the very behaviour being measured.

    Comparing each point against its local window instead catches switching
    artefacts and sensor dropouts while leaving genuine seasonal peaks intact.
    Points are flagged, not removed - Ausgrid documents switching-induced
    fluctuations as real network events, not errors.
    """
    w = cfg.outlier_window_hours

    def _flag(s: pd.Series) -> pd.Series:
        med = s.rolling(w, center=True, min_periods=w // 4).median()
        mad = (s - med).abs().rolling(w, center=True,
                                      min_periods=w // 4).median()
        sigma = (1.4826 * mad).replace(0, np.nan)
        return ((s - med).abs() > cfg.outlier_threshold * sigma).fillna(False)

    df = df.copy()
    df["is_outlier"] = (df.groupby("substation")["load_mw"]
                          .transform(_flag)
                          .astype(bool))
    df["is_outlier"] |= df["load_mw"].le(0).fillna(False)

    log(cfg, "outliers", f"flagged {int(df['is_outlier'].sum()):,} points "
                         f"({100 * df['is_outlier'].mean():.2f}%)")
    return df


def run(cfg: Config, long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    report = quality_report(cfg, long_df)
    kept = exclude_poor_substations(cfg, long_df, report)
    kept = deduplicate(cfg, kept)
    hourly = resample(cfg, kept)
    hourly = complete_index(cfg, hourly)
    hourly = fill_gaps(cfg, hourly)
    hourly = flag_outliers(cfg, hourly)
    return hourly, report
