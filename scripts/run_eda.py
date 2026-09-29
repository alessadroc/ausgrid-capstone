#!/usr/bin/env python3
"""Week 1 - exploratory data analysis and substation selection.

Produces the figures and diagnostics that go into the report, and applies a
documented, reproducible rule for choosing the substations used in the
controlled model comparison.

    python scripts/run_eda.py --data data/processed/hourly_features

Outputs to results/figures/ and results/metrics/:
    load_profiles.png        daily and weekly load shape
    seasonal_year.png        a full year of demand
    week_detail.png          one week at hourly resolution
    autocorrelation.png      ACF and PACF - empirical justification for lags
    stationarity.csv         ADF test results
    substation_selection.csv the chosen substations and why
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FIGDIR = Path("results/figures")
METDIR = Path("results/metrics")


def load(path: str) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_parquet(p)
    df = df.sort_values(["substation", "timestamp"]).reset_index(drop=True)
    print(f"loaded {len(df):,} rows, {df.substation.nunique()} substations")
    print(f"period {df.timestamp.min()} -> {df.timestamp.max()}")
    return df


# ---------------------------------------------------------------------------
# Substation selection
# ---------------------------------------------------------------------------

def select_substations(df: pd.DataFrame, n_per_stratum: int = 1) -> pd.DataFrame:
    """Stratified selection by mean load, filtered on data quality.

    Random selection would risk drawing three substations of similar size and
    would not support any claim about how model performance varies with load
    magnitude. Stratifying by tercile of mean demand and taking the highest
    quality site in each gives a small but defensible sample: results can be
    reported separately for small, medium and large substations.

    The quality filter comes first so the comparison is never confounded by
    one site having far more imputed data than another.
    """
    stats = (df.groupby("substation")
               .agg(mean_mw=("load_mw", "mean"),
                    max_mw=("load_mw", "max"),
                    n_hours=("load_mw", "size"),
                    min_mw=("load_mw", "min"),
                    imputed_frac=("was_imputed", "mean"),
                    outlier_frac=("is_outlier", "mean"),
                    reverse_flow_frac=("load_mw", lambda x: float((x < 0).mean())))
               .reset_index())

    # Quality gate: full coverage and little synthetic data
    max_hours = stats.n_hours.max()
    eligible = stats[(stats.n_hours >= 0.95 * max_hours) &
                     (stats.imputed_frac <= 0.05)].copy()
    print(f"\n{len(eligible)} of {len(stats)} substations pass the quality gate "
          f"(>=95% coverage, <=5% imputed)")

    if len(eligible) < 3:
        print("WARNING relaxing the quality gate - too few eligible sites")
        eligible = stats.copy()

    eligible["stratum"] = pd.qcut(eligible.mean_mw, 3,
                                  labels=["small", "medium", "large"])

    # Take the MEDIAN-load site in each tercile, not the cleanest. Picking on
    # data quality alone lands on whichever site happens to have the fewest
    # gaps, which can be unrepresentative of its stratum; every eligible site
    # has already passed the quality gate, so the tie-break should instead
    # favour typicality.
    picks = []
    for stratum, grp in eligible.groupby("stratum", observed=True):
        grp = grp.sort_values("mean_mw").reset_index(drop=True)
        mid = len(grp) // 2
        picks.append(grp.iloc[max(mid - n_per_stratum // 2, 0):
                              max(mid - n_per_stratum // 2, 0) + n_per_stratum])
    chosen = (pd.concat(picks).sort_values("mean_mw").reset_index(drop=True))

    chosen["selection_rule"] = (
        "median mean-load site within its tercile, among substations with "
        ">=95% coverage and <=5% imputed hours")

    METDIR.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(METDIR / "substation_selection.csv", index=False)

    print("\nNETWORK LOAD DISTRIBUTION (eligible substations)")
    print(eligible.mean_mw.describe().round(2).to_string())

    rf = eligible[eligible.reverse_flow_frac > 0]
    print(f"\n{len(rf)} of {len(eligible)} substations show reverse power flow "
          f"(negative net load from rooftop PV export)")
    if len(rf):
        print(f"  worst: {rf.reverse_flow_frac.max() * 100:.1f}% of hours, "
              f"minimum {rf.min_mw.min():.2f} MW")

    print("\nSELECTED SUBSTATIONS")
    print(chosen[["substation", "stratum", "mean_mw", "max_mw", "min_mw",
                  "imputed_frac", "reverse_flow_frac"]]
          .round(4).to_string(index=False))
    return chosen


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_profiles(df: pd.DataFrame, subs: list[str]) -> None:
    """Average load by hour of day and by day of week.

    The first check on any load dataset. A plausible urban profile has an
    overnight trough, a morning rise and a late-afternoon or evening peak.
    If this does not appear, something upstream is wrong and no model will
    fix it.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for s in subs:
        g = df[df.substation == s]
        norm = g.load_mw / g.load_mw.mean()
        axes[0].plot(g.groupby(g.hour).apply(
            lambda x: (x.load_mw / g.load_mw.mean()).mean()), label=s)
        axes[1].plot(norm.groupby(g.dayofweek).mean(), marker="o", label=s)

    axes[0].set(xlabel="hour of day", ylabel="load / mean",
                title="Daily load profile")
    axes[1].set(xlabel="day of week (0=Mon)", ylabel="load / mean",
                title="Weekly load profile")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGDIR / "load_profiles.png", dpi=150)
    plt.close(fig)


def plot_year(df: pd.DataFrame, sub: str) -> None:
    g = df[df.substation == sub].set_index("timestamp")
    daily = g.load_mw.resample("D").mean()
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(daily.index, daily.values, lw=0.8)
    ax.set(title=f"Daily mean demand - {sub}", ylabel="MW")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "seasonal_year.png", dpi=150)
    plt.close(fig)


def plot_week(df: pd.DataFrame, sub: str) -> None:
    g = df[df.substation == sub].set_index("timestamp")
    start = g.index.min() + pd.Timedelta(days=200)
    w = g.loc[start:start + pd.Timedelta(days=7)]
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(w.index, w.load_mw, lw=1)
    ax.set(title=f"One week at hourly resolution - {sub}", ylabel="MW")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "week_detail.png", dpi=150)
    plt.close(fig)


def plot_acf(df: pd.DataFrame, sub: str, lags: int = 200) -> None:
    """ACF and PACF.

    These give empirical justification for the lag features rather than
    choosing 24 and 168 by convention. Expect strong spikes at 24 and 168,
    which is the evidence that daily and weekly lags belong in the model.
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    s = df[df.substation == sub].load_mw.dropna()
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    plot_acf(s, lags=lags, ax=axes[0])
    axes[0].set_title(f"Autocorrelation - {sub}")
    plot_pacf(s, lags=min(lags, 100), ax=axes[1], method="ywm")
    axes[1].set_title("Partial autocorrelation")
    for ax in axes:
        for h in (24, 48, 168):
            if h <= lags:
                ax.axvline(h, color="red", ls="--", alpha=0.4, lw=0.8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "autocorrelation.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stationarity
# ---------------------------------------------------------------------------

def stationarity(df: pd.DataFrame, subs: list[str]) -> pd.DataFrame:
    """Augmented Dickey-Fuller test.

    Needed to choose ARIMA's differencing order d. A p-value below 0.05
    rejects the unit-root null, indicating the series is already stationary
    and d=0 is appropriate. Load series are usually stationary in level but
    strongly seasonal, which is why seasonal handling matters more than
    differencing here.
    """
    from statsmodels.tsa.stattools import adfuller

    rows = []
    for s in subs:
        x = df[df.substation == s].load_mw.dropna().to_numpy()
        x = x[:20000]  # ADF is slow on very long series
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            stat, p, lags, nobs, crit, _ = adfuller(x, autolag="AIC")
        rows.append({"substation": s, "adf_statistic": round(stat, 4),
                     "p_value": round(p, 6), "lags_used": lags, "n_obs": nobs,
                     "crit_5pct": round(crit["5%"], 4),
                     "stationary_at_5pct": p < 0.05,
                     "suggested_d": 0 if p < 0.05 else 1})
    out = pd.DataFrame(rows)
    out.to_csv(METDIR / "stationarity.csv", index=False)
    print("\nSTATIONARITY (ADF)")
    print(out.to_string(index=False))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/hourly_features")
    ap.add_argument("--acf-lags", type=int, default=200)
    a = ap.parse_args()

    FIGDIR.mkdir(parents=True, exist_ok=True)
    METDIR.mkdir(parents=True, exist_ok=True)

    df = load(a.data)
    chosen = select_substations(df)
    subs = chosen.substation.tolist()
    focus = subs[len(subs) // 2]

    print("\ngenerating figures...")
    plot_profiles(df, subs)
    plot_year(df, focus)
    plot_week(df, focus)
    plot_acf(df, focus, a.acf_lags)
    stationarity(df, subs)

    print(f"\nfigures -> {FIGDIR}/")
    print(f"metrics -> {METDIR}/")
    print("\nNext: python scripts/run_baselines.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())