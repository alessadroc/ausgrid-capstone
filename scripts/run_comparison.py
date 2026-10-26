#!/usr/bin/env python3
"""Week 5 - final model comparison.

Consolidates every result file into the tables and figures the report needs,
and runs the full set of pairwise significance tests.

    python scripts/run_comparison.py

Inputs (whichever exist):
    results/metrics/baseline_results.csv
    results/metrics/arima_results.csv
    results/metrics/lstm_results.csv
    results/metrics/transformer_results.csv
    results/metrics/{arima,lstm,transformer}_predictions.csv

Outputs:
    results/metrics/comparison_table.csv    headline accuracy table
    results/metrics/comparison_table.md     same, ready to paste into the report
    results/metrics/pairwise_dm.csv         every model pair, both horizons
    results/metrics/reliability_table.csv   proxy reliability metrics
    results/figures/comparison_mase.png
    results/figures/comparison_peak_bias.png
    results/figures/predictions_overlay.png
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.metrics import diebold_mariano  # noqa: E402


def short_name(substation: str) -> str:
    """Drop the voltage suffix for axis labels ('West Gosford 132_11kV')."""
    parts = str(substation).split()
    return " ".join(parts[:-1]) if len(parts) > 1 else str(substation)

METDIR = Path("results/metrics")
FIGDIR = Path("results/figures")

# Display order: simplest model first, so the table reads as increasing
# sophistication and the reader can see where the gains stop.
MODEL_ORDER = ["persistence", "seasonal_naive_daily", "seasonal_naive_weekly",
               "drift_adjusted_seasonal", "ARIMA", "LSTM", "Transformer"]


def family(model_name: str) -> str:
    """Strip configuration detail so models group across runs."""
    s = str(model_name)
    for key in MODEL_ORDER:
        if s.lower().startswith(key.lower()):
            return key
    return s


def load_results() -> pd.DataFrame:
    frames = []
    for stem in ("baseline", "arima", "lstm", "transformer"):
        f = METDIR / f"{stem}_results.csv"
        if f.exists():
            d = pd.read_csv(f)
            d["source"] = stem
            frames.append(d)
            print(f"  loaded {stem}_results.csv ({len(d)} rows)")
        else:
            print(f"  MISSING {stem}_results.csv - skipping")
    if not frames:
        raise FileNotFoundError("no result files found in results/metrics/")

    df = pd.concat(frames, ignore_index=True)
    df["family"] = df["model"].map(family)
    df["order"] = df["family"].map(
        {k: i for i, k in enumerate(MODEL_ORDER)}).fillna(99)
    return df.sort_values(["substation", "horizon_h", "order"])


def best_baseline_only(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the four naive methods to the strongest per cell.

    Reporting all four in the headline table buries the comparison that
    matters. The full baseline set stays in the appendix table.
    """
    naive = df[df["source"] == "baseline"]
    keep = naive.loc[naive.groupby(["substation", "horizon_h"])["MAE"].idxmin()]
    keep = keep.copy()
    keep["family"] = "Best naive"
    keep["order"] = -1
    return pd.concat([keep, df[df["source"] != "baseline"]], ignore_index=True)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["substation", "horizon_h", "family", "MAE", "RMSE", "MASE"]
    if "mae_seed_sd" in df.columns:
        cols.append("mae_seed_sd")
    t = df[[c for c in cols if c in df.columns]].copy()
    return t.sort_values(["substation", "horizon_h", "MAE"])


def reliability_table(df: pd.DataFrame) -> pd.DataFrame:
    """The proxy reliability metrics, which are the point of the study."""
    cols = ["substation", "horizon_h", "family", "peak_MAE", "peak_RMSE",
            "underestimation_rate", "bias_pvalue", "biased_low"]
    t = df[[c for c in cols if c in df.columns]].copy()
    return t.sort_values(["substation", "horizon_h", "peak_MAE"])


def to_markdown(t: pd.DataFrame, float_fmt: str = "%.3f") -> str:
    d = t.copy()
    for c in d.select_dtypes("float").columns:
        d[c] = d[c].map(lambda v: "" if pd.isna(v) else float_fmt % v)
    return d.to_markdown(index=False)


# ---------------------------------------------------------------------------
# Pairwise significance
# ---------------------------------------------------------------------------

def load_predictions() -> dict:
    out = {}
    for stem in ("arima", "lstm", "transformer"):
        f = METDIR / f"{stem}_predictions.csv"
        if f.exists():
            out[stem] = pd.read_csv(f, parse_dates=["timestamp"])
            print(f"  loaded {stem}_predictions.csv")
    return out


def pairwise_dm(preds: dict) -> pd.DataFrame:
    """Diebold-Mariano for every model pair on identical target timestamps.

    Raw error differences cannot establish that one architecture beats
    another; this is what licenses the claim. Models are compared only on
    timestamps both predicted, so sample differences between runs cannot
    distort the test.
    """
    rows = []
    for a, b in itertools.combinations(sorted(preds), 2):
        da, db = preds[a], preds[b]
        for (sub, h), ga in da.groupby(["substation", "horizon_h"]):
            gb = db[(db.substation == sub) & (db.horizon_h == h)]
            if not len(gb):
                continue
            ga = ga.set_index("timestamp")
            gb = gb.set_index("timestamp")
            common = ga.index.intersection(gb.index)
            if len(common) < 100:
                continue
            actual = ga.loc[common, "actual"]
            r = diebold_mariano(actual, ga.loc[common, "prediction"],
                                gb.loc[common, "prediction"], horizon=int(h))
            rows.append({
                "substation": sub, "horizon_h": int(h),
                "model_A": a.upper(), "model_B": b.upper(),
                "MAE_A": float(np.mean(np.abs(actual - ga.loc[common, "prediction"]))),
                "MAE_B": float(np.mean(np.abs(actual - gb.loc[common, "prediction"]))),
                "dm_statistic": r["dm_statistic"], "p_value": r["p_value"],
                "verdict": r["verdict"].replace("A", a.upper()).replace(
                    "B", b.upper()),
                "n": r["n"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_mase(df: pd.DataFrame) -> None:
    """MASE by model and horizon. Scale-free, so substations are comparable."""
    fams = [f for f in ["Best naive", "ARIMA", "LSTM", "Transformer"]
            if f in df["family"].unique()]
    horizons = sorted(df["horizon_h"].unique())
    subs = sorted(df["substation"].unique())

    fig, axes = plt.subplots(1, len(horizons), figsize=(6 * len(horizons), 4.2),
                             squeeze=False)
    width = 0.8 / max(len(fams), 1)

    for ax, h in zip(axes[0], horizons):
        for i, fam in enumerate(fams):
            vals = [df[(df.substation == s) & (df.horizon_h == h) &
                       (df.family == fam)]["MASE"].mean() for s in subs]
            ax.bar(np.arange(len(subs)) + i * width, vals, width, label=fam)
        ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.6)
        ax.annotate("seasonal naive", (0.01, 1.0), xycoords=("axes fraction", "data"),
                    fontsize=7, va="bottom")
        ax.set_xticks(np.arange(len(subs)) + 0.4 - width / 2)
        ax.set_xticklabels([short_name(s) for s in subs], fontsize=8)
        ax.set(title=f"{h}-hour ahead", ylabel="MASE")
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Forecast accuracy (MASE — lower is better; "
                 "below 1.0 beats seasonal naive)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "comparison_mase.png", dpi=150)
    plt.close(fig)


def plot_peak_bias(df: pd.DataFrame) -> None:
    """Peak underestimation rate: the accuracy/reliability dissociation.

    An unbiased forecaster sits at 0.5. Points well above it underestimate
    demand during summer weekday evening peaks, which is the operationally
    dangerous direction.
    """
    if "underestimation_rate" not in df.columns:
        return
    fams = [f for f in ["Best naive", "ARIMA", "LSTM", "Transformer"]
            if f in df["family"].unique()]
    horizons = sorted(df["horizon_h"].unique())
    subs = sorted(df["substation"].unique())

    fig, axes = plt.subplots(1, len(horizons), figsize=(6 * len(horizons), 4.2),
                             squeeze=False)
    width = 0.8 / max(len(fams), 1)

    for ax, h in zip(axes[0], horizons):
        for i, fam in enumerate(fams):
            vals = [df[(df.substation == s) & (df.horizon_h == h) &
                       (df.family == fam)]["underestimation_rate"].mean()
                    for s in subs]
            ax.bar(np.arange(len(subs)) + i * width, vals, width, label=fam)
        ax.axhline(0.5, color="k", ls="--", lw=1, alpha=0.6)
        ax.annotate("unbiased", (0.01, 0.5), xycoords=("axes fraction", "data"),
                    fontsize=7, va="bottom")
        ax.set_xticks(np.arange(len(subs)) + 0.4 - width / 2)
        ax.set_xticklabels([short_name(s) for s in subs], fontsize=8)
        ax.set(title=f"{h}-hour ahead", ylabel="P(under-forecast at peak)",
               ylim=(0, 1))
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Peak-period forecast bias (summer weekday evenings)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "comparison_peak_bias.png", dpi=150)
    plt.close(fig)


def plot_overlay(preds: dict, horizon: int = 24, days: int = 7) -> None:
    """One week of predictions against actual, for the largest substation."""
    if not preds:
        return
    any_df = next(iter(preds.values()))
    g = any_df[any_df.horizon_h == horizon]
    if not len(g):
        return
    sub = g.groupby("substation")["actual"].mean().idxmax()

    fig, ax = plt.subplots(figsize=(12, 4))
    drawn = False
    for name, d in preds.items():
        d = d[(d.substation == sub) & (d.horizon_h == horizon)]
        if not len(d):
            continue
        d = d.sort_values("timestamp").head(days * 24)
        if not drawn:
            ax.plot(d.timestamp, d.actual, "k-", lw=1.8, label="actual")
            drawn = True
        ax.plot(d.timestamp, d.prediction, lw=1.1, alpha=0.85,
                label=name.upper())
    ax.set(title=f"{sub} — {horizon}h ahead, first test week", ylabel="MW")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGDIR / "predictions_overlay.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay-horizon", type=int, default=24)
    a = ap.parse_args()

    METDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    print("loading results...")
    raw = load_results()
    df = best_baseline_only(raw)

    acc = accuracy_table(df)
    acc.to_csv(METDIR / "comparison_table.csv", index=False)
    (METDIR / "comparison_table.md").write_text(to_markdown(acc))

    rel = reliability_table(df)
    rel.to_csv(METDIR / "reliability_table.csv", index=False)

    print("\nloading predictions...")
    preds = load_predictions()
    dm = pairwise_dm(preds)
    if len(dm):
        dm.to_csv(METDIR / "pairwise_dm.csv", index=False)

    plot_mase(df)
    plot_peak_bias(df)
    plot_overlay(preds, a.overlay_horizon)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 78)
    print("ACCURACY")
    print("=" * 78)
    print(acc.round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("WINNER BY SUBSTATION AND HORIZON")
    print("=" * 78)
    win = acc.loc[acc.groupby(["substation", "horizon_h"])["MAE"].idxmin()]
    print(win[["substation", "horizon_h", "family", "MAE", "MASE"]]
          .round(4).to_string(index=False))

    if len(dm):
        print("\n" + "=" * 78)
        print("PAIRWISE SIGNIFICANCE (Diebold-Mariano)")
        print("=" * 78)
        print(dm[["substation", "horizon_h", "model_A", "model_B",
                  "MAE_A", "MAE_B", "p_value", "verdict"]]
              .round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("PEAK-PERIOD BIAS  (0.5 = unbiased; higher = under-forecasts peaks)")
    print("=" * 78)
    print(rel.round(4).to_string(index=False))

    print(f"\ntables  -> {METDIR}/comparison_table.{{csv,md}}, "
          f"reliability_table.csv, pairwise_dm.csv")
    print(f"figures -> {FIGDIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())