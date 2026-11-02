#!/usr/bin/env python3
"""Regenerate every results CSV from saved artifacts, without retraining.

Baselines are recomputed directly from the processed data, exactly as
scripts/run_baselines.py does. ARIMA, LSTM and Transformer metrics are
recomputed from their saved ensemble predictions. Columns that cannot be
recovered from those predictions (seed counts and per-seed MAE spread, plus
the DM-vs-baseline columns written at training time) are carried over from
the existing CSVs on (model, substation, horizon_h).

    python scripts/recompute_metrics.py

Outputs:
    results/metrics/{baseline,arima,lstm,transformer}_results.csv
    results/metrics/all_results.csv          baselines + ARIMA, as run_arima.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluation.metrics import evaluate, results_frame  # noqa: E402
from run_baselines import evaluation_mask                # noqa: E402
from run_baselines import run as run_baselines           # noqa: E402

METDIR = Path("results/metrics")
KEY = ["model", "substation", "horizon_h"]

# Fields evaluate() returns beyond results_frame's column list. They are
# recomputed, not preserved, wherever the existing CSV already carries them.
EXTRA_RECOMPUTED = ("peak_threshold_MW", "mean_peak_error_MW")


def train_series(df: pd.DataFrame, sub: str) -> pd.Series:
    """Training-split load for one substation, imputed and outlier hours out.

    Every model at a substation shares this MASE denominator. The original
    deep-model runs scaled by their windowed training targets instead (which
    drop the first lookback hours), so their MASE differed by up to 0.45%.
    """
    g = (df[df.substation == sub]
         .sort_values("timestamp")
         .set_index("timestamp"))
    return g["load_mw"][(g["split"] == "train") & evaluation_mask(g)]


def recompute_model(stem: str, df: pd.DataFrame,
                    substations: list[str]) -> pd.DataFrame:
    # round_trip so preserved floats survive repeated read/write unchanged
    old = pd.read_csv(METDIR / f"{stem}_results.csv",
                      float_precision="round_trip")
    preds = pd.read_csv(METDIR / f"{stem}_predictions.csv",
                        parse_dates=["timestamp"])

    rows = []
    for (sub, h), g in preds.groupby(["substation", "horizon_h"], sort=False):
        if sub not in substations:
            continue
        name = old.loc[(old.substation == sub) & (old.horizon_h == h), "model"]
        if len(name) != 1:
            raise ValueError(f"{stem}: expected one model row for {sub} h={h}, "
                             f"found {len(name)}")
        g = g.sort_values("timestamp").set_index("timestamp")
        rows.append(evaluate(g["actual"], g["prediction"], name.iloc[0], sub,
                             int(h), train_series=train_series(df, sub),
                             timestamps=g.index))

    new = results_frame(rows)
    extra = pd.DataFrame(rows)
    for c in EXTRA_RECOMPUTED:
        if c in old.columns:
            new[c] = extra[c].values

    # Carry over whatever the predictions cannot reproduce
    keep = [c for c in old.columns
            if c not in new.columns and c != "n_peak"]
    out = new.merge(old[KEY + keep], on=KEY, how="left", validate="1:1")
    missing = out[keep].isna().all(axis=1) if keep else pd.Series(False)
    if missing.any():
        print(f"  WARNING {stem}: {int(missing.sum())} rows had no match in "
              f"the existing CSV for preserved columns")

    # Old column order, with n_peak replaced by the two explicit counts
    order = []
    for c in old.columns:
        order += (["n_peak_quantile", "n_peak_window"] if c == "n_peak" else [c])
    order += [c for c in out.columns if c not in order]
    return out[[c for c in order if c in out.columns]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/hourly_features")
    ap.add_argument("--selection",
                    default="results/metrics/substation_selection.csv")
    ap.add_argument("--horizons", default="1,24")
    a = ap.parse_args()

    df = pd.read_parquet(a.data)
    substations = pd.read_csv(a.selection).substation.tolist()
    horizons = tuple(int(x) for x in a.horizons.split(","))
    print(f"substations: {substations}")

    base, _, _ = run_baselines(df, substations, horizons)
    base.to_csv(METDIR / "baseline_results.csv", index=False)
    print(f"  baseline_results.csv     {len(base)} rows")

    for stem in ("arima", "lstm", "transformer"):
        res = recompute_model(stem, df, substations)
        res.to_csv(METDIR / f"{stem}_results.csv", index=False)
        print(f"  {stem}_results.csv{' ' * (14 - len(stem))}{len(res)} rows")
        if stem == "arima":
            pd.concat([base, res], ignore_index=True).to_csv(
                METDIR / "all_results.csv", index=False)

    print(f"\n-> {METDIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
