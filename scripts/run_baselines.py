#!/usr/bin/env python3
"""Week 1 - naive baseline forecasts.

Evaluates persistence and seasonal naive forecasters on the test split, at
both horizons, for the selected substations. This produces the first real
numbers in the project and the benchmark every later model must beat.

    python scripts/run_baselines.py

Outputs:
    results/metrics/baseline_results.csv     one row per model/substation/horizon
    results/metrics/baseline_dm_tests.csv    pairwise significance tests
    results/figures/baseline_predictions.png a week of predictions vs actual
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.baselines import all_baselines          # noqa: E402
from evaluation.metrics import (evaluate, results_frame,  # noqa: E402
                                diebold_mariano)

METDIR = Path("results/metrics")
FIGDIR = Path("results/figures")


def evaluation_mask(g: pd.DataFrame) -> pd.Series:
    """Rows admissible as evaluation targets.

    Imputed and outlier-flagged hours are excluded. Scoring a model on an
    imputed value measures how well it reproduces our own interpolation, not
    how well it forecasts demand.
    """
    mask = ~(g["was_imputed"].fillna(False) | g["is_outlier"].fillna(False))
    # Reverse-flow hours are deliberately RETAINED. Negative net load is real
    # rooftop-PV export, and minimum-demand conditions are an operational
    # concern in their own right - excluding them would bias the evaluation
    # away from exactly the hours the network cares about.
    return mask


def run(df: pd.DataFrame, substations: list[str],
        horizons: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, dm_rows, stored = [], [], {}

    for sub in substations:
        g = (df[df.substation == sub]
             .sort_values("timestamp")
             .set_index("timestamp"))
        series = g["load_mw"]

        for h in horizons:
            preds = all_baselines(series, h)

            test = (g["split"] == "test") & evaluation_mask(g)
            actual = series[test]
            train_series = series[(g["split"] == "train") & evaluation_mask(g)]

            for name, pred in preds.items():
                p = pred[test]
                ok = ~(actual.isna() | p.isna())
                if ok.sum() < 100:
                    continue
                rows.append(evaluate(actual[ok], p[ok], name, sub, h,
                                     train_series=train_series,
                                     timestamps=actual[ok].index))
                stored[(sub, h, name)] = (actual[ok], p[ok])

            # Pairwise significance against the weakest baseline
            names = [n for n in preds if (sub, h, n) in stored]
            if "persistence(lag=%dh)" % h in names:
                ref = "persistence(lag=%dh)" % h
                for other in names:
                    if other == ref:
                        continue
                    a1, p1 = stored[(sub, h, other)]
                    a2, p2 = stored[(sub, h, ref)]
                    common = a1.index.intersection(a2.index)
                    r = diebold_mariano(a1.loc[common], p1.loc[common],
                                        p2.loc[common], horizon=h)
                    dm_rows.append({"substation": sub, "horizon_h": h,
                                    "model_A": other, "model_B": ref, **r})

    return results_frame(rows), pd.DataFrame(dm_rows), stored


def plot_predictions(stored: dict, sub: str, horizon: int) -> None:
    keys = [k for k in stored if k[0] == sub and k[1] == horizon]
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    actual, _ = stored[keys[0]]
    window = actual.index[:168]
    ax.plot(window, actual.loc[window], "k-", lw=1.6, label="actual")
    for k in keys:
        _, pred = stored[k]
        ax.plot(window, pred.loc[window], lw=0.9, alpha=0.8, label=k[2])
    ax.set(title=f"{sub} - {horizon}h ahead, first test week", ylabel="MW")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "baseline_predictions.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/hourly_features")
    ap.add_argument("--selection", default="results/metrics/substation_selection.csv")
    ap.add_argument("--horizons", default="1,24")
    a = ap.parse_args()

    METDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(a.data)
    horizons = tuple(int(x) for x in a.horizons.split(","))

    sel = Path(a.selection)
    if sel.exists():
        substations = pd.read_csv(sel).substation.tolist()
    else:
        print("no selection file - run scripts/run_eda.py first; "
              "falling back to the three largest substations")
        substations = (df.groupby("substation").load_mw.mean()
                         .nlargest(3).index.tolist())

    print(f"substations: {substations}")
    print(f"horizons: {horizons}\n")

    results, dm, stored = run(df, substations, horizons)
    results.to_csv(METDIR / "baseline_results.csv", index=False)
    dm.to_csv(METDIR / "baseline_dm_tests.csv", index=False)
    plot_predictions(stored, substations[-1], horizons[0])

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("BASELINE RESULTS")
    cols = ["model", "substation", "horizon_h", "MAE", "RMSE", "MASE",
            "peak_MAE", "underestimation_rate"]
    print(results[[c for c in cols if c in results.columns]]
          .round(4).to_string(index=False))

    print("\nBEST BASELINE PER SUBSTATION / HORIZON  (the bar to beat)")
    best = results.loc[results.groupby(["substation", "horizon_h"]).MAE.idxmin()]
    print(best[["substation", "horizon_h", "model", "MAE", "RMSE"]]
          .round(4).to_string(index=False))

    print(f"\n-> {METDIR}/baseline_results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())