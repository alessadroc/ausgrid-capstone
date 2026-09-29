#!/usr/bin/env python3
"""Week 2 - ARIMA with Fourier seasonality.

Fits one ARIMA per substation, rolls it forward through the test period at
both horizons, and tests it against the best naive baseline.

    python scripts/run_arima.py                      # select order, then run
    python scripts/run_arima.py --order 2,0,1        # skip order selection

Outputs:
    results/metrics/arima_results.csv    ARIMA rows, same schema as baselines
    results/metrics/arima_dm_tests.csv   ARIMA vs best baseline
    results/metrics/all_results.csv      baselines + ARIMA combined
    results/figures/arima_predictions.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.arima import (ArimaForecaster, calendar_exog,  # noqa: E402
                          select_order)
from models.baselines import all_baselines                 # noqa: E402
from evaluation.metrics import (evaluate, results_frame,   # noqa: E402
                                diebold_mariano)

METDIR = Path("results/metrics")
FIGDIR = Path("results/figures")


def evaluation_mask(g: pd.DataFrame) -> pd.Series:
    """Imputed and outlier hours are excluded; reverse-flow hours retained."""
    return ~(g["was_imputed"].fillna(False) | g["is_outlier"].fillna(False))


def prepare(df: pd.DataFrame, sub: str):
    g = (df[df.substation == sub].sort_values("timestamp")
         .set_index("timestamp"))
    series = g["load_mw"].astype(float)

    # Interpolate any residual gaps: the state-space filter cannot accept NaN
    # in the endogenous series. These hours are excluded from evaluation
    # anyway by evaluation_mask, so this affects conditioning only.
    series = series.interpolate(limit_direction="both")

    holidays = g["is_holiday"] if "is_holiday" in g.columns else None
    exog = calendar_exog(g.index, holidays)
    return g, series, exog


def run_substation(df: pd.DataFrame, sub: str, order: tuple,
                   horizons: tuple[int, ...], verbose: bool = True,
                   origin_step: int = 1):
    g, series, exog = prepare(df, sub)

    train_idx = g.index[g["split"] == "train"]
    test_idx = g.index[g["split"] == "test"]
    keep = evaluation_mask(g)

    if verbose:
        print(f"\n{sub}: train {len(train_idx):,}h, test {len(test_idx):,}h")

    t0 = time.time()
    model = ArimaForecaster(order=order).fit(series.loc[train_idx], exog)
    if verbose:
        print(f"  fitted in {time.time() - t0:.1f}s; "
              f"AIC={model.result.aic:,.1f}")

    rows, stored = [], {}
    for h in horizons:
        t1 = time.time()
        pred = model.forecast_rolling(series, exog, test_idx, h, verbose,
                                      origin_step=origin_step)

        actual = series.loc[pred.index]
        ok = keep.loc[pred.index] & ~pred.isna() & ~actual.isna()
        if ok.sum() < 50:
            print(f"  h={h}: too few evaluable points, skipping")
            continue

        a, p = actual[ok], pred[ok]
        train_series = series.loc[train_idx][keep.loc[train_idx]]
        rows.append(evaluate(a, p, f"ARIMA{order}", sub, h,
                             train_series=train_series, timestamps=a.index))
        stored[(sub, h, "ARIMA")] = (a, p)

        # Best naive baseline on exactly the same target timestamps
        best_name, best_mae, best_pred = None, np.inf, None
        for name, bp in all_baselines(series, h).items():
            bp = bp.loc[a.index]
            if bp.isna().mean() > 0.2:
                continue
            m = ~bp.isna()
            err = float(np.mean(np.abs(a[m] - bp[m])))
            if err < best_mae:
                best_name, best_mae, best_pred = name, err, bp

        if best_pred is not None:
            m = ~best_pred.isna()
            dm = diebold_mariano(a[m], p[m], best_pred[m], horizon=h)
            rows[-1]["vs_baseline"] = best_name
            rows[-1]["dm_vs_baseline"] = dm["dm_statistic"]
            rows[-1]["dm_pvalue"] = dm["p_value"]
            rows[-1]["dm_verdict"] = dm["verdict"].replace("A", "ARIMA").replace(
                "B", "baseline")
            stored[(sub, h, best_name)] = (a[m], best_pred[m])

        if verbose:
            r = rows[-1]
            print(f"  h={h:2d}: MAE={r['MAE']:.4f} MASE={r.get('MASE', np.nan):.4f} "
                  f"vs {best_name} MAE={best_mae:.4f} "
                  f"[{r.get('dm_verdict', 'n/a')}]  ({time.time() - t1:.0f}s)")

    return rows, stored


def plot(stored: dict, sub: str, horizon: int) -> None:
    keys = [k for k in stored if k[0] == sub and k[1] == horizon]
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    a, _ = stored[keys[0]]
    w = a.index[:168]
    ax.plot(w, a.loc[w], "k-", lw=1.6, label="actual")
    for k in keys:
        _, p = stored[k]
        common = p.index.intersection(w)
        ax.plot(common, p.loc[common], lw=1.1, alpha=0.85, label=k[2])
    ax.set(title=f"{sub} - {horizon}h ahead, first test week", ylabel="MW")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / f"arima_predictions_h{horizon}.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/hourly_features")
    ap.add_argument("--selection", default="results/metrics/substation_selection.csv")
    ap.add_argument("--horizons", default="1,24")
    ap.add_argument("--order", default=None,
                    help="p,d,q - skips order selection")
    ap.add_argument("--origin-step", type=int, default=1,
                    help="hours between forecast origins for h>1 "
                         "(1 = every hour, most thorough; larger = faster)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    METDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    verbose = not a.quiet

    if a.origin_step % 24 == 0 and a.origin_step > 1:
        print(f"WARNING --origin-step {a.origin_step} is a multiple of 24, so "
              "every forecast target lands on the SAME hour of day. Peak-window "
              "metrics become degenerate. Use a step coprime with 24 (e.g. 5) "
              "or the default 1.")

    df = pd.read_parquet(a.data)
    horizons = tuple(int(x) for x in a.horizons.split(","))
    subs = pd.read_csv(a.selection).substation.tolist()

    if a.order:
        order = tuple(int(x) for x in a.order.split(","))
        print(f"using ARIMA{order}")
    else:
        # Select once on the MEDIAN substation and reuse everywhere, so the
        # model specification is identical across sites.
        ref = subs[len(subs) // 2]
        print(f"selecting order on {ref} (d=0 per ADF test)...")
        _, series, exog = prepare(df, ref)
        train = series[df[df.substation == ref].set_index("timestamp")["split"] == "train"]
        order = select_order(train, exog, verbose=verbose)

    all_rows, all_stored = [], {}
    for sub in subs:
        rows, stored = run_substation(df, sub, order, horizons, verbose,
                                      a.origin_step)
        all_rows += rows
        all_stored.update(stored)

    res = results_frame(all_rows)
    for c in ("vs_baseline", "dm_vs_baseline", "dm_pvalue", "dm_verdict"):
        if c in pd.DataFrame(all_rows).columns:
            res[c] = pd.DataFrame(all_rows)[c].values
    res.to_csv(METDIR / "arima_results.csv", index=False)

    for h in horizons:
        plot(all_stored, subs[-1], h)

    # Combine with baselines for the running comparison table
    base_path = METDIR / "baseline_results.csv"
    if base_path.exists():
        combined = pd.concat([pd.read_csv(base_path), res], ignore_index=True)
        combined.to_csv(METDIR / "all_results.csv", index=False)

    pd.set_option("display.width", 210)
    cols = ["model", "substation", "horizon_h", "MAE", "RMSE", "MASE",
            "peak_MAE", "underestimation_rate", "vs_baseline", "dm_verdict"]
    print("\nARIMA RESULTS")
    print(res[[c for c in cols if c in res.columns]].round(4).to_string(index=False))
    print(f"\n-> {METDIR}/arima_results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())