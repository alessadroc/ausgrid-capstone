#!/usr/bin/env python3
"""Week 3 - LSTM forecaster.

Trains an LSTM per substation per horizon under several random seeds, and
tests it against ARIMA and the best naive baseline.

    python scripts/run_lstm.py --backend ridge     # fast smoke test, no torch
    python scripts/run_lstm.py                     # the real run

WHY --backend ridge EXISTS
--------------------------
It swaps the network for a linear model on the flattened window. It trains in
seconds and is not a research result - its purpose is to prove the harness is
sound (windows aligned, splits respected, metrics wired up) before committing
to an hour of training. Run it first whenever anything upstream changes.

MULTI-SEED
----------
Deep model run-to-run variance is often larger than the gap between
architectures, so every configuration is trained under `--seeds` seeds and
reported as mean +/- sd. A single-run comparison cannot support a claim that
one architecture beats another.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.lstm import (build_windows, split_arrays,  # noqa: E402
                         assert_no_leakage)
from models.baselines import all_baselines             # noqa: E402
from evaluation.metrics import (evaluate, results_frame,  # noqa: E402
                                diebold_mariano)

METDIR = Path("results/metrics")
FIGDIR = Path("results/figures")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def fit_predict_torch(tr, va, te, n_wf, n_tf, seed, args):
    from models.lstm import train_model, predict
    model, val_loss = train_model(
        tr, va, n_wf, n_tf, seed=seed, hidden=args.hidden,
        layers=args.layers, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience, verbose=not args.quiet)
    return predict(model, te, args._scaler), val_loss


def fit_predict_ridge(tr, va, te, n_wf, n_tf, seed, args):
    """Linear model on the flattened window - smoke test only."""
    from sklearn.linear_model import Ridge

    def flat(d):
        return np.hstack([d["X_seq"].reshape(len(d["y"]), -1), d["X_tgt"]])

    m = Ridge(alpha=1.0, random_state=seed) if "random_state" in \
        Ridge().get_params() else Ridge(alpha=1.0)
    m.fit(flat(tr), tr["y"])
    val_loss = float(np.mean(np.abs(m.predict(flat(va)) - va["y"])))
    mu, sd = args._scaler
    return m.predict(flat(te)) * sd + mu, val_loss


BACKENDS = {"torch": fit_predict_torch, "ridge": fit_predict_ridge}


# ---------------------------------------------------------------------------
# Per substation / horizon
# ---------------------------------------------------------------------------

def run_one(df: pd.DataFrame, sub: str, horizon: int, args):
    g = df[df.substation == sub].copy()

    data = build_windows(g, horizon=horizon, lookback=args.lookback)
    assert_no_leakage(g, data, horizon, args.lookback)   # fails loudly, early
    args._scaler = data["scaler"]

    tr = split_arrays(data, "train")
    va = split_arrays(data, "val")
    te = split_arrays(data, "test")
    if min(len(tr["y"]), len(va["y"]), len(te["y"])) < 100:
        print(f"  {sub} h={horizon}: insufficient samples, skipping")
        return [], {}

    n_wf = data["X_seq"].shape[2]
    n_tf = data["X_tgt"].shape[1]
    if not args.quiet:
        print(f"\n{sub} h={horizon}: train {len(tr['y']):,} / "
              f"val {len(va['y']):,} / test {len(te['y']):,} samples, "
              f"{n_wf} window + {n_tf} target features")

    backend = BACKENDS[args.backend]
    preds, val_losses = [], []
    for seed in range(args.seeds):
        t0 = time.time()
        p, vl = backend(tr, va, te, n_wf, n_tf, seed, args)
        preds.append(p)
        val_losses.append(vl)
        if not args.quiet:
            mae = float(np.mean(np.abs(te["y_raw"] - p)))
            print(f"    seed {seed}: test MAE={mae:.4f} "
                  f"val={vl:.5f} ({time.time() - t0:.0f}s)")

    actual = pd.Series(te["y_raw"], index=te["index"])
    per_seed_mae = [float(np.mean(np.abs(te["y_raw"] - p))) for p in preds]

    # Ensemble mean across seeds is the headline prediction; the spread across
    # seeds is reported alongside so run-to-run variance is visible.
    mean_pred = pd.Series(np.mean(preds, axis=0), index=te["index"])

    mu, sd = data["scaler"]
    train_raw = pd.Series(tr["y_raw"], index=tr["index"])

    name = f"LSTM({args.hidden}x{args.layers})" if args.backend == "torch" \
        else "Ridge(smoke-test)"
    row = evaluate(actual, mean_pred, name, sub, horizon,
                   train_series=train_raw, timestamps=actual.index)
    row["seeds"] = args.seeds
    row["mae_seed_mean"] = float(np.mean(per_seed_mae))
    row["mae_seed_sd"] = float(np.std(per_seed_mae))

    stored = {(sub, horizon, name): (actual, mean_pred)}

    # Compare against the best naive baseline on the same targets
    full = (df[df.substation == sub].sort_values("timestamp")
            .set_index("timestamp")["load_mw"].astype(float))
    best, best_mae, best_p = None, np.inf, None
    for bname, bp in all_baselines(full, horizon).items():
        bp = bp.reindex(actual.index)
        if bp.isna().mean() > 0.2:
            continue
        m = ~bp.isna()
        e = float(np.mean(np.abs(actual[m] - bp[m])))
        if e < best_mae:
            best, best_mae, best_p = bname, e, bp
    if best_p is not None:
        m = ~best_p.isna()
        dm = diebold_mariano(actual[m], mean_pred[m], best_p[m], horizon=horizon)
        row["vs_baseline"] = best
        row["baseline_MAE"] = best_mae
        row["dm_vs_baseline_p"] = dm["p_value"]
        row["dm_vs_baseline"] = dm["verdict"].replace("A", "LSTM").replace(
            "B", "baseline")

    # Compare against ARIMA if it has been run
    ap = METDIR / "arima_predictions.csv"
    if ap.exists():
        arima = pd.read_csv(ap, parse_dates=["timestamp"])
        arima = arima[(arima.substation == sub) & (arima.horizon_h == horizon)]
        if len(arima):
            s = arima.set_index("timestamp")["prediction"].reindex(actual.index)
            m = ~s.isna()
            if m.sum() > 100:
                dm = diebold_mariano(actual[m], mean_pred[m], s[m],
                                     horizon=horizon)
                row["arima_MAE"] = float(np.mean(np.abs(actual[m] - s[m])))
                row["dm_vs_arima_p"] = dm["p_value"]
                row["dm_vs_arima"] = dm["verdict"].replace(
                    "A", "LSTM").replace("B", "ARIMA")

    return [row], stored


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/hourly_features")
    ap.add_argument("--selection",
                    default="results/metrics/substation_selection.csv")
    ap.add_argument("--horizons", default="1,24")
    ap.add_argument("--backend", default="torch", choices=list(BACKENDS))
    ap.add_argument("--lookback", type=int, default=168)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    METDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data)
    horizons = tuple(int(x) for x in args.horizons.split(","))
    subs = pd.read_csv(args.selection).substation.tolist()

    print(f"backend={args.backend}  seeds={args.seeds}  "
          f"lookback={args.lookback}h")
    if args.backend == "ridge":
        print("NOTE ridge is a harness smoke test, not a research result\n")

    rows, stored = [], {}
    for sub in subs:
        for h in horizons:
            r, s = run_one(df, sub, h, args)
            rows += r
            stored.update(s)

    if not rows:
        print("no results")
        return 1

    res = results_frame(rows)
    extra = pd.DataFrame(rows)
    for c in ("seeds", "mae_seed_mean", "mae_seed_sd", "vs_baseline",
              "baseline_MAE", "dm_vs_baseline_p", "dm_vs_baseline",
              "arima_MAE", "dm_vs_arima_p", "dm_vs_arima"):
        if c in extra.columns:
            res[c] = extra[c].values

    out = METDIR / ("lstm_results.csv" if args.backend == "torch"
                    else "smoke_test_results.csv")
    res.to_csv(out, index=False)

    pd.set_option("display.width", 220)
    cols = ["model", "substation", "horizon_h", "MAE", "RMSE", "MASE",
            "mae_seed_sd", "peak_MAE", "baseline_MAE", "dm_vs_baseline",
            "arima_MAE", "dm_vs_arima"]
    print("\nRESULTS")
    print(res[[c for c in cols if c in res.columns]].round(4).to_string(index=False))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())