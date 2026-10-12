#!/usr/bin/env python3
"""Weeks 3-4 - deep models (LSTM and Transformer).

One script runs both architectures through an identical pipeline: same
windows, same splits, same scaler, same optimiser, loss, early stopping and
seeds. Only the encoder differs. That is what makes the comparison
controlled - two separate scripts would leave room for an unnoticed
difference in protocol to masquerade as an architecture effect.

    python scripts/run_deep.py --model transformer --backend ridge   # smoke test
    python scripts/run_deep.py --model transformer                   # real run
    python scripts/run_deep.py --model lstm                          # reproduces week 3

Outputs:
    results/metrics/{model}_results.csv
    results/metrics/{model}_predictions.csv   for later pairwise DM tests
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.lstm import (build_windows, split_arrays,       # noqa: E402
                         assert_no_leakage)
from models.transformer import (lstm_parameter_count,       # noqa: E402
                                transformer_parameter_count)
from models.baselines import all_baselines                  # noqa: E402
from evaluation.metrics import (evaluate, results_frame,    # noqa: E402
                                diebold_mariano)

METDIR = Path("results/metrics")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def fit_lstm(tr, va, te, n_wf, n_tf, seed, a):
    from models.lstm import train_model, predict
    model, vl = train_model(tr, va, n_wf, n_tf, seed=seed, hidden=a.hidden,
                            layers=a.layers, epochs=a.epochs,
                            batch_size=a.batch_size, lr=a.lr,
                            patience=a.patience, verbose=not a.quiet)
    return predict(model, te, a._scaler), vl


def fit_transformer(tr, va, te, n_wf, n_tf, seed, a):
    from models.transformer import train_model
    from models.lstm import predict
    model, vl = train_model(tr, va, n_wf, n_tf, seed=seed, d_model=a.d_model,
                            nhead=a.nhead, layers=a.layers, ff=a.d_model * 2,
                            epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                            patience=a.patience, verbose=not a.quiet)
    return predict(model, te, a._scaler), vl


def fit_ridge(tr, va, te, n_wf, n_tf, seed, a):
    """Linear smoke test - validates the harness, not a research result."""
    from sklearn.linear_model import Ridge

    def flat(d):
        return np.hstack([d["X_seq"].reshape(len(d["y"]), -1), d["X_tgt"]])

    m = Ridge(alpha=1.0).fit(flat(tr), tr["y"])
    vl = float(np.mean(np.abs(m.predict(flat(va)) - va["y"])))
    mu, sd = a._scaler
    return m.predict(flat(te)) * sd + mu, vl


# ---------------------------------------------------------------------------

def model_label(a) -> str:
    if a.backend == "ridge":
        return "Ridge(smoke-test)"
    if a.model == "lstm":
        return f"LSTM({a.hidden}x{a.layers})"
    return f"Transformer(d{a.d_model}x{a.layers})"


def run_one(df, sub, horizon, a):
    g = df[df.substation == sub].copy()
    data = build_windows(g, horizon=horizon, lookback=a.lookback)
    assert_no_leakage(g, data, horizon, a.lookback)
    a._scaler = data["scaler"]

    tr, va, te = (split_arrays(data, s) for s in ("train", "val", "test"))
    if min(len(tr["y"]), len(va["y"]), len(te["y"])) < 100:
        print(f"  {sub} h={horizon}: too few samples, skipping")
        return [], None

    n_wf, n_tf = data["X_seq"].shape[2], data["X_tgt"].shape[1]
    if not a.quiet:
        print(f"\n{sub} h={horizon}: train {len(tr['y']):,} / "
              f"val {len(va['y']):,} / test {len(te['y']):,}")

    fit = {"lstm": fit_lstm, "transformer": fit_transformer,
           "ridge": fit_ridge}[a.backend if a.backend == "ridge" else a.model]

    preds, maes = [], []
    for seed in range(a.seeds):
        t0 = time.time()
        p, vl = fit(tr, va, te, n_wf, n_tf, seed, a)
        preds.append(p)
        maes.append(float(np.mean(np.abs(te["y_raw"] - p))))
        if not a.quiet:
            print(f"    seed {seed}: test MAE={maes[-1]:.4f} val={vl:.5f} "
                  f"({time.time() - t0:.0f}s)")

    actual = pd.Series(te["y_raw"], index=te["index"])
    mean_pred = pd.Series(np.mean(preds, axis=0), index=te["index"])
    train_raw = pd.Series(tr["y_raw"], index=tr["index"])

    name = model_label(a)
    row = evaluate(actual, mean_pred, name, sub, horizon,
                   train_series=train_raw, timestamps=actual.index)
    row["seeds"] = a.seeds
    row["mae_seed_mean"] = float(np.mean(maes))
    row["mae_seed_sd"] = float(np.std(maes))

    # vs best naive baseline
    full = (df[df.substation == sub].sort_values("timestamp")
            .set_index("timestamp")["load_mw"].astype(float))
    best, best_mae, best_p = None, np.inf, None
    for bn, bp in all_baselines(full, horizon).items():
        bp = bp.reindex(actual.index)
        if bp.isna().mean() > 0.2:
            continue
        m = ~bp.isna()
        e = float(np.mean(np.abs(actual[m] - bp[m])))
        if e < best_mae:
            best, best_mae, best_p = bn, e, bp
    if best_p is not None:
        m = ~best_p.isna()
        dm = diebold_mariano(actual[m], mean_pred[m], best_p[m], horizon=horizon)
        row.update({"vs_baseline": best, "baseline_MAE": best_mae,
                    "dm_vs_baseline_p": dm["p_value"],
                    "dm_vs_baseline": dm["verdict"].replace("A", name)
                    .replace("B", "baseline")})

    # vs every previously saved model's predictions
    for other in ("arima", "lstm", "transformer"):
        if other == a.model and a.backend != "ridge":
            continue
        f = METDIR / f"{other}_predictions.csv"
        if not f.exists():
            continue
        o = pd.read_csv(f, parse_dates=["timestamp"])
        o = o[(o.substation == sub) & (o.horizon_h == horizon)]
        if not len(o):
            continue
        s = o.set_index("timestamp")["prediction"].reindex(actual.index)
        m = ~s.isna()
        if m.sum() < 100:
            continue
        dm = diebold_mariano(actual[m], mean_pred[m], s[m], horizon=horizon)
        row[f"{other}_MAE"] = float(np.mean(np.abs(actual[m] - s[m])))
        row[f"dm_vs_{other}_p"] = dm["p_value"]
        row[f"dm_vs_{other}"] = dm["verdict"].replace("A", name).replace(
            "B", other.upper())

    pred_df = pd.DataFrame({"timestamp": actual.index, "substation": sub,
                            "horizon_h": horizon,
                            "prediction": mean_pred.values,
                            "actual": actual.values})
    return [row], pred_df


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="transformer",
                   choices=["lstm", "transformer"])
    p.add_argument("--backend", default=None,
                   choices=["ridge"], help="ridge = harness smoke test")
    p.add_argument("--data", default="data/processed/hourly_features")
    p.add_argument("--selection",
                   default="results/metrics/substation_selection.csv")
    p.add_argument("--horizons", default="1,24")
    p.add_argument("--lookback", type=int, default=168)
    p.add_argument("--hidden", type=int, default=64)      # LSTM
    p.add_argument("--d-model", type=int, default=48)     # Transformer
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()
    if a.backend is None:
        a.backend = a.model

    METDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(a.data)
    horizons = tuple(int(x) for x in a.horizons.split(","))
    subs = pd.read_csv(a.selection).substation.tolist()

    # Capacity parity is a methodological claim - state the numbers.
    nwf, ntf = 9, 8
    lp = lstm_parameter_count(nwf, ntf, a.hidden, a.layers)
    tp = transformer_parameter_count(nwf, ntf, a.d_model, a.d_model * 2,
                                     a.layers)
    print(f"model={model_label(a)}  seeds={a.seeds}  lookback={a.lookback}h")
    print(f"capacity: LSTM ~{lp:,} params | Transformer ~{tp:,} params "
          f"({100 * tp / lp - 100:+.1f}%)")
    if a.backend == "ridge":
        print("NOTE ridge is a harness smoke test, not a research result")

    rows, preds = [], []
    for sub in subs:
        for h in horizons:
            r, pdf = run_one(df, sub, h, a)
            rows += r
            if pdf is not None:
                preds.append(pdf)

    if not rows:
        print("no results")
        return 1

    res = results_frame(rows)
    extra = pd.DataFrame(rows)
    for c in extra.columns:
        if c not in res.columns:
            res[c] = extra[c].values

    tag = "smoke_test" if a.backend == "ridge" else a.model
    res.to_csv(METDIR / f"{tag}_results.csv", index=False)
    if preds and a.backend != "ridge":
        pd.concat(preds, ignore_index=True).to_csv(
            METDIR / f"{tag}_predictions.csv", index=False)

    pd.set_option("display.width", 240)
    cols = ["model", "substation", "horizon_h", "MAE", "RMSE", "MASE",
            "mae_seed_sd", "peak_MAE", "underestimation_rate",
            "baseline_MAE", "arima_MAE", "lstm_MAE", "dm_vs_lstm"]
    print("\nRESULTS")
    print(res[[c for c in cols if c in res.columns]].round(4)
          .to_string(index=False))
    print(f"\n-> {METDIR}/{tag}_results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())