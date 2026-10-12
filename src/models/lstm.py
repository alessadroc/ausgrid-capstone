"""LSTM forecaster for substation demand.

STRUCTURE
---------
The module is split deliberately:

  * `build_windows` and friends are pure numpy/pandas. Every subtle bug in a
    sequence model lives here - off-by-one windows, leaked features, scaling
    fitted on the wrong split - so this layer is written to be testable in
    isolation and is covered by tests/test_lstm_data.py.
  * `LoadLSTM` and `train_model` are standard PyTorch and are imported lazily,
    so the data layer can be tested without torch installed.

FEATURE AVAILABILITY
--------------------
A forecast for t+h is issued at t. Two kinds of input are therefore legitimate:

  1. The observed window  y(t-L+1) ... y(t), plus its calendar features.
     This is the sequence fed to the LSTM.
  2. Calendar features OF THE TARGET, t+h. Hour of day, day of week and
     public holidays are deterministic functions of the clock, known
     arbitrarily far ahead, so using them at any horizon is not leakage -
     and they matter enormously at h=24, where the model must otherwise infer
     the target's time of day from the window's phase alone.

What is NOT legitimate is any observed load between t+1 and t+h. The window
always ends at t, and the gap to the target is exactly h, so those values are
never seen. `assert_no_leakage` checks this property directly.

MULTI-SEED PROTOCOL
-------------------
Run-to-run variance in deep models is frequently larger than the difference
between architectures, so a single run cannot support a claim that one model
beats another. Every configuration is trained under several seeds and results
are reported as mean +/- standard deviation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Sequence inputs: the load history plus its calendar context
WINDOW_FEATURES = ["load_scaled", "hour_sin", "hour_cos",
                   "dayofweek_sin", "dayofweek_cos",
                   "dayofyear_sin", "dayofyear_cos",
                   "is_weekend", "is_holiday"]

# Known-ahead features describing the TARGET timestamp
TARGET_FEATURES = ["hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos",
                   "dayofyear_sin", "dayofyear_cos", "is_weekend", "is_holiday"]


# ---------------------------------------------------------------------------
# Data preparation (pure numpy - fully testable without torch)
# ---------------------------------------------------------------------------

def scaling_stats(df: pd.DataFrame, target_col: str = "load_mw") -> tuple:
    """Mean and standard deviation from the TRAINING split only.

    Fitting on the whole series leaks test-period level and spread into the
    inputs, which inflates validation scores in a way that never reproduces
    on genuinely unseen data.
    """
    train = df.loc[df["split"] == "train", target_col].dropna()
    mu = float(train.mean())
    sd = float(train.std())
    return mu, (sd if sd > 0 else 1.0)


def build_windows(df: pd.DataFrame, horizon: int, lookback: int = 168,
                  target_col: str = "load_mw",
                  window_features=WINDOW_FEATURES,
                  target_features=TARGET_FEATURES,
                  require_valid_target: bool = True):
    """Construct (X_seq, X_target, y, target_index, split_labels).

    For a window ending at position i-1 the target sits at i-1+h, so the last
    observed input and the target are exactly `horizon` steps apart. Nothing
    between them enters the model.

    A sample is kept only if its window contains no NaN and its target is a
    real observation - imputed and outlier-flagged targets are dropped, since
    scoring against our own interpolation measures the wrong thing.
    """
    d = df.sort_values("timestamp").reset_index(drop=True)

    mu, sd = scaling_stats(d, target_col)
    d = d.copy()
    d["load_scaled"] = (d[target_col] - mu) / sd

    wf = [c for c in window_features if c in d.columns]
    tf = [c for c in target_features if c in d.columns]

    W = d[wf].to_numpy(dtype=np.float32)
    T = d[tf].to_numpy(dtype=np.float32)
    y = d["load_scaled"].to_numpy(dtype=np.float32)
    raw = d[target_col].to_numpy(dtype=np.float64)

    bad_target = np.zeros(len(d), dtype=bool)
    for col in ("was_imputed", "is_outlier"):
        if col in d.columns:
            bad_target |= d[col].fillna(False).to_numpy(dtype=bool)

    n = len(d)
    X_seq, X_tgt, Y, idx, splits, raw_y = [], [], [], [], [], []

    for i in range(lookback, n - horizon + 1):
        t_idx = i - 1 + horizon              # target position
        if t_idx >= n:
            break
        win = W[i - lookback:i]              # ends at i-1
        if np.isnan(win).any() or np.isnan(y[t_idx]):
            continue
        if require_valid_target and bad_target[t_idx]:
            continue
        X_seq.append(win)
        X_tgt.append(T[t_idx])
        Y.append(y[t_idx])
        raw_y.append(raw[t_idx])
        idx.append(d["timestamp"].iloc[t_idx])
        splits.append(d["split"].iloc[t_idx])

    return {
        "X_seq": np.asarray(X_seq, dtype=np.float32),
        "X_tgt": np.asarray(X_tgt, dtype=np.float32),
        "y": np.asarray(Y, dtype=np.float32),
        "y_raw": np.asarray(raw_y, dtype=np.float64),
        "index": pd.DatetimeIndex(idx),
        "split": np.asarray(splits),
        "scaler": (mu, sd),
        "window_features": wf,
        "target_features": tf,
    }


def split_arrays(data: dict, split: str) -> dict:
    m = data["split"] == split
    return {"X_seq": data["X_seq"][m], "X_tgt": data["X_tgt"][m],
            "y": data["y"][m], "y_raw": data["y_raw"][m],
            "index": data["index"][m]}


def assert_no_leakage(df: pd.DataFrame, data: dict, horizon: int,
                      lookback: int, target_col: str = "load_mw") -> None:
    """Verify the gap between last input and target is exactly `horizon`.

    Checks the property directly against the source frame rather than trusting
    the loop: for a sample of windows, the final load value in the window must
    equal the observed load exactly `horizon` steps before the target.
    """
    d = df.sort_values("timestamp").reset_index(drop=True)
    mu, sd = data["scaler"]
    load_pos = data["window_features"].index("load_scaled")
    ts_to_pos = {t: i for i, t in enumerate(d["timestamp"])}

    rng = np.random.default_rng(0)
    for k in rng.choice(len(data["y"]), size=min(50, len(data["y"])),
                        replace=False):
        target_time = data["index"][k]
        tpos = ts_to_pos[target_time]
        last_input = data["X_seq"][k, -1, load_pos] * sd + mu
        expected = d[target_col].iloc[tpos - horizon]
        if not np.isclose(last_input, expected, rtol=1e-4, atol=1e-6):
            raise AssertionError(
                f"leakage check failed at {target_time}: last window value "
                f"{last_input:.6f} != load at t-{horizon} ({expected:.6f})")


# ---------------------------------------------------------------------------
# Model (PyTorch - imported lazily)
# ---------------------------------------------------------------------------

def build_model(n_window_features: int, n_target_features: int,
                hidden: int = 64, layers: int = 1, dropout: float = 0.1):
    """LSTM encoder over the history, concatenated with target calendar.

    Deliberately small. The research question is whether architecture matters,
    not whether a larger model fits better, so the LSTM and the Transformer
    must be kept to a comparable parameter count. A single 64-unit layer also
    trains in minutes on CPU, which matters on a laptop.
    """
    import torch
    from torch import nn

    class LoadLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_window_features, hidden, num_layers=layers,
                                batch_first=True,
                                dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Sequential(
                nn.Linear(hidden + n_target_features, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

        def forward(self, x_seq, x_tgt):
            out, _ = self.lstm(x_seq)
            last = out[:, -1, :]                    # final hidden state
            return self.head(torch.cat([last, x_tgt], dim=1)).squeeze(-1)

    return LoadLSTM()


def train_model(train: dict, val: dict, n_window_features: int,
                n_target_features: int, seed: int = 0, hidden: int = 64,
                layers: int = 1, epochs: int = 40, batch_size: int = 256,
                lr: float = 1e-3, patience: int = 5, verbose: bool = True):
    """Train with early stopping on validation MAE.

    Early stopping uses the validation split only; the test split is never
    touched during training or model selection.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(n_window_features, n_target_features, hidden, layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()          # MAE - matches the reported metric

    def tensors(d):
        return (torch.from_numpy(d["X_seq"]), torch.from_numpy(d["X_tgt"]),
                torch.from_numpy(d["y"]))

    tr = DataLoader(TensorDataset(*tensors(train)), batch_size=batch_size,
                    shuffle=True)
    vx, vt, vy = tensors(val)

    best_loss, best_state, bad_epochs = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        for xb, tb, yb in tr:
            opt.zero_grad()
            loss = loss_fn(model(xb, tb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(model(vx, vt), vy))

        if vloss < best_loss - 1e-5:
            best_loss, bad_epochs = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        if verbose and (epoch % 5 == 0 or bad_epochs >= patience):
            print(f"      epoch {epoch:3d}  val_MAE(scaled)={vloss:.5f}"
                  f"{'  *' if bad_epochs == 0 else ''}", flush=True)

        if bad_epochs >= patience:
            if verbose:
                print(f"      early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss


def predict(model, data: dict, scaler: tuple) -> np.ndarray:
    """Predictions in original MW units."""
    import torch

    mu, sd = scaler
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(data["X_seq"]),
                    torch.from_numpy(data["X_tgt"])).numpy()
    return out * sd + mu