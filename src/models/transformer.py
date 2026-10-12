"""Transformer encoder for substation demand forecasting.
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Capacity accounting
# ---------------------------------------------------------------------------

def lstm_parameter_count(n_window_features: int, n_target_features: int,
                         hidden: int = 64, layers: int = 1) -> int:
    """Analytic parameter count for the LSTM, for capacity matching."""
    total = 0
    in_size = n_window_features
    for _ in range(layers):
        # 4 gates: input-hidden, hidden-hidden, and two bias vectors
        total += 4 * (in_size * hidden + hidden * hidden + 2 * hidden)
        in_size = hidden
    # head: Linear(hidden + target -> hidden), Linear(hidden -> 1)
    total += (hidden + n_target_features) * hidden + hidden
    total += hidden + 1
    return total


def transformer_parameter_count(n_window_features: int, n_target_features: int,
                                d_model: int = 48, ff: int = 96,
                                layers: int = 1, head_hidden: int = 64) -> int:
    total = n_window_features * d_model + d_model          # input projection
    for _ in range(layers):
        total += 4 * (d_model * d_model + d_model)         # q,k,v,out
        total += d_model * ff + ff + ff * d_model + d_model  # feed-forward
        total += 4 * d_model                               # two layer norms
    total += (d_model + n_target_features) * head_hidden + head_hidden
    total += head_hidden + 1
    return total


def suggest_d_model(n_window_features: int, n_target_features: int,
                    target_params: int, nhead: int = 4,
                    ff_ratio: int = 2, layers: int = 1) -> int:
    """Smallest d_model (divisible by nhead) whose count is nearest target."""
    best, best_gap = nhead, math.inf
    for d in range(nhead, 257, nhead):
        n = transformer_parameter_count(n_window_features, n_target_features,
                                        d, d * ff_ratio, layers)
        gap = abs(n - target_params)
        if gap < best_gap:
            best, best_gap = d, gap
    return best


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(n_window_features: int, n_target_features: int,
                d_model: int = 48, nhead: int = 4, layers: int = 1,
                ff: int = 96, dropout: float = 0.1, head_hidden: int = 64,
                pooling: str = "last"):
    """Encoder-only Transformer over the lookback window.

    Encoder-only because this is a regression onto a single scalar, not a
    sequence-to-sequence generation task: there is nothing for a decoder to
    autoregress over. The window is projected to d_model, positionally
    encoded, passed through self-attention layers, pooled, concatenated with
    the target's known-ahead calendar features, and mapped to one output -
    structurally identical to the LSTM apart from the encoder itself.
    """
    import torch
    from torch import nn

    class PositionalEncoding(nn.Module):
        """Fixed sinusoidal encoding (Vaswani et al., 2017).

        Fixed rather than learned: with ~14k training windows a learned
        embedding for 168 positions is a lot of free parameters to spend on
        something the sinusoid already represents well, and it would break
        the capacity matching.
        """

        def __init__(self, d: int, max_len: int = 1024):
            super().__init__()
            pe = torch.zeros(max_len, d)
            pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(torch.arange(0, d, 2).float()
                            * (-math.log(10000.0) / d))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1), :]

    class LoadTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(n_window_features, d_model)
            self.pos = PositionalEncoding(d_model)
            enc = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=ff,
                dropout=dropout, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
            self.pooling = pooling
            self.head = nn.Sequential(
                nn.Linear(d_model + n_target_features, head_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )

        def forward(self, x_seq, x_tgt):
            h = self.pos(self.proj(x_seq))
            h = self.encoder(h)
            # "last" mirrors the LSTM's final hidden state, keeping the only
            # difference between the two models the encoder itself.
            h = h[:, -1, :] if self.pooling == "last" else h.mean(dim=1)
            return self.head(torch.cat([h, x_tgt], dim=1)).squeeze(-1)

    return LoadTransformer()


def train_model(train: dict, val: dict, n_window_features: int,
                n_target_features: int, seed: int = 0, d_model: int = 48,
                nhead: int = 4, layers: int = 1, ff: int = 96,
                epochs: int = 40, batch_size: int = 256, lr: float = 1e-3,
                patience: int = 5, verbose: bool = True):
    """Same optimiser, loss, early stopping and seeding as the LSTM.

    Identical training protocol is part of the controlled comparison: if the
    two models differed in loss function or stopping rule, a performance gap
    could not be attributed to architecture.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(n_window_features, n_target_features, d_model,
                        nhead, layers, ff)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    def tensors(d):
        return (torch.from_numpy(d["X_seq"]), torch.from_numpy(d["X_tgt"]),
                torch.from_numpy(d["y"]))

    tr = DataLoader(TensorDataset(*tensors(train)), batch_size=batch_size,
                    shuffle=True)
    vx, vt, vy = tensors(val)

    best_loss, best_state, bad = np.inf, None, 0
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
            best_loss, bad = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        if verbose and (epoch % 5 == 0 or bad >= patience):
            print(f"      epoch {epoch:3d}  val_MAE(scaled)={vloss:.5f}"
                  f"{'  *' if bad == 0 else ''}", flush=True)
        if bad >= patience:
            if verbose:
                print(f"      early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss