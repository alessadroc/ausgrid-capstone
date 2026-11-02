#!/usr/bin/env python3
"""Publication figures for the results chapter.

Everything is derived from the processed dataset and the saved predictions;
no model is retrained. Figures are sized for a half-page-width column in the
report (3.5 in wide), so fonts are set at their printed size.

    python scripts/make_report_figures.py

Outputs to results/figures/:
    fig_network_distribution.png   mean demand across eligible substations
    fig_reverse_flow.png           seasonal daily profile, Telarah 33/11 kV
    fig_error_by_hour.png          MAE by hour of day, 24-hour horizon
    fig_peak_scatter.png           Transformer predicted vs actual at peak
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluation.metrics import operational_peak_mask  # noqa: E402
from run_baselines import evaluation_mask              # noqa: E402
from run_baselines import run as run_baselines         # noqa: E402
from run_comparison import short_name                  # noqa: E402

METDIR = Path("results/metrics")
FIGDIR = Path("results/figures")

WIDTH = 3.5          # inches: half of an A4 text block
DPI = 150

# Categorical slots in fixed order (validated: adjacent CVD dE >= 9.1). The
# lighter two sit below 3:1 on white, so every figure also carries a legend
# and a marker or dash pattern per series.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"
SHADE = "#e9e8e4"

SEASONS = {"Summer (Dec-Feb)": (12, 1, 2), "Autumn (Mar-May)": (3, 4, 5),
           "Winter (Jun-Aug)": (6, 7, 8), "Spring (Sep-Nov)": (9, 10, 11)}
PEAK_HOURS = (16, 20)

COLUMNS = ["timestamp", "substation", "load_mw", "was_imputed", "is_outlier",
           "split", "hour", "month"]


def style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "figure.titlesize": 9, "axes.edgecolor": INK_2,
        "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": SHADE, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "lines.linewidth": 1.2,
        "legend.frameon": False, "savefig.dpi": DPI,
    })


def save(fig, name: str) -> None:
    fig.savefig(FIGDIR / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {FIGDIR / name}")


# ---------------------------------------------------------------------------
# Shared derivations (also used by make_chapter4_summary.py)
# ---------------------------------------------------------------------------

def substation_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-substation summary, computed as run_eda.select_substations does."""
    return (df.groupby("substation")
              .agg(mean_mw=("load_mw", "mean"),
                   max_mw=("load_mw", "max"),
                   n_hours=("load_mw", "size"),
                   min_mw=("load_mw", "min"),
                   imputed_frac=("was_imputed", "mean"),
                   outlier_frac=("is_outlier", "mean"),
                   reverse_flow_frac=("load_mw",
                                      lambda x: float((x < 0).mean())))
              .reset_index())


def eligible_substations(df: pd.DataFrame) -> pd.DataFrame:
    """Sites passing the quality gate (>=95% coverage, <=5% imputed hours)."""
    stats = substation_stats(df)
    max_hours = stats.n_hours.max()
    return stats[(stats.n_hours >= 0.95 * max_hours) &
                 (stats.imputed_frac <= 0.05)].copy()


def season_of(month: pd.Series) -> pd.Series:
    lookup = {m: s for s, months in SEASONS.items() for m in months}
    return month.map(lookup)


def best_baseline_name(horizon: int) -> pd.Series:
    """Lowest-MAE naive method per substation, as run_comparison picks it."""
    b = pd.read_csv(METDIR / "baseline_results.csv")
    b = b[b.horizon_h == horizon]
    return b.loc[b.groupby("substation").MAE.idxmin()].set_index(
        "substation")["model"]


def model_predictions(df: pd.DataFrame, subs: list[str],
                      horizon: int) -> dict:
    """{(substation, label): (actual, prediction)} for the four families."""
    _, _, stored = run_baselines(df, subs, (horizon,))
    best = best_baseline_name(horizon)
    out = {}
    for sub in subs:
        out[(sub, "Best naive")] = stored[(sub, horizon, best[sub])]
    for stem, label in (("arima", "ARIMA"), ("lstm", "LSTM"),
                        ("transformer", "Transformer")):
        p = pd.read_csv(METDIR / f"{stem}_predictions.csv",
                        parse_dates=["timestamp"])
        p = p[p.horizon_h == horizon]
        for sub, g in p.groupby("substation"):
            g = g.set_index("timestamp").sort_index()
            out[(sub, label)] = (g["actual"], g["prediction"])
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_network_distribution(df: pd.DataFrame, selection: pd.DataFrame) -> None:
    elig = eligible_substations(df)
    edges = pd.qcut(elig.mean_mw, 3, retbins=True)[1]

    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    ax.hist(elig.mean_mw, bins=np.arange(0, elig.mean_mw.max() + 2, 2),
            color=SERIES[0], edgecolor="white", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ymax = ax.get_ylim()[1]
    xmin, xmax = ax.get_xlim()
    top = ymax * 1.45
    ax.set_ylim(0, top)

    # Tercile names sit just above the frame; selected sites are labelled
    # in the headroom at staggered heights so neighbours never collide.
    for x in edges[1:-1]:
        ax.axvline(x, color=MUTED, ls="--", lw=0.9)
    for label, lo, hi in zip(("small", "medium", "large"), edges[:-1],
                             edges[1:]):
        ax.text((max(lo, xmin) + min(hi, xmax)) / 2, top, label,
                ha="center", va="bottom", fontsize=7, color=INK_2)

    for i, (_, r) in enumerate(selection.sort_values("mean_mw").iterrows()):
        y = ymax * (1.36 - 0.13 * i)
        ax.plot([r.mean_mw, r.mean_mw], [0, y], color=INK, lw=1.2)
        ax.text(r.mean_mw + 0.6, y, short_name(r.substation), ha="left",
                va="center", fontsize=7, color=INK)

    ax.set(xlabel="Mean hourly demand (MW)",
           ylabel=f"Substations (n = {len(elig)})")
    ax.set_title("Distribution of mean demand across eligible substations",
                 loc="left", pad=12)
    save(fig, "fig_network_distribution.png")


def fig_reverse_flow(df: pd.DataFrame, sub: str = "Telarah 33_11kV") -> None:
    g = df[df.substation == sub]
    g = g[evaluation_mask(g)]
    season = season_of(g.month)
    mean = g.groupby([season, g.hour]).load_mw.mean().unstack(0)
    neg = (g.load_mw < 0).groupby([season, g.hour]).mean().unstack(0) * 100

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(WIDTH, 4.6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 2]})
    lo = min(-0.5, float(mean.min().min()) - 0.5)
    ax.axhspan(lo, 0, color=SHADE, lw=0)
    ax.axhline(0, color=INK_2, lw=0.9)
    markers = ["o", "s", "^", "D"]
    for (name, color, mk) in zip(SEASONS, SERIES, markers):
        ax.plot(mean.index, mean[name], color=color, marker=mk, ms=3,
                label=name)
        ax2.plot(neg.index, neg[name], color=color, marker=mk, ms=3,
                 label=name)
    ax.set_ylim(lo, mean.max().max() * 1.08)
    ax.text(0.3, lo / 2, "net export (reverse flow)", fontsize=7,
            color=INK_2, va="center")
    ax.set(ylabel="Mean demand (MW)")
    ax.set_title("Seasonal daily demand profile, Telarah 33/11 kV",
                 loc="left")

    ax2.set(xlabel="Hour of day", ylabel="Hours below 0 MW (%)",
            xticks=range(0, 24, 3), xlim=(-0.5, 23.5))
    ax2.set_ylim(0, None)
    fig.align_ylabels((ax, ax2))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="lower center",
               columnspacing=1.2, handlelength=1.8, bbox_to_anchor=(0.5, 0))
    save(fig, "fig_reverse_flow.png")


def fig_error_by_hour(preds: dict, subs: list[str]) -> None:
    styles = {"Best naive": dict(color=MUTED, ls="--", marker=None),
              "ARIMA": dict(color=SERIES[0], ls="-", marker="o"),
              "LSTM": dict(color=SERIES[1], ls="-", marker="s"),
              "Transformer": dict(color=SERIES[2], ls="-", marker="^")}

    fig, axes = plt.subplots(len(subs), 1, figsize=(WIDTH, 2.0 * len(subs)),
                             sharex=True)
    for ax, sub in zip(np.atleast_1d(axes), subs):
        # Score every model on the timestamps all four predicted
        common = None
        for fam in styles:
            idx = preds[(sub, fam)][0].index
            common = idx if common is None else common.intersection(idx)
        ax.axvspan(PEAK_HOURS[0] - 0.5, PEAK_HOURS[1] + 0.5, color=SHADE,
                   lw=0)
        for fam, st in styles.items():
            a, p = preds[(sub, fam)]
            err = (a.loc[common] - p.loc[common]).abs()
            by_hour = err.groupby(common.hour).mean()
            ax.plot(by_hour.index, by_hour.values, label=fam, ms=2.5, **st)
        ax.set_title(short_name(sub), loc="left")
        ax.set(ylabel="MAE (MW)")
        ax.set_ylim(0, None)
    axes[-1].set(xlabel="Hour of day", xticks=range(0, 24, 3),
                 xlim=(-0.5, 23.5))
    axes[0].text(PEAK_HOURS[0] - 0.3, axes[0].get_ylim()[1] * 0.03,
                 "16:00-20:00", fontsize=6.5, color=INK_2, va="bottom")
    fig.suptitle("Forecast error by hour of day (24-hour horizon)",
                 x=0.02, ha="left")
    fig.align_ylabels(axes)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center",
               columnspacing=1.0, handlelength=2.0, bbox_to_anchor=(0.5, 0))
    save(fig, "fig_error_by_hour.png")


def fig_peak_scatter(preds: dict, subs: list[str]) -> None:
    fig, axes = plt.subplots(len(subs), 1, figsize=(WIDTH, 2.7 * len(subs)))
    for ax, sub in zip(np.atleast_1d(axes), subs):
        a, p = preds[(sub, "Transformer")]
        m = operational_peak_mask(a.index)
        a, p = a[m], p[m]
        under = float((p < a).mean() * 100)

        lo = float(min(a.min(), p.min()))
        hi = float(max(a.max(), p.max()))
        pad = 0.05 * (hi - lo)
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, color=INK_2, lw=0.9, zorder=1)
        ax.scatter(a, p, s=9, color=SERIES[0], alpha=0.55,
                   edgecolors="white", linewidths=0.3, zorder=2)
        ax.set(xlim=lim, ylim=lim, xlabel="Actual demand (MW)",
               ylabel="Predicted (MW)")
        ax.set_aspect("equal")
        ax.set_title(short_name(sub), loc="left")
        ax.text(0.97, 0.04, f"under-forecast: {under:.1f}%\nn = {len(a)} h",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                color=INK)
        ax.text(0.04, 0.96, "y = x", transform=ax.transAxes, ha="left",
                va="top", fontsize=7, color=INK_2)
    fig.suptitle("Peak-period forecast accuracy\n"
                 "(24-hour horizon, Transformer)", x=0.02, ha="left")
    fig.tight_layout()
    save(fig, "fig_peak_scatter.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/hourly_features")
    ap.add_argument("--selection",
                    default="results/metrics/substation_selection.csv")
    a = ap.parse_args()

    FIGDIR.mkdir(parents=True, exist_ok=True)
    style()

    df = pd.read_parquet(a.data, columns=COLUMNS)
    selection = pd.read_csv(a.selection)
    subs = selection.substation.tolist()

    fig_network_distribution(df, selection)
    fig_reverse_flow(df)
    preds = model_predictions(df, subs, horizon=24)
    fig_error_by_hour(preds, subs)
    fig_peak_scatter(preds, subs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
