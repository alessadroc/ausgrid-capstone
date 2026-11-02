#!/usr/bin/env python3
"""Collect every number the results chapter needs into one markdown file.

Reads only regenerated artifacts (the processed dataset, the raw-file index
and the results CSVs), so it should be run after recompute_metrics.py and
run_comparison.py:

    python scripts/recompute_metrics.py
    python scripts/run_comparison.py
    python scripts/make_report_figures.py
    python scripts/make_chapter4_summary.py

Output:
    results/metrics/chapter4_summary.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ausgrid_pipeline import ingest, stream          # noqa: E402
from ausgrid_pipeline.config import Config           # noqa: E402
from evaluation.metrics import operational_peak_mask  # noqa: E402
from make_report_figures import (COLUMNS, PEAK_HOURS, SEASONS,  # noqa: E402
                                 eligible_substations, model_predictions,
                                 season_of)
from run_baselines import evaluation_mask            # noqa: E402
from run_comparison import short_name                # noqa: E402

METDIR = Path("results/metrics")
N_DM_TESTS = 18
ALPHA = 0.05
FAMILIES = ["Best naive", "ARIMA", "LSTM", "Transformer"]


def md(t: pd.DataFrame, digits: int = 4) -> str:
    d = t.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}g}"
                            if abs(v) < 1e-3 and v != 0 else f"{v:.{digits}f}")
    return d.to_markdown(index=False)


def safe_name(sub: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in sub)


def all_results() -> pd.DataFrame:
    frames = []
    for stem in ("baseline", "arima", "lstm", "transformer"):
        d = pd.read_csv(METDIR / f"{stem}_results.csv")
        d["source"] = stem
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def section_dataset(df: pd.DataFrame, raw_dir: Path, data_dir: Path) -> str:
    cfg = Config(input_dir=raw_dir, verbose=False)
    files = ingest.discover_files(cfg)
    index, lo, hi = stream.build_index(cfg, files)
    processed = {p.stem for p in data_dir.glob("*.parquet")}
    excluded = sorted(s for s in index if safe_name(s) not in processed)
    elig = eligible_substations(df)

    t = pd.DataFrame([
        ("Raw CSV files", len(files)),
        ("Substations indexed", len(index)),
        ("Substations processed", df.substation.nunique()),
        ("Substations excluded at ingestion", len(excluded)),
        ("Substations passing quality gate (>=95% coverage, <=5% imputed)",
         len(elig)),
        ("Total hourly rows", f"{len(df):,}"),
        ("Raw data span", f"{lo:%Y-%m-%d} to {hi:%Y-%m-%d}"),
        ("Study period start", f"{df.timestamp.min():%Y-%m-%d %H:%M}"),
        ("Study period end", f"{df.timestamp.max():%Y-%m-%d %H:%M}"),
    ], columns=["Item", "Value"])
    out = "## 1. Dataset\n\n" + t.to_markdown(index=False)
    if excluded:
        out += ("\n\nExcluded at ingestion (indexed in the raw files but "
                "absent from the processed output): " + ", ".join(excluded)
                + ".")
    return out


def section_distribution(elig: pd.DataFrame) -> str:
    d = elig.mean_mw.describe()
    t = pd.DataFrame({"Statistic": d.index, "Mean hourly demand (MW)": d.values})
    edges = pd.qcut(elig.mean_mw, 3, retbins=True)[1]
    return ("## 2. Network load distribution (eligible substations)\n\n"
            + md(t, 3)
            + f"\n\nTercile boundaries: {edges[1]:.3f} MW and {edges[2]:.3f} MW.")


def section_reverse_flow(elig: pd.DataFrame) -> str:
    rf = elig[elig.reverse_flow_frac > 0]
    worst = rf.loc[rf.reverse_flow_frac.idxmax()] if len(rf) else None
    lowest = elig.loc[elig.min_mw.idxmin()]
    rows = [("Eligible substations with any negative net demand",
             f"{len(rf)} of {len(elig)}")]
    if worst is not None:
        rows.append(("Highest share of negative hours",
                     f"{worst.reverse_flow_frac * 100:.2f}% "
                     f"({worst.substation})"))
    rows.append(("Network minimum hourly demand",
                 f"{lowest.min_mw:.3f} MW ({lowest.substation})"))
    t = pd.DataFrame(rows, columns=["Item", "Value"])
    return "## 3. Reverse power flow\n\n" + t.to_markdown(index=False)


def section_selected(sel: pd.DataFrame) -> str:
    t = sel[["substation", "stratum", "mean_mw", "max_mw", "min_mw",
             "imputed_frac", "reverse_flow_frac"]]
    return "## 4. Selected substations\n\n" + md(t, 4)


def section_zero_hours(data_dir: Path, subs: list[str]) -> str:
    """Hours of exactly 0.0 MW: count, contiguity and how they were flagged.

    Blank days exported as a row carrying a lone 0.0 are treated as missing
    at ingestion (ingest.file_to_long), so any zero counted here is a
    recorded reading, not that placeholder.
    """
    g = pd.read_parquet(data_dir, filters=[("substation", "in", subs)],
                        columns=["timestamp", "substation", "load_mw",
                                 "n_intervals", "was_imputed", "is_outlier",
                                 "split"])
    g = g.sort_values(["substation", "timestamp"]).reset_index(drop=True)
    zero = g.load_mw == 0
    # Length of the imputed run touching each hour, on either side
    run = (g.was_imputed != g.was_imputed.shift()).cumsum()
    run_len = g.was_imputed.groupby(run).transform("sum")
    adjacent = (run_len.shift(1).where(g.was_imputed.shift(1, fill_value=False), 0)
                + run_len.shift(-1).where(g.was_imputed.shift(-1, fill_value=False), 0))

    rows = []
    for sub in subs:
        z = g[zero & (g.substation == sub)]
        run_id = (zero != zero.shift()).cumsum()[z.index]
        rows.append({
            "substation": sub,
            "zero_hours": len(z),
            "longest_run_h": int(run_id.value_counts().max()) if len(z) else 0,
            "hour_of_day": ", ".join(f"{h:02d}:00" for h in
                                     sorted(z.timestamp.dt.hour.unique())),
            "single_reading": int((z.n_intervals == 1).sum()),
            "flagged_imputed": int(z.was_imputed.sum()),
            "flagged_outlier": int(z.is_outlier.sum()),
            "imputed_hours_adjacent": int(adjacent[z.index].sum()),
            "splits": ", ".join(sorted(z.split.unique())),
            "dates": ", ".join(f"{t:%Y-%m-%d}" for t in z.timestamp),
        })
    t = pd.DataFrame(rows)
    return ("### 4a. Exact-zero hours in the selected substations\n\n"
            + t.to_markdown(index=False)
            + "\n\nimputed_hours_adjacent: length of the imputed gaps "
              "immediately before and after each zero, summed per substation. "
              "Ausgrid's blank-day placeholder (a lone 0.0 in an otherwise "
              "empty day row) is treated as missing at ingestion, so zeros "
              "counted here are recorded readings. Zero hours not flagged as imputed or outlier remain in the training "
              "series used for model fitting and for the MASE denominator.")


def section_adf(adf: pd.DataFrame) -> str:
    t = adf[["substation", "adf_statistic", "p_value", "lags_used", "n_obs",
             "crit_5pct", "suggested_d"]]
    return ("## 5. Augmented Dickey-Fuller test\n\n" + md(t, 4)
            + "\n\nCritical value shown is the 5% level. A p-value of 0.0 is "
              "below the precision stored in stationarity.csv (6 d.p.).")


def section_partition(df: pd.DataFrame, res: pd.DataFrame,
                      subs: list[str]) -> str:
    g = df[df.substation.isin(subs)]
    b = (g.groupby("split").timestamp.agg(["min", "max"])
          .reindex(["train", "val", "test"]).reset_index())
    b.columns = ["Split", "First hour", "Last hour"]
    counts = g.groupby("split").substation.value_counts().unstack(0)[
        ["train", "val", "test"]]

    r = res.copy()
    r["family"] = r["source"].map({"baseline": "Naive baselines",
                                   "arima": "ARIMA", "lstm": "LSTM",
                                   "transformer": "Transformer"})
    n = (r.groupby(["substation", "horizon_h", "family"]).n.max()
          .unstack("family")[["Naive baselines", "ARIMA", "LSTM",
                              "Transformer"]]
          .reset_index())

    return ("## 6. Partition and evaluation targets\n\n"
            "Boundaries for the three selected substations:\n\n"
            + b.to_markdown(index=False)
            + "\n\nHours per split:\n\n"
            + counts.reset_index().rename(columns={"substation": "Substation"})
                    .to_markdown(index=False)
            + "\n\nEvaluation targets (test hours after excluding imputed and "
              "outlier-flagged targets and any without a prediction):\n\n"
            + n.to_markdown(index=False))


def section_accuracy(res: pd.DataFrame) -> str:
    t = res[["model", "substation", "horizon_h", "MAE", "RMSE", "MASE",
             "MAPE", "mae_seed_sd"]].sort_values(
        ["substation", "horizon_h", "MAE"])
    return ("## 7. Accuracy (all models)\n\n" + md(t, 4)
            + "\n\nMAE and RMSE in MW, MAPE in %. mae_seed_sd is the standard "
              "deviation of test MAE across training seeds (deep models "
              "only); headline metrics are for the seed-ensemble mean.")


def section_reliability(res: pd.DataFrame) -> str:
    t = res[["model", "substation", "horizon_h", "peak_MAE", "peak_RMSE",
             "n_peak_quantile", "underestimation_rate", "bias_pvalue",
             "biased_low", "n_peak_window"]].sort_values(
        ["substation", "horizon_h", "peak_MAE"])
    return ("## 8. Reliability (all models)\n\n" + md(t, 4)
            + "\n\npeak_MAE / peak_RMSE / n_peak_quantile: top 5% of test hours "
              "by actual load. underestimation_rate / bias_pvalue / biased_low "
              "/ n_peak_window: exogenous window (Dec-Feb, Mon-Fri, "
              "16:00-20:00), two-sided binomial test against 0.5.")


def section_dm(dm: pd.DataFrame) -> str:
    threshold = ALPHA / N_DM_TESTS
    t = dm[["substation", "horizon_h", "model_A", "model_B", "MAE_A",
            "MAE_B", "dm_statistic", "p_value", "verdict", "n"]].copy()
    t["significant_bonferroni"] = t.p_value < threshold
    t = t.sort_values(["substation", "horizon_h", "model_A", "model_B"])
    note = ""
    if len(t) != N_DM_TESTS:
        note = (f"\n\n**Note:** pairwise_dm.csv has {len(t)} rows, not "
                f"{N_DM_TESTS}; the threshold above assumes {N_DM_TESTS}.")
    return (f"## 9. Pairwise Diebold-Mariano tests\n\n"
            f"Bonferroni threshold: {ALPHA}/{N_DM_TESTS} = {threshold:.5f}. "
            f"{int(t.significant_bonferroni.sum())} of {len(t)} remain "
            f"significant. A negative statistic means model A has lower "
            f"squared-error loss. p-values of 0.0 are below double precision."
            "\n\n" + md(t, 4) + note)


# ---------------------------------------------------------------------------
# Figure descriptions: numbers are computed, wording is templated
# ---------------------------------------------------------------------------

def describe_fig1(elig: pd.DataFrame, sel: pd.DataFrame) -> str:
    d = elig.mean_mw
    edges = pd.qcut(d, 3, retbins=True)[1]
    above = int((d > 2 * d.median()).sum())
    picks = "; ".join(f"{short_name(r.substation)} {r.mean_mw:.1f} MW"
                      for _, r in sel.iterrows())
    return (f"**fig_network_distribution.png.** The histogram shows mean "
            f"hourly demand for the {len(d)} eligible substations, ranging "
            f"from {d.min():.2f} to {d.max():.2f} MW with a median of "
            f"{d.median():.2f} MW and a mean of {d.mean():.2f} MW; "
            f"{above} substations exceed twice the median. Tercile "
            f"boundaries fall at {edges[1]:.2f} and {edges[2]:.2f} MW, and "
            f"the three selected substations sit at {picks}.")


def describe_fig2(df: pd.DataFrame, sub: str = "Telarah 33_11kV") -> str:
    g = df[df.substation == sub]
    g = g[evaluation_mask(g)]
    season = season_of(g.month)
    mean = g.groupby([season, g.hour]).load_mw.mean().unstack(0)
    neg = (g.load_mw < 0).groupby([season, g.hour]).mean().unstack(0) * 100
    summer, spring = list(SEASONS)[0], list(SEASONS)[3]

    lowest_season = mean.min().idxmin()
    worst_neg_season = neg.max().idxmax()
    return (f"**fig_reverse_flow.png.** No seasonal mean profile at "
            f"{short_name(sub)} falls below 0 MW; the lowest hourly mean is "
            f"{mean.min().min():.2f} MW at {int(mean[lowest_season].idxmin()):02d}:00 "
            f"in {lowest_season.split(' ')[0].lower()}, while the summer "
            f"midday minimum is {mean[summer].loc[9:15].min():.2f} MW. The "
            f"lower panel shows the share of hours below 0 MW: it peaks at "
            f"{neg.max().max():.0f}% at {int(neg[worst_neg_season].idxmax()):02d}:00 in "
            f"{worst_neg_season.split(' ')[0].lower()} and does not exceed "
            f"{neg[summer].max():.0f}% in any summer hour. The evening peak "
            f"is highest in summer at {mean[summer].max():.2f} MW "
            f"({int(mean[summer].idxmax()):02d}:00).")


def describe_fig3(preds: dict, subs: list[str]) -> str:
    lo, hi = PEAK_HOURS
    parts, peaks, troughs = [], [], []
    for sub in subs:
        common = None
        for fam in FAMILIES:
            idx = preds[(sub, fam)][0].index
            common = idx if common is None else common.intersection(idx)
        by_hour = {}
        for fam in FAMILIES:
            a, p = preds[(sub, fam)]
            e = (a.loc[common] - p.loc[common]).abs()
            by_hour[fam] = e.groupby(common.hour).mean()
        bh = pd.DataFrame(by_hour)
        peaks += list(bh.idxmax())
        troughs += list(bh.idxmin())
        window = bh.loc[lo:hi].mean()
        night = bh.loc[0:5].mean()
        parts.append(
            f"At {short_name(sub)} the Transformer error peaks at "
            f"{int(bh['Transformer'].idxmax()):02d}:00 "
            f"({bh['Transformer'].max():.2f} MW); in the 16:00-20:00 window "
            f"mean MAE is {window['Best naive']:.2f} (best naive), "
            f"{window['ARIMA']:.2f} (ARIMA), {window['LSTM']:.2f} (LSTM) and "
            f"{window['Transformer']:.2f} MW (Transformer); between 00:00 and "
            f"05:00 ARIMA MAE is {night['ARIMA']:.2f} MW against "
            f"{night['Transformer']:.2f} MW for the Transformer")
    return (f"**fig_error_by_hour.png.** Across the three substations and "
            f"four models, hourly MAE is lowest between "
            f"{min(troughs):02d}:00 and {max(troughs):02d}:00 and highest "
            f"between {min(peaks):02d}:00 and {max(peaks):02d}:00. "
            + ". ".join(parts)
            + ". Each model is scored on the timestamps all four predicted.")


def describe_fig4(preds: dict, subs: list[str]) -> str:
    parts = []
    for sub in subs:
        a, p = preds[(sub, "Transformer")]
        m = operational_peak_mask(a.index)
        a, p = a[m], p[m]
        top = a >= a.quantile(0.9)
        bottom = a <= a.quantile(0.1)
        parts.append(
            f"{short_name(sub)} {(p < a).mean() * 100:.1f}% under-forecast "
            f"(n = {len(a)}), actual range {a.min():.1f}-{a.max():.1f} MW "
            f"against predicted {p.min():.1f}-{p.max():.1f} MW, with "
            f"{(p[top] < a[top]).mean() * 100:.0f}% under-forecast among the "
            f"top decile of actual window hours and "
            f"{(p[bottom] < a[bottom]).mean() * 100:.0f}% among the bottom "
            f"decile")
    return ("**fig_peak_scatter.png.** Within the exogenous peak window the "
            "Transformer's under-forecast share is: " + "; ".join(parts)
            + ".")


def section_figures(df, elig, sel, preds, subs) -> str:
    return "\n\n".join([
        "## 10. Figure descriptions",
        describe_fig1(elig, sel),
        describe_fig2(df),
        describe_fig3(preds, subs),
        describe_fig4(preds, subs),
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/hourly_features")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--selection",
                    default="results/metrics/substation_selection.csv")
    a = ap.parse_args()

    df = pd.read_parquet(a.data, columns=COLUMNS)
    sel = pd.read_csv(a.selection)
    subs = sel.substation.tolist()
    elig = eligible_substations(df)
    res = all_results()
    adf = pd.read_csv(METDIR / "stationarity.csv")
    dm = pd.read_csv(METDIR / "pairwise_dm.csv")
    preds = model_predictions(df, subs, horizon=24)

    sections = [
        "# Chapter 4 results summary\n\n"
        "Generated by scripts/make_chapter4_summary.py from the processed "
        "dataset and results/metrics/. Do not edit by hand.",
        section_dataset(df, Path(a.raw), Path(a.data)),
        section_distribution(elig),
        section_reverse_flow(elig),
        section_selected(sel),
        section_zero_hours(Path(a.data), subs),
        section_adf(adf),
        section_partition(df, res, subs),
        section_accuracy(res),
        section_reliability(res),
        section_dm(dm),
        section_figures(df, elig, sel, preds, subs),
    ]
    out = METDIR / "chapter4_summary.md"
    out.write_text("\n\n".join(sections) + "\n")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
