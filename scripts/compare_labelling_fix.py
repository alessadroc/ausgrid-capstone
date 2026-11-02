#!/usr/bin/env python3
"""Before/after comparison of two runs of the full experimental chain.

Written for the FY2024 interval-labelling fix and the blank-day placeholder
masking that followed it. Two steps, because reprocessing overwrites
data/processed:

    # 1. before reprocessing, record per-substation stats of the old data
    python scripts/compare_labelling_fix.py snapshot --out results_old_labelling

    # 2. after the full chain has re-run
    python scripts/compare_labelling_fix.py

Output (default):
    results/metrics/labelling_fix_comparison.md
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

from make_report_figures import substation_stats  # noqa: E402
from run_comparison import family                 # noqa: E402

KEY = ["family", "substation", "horizon_h"]
METRICS = ["MAE", "MASE", "peak_MAE", "peak_RMSE", "underestimation_rate"]
N_DM_TESTS = 18
ALPHA = 0.05


def network_stats(data: str) -> pd.DataFrame:
    df = pd.read_parquet(data, columns=["substation", "load_mw",
                                        "was_imputed", "is_outlier"])
    stats = substation_stats(df)
    zeros = (df.load_mw == 0).groupby(df.substation).sum()
    stats["zero_hours"] = stats.substation.map(zeros).astype(int)
    max_hours = stats.n_hours.max()
    stats["eligible"] = ((stats.n_hours >= 0.95 * max_hours) &
                         (stats.imputed_frac <= 0.05))
    return stats


def snapshot(data: str, out: Path) -> None:
    dest = out / "metrics" / "network_stats.csv"
    network_stats(data).to_csv(dest, index=False)
    print(f"-> {dest}")


# ---------------------------------------------------------------------------

def load_results(metdir: Path) -> pd.DataFrame:
    frames = []
    for stem in ("baseline", "arima", "lstm", "transformer"):
        d = pd.read_csv(metdir / f"{stem}_results.csv")
        d["source"] = stem
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["family"] = df["model"].map(family)
    return df


def best_naive(df: pd.DataFrame) -> pd.DataFrame:
    """Headline families: the lowest-MAE naive method stands for all four."""
    naive = df[df.source == "baseline"]
    best = naive.loc[naive.groupby(["substation", "horizon_h"]).MAE.idxmin()]
    return pd.concat([best.assign(family="Best naive"),
                      df[df.source != "baseline"]], ignore_index=True)


def with_best_naive(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([best_naive(df), df[df.source == "baseline"]],
                     ignore_index=True)


def pct(old, new):
    return np.where(old != 0, 100 * (new - old) / np.abs(old), np.nan)


def metrics_table(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    m = old[KEY + METRICS].merge(new[KEY + METRICS], on=KEY, how="outer",
                                 suffixes=("_old", "_new"))
    cols = list(KEY)
    for c in METRICS:
        m[f"{c}_pct"] = pct(m[f"{c}_old"], m[f"{c}_new"])
        cols += [f"{c}_old", f"{c}_new", f"{c}_pct"]
    return m[cols].sort_values(KEY)


def rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Order of the four headline families by MAE in each cell."""
    h = best_naive(df)
    h = h[h.family.isin(["Best naive", "ARIMA", "LSTM", "Transformer"])]
    return (h.sort_values("MAE").groupby(["substation", "horizon_h"])
             .family.apply(" < ".join).rename("ranking").reset_index())


def dm_table(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    k = ["substation", "horizon_h", "model_A", "model_B"]
    threshold = ALPHA / N_DM_TESTS
    m = old[k + ["p_value", "verdict"]].merge(
        new[k + ["p_value", "verdict"]], on=k, how="outer",
        suffixes=("_old", "_new"))
    for s in ("old", "new"):
        m[f"bonf_{s}"] = m[f"p_value_{s}"] < threshold
    m["verdict_changed"] = m.verdict_old != m.verdict_new
    m["bonferroni_changed"] = m.bonf_old != m.bonf_new
    return m.sort_values(k)


def fmt(t: pd.DataFrame, digits: int = 4) -> str:
    d = t.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            dd = 2 if c.endswith("_pct") else digits
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else
                            (f"{v:+.{dd}f}" if c.endswith("_pct")
                             else f"{v:.{dd}g}" if 0 < abs(v) < 1e-3
                             else f"{v:.{dd}f}"))
    return d.to_markdown(index=False)


def network_summary(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    def counts(s):
        e = s[s.eligible]
        return {
            "substations processed": len(s),
            "pass quality gate": int(s.eligible.sum()),
            "eligible with reverse flow (any hour < 0 MW)":
                int((e.reverse_flow_frac > 0).sum()),
            "eligible with minimum exactly 0.00 MW": int((e.min_mw == 0).sum()),
            "processed with minimum exactly 0.00 MW": int((s.min_mw == 0).sum()),
            "exact 0.00 MW hours, all processed": int(s.zero_hours.sum()),
        }
    o, n = counts(old), counts(new)
    return pd.DataFrame({"measure": list(o), "old": list(o.values()),
                         "new": list(n.values()),
                         "changed": [o[k] != n[k] for k in o]})


def compare(old_dir: Path, new_dir: Path, data: str, title: str,
            old_desc: str, new_desc: str) -> str:
    om, nm = old_dir / "metrics", new_dir / "metrics"
    sections = [f"# {title}\n\n"
                f"Generated by scripts/compare_labelling_fix.py. Old: "
                f"`{old_dir}/` ({old_desc}). New: `{new_dir}/` ({new_desc}). "
                "Do not edit by hand."]

    old_sel = pd.read_csv(om / "substation_selection.csv")
    new_sel = pd.read_csv(nm / "substation_selection.csv")
    same = old_sel.substation.tolist() == new_sel.substation.tolist()
    sections.append(
        "## Substation selection\n\n"
        + ("Unchanged: " + ", ".join(new_sel.substation) + "." if same else
           "**Changed.** Old: " + ", ".join(old_sel.substation)
           + ". New: " + ", ".join(new_sel.substation)
           + ". Per-cell comparisons below only pair identical substations."))

    old_r, new_r = load_results(om), load_results(nm)
    t = metrics_table(with_best_naive(old_r), with_best_naive(new_r))
    sections.append(
        "## Accuracy and peak metrics\n\n"
        "_pct is the percentage change from old to new. 'Best naive' is "
        "the lowest-MAE naive method in each cell; the individual naive "
        "methods follow. underestimation_rate is on the exogenous peak "
        "window; peak_MAE / peak_RMSE on the top 5% of actual load.\n\n"
        + fmt(t))

    ro, rn = rankings(old_r), rankings(new_r)
    rk = ro.merge(rn, on=["substation", "horizon_h"], how="outer",
                  suffixes=("_old", "_new"))
    rk["changed"] = rk.ranking_old != rk.ranking_new
    sections.append(
        f"## Model rankings by MAE\n\n"
        f"{int(rk.changed.sum())} of {len(rk)} substation/horizon cells "
        f"changed order.\n\n" + rk.to_markdown(index=False))

    dm = dm_table(pd.read_csv(om / "pairwise_dm.csv"),
                  pd.read_csv(nm / "pairwise_dm.csv"))
    sections.append(
        f"## Diebold-Mariano verdicts\n\n"
        f"Bonferroni threshold {ALPHA}/{N_DM_TESTS} = "
        f"{ALPHA / N_DM_TESTS:.5f}. Verdicts changed (at 0.05): "
        f"{int(dm.verdict_changed.sum())} of {len(dm)}. Significance "
        f"after Bonferroni changed: {int(dm.bonferroni_changed.sum())} of "
        f"{len(dm)}.\n\n" + fmt(dm))

    ns = network_summary(pd.read_csv(om / "network_stats.csv"),
                         network_stats(data))
    sections.append(
        "## Network-level counts\n\n"
        "Quality gate: >=95% coverage and <=5% imputed hours.\n\n"
        + ns.to_markdown(index=False))

    return "\n\n".join(sections) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", default="compare",
                    choices=["compare", "snapshot"])
    ap.add_argument("--data", default="data/processed/hourly_features")
    ap.add_argument("--old", default="results_old_labelling")
    ap.add_argument("--new", default="results")
    ap.add_argument("--out", default="results_old_labelling",
                    help="snapshot destination (results directory)")
    ap.add_argument("--name", default="labelling_fix_comparison.md",
                    help="report filename, written to <new>/metrics/")
    ap.add_argument("--title", default="Labelling fix: before and after")
    ap.add_argument("--old-desc",
                    default="FY2024 files read as start-labelled")
    ap.add_argument("--new-desc", default="trailing 00:00 read as 24:00")
    a = ap.parse_args()

    if a.command == "snapshot":
        snapshot(a.data, Path(a.out))
        return 0

    text = compare(Path(a.old), Path(a.new), a.data, a.title, a.old_desc,
                   a.new_desc)
    out = Path(a.new) / "metrics" / a.name
    out.write_text(text)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
