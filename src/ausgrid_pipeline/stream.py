"""Stage 4 (optional) - Streaming execution, keyed on substation.

Processes one SUBSTATION at a time and writes each to its own parquet file, so
peak memory stays roughly constant regardless of dataset size.

Why substation and not file: in the published archive a single substation is
split across several files (one per financial year). Lags, rolling windows,
gap filling and deduplication all need that substation's history contiguous.
Processing file-by-file would fragment it, producing NaN lags at every year
boundary and silently splicing non-adjacent hours together. So a first pass
indexes which files contain which substations, and the second pass reassembles
each substation from its own files before doing any work.

Results are identical to the in-memory path. Every operation in `clean` and
`features` is computed within a substation group, so isolating substations
changes nothing mathematically. The one exception is the train/val/test
boundary, which the in-memory path takes from quantiles of the whole dataset.
Here it is resolved once up front and applied identically to every substation
- the better design regardless, since it guarantees a shared split even when
substations differ in coverage.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd

from .config import Config, log
from . import ingest, clean, features


# ---------------------------------------------------------------------------
# Pass 1 - index
# ---------------------------------------------------------------------------

def build_index(cfg: Config, files: list) -> tuple[dict, pd.Timestamp, pd.Timestamp]:
    """Map substation -> files, and find the global date range.

    Reads only the two columns needed, so a 545-file archive is indexed in
    seconds rather than by loading every interval value.
    """
    index = defaultdict(list)
    lo = hi = None

    for i, p in enumerate(files, 1):
        try:
            d = pd.read_csv(p, nrows=0)
            if "Zone Substation" not in ingest.normalise_identifiers(d).columns:
                d = pd.read_csv(p, nrows=0, skiprows=1)
                skip = 1
            else:
                skip = 0
            cols = [c for c in d.columns
                    if str(c).strip().lower() in ("zone substation", "date")]
            d = ingest.normalise_identifiers(
                pd.read_csv(p, usecols=cols, skiprows=skip))
        except Exception:
            log(cfg, "index", f"WARNING unreadable, skipping: {p.name}")
            continue

        for sub in d["Zone Substation"].astype(str).str.strip().unique():
            index[sub].append(p)

        dates = ingest.parse_dates(d["Date"]).dropna()
        if not dates.empty:
            lo = dates.min() if lo is None else min(lo, dates.min())
            hi = dates.max() if hi is None else max(hi, dates.max())

        if cfg.verbose and i % 100 == 0:
            log(cfg, "index", f"[{i}/{len(files)}] indexed")

    if lo is None:
        raise ValueError("No parseable dates found across input files.")

    log(cfg, "index", f"{len(index)} substations across {len(files)} files "
                      f"({lo:%Y-%m-%d} -> {hi:%Y-%m-%d})")
    return dict(index), lo, hi


def resolve_split_dates(cfg: Config, lo, hi) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Global train/val cut dates, honouring explicit overrides."""
    if cfg.split_train_end and cfg.split_val_end:
        t_train = pd.Timestamp(cfg.split_train_end)
        t_val = pd.Timestamp(cfg.split_val_end)
    else:
        lo = max(lo, pd.Timestamp(cfg.start_date)) if cfg.start_date else lo
        hi = min(hi, pd.Timestamp(cfg.end_date)) if cfg.end_date else hi
        span = hi - lo
        t_train = lo + span * cfg.train_fraction
        t_val = lo + span * (cfg.train_fraction + cfg.val_fraction)

    log(cfg, "split", f"train<= {t_train:%Y-%m-%d}, val<= {t_val:%Y-%m-%d}")
    return t_train, t_val


def apply_fixed_split(df: pd.DataFrame, t_train, t_val) -> pd.DataFrame:
    df = df.copy()
    t = df["timestamp"]
    if t.dt.tz is not None:
        t = t.dt.tz_localize(None)
    df["split"] = np.select([t <= t_train, t <= t_val],
                            ["train", "val"], default="test")
    return df


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Halve the float footprint.

    Load sits in the single-digit to low-hundreds MW range with six decimals;
    float32 carries about seven significant digits, so this costs nothing.
    Neural networks train in float32 regardless.
    """
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes("int64").columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


# ---------------------------------------------------------------------------
# Pass 2 - per-substation processing
# ---------------------------------------------------------------------------

def process_substation(cfg: Config, substation: str, files: list,
                       t_train, t_val) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reassemble one substation from all its files, then run the pipeline."""
    parts = []
    for p in files:
        long_df = ingest.file_to_long(cfg, ingest.read_one(p), p.name)
        if long_df.empty:
            continue
        long_df = long_df[long_df["substation"] == substation]
        long_df = ingest.apply_date_filter(cfg, long_df)
        if not long_df.empty:
            parts.append(long_df)

    if not parts:
        return pd.DataFrame(), pd.DataFrame()

    long_df = (pd.concat(parts, ignore_index=True)
                 .sort_values("timestamp", ignore_index=True))
    long_df = ingest.localise(cfg, long_df)

    report = clean.quality_report(cfg, long_df)
    long_df = clean.exclude_poor_substations(cfg, long_df, report)
    if long_df.empty:
        return pd.DataFrame(), report

    long_df = clean.deduplicate(cfg, long_df)
    hourly = clean.resample(cfg, long_df)
    hourly = clean.complete_index(cfg, hourly)
    hourly = clean.fill_gaps(cfg, hourly)
    hourly = clean.flag_outliers(cfg, hourly)

    df = features.add_calendar(cfg, hourly)
    df = features.add_lags(cfg, df)
    df = features.add_weather(cfg, df)
    df = apply_fixed_split(df, t_train, t_val)

    train = df[df["split"] == "train"]["load_mw"]
    stats = {substation: {"load_mean": float(train.mean()),
                          "load_std": float(train.std() or 1.0)}}
    df = features.apply_scaling(df, stats)

    return (downcast(df) if cfg.float32 else df), report


def run(cfg: Config) -> None:
    files = ingest.discover_files(cfg)
    index, lo, hi = build_index(cfg, files)
    t_train, t_val = resolve_split_dates(cfg, lo, hi)

    dest = cfg.output_dir / "hourly_features"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    quiet = Config(**{**cfg.__dict__, "verbose": False})
    reports, scalers, total, skipped = [], {}, 0, []

    for i, (sub, subfiles) in enumerate(sorted(index.items()), 1):
        try:
            df, rep = process_substation(quiet, sub, subfiles, t_train, t_val)
        except Exception as exc:
            log(cfg, "stream", f"WARNING {sub}: {type(exc).__name__}: {exc}")
            skipped.append(sub)
            continue

        if not rep.empty:
            reports.append(rep)
        if df.empty:
            skipped.append(sub)
            continue

        safe = "".join(ch if ch.isalnum() else "_" for ch in sub)
        df.to_parquet(dest / f"{safe}.parquet", index=False)

        train = df[df["split"] == "train"]["load_mw"]
        scalers[sub] = {"load_mean": float(train.mean()),
                        "load_std": float(train.std() or 1.0)}
        total += len(df)

        if cfg.verbose and (i % 20 == 0 or i == len(index)):
            log(cfg, "stream", f"[{i}/{len(index)}] {total:,} rows written")

    if reports:
        (pd.concat(reports, ignore_index=True)
           .sort_values("completeness_pct")
           .to_csv(cfg.output_dir / "qc_report.csv", index=False))
    (cfg.output_dir / "scaler_params.json").write_text(json.dumps(scalers, indent=2))

    log(cfg, "stream", f"done: {total:,} rows, "
                       f"{len(list(dest.glob('*.parquet')))} substations"
                       + (f", {len(skipped)} skipped" if skipped else ""))
    log(cfg, "stream", f"load with: pd.read_parquet('{dest}')")