#!/usr/bin/env python3
"""Run the full Ausgrid preprocessing pipeline.

    python run_pipeline.py --input data --output processed
    python run_pipeline.py --input data --max-files 3        # quick smoke test
    python run_pipeline.py --input data --weather sydney.csv --tz

Outputs written to --output:

    qc_report.csv                    per-substation quality summary
    interval_long.parquet            cleaned native-resolution long series
    hourly_features.parquet          modelling dataset
    scaler_params.json               per-substation train-only scaling
    pipeline_summary.txt             run log for the methodology appendix
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))
import sys
import time
from pathlib import Path

import pandas as pd

from ausgrid_pipeline import Config, ingest, clean, features


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data")
    p.add_argument("--output", default="processed")
    p.add_argument("--weather", default=None)
    p.add_argument("--tz", action="store_true",
                   help="localise to Australia/Sydney (default: naive local time)")
    p.add_argument("--freq", default="1h", help="target resolution")
    p.add_argument("--native-minutes", type=int, default=15)
    p.add_argument("--force-resolution", action="store_true",
                   help="use --native-minutes instead of detecting per file")
    p.add_argument("--min-completeness", type=float, default=80.0)
    p.add_argument("--start-date", default=None, help="e.g. 2023-01-01")
    p.add_argument("--end-date", default=None, help="e.g. 2025-12-31")
    p.add_argument("--stream", action="store_true",
                   help="process one file at a time; bounded memory")
    p.add_argument("--split-train-end", default=None)
    p.add_argument("--split-val-end", default=None)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    return Config(
        input_dir=Path(a.input),
        output_dir=Path(a.output),
        weather_csv=Path(a.weather) if a.weather else None,
        timezone="Australia/Sydney" if a.tz else None,
        target_freq=a.freq,
        native_interval_minutes=a.native_minutes,
        force_resolution=a.force_resolution,
        min_completeness_pct=a.min_completeness,
        start_date=a.start_date,
        end_date=a.end_date,
        split_train_end=a.split_train_end,
        split_val_end=a.split_val_end,
        stream=a.stream,
        max_files=a.max_files,
        verbose=not a.quiet,
    ).resolve()


def main() -> int:
    cfg = parse_args()
    t0 = time.time()

    print("=" * 68)
    print("AUSGRID ZONE SUBSTATION PREPROCESSING")
    print("=" * 68)

    if cfg.stream:
        from ausgrid_pipeline import stream
        stream.run(cfg)
        print(f"\nruntime {time.time() - t0:.1f}s")
        return 0

    # Stage 1 - ingestion
    long_df = ingest.run(cfg)
    long_df.to_parquet(cfg.output_dir / "interval_long.parquet", index=False)

    # Stage 2 - cleaning and standardisation
    hourly, report = clean.run(cfg, long_df)
    report.to_csv(cfg.output_dir / "qc_report.csv", index=False)

    # Stage 3 - features
    modelling = features.run(cfg, hourly)
    out = cfg.output_dir / "hourly_features.parquet"
    modelling.to_parquet(out, index=False)

    elapsed = time.time() - t0
    summary = "\n".join([
        f"rows                {len(modelling):,}",
        f"columns             {modelling.shape[1]}",
        f"substations         {modelling['substation'].nunique()}",
        f"period              {modelling['timestamp'].min()} -> "
        f"{modelling['timestamp'].max()}",
        f"imputed values      {int(modelling['was_imputed'].sum()):,} "
        f"({100 * modelling['was_imputed'].mean():.2f}%)",
        f"outliers flagged    {int(modelling['is_outlier'].sum()):,} "
        f"({100 * modelling['is_outlier'].mean():.2f}%)",
        f"target frequency    {cfg.target_freq}",
        f"runtime             {elapsed:.1f}s",
    ])
    (cfg.output_dir / "pipeline_summary.txt").write_text(summary)

    print("\n" + "=" * 68)
    print(summary)
    print("=" * 68)
    print(f"\nwritten to {out}")

    with pd.option_context("display.width", 200):
        first = modelling["substation"].iloc[0]
        print(f"\nsample - {first}:")
        print(modelling[modelling["substation"] == first]
              .head(6)[["timestamp", "load_mw", "n_intervals", "hour",
                        "was_imputed", "is_outlier", "split"]]
              .to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
