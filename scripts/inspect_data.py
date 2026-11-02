#!/usr/bin/env python3
"""Inspect the real Ausgrid archive without loading interval data.

Run this FIRST. It reads only headers and identifier columns, so it finishes
in seconds on 545 files, and it answers the questions that determine how the
pipeline should be configured:

    python inspect_data.py --input data

  * How many substations, and how many files each spans?
  * What resolutions are present (15-min, 30-min, mixed)?
  * Are time labels zero-padded consistently?
  * Are labels interval-start or interval-end?
  * What date range does each substation actually cover?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ausgrid_pipeline import ingest  # noqa: E402

TIME_PATTERN = re.compile(r"^\s*(\d{1,2}):([0-5]\d)\s*$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data")
    ap.add_argument("--sample", type=int, default=None,
                    help="only inspect the first N files")
    a = ap.parse_args()

    files = sorted(Path(a.input).rglob("*.csv"))
    if a.sample:
        files = files[: a.sample]
    if not files:
        raise SystemExit(f"No CSVs found in {a.input}")

    print(f"Found {len(files)} CSV files\n")

    resolutions = Counter()
    labellings = Counter()
    padding = Counter()
    colcounts = Counter()
    id_schemas = Counter()
    sub_files = defaultdict(list)
    sub_dates = defaultdict(list)
    problems = []

    for p in files:
        try:
            head = pd.read_csv(p, nrows=1)
            head.columns = [str(c).strip() for c in head.columns]
        except Exception as exc:
            problems.append(f"{p.name}: unreadable ({type(exc).__name__})")
            continue

        tcols = [c for c in head.columns if TIME_PATTERN.match(str(c))]
        if not tcols:
            problems.append(f"{p.name}: no time columns in header")
            continue

        # Same mapping as the pipeline, so a trailing 00:00 reads as 24:00
        mins = sorted(ingest.interval_map(head[tcols]).values())
        colcounts[len(tcols)] += 1

        # resolution from modal gap
        res = int(pd.Series(np.diff(mins)).mode().iloc[0]) if len(mins) > 1 else 60
        resolutions[res] += 1

        # labelling convention
        if 1440 in mins and 0 not in mins:
            labellings["end (…24:00, no 00:00)"] += 1
        elif 0 in mins and 1440 not in mins:
            labellings["start (00:00…, no 24:00)"] += 1
        else:
            labellings["ambiguous (both endpoints)"] += 1

        # zero padding
        padded = [c for c in tcols if re.match(r"^\d{2}:", str(c).strip())]
        padding["padded" if len(padded) == len(tcols)
                else "unpadded" if not padded else "mixed within file"] += 1

        id_cols = [c for c in head.columns if not TIME_PATTERN.match(str(c))]
        id_schemas[tuple(id_cols)] += 1

        try:
            d = pd.read_csv(p, usecols=["Zone Substation", "Date"])
            for sub in d["Zone Substation"].astype(str).str.strip().unique():
                sub_files[sub].append(p.name)
            dates = pd.to_datetime(d["Date"], errors="coerce", format="mixed").dropna()
            if not dates.empty:
                for sub in d["Zone Substation"].astype(str).str.strip().unique():
                    sub_dates[sub] += [dates.min(), dates.max()]
        except Exception as exc:
            problems.append(f"{p.name}: {type(exc).__name__} reading id columns")

    def show(title, counter):
        print(f"{title}")
        for k, v in counter.most_common():
            print(f"   {str(k):<40} {v:>5} files")
        print()

    show("INTERVAL COLUMN COUNT", colcounts)
    show("DETECTED RESOLUTION (minutes)", resolutions)
    show("LABELLING CONVENTION", labellings)
    show("ZERO PADDING OF TIME LABELS", padding)

    print("IDENTIFIER COLUMN SCHEMAS")
    for cols, n in id_schemas.most_common():
        print(f"   {n:>5} files: {list(cols)}")
    print()

    if sub_dates:
        rows = []
        for sub, ds in sub_dates.items():
            rows.append({"substation": sub,
                         "n_files": len(sub_files[sub]),
                         "start": min(ds), "end": max(ds),
                         "years": round((max(ds) - min(ds)).days / 365.25, 2)})
        cov = pd.DataFrame(rows).sort_values("years")
        print(f"SUBSTATION COVERAGE  ({len(cov)} substations)")
        print(f"   files per substation : min {cov.n_files.min()}, "
              f"median {int(cov.n_files.median())}, max {cov.n_files.max()}")
        print(f"   years per substation : min {cov.years.min()}, "
              f"median {cov.years.median()}, max {cov.years.max()}")
        print(f"   earliest date        : {cov.start.min():%Y-%m-%d}")
        print(f"   latest date          : {cov.end.max():%Y-%m-%d}")
        print(f"\n   shortest coverage:")
        print(cov.head(5).to_string(index=False))
        cov.to_csv("substation_coverage.csv", index=False)
        print("\n   full table -> substation_coverage.csv")

    if problems:
        print(f"\nPROBLEM FILES ({len(problems)})")
        for x in problems[:15]:
            print("   " + x)

    print("\n" + "=" * 60)
    print("SUGGESTED COMMAND")
    print("=" * 60)
    if sub_dates:
        lo, hi = cov.start.min(), cov.end.max()
        print(f"python run_pipeline.py --input {a.input} --output processed \\")
        print(f"    --start-date 2023-01-01 --end-date {hi:%Y-%m-%d} --stream")
        if len(resolutions) > 1:
            print("\nNote: mixed resolutions detected. This is handled "
                  "automatically -\n      each file is read at its own "
                  "resolution and resampled to hourly.")


if __name__ == "__main__":
    main()
