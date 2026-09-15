"""Stage 1 - Ingestion.

Raw wide-format Ausgrid CSVs in, tidy long-format interval series out.

Three properties of the real published data drive the design here:

  * Time labels are not written consistently. Some files use "00:15", others
    "0:15". These are the same interval, so labels are normalised before
    anything else happens. Skipping this splits every pre-10am hour across two
    columns when files are concatenated.

  * Resolution varies between files. Older vintages are 30-minute (48
    columns), newer ones 15-minute (96). Resolution is detected per file from
    its own columns rather than assumed globally.

  * Dates are parsed on the wide frame, before melting. Parsing after the melt
    repeats the work once per interval instead of once per day - on this
    dataset that is over 100x more parsing than necessary, slow enough to look
    like a hang.

Nothing downstream depends on physical column order.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, log

TIME_PATTERN = re.compile(r"^\s*(\d{1,2}):([0-5]\d)\s*$")

# Tried in order before falling back to dateutil. The fallback parses
# element-by-element and is orders of magnitude slower.
DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d",
                "%d-%m-%Y", "%m/%d/%Y", "%d %b %Y", "%d-%b-%y",
                "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M", "%d %B %Y",
                "%b %d %Y", "%Y%m%d")

# The published archive changed identifier capitalisation between vintages:
# 'unit' vs 'Unit', 'year' vs 'Year'. Column names are therefore matched
# case-insensitively and renamed to a single canonical form on read.
CANONICAL_IDS = {
    "year": "year",
    "zone substation": "Zone Substation",
    "zonesubstation": "Zone Substation",
    "substation": "Zone Substation",
    "date": "Date",
    "unit": "Unit",
    "units": "Unit",
}


# ---------------------------------------------------------------------------
# Time labels
# ---------------------------------------------------------------------------

def label_minutes(label) -> int | None:
    """Minutes from midnight for a time label, or None if not a time."""
    m = TIME_PATTERN.match(str(label))
    if not m:
        return None
    h, mm = m.groups()
    return int(h) * 60 + int(mm)


def canonical_label(label) -> str | None:
    """Normalise a time label to zero-padded HH:MM.

    '0:15', '00:15' and ' 0:15 ' all become '00:15'. Without this, files
    differing only in zero-padding produce disjoint column sets when
    concatenated, inflating the interval count and splitting affected hours.
    """
    mins = label_minutes(label)
    return None if mins is None else f"{mins // 60:02d}:{mins % 60:02d}"


def interval_map(df: pd.DataFrame) -> dict:
    """Map each interval column in this frame to minutes-from-midnight."""
    mapping = {c: label_minutes(c) for c in df.columns}
    mapping = {c: m for c, m in mapping.items() if m is not None}
    if not mapping:
        raise ValueError("No HH:MM interval columns found - check the header row.")
    return dict(sorted(mapping.items(), key=lambda kv: kv[1]))


def detect_resolution(minutes) -> int:
    """Infer native resolution from the modal gap between interval labels."""
    uniq = sorted(set(minutes))
    if len(uniq) < 2:
        return 60
    return int(pd.Series(np.diff(uniq)).mode().iloc[0])


def detect_labelling(minutes) -> str:
    """Decide whether labels are interval start or end times.

    A day labelled by closing time runs to 24:00 and has no 00:00. A day
    labelled by opening time starts at 00:00 and stops one interval short of
    midnight. If both endpoints appear, treat as end-labelled per Ausgrid's
    documented convention.
    """
    has_start = 0 in minutes
    has_2400 = 1440 in minutes
    if has_2400 and not has_start:
        return "end"
    if has_start and not has_2400:
        return "start"
    return "end"


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

def parse_dates(series: pd.Series) -> pd.Series:
    """Parse a date column, trying explicit formats before dateutil.

    Parses only the DISTINCT values then maps back. A multi-year file has a
    few thousand unique dates, not millions of rows, so even the fallback
    path stays cheap.
    """
    s = series.astype(str).str.strip()
    uniq = pd.Series(s.unique())

    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(uniq, format=fmt, errors="coerce")
        if parsed.notna().mean() > 0.99:
            return s.map(dict(zip(uniq, parsed)))

    # Deliberate fallback: parse the distinct values only. The warning pandas
    # emits here is expected, not a problem, so it is silenced rather than
    # printed once per file across hundreds of files.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(uniq, errors="coerce", dayfirst=True)
    return s.map(dict(zip(uniq, parsed)))


# ---------------------------------------------------------------------------
# Discovery and reading
# ---------------------------------------------------------------------------

def discover_files(cfg: Config) -> list:
    files = sorted(cfg.input_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files in {cfg.input_dir.resolve()}. "
            "Point --input at the folder holding the Ausgrid exports."
        )
    if cfg.max_files:
        files = files[: cfg.max_files]
    log(cfg, "discover", f"{len(files)} CSV files")
    return files


def read_one(path) -> pd.DataFrame:
    """Read a single export, tolerating a preamble row if present."""
    df = pd.read_csv(path, low_memory=False)
    if not any(TIME_PATTERN.match(str(c)) for c in df.columns):
        df = pd.read_csv(path, low_memory=False, skiprows=1)
    df.columns = [str(c).strip() for c in df.columns]
    return normalise_identifiers(df)


def normalise_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Rename identifier columns to a canonical form, case-insensitively.

    Ausgrid's vintages disagree on capitalisation ('unit' vs 'Unit', 'year'
    vs 'Year'), so an exact-match lookup silently rejects entire financial
    years. Time columns are left untouched - they are matched by pattern.
    """
    rename = {}
    for c in df.columns:
        key = str(c).strip().lower()
        if key in CANONICAL_IDS and CANONICAL_IDS[key] != c:
            rename[c] = CANONICAL_IDS[key]
    return df.rename(columns=rename) if rename else df


# ---------------------------------------------------------------------------
# Per-file conversion
# ---------------------------------------------------------------------------

def file_to_long(cfg: Config, raw: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """Convert ONE file's wide frame to a timestamped long series.

    Done per file because resolution and labelling convention are properties
    of the file, not of the dataset. Concatenating first would force a single
    global assumption onto a mixed-vintage archive.
    """
    raw = normalise_identifiers(raw)

    # Only substation and date are structurally required. A missing Unit is
    # recoverable: the archive is published in MW throughout, and losing a
    # whole financial year over an absent metadata column is the worse error.
    required = ["Zone Substation", "Date"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"{source}: missing required columns {missing}")
    if "Unit" not in raw.columns:
        raw = raw.copy()
        raw["Unit"] = "MW"

    imap = interval_map(raw)
    minutes = list(imap.values())
    resolution = (cfg.native_interval_minutes if cfg.force_resolution
                  else detect_resolution(minutes))
    labelling = (cfg.interval_label if cfg.force_labelling
                 else detect_labelling(minutes))

    raw = raw.copy()
    raw["_date"] = parse_dates(raw["Date"])
    raw = raw.dropna(subset=["_date"])
    if raw.empty:
        return pd.DataFrame()

    id_vars = [c for c in cfg.id_columns if c in raw.columns] + ["_date"]
    long_df = raw.melt(id_vars=id_vars, value_vars=list(imap.keys()),
                       var_name="interval_label", value_name="load_mw")

    long_df["load_mw"] = pd.to_numeric(long_df["load_mw"], errors="coerce")
    long_df = long_df.dropna(subset=["load_mw"])
    if long_df.empty:
        return long_df

    offset = pd.to_timedelta(long_df["interval_label"].map(imap).to_numpy(),
                             unit="m")
    stamp = long_df["_date"] + offset
    if labelling == "end":
        stamp = stamp - pd.Timedelta(minutes=resolution)
    long_df["timestamp"] = stamp

    long_df = long_df.rename(columns={"Zone Substation": "substation"})
    long_df["substation"] = long_df["substation"].astype(str).str.strip()
    long_df["resolution_min"] = resolution
    long_df["source_file"] = source

    return (long_df[["substation", "timestamp", "load_mw", "Unit",
                     "resolution_min", "source_file"]]
            .sort_values(["substation", "timestamp"], ignore_index=True))


def apply_date_filter(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the configured study period, before any heavy work."""
    if df.empty or (not cfg.start_date and not cfg.end_date):
        return df
    ts = df["timestamp"]
    naive = ts.dt.tz_localize(None) if ts.dt.tz is not None else ts
    keep = pd.Series(True, index=df.index)
    if cfg.start_date:
        keep &= naive >= pd.Timestamp(cfg.start_date)
    if cfg.end_date:
        keep &= naive < pd.Timestamp(cfg.end_date) + pd.Timedelta(days=1)
    return df[keep].reset_index(drop=True)


def localise(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    """Optionally attach the local timezone (handles DST transitions)."""
    if cfg.timezone is None or df.empty:
        return df
    out = []
    for sub, g in df.groupby("substation", sort=False):
        g = g.copy()
        try:
            g["timestamp"] = g["timestamp"].dt.tz_localize(
                cfg.timezone, ambiguous="infer", nonexistent="shift_forward")
        except Exception:
            g["timestamp"] = g["timestamp"].dt.tz_localize(
                cfg.timezone, ambiguous=True, nonexistent="shift_forward")
        out.append(g)
    return pd.concat(out, ignore_index=True)


def run(cfg: Config) -> pd.DataFrame:
    """Non-streaming path: convert every file, then concatenate."""
    files = discover_files(cfg)
    parts, resolutions = [], {}

    for i, p in enumerate(files, 1):
        part = file_to_long(cfg, read_one(p), p.name)
        part = apply_date_filter(cfg, part)
        if part.empty:
            continue
        res = int(part["resolution_min"].iloc[0])
        resolutions[res] = resolutions.get(res, 0) + 1
        parts.append(part)
        if cfg.verbose and (i % 50 == 0 or i == len(files)):
            log(cfg, "ingest", f"[{i}/{len(files)}] files read")

    if not parts:
        raise ValueError("No data survived ingestion - check date filters.")

    df = pd.concat(parts, ignore_index=True)
    log(cfg, "resolution", ", ".join(f"{k}-min: {v} files"
                                     for k, v in sorted(resolutions.items())))
    log(cfg, "ingest", f"{len(df):,} observations, "
                       f"{df['substation'].nunique()} substations, "
                       f"{df['timestamp'].min():%Y-%m-%d} -> "
                       f"{df['timestamp'].max():%Y-%m-%d}")
    return localise(cfg, df)