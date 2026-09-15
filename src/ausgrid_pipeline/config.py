"""Central configuration.

Every tunable lives here so the pipeline can be re-run under different
assumptions without editing logic. Record the values you used in your
methodology chapter - this file IS your parameter table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ---- paths -----------------------------------------------------------
    input_dir: Path = Path("data")
    output_dir: Path = Path("processed")
    weather_csv: Path | None = None

    # ---- source data description ----------------------------------------
    # Columns that identify the observation rather than a time interval.
    id_columns: tuple[str, ...] = ("year", "Zone Substation", "Date", "Unit")

    # Native resolution of the export, in minutes.
    native_interval_minutes: int = 15

    # Ausgrid labels each column with the CLOSING time of the interval:
    # the "00:15" column covers 00:00-00:15, and "24:00" covers 23:45-00:00
    # of the following day. A day therefore has 96 columns running 00:15
    # through 24:00, with no 00:00 column.
    #   "end"   -> labels are closing times (Ausgrid convention)
    #   "start" -> labels are opening times
    interval_label: str = "end"

    # Resolution and labelling are DETECTED per file by default, because the
    # published archive mixes 15-minute and 30-minute vintages and mixes
    # zero-padded with non-padded time labels. Set these to True only to
    # override detection with the values above.
    force_resolution: bool = False
    force_labelling: bool = False

    # ---- date range restriction ------------------------------------------
    # Restrict to a study period at ingestion time. Applied before any heavy
    # work, so it cuts both memory and runtime proportionally.
    start_date: str | None = None        # e.g. "2023-01-01"
    end_date: str | None = None          # e.g. "2025-12-31"

    # ---- temporal standardisation ---------------------------------------
    target_freq: str = "1h"
    timezone: str | None = None          # e.g. "Australia/Sydney"; None = naive

    # ---- cleaning --------------------------------------------------------
    short_gap_hours: int = 3             # linear interpolation limit
    outlier_window_hours: int = 168      # rolling window for MAD test
    outlier_threshold: float = 5.0       # robust sigmas
    min_completeness_pct: float = 80.0   # substations below this are excluded

    # ---- feature engineering --------------------------------------------
    lags: tuple[int, ...] = (1, 2, 3, 24, 25, 48, 168)
    rolling_windows: tuple[int, ...] = (24, 168)
    cooling_base_c: float = 22.0
    heating_base_c: float = 18.0

    # ---- splitting -------------------------------------------------------
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    # Explicit cut dates override the fractions. Set these once you have
    # chosen a study period so the split is reproducible and quotable.
    split_train_end: str | None = None   # e.g. "2025-03-31"
    split_val_end: str | None = None     # e.g. "2025-09-30"

    # ---- sequence windows (for LSTM / Transformer) -----------------------
    lookback_hours: int = 168            # one week of history per sample
    horizons: tuple[int, ...] = (1, 24)  # the two horizons in the proposal

    # ---- runtime ---------------------------------------------------------
    max_files: int | None = None         # cap for quick test runs
    stream: bool = False                 # per-file processing, bounded memory
    float32: bool = True                 # halve float memory footprint
    verbose: bool = True

    def resolve(self) -> "Config":
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.weather_csv is not None:
            self.weather_csv = Path(self.weather_csv)
        return self


def log(cfg: Config, stage: str, message: str) -> None:
    if cfg.verbose:
        print(f"[{stage:<10}] {message}", flush=True)
