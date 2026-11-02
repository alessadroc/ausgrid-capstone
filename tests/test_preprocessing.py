"""Correctness checks for the preprocessing logic.

These exist because the failures that matter here are silent: a lag that is
off by one, a 24:00 column landing on the wrong day, or a substation's history
fragmented at a file boundary all produce plausible-looking output and wrong
results. Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ausgrid_pipeline import Config, ingest  # noqa: E402


def make_wide(padded=True, resolution=15, labelling="end", days=3):
    """Build a small wide-format frame matching the Ausgrid export shape."""
    step = resolution
    mins = (list(range(step, 1441, step)) if labelling == "end"
            else list(range(0, 1440, step)))
    cols = [f"{m // 60:02d}:{m % 60:02d}" if padded else f"{m // 60}:{m % 60:02d}"
            for m in mins]
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    vals = np.arange(len(dates) * len(cols), dtype=float).reshape(len(dates), -1)
    df = pd.DataFrame(vals, columns=cols)
    df.insert(0, "Unit", "MW")
    df.insert(0, "Date", dates.strftime("%Y-%m-%d"))
    df.insert(0, "Zone Substation", "Test 66_11kV")
    df.insert(0, "year", 2024)
    return df


def test_2400_maps_to_end_of_same_day():
    """24:00 closes the labelled day, so as an interval START it is 23:45."""
    cfg = Config(verbose=False)
    out = ingest.file_to_long(cfg, make_wide(days=1), "test.csv")
    assert out["timestamp"].max() == pd.Timestamp("2024-01-01 23:45")
    assert out["timestamp"].min() == pd.Timestamp("2024-01-01 00:00")


def test_padded_and_unpadded_labels_agree():
    """'0:15' and '00:15' are the same interval and must produce the same series."""
    cfg = Config(verbose=False)
    a = ingest.file_to_long(cfg, make_wide(padded=True), "a.csv")
    b = ingest.file_to_long(cfg, make_wide(padded=False), "b.csv")
    pd.testing.assert_series_equal(a["timestamp"], b["timestamp"])


def test_resolution_detected_per_file():
    cfg = Config(verbose=False)
    assert ingest.file_to_long(cfg, make_wide(resolution=15), "a")["resolution_min"].iloc[0] == 15
    assert ingest.file_to_long(cfg, make_wide(resolution=30), "b")["resolution_min"].iloc[0] == 30


def test_column_order_is_ignored():
    """Header order is not chronological in the real export."""
    cfg = Config(verbose=False)
    wide = make_wide()
    shuffled = wide[list(wide.columns[:4]) +
                    list(np.random.permutation(wide.columns[4:]))]
    a = ingest.file_to_long(cfg, wide, "a").sort_values("timestamp").reset_index(drop=True)
    b = ingest.file_to_long(cfg, shuffled, "b").sort_values("timestamp").reset_index(drop=True)
    pd.testing.assert_series_equal(a["load_mw"], b["load_mw"])


def test_timestamps_are_regular_and_unique():
    cfg = Config(verbose=False)
    out = ingest.file_to_long(cfg, make_wide(days=5), "test.csv")
    gaps = out["timestamp"].diff().dropna().unique()
    assert len(gaps) == 1 and gaps[0] == pd.Timedelta(minutes=15)
    assert not out["timestamp"].duplicated().any()


@pytest.mark.parametrize("fmt", ["%Y-%m-%d", "%d/%m/%Y"])
def test_date_formats(fmt):
    cfg = Config(verbose=False)
    wide = make_wide()
    wide["Date"] = pd.to_datetime(wide["Date"]).dt.strftime(fmt)
    out = ingest.file_to_long(cfg, wide, "test.csv")
    assert out["timestamp"].dt.year.eq(2024).all()


def test_identifier_columns_are_case_insensitive():
    """Ausgrid vintages disagree on capitalisation: 'unit' vs 'Unit', 'year'
    vs 'Year'. An exact-match lookup silently rejected entire financial years."""
    cfg = Config(verbose=False)
    base = make_wide()
    expected = ingest.file_to_long(cfg, base, "base.csv")

    for renames in ({"Unit": "unit"}, {"year": "Year"},
                    {"Unit": "unit", "year": "Year"},
                    {"Unit": "UNIT", "year": "YEAR"}):
        variant = base.rename(columns=renames)
        out = ingest.file_to_long(cfg, variant, "variant.csv")
        pd.testing.assert_series_equal(
            out["timestamp"], expected["timestamp"],
            obj=f"timestamps with {renames}")


def test_missing_unit_column_is_tolerated():
    """Losing a whole financial year over an absent metadata column is worse
    than assuming MW, which is what the archive publishes throughout."""
    cfg = Config(verbose=False)
    out = ingest.file_to_long(cfg, make_wide().drop(columns=["Unit"]), "no_unit.csv")
    assert len(out) > 0
    assert (out["Unit"] == "MW").all()


def fy2024_wide(days=3):
    """End-labelled export whose closing column is headed '00:00', not '24:00',
    and sits last - the layout of every FY2024 file in the archive."""
    return make_wide(days=days).rename(columns={"24:00": "00:00"})


def test_trailing_0000_is_the_closing_interval():
    """Read as the opening of the day, the trailing 00:00 flipped whole files
    to start-labelling and stamped each day's closing reading 24h early."""
    wide = fy2024_wide()
    tcols = [c for c in wide.columns if ingest.TIME_PATTERN.match(c)]
    assert tcols[-1] == "00:00"
    minutes = list(ingest.interval_map(wide).values())
    assert ingest.detect_labelling(minutes) == "end"
    assert max(minutes) == 1440 and 0 not in minutes


def test_fy2024_layout_matches_standard_end_labelling():
    cfg = Config(verbose=False)
    expected = ingest.file_to_long(cfg, make_wide(), "fy23.csv")
    out = ingest.file_to_long(cfg, fy2024_wide(), "fy24.csv")
    pd.testing.assert_series_equal(out["timestamp"], expected["timestamp"])
    pd.testing.assert_series_equal(out["load_mw"], expected["load_mw"])
    # The closing reading of 1 Jan belongs to 1 Jan 23:45, not 1 Jan 00:00
    closing = make_wide().loc[0, "24:00"]
    row = out[out["load_mw"] == closing]
    assert row["timestamp"].tolist() == [pd.Timestamp("2024-01-01 23:45")]


def test_leading_0000_stays_start_labelled():
    """The inverse case: a genuine start-labelled export, 00:00 first."""
    wide = make_wide(labelling="start")
    tcols = [c for c in wide.columns if ingest.TIME_PATTERN.match(c)]
    assert tcols[0] == "00:00"
    assert ingest.detect_labelling(list(ingest.interval_map(wide).values())) == "start"


def test_trailing_0000_alongside_2400_is_not_remapped():
    """With a real 24:00 present, a trailing 00:00 is not the closing interval."""
    wide = make_wide()
    wide["00:00"] = 0.0
    assert 0 in ingest.interval_map(wide).values()


def test_trailing_0000_after_a_gap_is_not_remapped():
    """Only remap when the previous label is one interval before midnight."""
    wide = fy2024_wide().drop(columns=["23:45"])
    assert 0 in ingest.interval_map(wide).values()


def test_trailing_0000_respects_resolution():
    """At 30-minute resolution the label before the closing one is 23:30."""
    wide = make_wide(resolution=30).rename(columns={"24:00": "00:00"})
    minutes = list(ingest.interval_map(wide).values())
    assert max(minutes) == 1440 and ingest.detect_labelling(minutes) == "end"


def test_blank_day_placeholder_is_missing():
    """A blank day exported with a lone 0.0 is not a measurement."""
    cfg = Config(verbose=False)
    wide = make_wide(days=3)
    tcols = [c for c in wide.columns if ingest.TIME_PATTERN.match(c)]
    wide[tcols] += 1                     # make_wide's first reading is 0.0
    wide.loc[1, tcols] = np.nan
    wide.loc[1, "24:00"] = 0.0
    out = ingest.file_to_long(cfg, wide, "blank.csv")
    assert not (out["load_mw"] == 0).any()
    assert not out["timestamp"].dt.date.eq(pd.Timestamp("2024-01-02").date()).any()
    assert len(out) == 2 * len(tcols)


def test_genuine_zero_readings_are_kept():
    """Zeros inside an otherwise populated day are real readings."""
    cfg = Config(verbose=False)
    wide = make_wide(days=2)
    tcols = [c for c in wide.columns if ingest.TIME_PATTERN.match(c)]
    wide[tcols] += 1                     # make_wide's first reading is 0.0
    wide.loc[0, "12:00"] = 0.0
    wide.loc[1, ["12:00", "12:15"]] = 0.0
    out = ingest.file_to_long(cfg, wide, "zeros.csv")
    assert (out["load_mw"] == 0).sum() == 3


def test_start_and_end_labelling_align():
    """A start-labelled file and an end-labelled file covering the same day
    must produce timestamps on the same grid, not offset by one interval."""
    cfg = Config(verbose=False)
    end = ingest.file_to_long(cfg, make_wide(labelling="end"), "e.csv")
    start = ingest.file_to_long(cfg, make_wide(labelling="start"), "s.csv")
    assert end["timestamp"].min() == start["timestamp"].min()
    assert end["timestamp"].dt.hour.tolist() == start["timestamp"].dt.hour.tolist()