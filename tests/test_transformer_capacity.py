"""Capacity-matching tests.

The controlled comparison requires the two architectures to have comparable
parameter counts; otherwise a performance difference is ambiguous between
architecture and capacity. These tests pin that property so a later change to
hidden size or d_model cannot silently break it.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.transformer import (lstm_parameter_count,  # noqa: E402
                                transformer_parameter_count, suggest_d_model)

NWF, NTF = 9, 8


def test_default_configs_are_within_10_percent():
    lp = lstm_parameter_count(NWF, NTF, hidden=64, layers=1)
    tp = transformer_parameter_count(NWF, NTF, d_model=48, ff=96, layers=1)
    assert abs(tp - lp) / lp < 0.10, f"LSTM {lp:,} vs Transformer {tp:,}"


def test_counts_are_plausible():
    assert 20_000 < lstm_parameter_count(NWF, NTF, 64, 1) < 30_000
    assert 20_000 < transformer_parameter_count(NWF, NTF, 48, 96, 1) < 30_000


def test_suggest_d_model_picks_48_for_default_lstm():
    lp = lstm_parameter_count(NWF, NTF, 64, 1)
    assert suggest_d_model(NWF, NTF, lp, nhead=4) == 48


@pytest.mark.parametrize("d", [16, 32, 48, 64])
def test_counts_increase_with_width(d):
    a = transformer_parameter_count(NWF, NTF, d, d * 2, 1)
    b = transformer_parameter_count(NWF, NTF, d + 8, (d + 8) * 2, 1)
    assert b > a


def test_d_model_divisible_by_nhead():
    lp = lstm_parameter_count(NWF, NTF, 64, 1)
    for nhead in (2, 4, 8):
        assert suggest_d_model(NWF, NTF, lp, nhead=nhead) % nhead == 0