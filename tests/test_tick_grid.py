import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_loader import resample_ticks_to_bars
from snap_harvester.utils.ticks import get_tick_size, price_to_tick, tick_to_price


def test_price_tick_roundtrip_alignment() -> None:
    tick_size = get_tick_size("BTCUSDT")
    prices = [93630.1, 93630.1000000001, 93630.15]

    ticks = [price_to_tick(p, tick_size) for p in prices]
    rebuilt = [tick_to_price(t, tick_size) for t in ticks]

    decimals = len(str(tick_size).split(".")[-1])
    for price in rebuilt:
        ratio = price / tick_size
        assert math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-9)
        assert price == round(price, decimals)

    # Near-equal prices should land on the same tick after quantization
    assert ticks[0] == ticks[1]


def test_resample_aligns_to_tick_grid_and_sides() -> None:
    tick_size = get_tick_size("ETHUSDT")
    ts = pd.date_range("2025-01-01", periods=3, freq="500ms", tz="UTC")
    ticks = pd.DataFrame(
        {
            "ts": ts,
            "price": [2500.0000004, 2500.0099999, 2500.0199],
            "qty": [1.0, 2.0, 1.5],
            "is_buyer_maker": [False, True, False],
        }
    )

    bars = resample_ticks_to_bars(ticks, timeframe="1s", symbol="ETHUSDT", tick_size=tick_size)

    decimals = len(str(tick_size).split(".")[-1])
    for col in ("open", "high", "low", "close"):
        assert (bars[col].round(decimals) == bars[col]).all()

    first_bar = bars.iloc[0]
    assert first_bar["buy_qty"] == 1.0  # only the first tick (is_buyer_maker=False)
    assert first_bar["sell_qty"] == 2.0  # second tick is aggressive seller
