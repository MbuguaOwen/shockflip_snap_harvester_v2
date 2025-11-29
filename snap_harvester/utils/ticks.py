from decimal import Decimal, ROUND_HALF_UP
from typing import Dict

# Canonical Binance tick sizes for supported symbols.
# Extend this mapping as new symbols are onboarded.
SYMBOL_TICK_SIZE: Dict[str, float] = {
    "BTCUSDT": 0.1,
    "ETHUSDT": 0.01,
    "SOLUSDT": 0.01,
}


def get_tick_size(symbol: str) -> float:
    """
    Return the tick size for a symbol (case-insensitive).

    Raises
    ------
    KeyError if the symbol is unknown.
    """
    sym = symbol.upper()
    if sym not in SYMBOL_TICK_SIZE:
        known = ", ".join(sorted(SYMBOL_TICK_SIZE))
        raise KeyError(f"Tick size for symbol '{symbol}' not configured. Known: {known}")
    return SYMBOL_TICK_SIZE[sym]


def price_to_tick(price: float, tick_size: float) -> int:
    """
    Map a float price to an integer tick index on the given tick grid.

    Decimal arithmetic is used to avoid floating-point artifacts and we use
    ROUND_HALF_UP to land on the nearest executable tick.
    """
    ts = Decimal(str(tick_size))
    if ts <= 0:
        raise ValueError(f"tick_size must be positive, got {tick_size!r}")
    p = Decimal(str(price))
    return int((p / ts).quantize(0, rounding=ROUND_HALF_UP))


def tick_to_price(tick: int, tick_size: float) -> float:
    """Convert an integer tick index back to a float price on the grid."""
    ts = Decimal(str(tick_size))
    if ts <= 0:
        raise ValueError(f"tick_size must be positive, got {tick_size!r}")
    return float(Decimal(int(tick)) * ts)
