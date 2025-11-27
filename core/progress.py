import sys
import time
from typing import Iterable, Iterator, Optional


class _NoopProgress:
    def update(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass


class Progress:
    def __init__(
        self,
        total: Optional[int],
        desc: str = "",
        width: int = 30,
        min_interval: float = 0.1,
        file = sys.stdout,
    ) -> None:
        self.total = total or 0
        self.desc = desc
        self.width = max(10, int(width))
        self.min_interval = max(0.05, float(min_interval))
        self.file = file
        self.count = 0
        self._last_print = 0.0

    def _render(self) -> None:
        if self.total <= 0:
            msg = f"{self.desc} {self.count}"
            print(f"\r{msg}", end="", file=self.file, flush=True)
            return
        ratio = min(1.0, self.count / self.total)
        filled = int(ratio * self.width)
        if filled <= 0:
            bar = ">" + " " * (self.width - 1)
        elif filled >= self.width:
            bar = "=" * self.width
        else:
            bar = "=" * (filled - 1) + ">" + " " * (self.width - filled)
        pct = f"{ratio*100:5.1f}%"
        msg = f"{self.desc} [{bar}] {pct} ({self.count}/{self.total})"
        print(f"\r{msg}", end="", file=self.file, flush=True)

    def update(self, n: int = 1) -> None:
        self.count += int(n)
        t = time.time()
        if (t - self._last_print) >= self.min_interval or (self.total and self.count >= self.total):
            self._render()
            self._last_print = t

    def close(self) -> None:
        # Ensure we end with a newline so subsequent prints don't overwrite the bar
        print("", file=self.file)


def get_progress(enabled: bool, total: Optional[int], desc: str = "") -> Progress | _NoopProgress:
    if not enabled:
        return _NoopProgress()
    return Progress(total=total, desc=desc)


def iter_with_progress(iterable: Iterable, total: Optional[int], desc: str = "", enabled: bool = True) -> Iterator:
    p = get_progress(enabled, total, desc)
    try:
        for item in iterable:
            p.update(1)
            yield item
    finally:
        p.close()

