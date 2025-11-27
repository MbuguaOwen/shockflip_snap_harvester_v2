import pandas as pd
from pathlib import Path


def main() -> None:
    path = Path("data/ticks/ETHUSDT/ETHUSDT-ticks-2025-09.csv")
    print(f"Loading {path}...")
    df = pd.read_csv(path)

    # Timestamps are in MICROseconds (≈1.7e15). Convert µs -> ms.
    df["timestamp"] = df["timestamp"] // 1_000  # integer division keeps int64

    print("After conversion sample:")
    print(df["timestamp"].head())
    print("min:", df["timestamp"].min(), "max:", df["timestamp"].max())

    df.to_csv(path, index=False)
    print("Normalized and saved back to:", path)


if __name__ == "__main__":
    main()
