import pandas as pd
from pathlib import Path


def main() -> None:
    path = Path("data/ticks/ETHUSDT/ETHUSDT-ticks-2025-09.csv")
    print(f"Sampling {path}...")
    df = pd.read_csv(path, nrows=10)
    print(df.dtypes)
    print(df.head())

    ts = df["timestamp"]
    print("min:", ts.min(), "max:", ts.max())


if __name__ == "__main__":
    main()
