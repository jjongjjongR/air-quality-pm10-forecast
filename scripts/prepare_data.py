from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts._bootstrap import ensure_src_on_path
except ImportError:  # running as a script: `python scripts/prepare_data.py`
    from _bootstrap import ensure_src_on_path


ensure_src_on_path()

from pm10_forecast.features import build_merged_df, make_train_test  # noqa: E402
from pm10_forecast.io import default_raw_paths, read_air_csv, read_weather_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare processed train/test CSVs from raw air/weather data.")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory containing raw CSVs.")
    p.add_argument(
        "--out-dir", type=Path, default=Path("data/processed"), help="Directory to write processed CSVs."
    )
    p.add_argument("--encoding", default="cp949", help="CSV encoding (default: cp949).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = default_raw_paths(args.raw_dir)

    air_24 = read_air_csv(paths.air_2024, encoding=args.encoding)
    air_25 = read_air_csv(paths.air_2025, encoding=args.encoding)
    weather_24 = read_weather_csv(paths.weather_2024, encoding=args.encoding)
    weather_25 = read_weather_csv(paths.weather_2025, encoding=args.encoding)

    df_24 = build_merged_df(weather_24, air_24)
    df_25 = build_merged_df(weather_25, air_25)

    ds = make_train_test(df_24, df_25)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ds.train_x.to_csv(args.out_dir / "train_x.csv", encoding=args.encoding, index=False)
    ds.train_y.to_frame("PM10_1").to_csv(args.out_dir / "train_y.csv", encoding=args.encoding, index=False)
    ds.test_x.to_csv(args.out_dir / "test_x.csv", encoding=args.encoding, index=False)
    ds.test_y.to_frame("PM10_1").to_csv(args.out_dir / "test_y.csv", encoding=args.encoding, index=False)

    print("Wrote:")
    print(f"- {args.out_dir / 'train_x.csv'}")
    print(f"- {args.out_dir / 'train_y.csv'}")
    print(f"- {args.out_dir / 'test_x.csv'}")
    print(f"- {args.out_dir / 'test_y.csv'}")


if __name__ == "__main__":
    main()
