from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from scripts._bootstrap import ensure_src_on_path
except ImportError:  # running as a script: `python scripts/train.py`
    from _bootstrap import ensure_src_on_path


ensure_src_on_path()

from pm10_forecast.features import build_merged_df, make_train_test  # noqa: E402
from pm10_forecast.io import default_raw_paths, read_air_csv, read_weather_csv  # noqa: E402
from pm10_forecast.modeling import create_model  # noqa: E402
from pm10_forecast.training import evaluate, predict, save_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a PM10 next-hour model (2024 train, 2025 test).")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory containing raw CSVs.")
    p.add_argument("--encoding", default="cp949", help="CSV encoding (default: cp949).")
    p.add_argument("--model", default="rf", help="Model: lr | rf | gbr | xgb")
    p.add_argument("--out-model", type=Path, default=Path("artifacts/model.joblib"), help="Model output path.")
    p.add_argument("--out-metrics", type=Path, default=Path("artifacts/metrics.json"), help="Metrics output path.")
    p.add_argument(
        "--out-pred",
        type=Path,
        default=Path("artifacts/predictions_2025.csv"),
        help="Prediction CSV output path.",
    )
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

    model = create_model(args.model)
    model.fit(ds.train_x, ds.train_y.values.ravel())

    y_pred = predict(model, ds.test_x)
    metrics = evaluate(ds.test_y, y_pred)

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, args.out_model)

    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.out_metrics.write_text(
        json.dumps({"mse": metrics.mse, "r2": metrics.r2, "model": args.model}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    args.out_pred.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": ds.test_y.values.reshape(-1), "y_pred": y_pred}).to_csv(
        args.out_pred, index=False, encoding="utf-8"
    )

    print(f"Model: {args.model}")
    print(f"MSE: {metrics.mse:.6f}")
    print(f"R2:  {metrics.r2:.6f}")
    print(f"Saved model:   {args.out_model}")
    print(f"Saved metrics: {args.out_metrics}")
    print(f"Saved preds:   {args.out_pred}")


if __name__ == "__main__":
    main()
