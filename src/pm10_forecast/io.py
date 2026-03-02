from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd

from .constants import DEFAULT_ENCODING, TIME_COLUMN


@dataclass(frozen=True)
class RawPaths:
    air_2024: Path
    air_2025: Path
    weather_2024: Path
    weather_2025: Path


def _parse_air_time(value: object, fmt: str) -> pd.Timestamp:
    """
    Parse 측정일시 like YYYYMMDDHH.

    Some rows may contain an hour like 24, which is not valid for strptime.
    In that case we convert ...24 -> ...00 and add 1 day (notebook logic).
    """

    text = str(value)
    try:
        return pd.to_datetime(text, format=fmt)
    except Exception:
        fixed = text[:-2] + "00"
        return pd.to_datetime(fixed, format=fmt) + timedelta(days=1)


def read_air_csv(path: Path, encoding: str = DEFAULT_ENCODING) -> pd.DataFrame:
    df = pd.read_csv(path, encoding=encoding)
    df[TIME_COLUMN] = df["측정일시"].map(lambda x: _parse_air_time(x, "%Y%m%d%H"))
    return df


def read_weather_csv(path: Path, encoding: str = DEFAULT_ENCODING) -> pd.DataFrame:
    df = pd.read_csv(path, encoding=encoding)
    df[TIME_COLUMN] = pd.to_datetime(df["일시"], format="%Y-%m-%d %H:%M")
    return df


def default_raw_paths(raw_dir: Path) -> RawPaths:
    return RawPaths(
        air_2024=raw_dir / "air_2024.csv",
        air_2025=raw_dir / "air_2025.csv",
        weather_2024=raw_dir / "weather_2024.csv",
        weather_2025=raw_dir / "weather_2025.csv",
    )

