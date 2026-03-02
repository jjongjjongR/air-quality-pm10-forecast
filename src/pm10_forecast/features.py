from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .constants import (
    DROP_COLS,
    PM10_COLUMN,
    PM10_LAG_COLUMN,
    RAIN_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
)


@dataclass(frozen=True)
class Dataset:
    train_x: pd.DataFrame
    train_y: pd.Series
    test_x: pd.DataFrame
    test_y: pd.Series


def build_merged_df(weather: pd.DataFrame, air: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(weather, air, on=TIME_COLUMN)
    df = df.drop(columns=DROP_COLS, errors="ignore").sort_values(by=TIME_COLUMN, ascending=True)

    if RAIN_COLUMN in df.columns:
        df[RAIN_COLUMN] = df[RAIN_COLUMN].fillna(0)

    df = df.ffill().bfill()

    df["month"] = df[TIME_COLUMN].dt.month
    df["day"] = df[TIME_COLUMN].dt.day
    df["hour"] = df[TIME_COLUMN].dt.hour
    df = df.drop(columns=[TIME_COLUMN])

    if PM10_COLUMN not in df.columns:
        raise KeyError(f"Missing required column: {PM10_COLUMN}")

    df[PM10_LAG_COLUMN] = df[PM10_COLUMN].shift(24)
    df[TARGET_COLUMN] = df[PM10_COLUMN].shift(-1)
    df = df.dropna(axis=0)

    return df


def make_train_test(df_2024: pd.DataFrame, df_2025: pd.DataFrame) -> Dataset:
    train_x = df_2024.drop(columns=[TARGET_COLUMN])
    train_y = df_2024[TARGET_COLUMN]
    test_x = df_2025.drop(columns=[TARGET_COLUMN])
    test_y = df_2025[TARGET_COLUMN]

    combined = pd.concat([train_x, test_x], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, drop_first=False)
    train_x_enc = combined.iloc[: len(train_x), :].copy()
    test_x_enc = combined.iloc[len(train_x) :, :].copy()

    return Dataset(train_x=train_x_enc, train_y=train_y, test_x=test_x_enc, test_y=test_y)

