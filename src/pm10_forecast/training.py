from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


@dataclass(frozen=True)
class Metrics:
    mse: float
    r2: float


def evaluate(y_true, y_pred) -> Metrics:
    return Metrics(
        mse=float(mean_squared_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
    )


def predict(model, x):
    y_pred = model.predict(x)
    return np.asarray(y_pred).reshape(-1)


def save_model(model, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)


def load_model(path: Path):
    return joblib.load(path)

