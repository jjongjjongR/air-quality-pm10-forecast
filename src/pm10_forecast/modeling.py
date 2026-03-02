from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression


def create_model(name: str, random_state: int = 42):
    key = name.lower().strip()
    if key in {"lr", "linear", "linear_regression"}:
        return LinearRegression()
    if key in {"rf", "random_forest", "randomforest"}:
        return RandomForestRegressor(random_state=random_state)
    if key in {"gbr", "gb", "gradient_boosting", "gradientboosting"}:
        return GradientBoostingRegressor(random_state=random_state)
    if key in {"xgb", "xgboost"}:
        try:
            from xgboost import XGBRegressor  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "xgboost is not installed. Install it (pip install xgboost) or choose another model."
            ) from exc
        return XGBRegressor(
            random_state=random_state,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=1.0,
        )
    raise ValueError(f"Unknown model name: {name}")

