from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import json

from src.features import (
    FEATURE_COLUMNS,
    TARGET,
)


def create_model() -> Pipeline:
    #Create Linear Regression pipeline.

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression()),
    ])

    return model


def train_model(train_data: pd.DataFrame) -> Pipeline:
    # Train Linear Regression model.

    required_columns = set(FEATURE_COLUMNS + [TARGET])
    missing_columns = required_columns - set(train_data.columns)


    if missing_columns:
        raise ValueError(f'Missing training columns: {sorted(missing_columns)}')

    if train_data.empty:
        raise ValueError('Training data is empty')

    X = train_data[FEATURE_COLUMNS]

    y = train_data[TARGET]

    if X.isna().any().any() or y.isna().any():
        raise ValueError('Training data contains missing values')

    model = create_model()

    model.fit(X, y)

    return model


def save_model(model: Pipeline, path: str | Path) -> None:
    # Save trained model.

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)


def load_model(path: str | Path) -> Pipeline:
    # Load trained model.

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    model = joblib.load(path)

    return model

def save_metrics(metrics: dict[str, float], path: str | Path) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(metrics, indent=2),
        encoding='utf-8',
    )


def load_metrics(path: str | Path) -> dict[str, float]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f'Metrics file not found: {path}'
        )

    return json.loads(
        path.read_text(encoding='utf-8')
    )