import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline

from src.features import (
    FEATURE_COLUMNS,
    TARGET,
    make_features,
)

FORECAST_HORIZON = 7


def forecast_7_days(model: Pipeline, history: pd.DataFrame, start_date: str | pd.Timestamp) -> pd.DataFrame:
    # Forecast the next 7 days recursively.
    # Predictions from previous forecast days are used
    # to create lag and rolling features for later days.

    history = history[
        ['date', 'restaurant_id', TARGET]
    ].copy()

    if history.empty:
        raise ValueError('History is empty')

    if history['restaurant_id'].nunique() != 1:
        raise ValueError('Forecast expects exactly one restaurant')

    history['date'] = pd.to_datetime(history['date'])
    history = history.sort_values('date').reset_index(drop=True)

    original_start_date = history['date'].min()
    requested_start_date = pd.Timestamp(start_date).normalize()
    last_history_date = history['date'].max().normalize()

    if requested_start_date <= last_history_date:
        raise ValueError(
            'Forecast date must be later than the last historical date. '
            f'Last historical date: {last_history_date.date()}, '
            f'received: {requested_start_date.date()}.'
        )

    first_recursive_date = last_history_date + pd.Timedelta(days=1)
    forecast_end_date = (
        requested_start_date
        + pd.Timedelta(days=FORECAST_HORIZON - 1)
    )

    restaurant_id = history['restaurant_id'].iloc[0]
    results: list[dict[str, object]] = []

    for forecast_date in pd.date_range(
        start=first_recursive_date,
        end=forecast_end_date,
        freq='D',
    ):
        new_row = pd.DataFrame({
            'date': [forecast_date],
            'restaurant_id': [restaurant_id],
            TARGET: [np.nan],
        })

        history = pd.concat(
            [history, new_row],
            ignore_index=True,
        )

        featured = make_features(
            history,
            start_date=original_start_date,
        )

        current_row = featured.iloc[[-1]]
        features = current_row[FEATURE_COLUMNS]

        if features.isna().any().any():
            raise ValueError(
                'Not enough historical data to create forecast features'
            )

        prediction = float(model.predict(features)[0])
        prediction = max(0.0, prediction)

        history.loc[history.index[-1], TARGET] = prediction

        if forecast_date >= requested_start_date:
            results.append({
                'date': forecast_date,
                'prediction': prediction,
                'horizon': len(results) + 1,
            })

    return pd.DataFrame(results)

def calculate_metrics(actual: pd.Series, prediction: pd.Series) -> dict[str, float]:
    # Calculate MAE and MAPE.

    if actual.empty:
        raise ValueError('Cannot calculate metrics on empty data')

    if (actual == 0).any():
        raise ValueError('MAPE is undefined when actual contains zero')

    mae = mean_absolute_error(actual, prediction)
    mape = mean_absolute_percentage_error(actual, prediction) * 100

    return {
        'MAE': float(mae),
        'MAPE': float(mape),
    }


def seven_day_backtest(
    train_history: pd.DataFrame,
    test: pd.DataFrame,
    model: Pipeline | None = None,
) -> pd.DataFrame:
    # If model is None, seasonal baseline is evaluated.

    if len(test) % FORECAST_HORIZON != 0:
        raise ValueError('Test period must contain complete 7-day windows')

    history = train_history[['date', 'restaurant_id', TARGET]].copy()

    history['date'] = pd.to_datetime(history['date'])

    test = test.copy()
    test['date'] = pd.to_datetime(test['date'])

    history = history.sort_values('date').reset_index(drop=True)

    test = test.sort_values('date').reset_index(drop=True)

    results = []

    for start in range(0, len(test), FORECAST_HORIZON,):
        
        week = test.iloc[start:start + FORECAST_HORIZON].copy()

        week_start = week['date'].iloc[0]

        if model is None:
            forecast = seasonal_baseline_7_days(
                history=history,
                start_date=week_start,
            )
        else:
            forecast = forecast_7_days(
                model=model,
                history=history,
                start_date=week_start,
            )

        for i in range(FORECAST_HORIZON):
            results.append({
                'date': week.iloc[i]['date'],
                'actual': week.iloc[i][TARGET],
                'prediction': forecast.iloc[i]['prediction'],
                'horizon': i + 1,
            })

        actual_week = week[['date', 'restaurant_id', TARGET]].copy()

        history = pd.concat([history, actual_week], ignore_index=True)

    return pd.DataFrame(results)

def seasonal_baseline_7_days(history: pd.DataFrame, start_date: str | pd.Timestamp,) -> pd.DataFrame:
    if len(history) < FORECAST_HORIZON:
        raise ValueError('At least 7 historical days are required')

    start_date = pd.Timestamp(start_date)

    predictions = (
        history
        .sort_values('date')[TARGET]
        .tail(FORECAST_HORIZON)
        .to_numpy(dtype=float)
    )

    return pd.DataFrame({
        'date': pd.date_range(
            start=start_date,
            periods=FORECAST_HORIZON,
            freq='D',
        ),
        'prediction': predictions,
        'horizon': range(1, FORECAST_HORIZON + 1),
    })


def calculate_horizon_metrics(backtest_results: pd.DataFrame) -> pd.DataFrame:
    # Calculate metrics separately for D+1 ... D+7.

    results = []

    for horizon in range(1, FORECAST_HORIZON + 1):
        horizon_data = backtest_results[backtest_results['horizon'] == horizon]

        metrics = calculate_metrics(
            actual=horizon_data['actual'],
            prediction=horizon_data['prediction'],
        )

        results.append({
            'horizon': horizon,
            'MAE': metrics['MAE'],
            'MAPE': metrics['MAPE'],
        })

    return pd.DataFrame(results)