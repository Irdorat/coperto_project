import argparse
from pathlib import Path

import pandas as pd

from src.data import load_daily_data
from src.model import load_model
from src.validation import forecast_7_days
from src.model import load_metrics, load_model


DATA_PATH = Path('data/processed/daily.csv')
MODEL_PATH = Path('models/linear_regression.joblib')

METRICS_PATH = Path('models/validation_metrics.json')

RESTAURANT_ID = 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Forecast restaurant guests for the next 7 days.'
    )

    parser.add_argument(
        '--date',
        type=str,
        required=True,
        help='First forecast date in YYYY-MM-DD format',
    )

    parser.add_argument(
        '--restaurant',
        type=int,
        required=True,
        help='Restaurant ID',
    )

    return parser.parse_args()


def validate_restaurant(restaurant_id: int) -> None:
    if restaurant_id != RESTAURANT_ID:
        raise ValueError(
            f'Unknown restaurant: {restaurant_id}. Available restaurant: {RESTAURANT_ID}'
        )


def parse_date(date_string: str) -> pd.Timestamp:
    try:
        date = pd.to_datetime(date_string, format='%Y-%m-%d')
    except ValueError as error:
        raise ValueError('Invalid date format. Use YYYY-MM-DD.') from error

    return date


def validate_forecast_date(forecast_date: pd.Timestamp, history: pd.DataFrame) -> None:
    last_history_date = history['date'].max()

    if forecast_date <= last_history_date:
        raise ValueError(f'Forecast date must be later than the last historical date.\n'
                         f'Last historical date: {last_history_date.date()}, '
                         f'received: {forecast_date.date()}.'
                         )


def main() -> None:
    args = parse_arguments()

    validate_restaurant(args.restaurant)

    forecast_date = parse_date(args.date)

    history = load_daily_data(DATA_PATH)

    validate_forecast_date(forecast_date, history)

    model = load_model(MODEL_PATH)
    metrics = load_metrics(METRICS_PATH)

    forecast = forecast_7_days(
        model=model,
        history=history,
        start_date=forecast_date,
    )

    forecast['prediction'] = forecast['prediction'].round().astype(int)

    print()
    print(f'Restaurant: {args.restaurant}')
    print(f'Forecast from: {forecast_date.date()}')
    print(f'Validation quality, last 6 weeks:\nMAE={metrics["MAE"]:.2f},\nMAPE={metrics["MAPE"]:.2f}%')
    print()

    print(forecast[['date', 'prediction', 'horizon']].to_string(index=False))


if __name__ == '__main__':
    main()