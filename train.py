from pathlib import Path

from src.data import load_daily_data
from src.features import prepare_model_data
from src.model import (
    save_metrics,
    save_model,
    train_model,
)
from src.validation import (
    calculate_metrics,
    seven_day_backtest,
)


DATA_PATH = Path('data/processed/daily.csv')
MODEL_PATH = Path(
    'models/linear_regression.joblib'
)
METRICS_PATH = Path(
    'models/validation_metrics.json'
)

VALIDATION_DAYS = 6 * 7


def main() -> None:
    print('Loading data...')

    daily = load_daily_data(DATA_PATH)

    print(
        f'History: {daily["date"].min().date()} '
        f'— {daily["date"].max().date()}'
    )

    if len(daily) <= VALIDATION_DAYS + 28:
        raise ValueError(
            'Not enough history for training '
            'and six-week validation'
        )

    print('Evaluating on the last 6 weeks...')

    train_history = (
        daily
        .iloc[:-VALIDATION_DAYS]
        .copy()
    )

    validation = (
        daily
        .iloc[-VALIDATION_DAYS:]
        .copy()
    )

    validation_train_data = prepare_model_data(
        train_history
    )

    validation_model = train_model(
        validation_train_data
    )

    validation_predictions = seven_day_backtest(
        train_history=train_history,
        test=validation,
        model=validation_model,
    )

    validation_metrics = calculate_metrics(
        actual=validation_predictions['actual'],
        prediction=validation_predictions['prediction'],
    )

    save_metrics(
        validation_metrics,
        METRICS_PATH,
    )

    print(
        f'Validation MAE: '
        f'{validation_metrics["MAE"]:.2f}'
    )

    print(
        f'Validation MAPE: '
        f'{validation_metrics["MAPE"]:.2f}%'
    )

    print('Creating final training features...')

    model_data = prepare_model_data(daily)

    print(f'Training rows: {len(model_data)}')
    print('Training final Linear Regression...')

    final_model = train_model(model_data)

    save_model(
        final_model,
        MODEL_PATH,
    )

    print(f'Model saved to: {MODEL_PATH}')
    print(f'Metrics saved to: {METRICS_PATH}')


if __name__ == '__main__':
    main()