import pandas as pd

from src.features import prepare_model_data
from src.model import train_model
from src.validation import forecast_7_days


def test_forecast_from_arbitrary_future_date() -> None:
    history = pd.DataFrame({
        'date': pd.date_range(
            start='2023-01-01',
            periods=100,
            freq='D',
        ),
        'restaurant_id': 1,
        'guests': [20 + index % 7 for index in range(100)],
    })

    training_data = prepare_model_data(history)
    model = train_model(training_data)

    requested_date = pd.Timestamp('2023-06-01')

    forecast = forecast_7_days(
        model=model,
        history=history,
        start_date=requested_date,
    )

    assert len(forecast) == 7
    assert forecast['date'].iloc[0] == requested_date
    assert forecast['date'].iloc[-1] == requested_date + pd.Timedelta(days=6)
    assert forecast['horizon'].tolist() == list(range(1, 8))
    assert (forecast['prediction'] >= 0).all()