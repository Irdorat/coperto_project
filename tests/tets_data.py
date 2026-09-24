import pandas as pd

from src.data import restore_daily_calendar


def create_data_with_gap() -> pd.DataFrame:
    return pd.DataFrame({
        'restaurant_id': [1, 1],
        'date': pd.to_datetime([
            '2024-01-01',
            '2024-01-03',
        ]),
        'guests': [20, 25],
        'revenue': [1000.0, 1200.0],
    })


def test_missing_export_is_restored() -> None:
    data = create_data_with_gap()

    restored = restore_daily_calendar(data)

    missing_day = restored.loc[
        restored['date'].eq('2024-01-02')
    ].iloc[0]

    assert missing_day['data_status'] == 'missing_export'
    assert pd.isna(missing_day['guests'])
    assert pd.isna(missing_day['revenue'])


def test_confirmed_closure_receives_zero() -> None:
    data = create_data_with_gap()

    restored = restore_daily_calendar(
        data,
        closed_dates=['2024-01-02'],
    )

    closed_day = restored.loc[
        restored['date'].eq('2024-01-02')
    ].iloc[0]

    assert closed_day['data_status'] == 'closed'
    assert closed_day['guests'] == 0
    assert closed_day['revenue'] == 0


def test_dates_outside_operating_period_are_not_created() -> None:
    data = create_data_with_gap()

    restored = restore_daily_calendar(data)

    assert restored['date'].min() == pd.Timestamp('2024-01-01')
    assert restored['date'].max() == pd.Timestamp('2024-01-03')