import holidays
import pandas as pd


FEATURE_COLUMNS = [
    'day_of_week',
    'month',
    'is_weekend',
    'is_holiday',
    'days_since_start',
    'guests_lag_1',
    'guests_lag_7',
    'guests_lag_14',
    'guests_lag_28',
    'guests_mean_7',
    'guests_mean_14',
    'guests_mean_28',
]

TARGET = 'guests'


def make_features(
    df: pd.DataFrame,
    start_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    # Create features for Linear Regression.

    result = df.copy()

    if result['restaurant_id'].nunique() != 1:
        raise ValueError('Make_features expects exactly one restaurant')

    result['date'] = pd.to_datetime(result['date'])

    result = result.sort_values('date').reset_index(drop=True)

    if start_date is None:
        start_date = result['date'].min()

    start_date = pd.Timestamp(start_date)

    # Calendar features
    result['day_of_week'] = (result['date'].dt.dayofweek)

    result['month'] = (
        result['date'].dt.month
    )

    result['is_weekend'] = (
        result['day_of_week']
        .isin([5, 6])
        .astype(int)
    )

    result['days_since_start'] = (
        result['date'] - start_date
    ).dt.days

    # Holidays
    years = result['date'].dt.year.unique().tolist()
    us_holidays = holidays.US(years=years)
    holiday_dates = set(us_holidays.keys())

    result['is_holiday'] = (
        result['date']
        .dt.date
        .isin(holiday_dates)
        .astype(int)
    )

    # Lag features
    result['guests_lag_1'] = (
        result['guests'].shift(1)
    )

    result['guests_lag_7'] = (
        result['guests'].shift(7)
    )

    result['guests_lag_14'] = (
        result['guests'].shift(14)
    )

    result['guests_lag_28'] = (
        result['guests'].shift(28)
    )

    # Rolling features
    result['guests_mean_7'] = (
        result['guests']
        .shift(1)
        .rolling(7)
        .mean()
    )

    result['guests_mean_14'] = (
        result['guests']
        .shift(1)
        .rolling(14)
        .mean()
    )

    result['guests_mean_28'] = (
        result['guests']
        .shift(1)
        .rolling(28)
        .mean()
    )

    return result


def prepare_model_data(df: pd.DataFrame, 
                       start_date: pd.Timestamp | None = None,) -> pd.DataFrame:
    
    featured = make_features(df, start_date=start_date)

    model_data = featured.dropna(subset=FEATURE_COLUMNS + [TARGET]).copy()

    return model_data.reset_index(drop=True)